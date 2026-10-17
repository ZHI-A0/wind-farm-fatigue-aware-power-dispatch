# Online Fatigue-Aware Power Dispatch

Code for wind-farm load modeling, online active-power allocation, and fatigue-index evaluation. Inputs include wind speed, power commands, historical load channels, and turbine states; outputs include turbine power allocations, reconstructed loads, and fatigue metrics.

Associated paper: **Online Fatigue-Aware Power Dispatch for Wind Farms with Hard Feasibility Guarantees**, FLINS–ISKE 2026.

[Paper PDF](paper/Published_FLINS2026_OnlinePowerDispatch.pdf) · [DOI](https://doi.org/10.1007/978-981-92-2369-5_36) · [中文说明](README.zh-CN.md)

## Installation

Use Python 3.12. Dependencies are listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run experiments

Run all four scenarios:

```bash
python run_experiments.py
```

The default configuration is `scripts/config.yaml`. The runner fits models, runs `mean`, `proportional`, and `optimized`, and writes models, schedules, metrics, and logs to `outputs/run_<timestamp>/`.

Run a single scenario:

```bash
python run_experiments.py --case wf1_temporal
python run_experiments.py --case wf2_temporal
python run_experiments.py --case wf1_to_wf2
python run_experiments.py --case wf2_to_wf1
```

Specify a configuration, strategies, seed, and output directory:

```bash
python run_experiments.py \
  --config scripts/config.yaml \
  --strategies mean,proportional,optimized \
  --seed 42 \
  --out-root outputs/my_experiment
```

Use the single-experiment entry point for custom NPZ inputs:

```bash
# Same farm: first 1000 steps for calibration, last 1000 for dispatch
python scripts/main.py \
  --data-npz data/data_wf1.npz \
  --out-root outputs --run-name wf1_temporal --skip-snapshot

# Transfer: calibrate on WF1, dispatch on the first 1000 steps of WF2
python scripts/main.py \
  --fit-data-npz data/data_wf1.npz \
  --data-npz data/data_wf2.npz --eval-start 0 \
  --out-root outputs --run-name wf1_to_wf2 --skip-snapshot
```

## Dataset contents and formats

Each of the two farms contains **100 turbines and 2000 samples at 1 Hz**. WF1 and WF2 correspond to Farm A and Farm B in the paper. Original time values run from 1 to 2000 seconds; NPZ arrays preserve time order and use zero-based indexing.

| File | Format | Contents |
|---|---|---|
| `data/raw/wind_farms.mat` | MATLAB MAT | Original time, input, output, and state channels for both farms |
| `data/data_wf1.npz` | Compressed NumPy archive | WF1 wind, power, load, and state arrays |
| `data/data_wf2.npz` | Compressed NumPy archive | WF2 wind, power, load, and state arrays |

### Original MAT structure

The root variable `data_TS_WF` contains `WF_1` and `WF_2`. Each farm has a `WT` collection with 100 turbines. Each turbine contains the following fields; column numbers below are one-based MATLAB indices.

| Field | Size per turbine | Contents |
|---|---|---|
| `time` | 2000 time values | Time in seconds |
| `inputs` | `(2000, 2)` | Column 1: power reference Pref (W); column 2: wind speed (m/s) |
| `outputs` | `(2000, 3)` | Column 1: shaft torque (N m); column 2: tower thrust (N); column 3: generated power (W) |
| `states` | `(2000, 3)` | Column 1: pitch (deg); column 2: low-speed shaft speed (rad/s); column 3: high-speed shaft speed (rad/s) |

### NPZ fields

Arrays are `float64`. Two-dimensional arrays are arranged as **time × turbine**, with shape `(2000, 100)`; `p_grid` has shape `(2000,)`. Power channels are converted from W to MW by dividing by `1e6`. Other channels retain their original units.

| Key | Meaning | Unit | Shape |
|---|---|---|---|
| `v` | Turbine inflow wind speed | m/s | `(2000, 100)` |
| `pref` | Turbine power reference | MW | `(2000, 100)` |
| `p_out` | Generated turbine power | MW | `(2000, 100)` |
| `pout` | Compatibility alias of `p_out` | MW | `(2000, 100)` |
| `tshaft` | Main-shaft torque | N m | `(2000, 100)` |
| `ft` | Tower/thrust load channel | N | `(2000, 100)` |
| `pitch` | Blade pitch angle | deg | `(2000, 100)` |
| `omega_r` | Low-speed shaft speed | rad/s | `(2000, 100)` |
| `omega_f` | High-speed shaft speed | rad/s | `(2000, 100)` |
| `p_grid` | Farm power instruction: sum of turbine `pref` | MW | `(2000,)` |

Load an NPZ file:

```python
import numpy as np

with np.load("data/data_wf1.npz", allow_pickle=False) as data:
    wind_speed = data["v"]        # [time, turbine]
    turbine_1_power = data["pref"][:, 0]
    farm_power = data["p_grid"]
```

Convert the original MAT to NPZ:

```bash
python tools/prepare_data.py --rebuild --output-dir data/rebuilt
```

See [data/README.md](data/README.md) for dataset files and channel mapping.

## Scenarios and strategies

The default configuration uses 1000 calibration steps and 1000 dispatch steps.

| `--case` | Calibration | Dispatch and evaluation |
|---|---|---|
| `wf1_temporal` | WF1, first 1000 steps | WF1, last 1000 steps |
| `wf2_temporal` | WF2, first 1000 steps | WF2, last 1000 steps |
| `wf1_to_wf2` | WF1, first 1000 steps | WF2, first 1000 steps |
| `wf2_to_wf1` | WF2, first 1000 steps | WF1, first 1000 steps |

| Strategy | Meaning |
|---|---|
| `mean` | Uniform allocation followed by feasible-set projection |
| `proportional` | Wind-speed-cubed allocation followed by projection |
| `optimized` | Fatigue-guided allocation, smoothing, dual updates, and projection |
| `optimized_legacy` | Optional blend of historical power references, wind-speed allocation, and smoothing |

Fatigue values are comparative indices rather than calibrated lifetime consumption.

## Outputs

Each scenario produces:

| Path | Contents |
|---|---|
| `01_offline_fit/` | Load model, state mapper, and fitting statistics |
| `02_online_and_offline/schedule_<strategy>.npz` | Power, predicted loads, online increments, and per-step timings |
| `02_online_and_offline/online_metrics_<strategy>.json` | Strategy-level dispatch and fatigue statistics |
| `final_metrics.json` | Configuration, fitting statistics, online and offline metrics |
| `final_table.csv` | Strategy metric summary |

Schedule arrays `p_sched`, `torque_hat`, `thrust_hat`, and `damage_step` have shape `(dispatch steps, turbines)` and represent power (MW), shaft torque (N m), tower load (N), and online damage increments. `solve_ms` records per-step latency in milliseconds. Batch runs also write `experiment_manifest.json` and `logs/`.

## Layout

```text
run_experiments.py    Batch experiment runner
scripts/             Scientific modules and configuration
tools/              MAT-to-NPZ data tools
data/               MAT and NPZ datasets
paper/              Publication PDF
tests/              Core unit tests
requirements.txt    Python dependencies
```

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{zhao2027online,
  author = {Zhao, Jiangtao and Zhang, Lin and He, Tiantian and Zhang, Tianyi and Wang, Ximeng and Wang, Bin and Hong, Feng},
  title = {Online Fatigue-Aware Power Dispatch for Wind Farms with Hard Feasibility Guarantees},
  booktitle = {Machine Learning and Knowledge Engineering for Decision Making},
  year = {2027},
  series = {Lecture Notes in Computer Science},
  volume = {16752},
  pages = {586--603},
  publisher = {Springer Nature Singapore},
  doi = {10.1007/978-981-92-2369-5_36},
  url = {https://doi.org/10.1007/978-981-92-2369-5_36}
}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).

## License

Code and project-authored documentation use the [MIT License](LICENSE). The data and publication PDF retain their respective terms.
