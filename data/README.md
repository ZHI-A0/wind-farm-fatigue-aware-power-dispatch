# Dataset

## Contents

`raw/wind_farms.mat` contains the measurements for both wind farms.

The two NPZ files contain the processed inputs for the two-farm experiments. `manifest.json` and `SHA256SUMS` record file information. WF1 maps to Farm A and WF2 to Farm B.

Each farm has 100 turbines and 2000 samples at 1-second intervals. The original MAT time values are 1 through 2000. NPZ arrays use zero-based indexing; the temporal calibration slice is `[0:1000]`, and its evaluation slice is `[1000:2000]`. Transfer evaluation uses the target farm slice `[0:1000]`.

## NPZ schema

All NPZ arrays are float64. Turbine channels have shape `(2000, 100)`; `p_grid` has shape `(2000,)`.

| Key | Description | Unit | Original MAT channel |
|---|---|---|---|
| `v` | Inflow wind speed | m/s | `inputs[:, 1]` |
| `pref` | Turbine power reference | MW | `inputs[:, 0] / 1e6` |
| `p_out` | Generated power | MW | `outputs[:, 2] / 1e6` |
| `pout` | Compatibility alias of `p_out` | MW | Same as `p_out` |
| `tshaft` | Main-shaft torque channel | N m | `outputs[:, 0]` |
| `ft` | Tower/thrust load channel | N | `outputs[:, 1]` |
| `pitch` | Blade pitch | deg | `states[:, 0]` |
| `omega_r` | Low-speed shaft speed | rad/s | `states[:, 1]` |
| `omega_f` | High-speed shaft speed | rad/s | `states[:, 2]` |
| `p_grid` | Farm power instruction | MW | Sum of `pref` across turbines |

Turbines retain their original MAT order. No samples or turbines are removed. Only the two power channels are rescaled from W to MW.

## Conversion

From the repository root:

```bash
python tools/prepare_data.py --rebuild --output-dir data/rebuilt
```

The conversion keeps turbine order, rescales power from W to MW, and writes one compressed NPZ archive per farm.

## Data terms

The MIT license in the repository root applies to the software and project-authored documentation. Dataset rights are separate; this repository grants no additional dataset license.
