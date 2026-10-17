# 风电场疲劳感知在线功率调度

本项目提供风电场载荷建模、在线有功功率分配及疲劳指标评估代码。输入为风速、功率指令、历史载荷和机组状态，输出为各机组功率分配、预测载荷和疲劳指标。

论文：**Online Fatigue-Aware Power Dispatch for Wind Farms with Hard Feasibility Guarantees**，FLINS–ISKE 2026。

[论文 PDF](paper/Published_FLINS2026_OnlinePowerDispatch.pdf) · [论文 DOI](https://doi.org/10.1007/978-981-92-2369-5_36) · [English README](README.md)

## 安装

Python 3.12，依赖版本见 `requirements.txt`。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows 使用 `.venv\Scripts\activate` 激活虚拟环境。

## 运行实验

运行全部四个场景：

```bash
python run_experiments.py
```

默认使用 `scripts/config.yaml`，运行 `mean`、`proportional`、`optimized` 三种策略，并将模型、调度轨迹、统计指标和日志保存到 `outputs/run_<运行时间>/`。

运行单个场景：

```bash
python run_experiments.py --case wf1_temporal
python run_experiments.py --case wf2_temporal
python run_experiments.py --case wf1_to_wf2
python run_experiments.py --case wf2_to_wf1
```

指定配置、策略、随机种子和输出目录：

```bash
python run_experiments.py \
  --config scripts/config.yaml \
  --strategies mean,proportional,optimized \
  --seed 42 \
  --out-root outputs/my_experiment
```

也可以直接调用单次实验入口，指定自己的 NPZ 文件：

```bash
# 同风场：前1000步标定，后1000步调度
python scripts/main.py \
  --data-npz data/data_wf1.npz \
  --out-root outputs --run-name wf1_temporal --skip-snapshot

# 跨风场：WF1标定，WF2前1000步调度
python scripts/main.py \
  --fit-data-npz data/data_wf1.npz \
  --data-npz data/data_wf2.npz --eval-start 0 \
  --out-root outputs --run-name wf1_to_wf2 --skip-snapshot
```

## 数据集内容与格式

包含两个风场，每个风场 **100 台机组、2000 个采样点、1 Hz 采样频率**。WF1、WF2 分别对应论文中的 Farm A、Farm B。原始时间为第1至第2000秒；NPZ 沿用时间顺序，使用从0开始的数组索引。

| 文件 | 格式 | 内容 |
|---|---|---|
| `data/raw/wind_farms.mat` | MATLAB MAT | 两个风场的原始时间、输入、输出和状态数据 |
| `data/data_wf1.npz` | NumPy 压缩归档 | WF1 的风速、功率、载荷和状态数组 |
| `data/data_wf2.npz` | NumPy 压缩归档 | WF2 的风速、功率、载荷和状态数组 |

### 原始 MAT 结构

根变量为 `data_TS_WF`，包含 `WF_1` 和 `WF_2`；每个风场的 `WT` 包含100台机组。每台机组包含以下字段，列序按原始MAT的一基列号说明：

| 字段 | 每台机组的数据规模 | 内容 |
|---|---|---|
| `time` | 2000个时间点 | 时间，单位s |
| `inputs` | `(2000, 2)` | 第1列功率指令 Pref（W）；第2列风速 v（m/s） |
| `outputs` | `(2000, 3)` | 第1列主轴扭矩 Tshaft（N·m）；第2列塔架推力 Ft（N）；第3列实际功率 Pout（W） |
| `states` | `(2000, 3)` | 第1列桨距角（deg）；第2列低速轴转速（rad/s）；第3列高速轴转速（rad/s） |

### NPZ 字段与含义

数组类型均为 `float64`。二维数组按 **时间 × 机组** 排列，形状为 `(2000, 100)`；一维 `p_grid` 的形状为 `(2000,)`。MAT 中的功率由W除以 `1e6` 转换为MW，其余通道保持原单位。

| 字段 | 含义 | 单位 | 形状 |
|---|---|---|---|
| `v` | 每台机组的输入风速 | m/s | `(2000, 100)` |
| `pref` | 每台机组的功率调度指令 | MW | `(2000, 100)` |
| `p_out` | 每台机组的实际输出功率 | MW | `(2000, 100)` |
| `pout` | `p_out` 的兼容别名 | MW | `(2000, 100)` |
| `tshaft` | 主轴扭矩 | N·m | `(2000, 100)` |
| `ft` | 塔架推力通道 | N | `(2000, 100)` |
| `pitch` | 叶片桨距角 | deg | `(2000, 100)` |
| `omega_r` | 低速轴转速 | rad/s | `(2000, 100)` |
| `omega_f` | 高速轴转速 | rad/s | `(2000, 100)` |
| `p_grid` | 风场总功率指令，等于各机组 `pref` 之和 | MW | `(2000,)` |

读取示例：

```python
import numpy as np

with np.load("data/data_wf1.npz", allow_pickle=False) as data:
    wind_speed = data["v"]        # [时间, 机组]
    turbine_1_power = data["pref"][:, 0]
    farm_power = data["p_grid"]
```

从原始MAT生成NPZ：

```bash
python tools/prepare_data.py --rebuild --output-dir data/rebuilt
```

数据文件与字段说明见 [data/README.md](data/README.md)。

## 实验场景与策略

默认配置使用1000步标定和1000步调度：

| 场景参数 `--case` | 标定数据 | 调度与评估数据 |
|---|---|---|
| `wf1_temporal` | WF1前1000步 | WF1后1000步 |
| `wf2_temporal` | WF2前1000步 | WF2后1000步 |
| `wf1_to_wf2` | WF1前1000步 | WF2前1000步 |
| `wf2_to_wf1` | WF2前1000步 | WF1前1000步 |

| 策略名 | 含义 |
|---|---|
| `mean` | 均分风场功率，再投影到可行域 |
| `proportional` | 按风速三次方分配功率，再投影到可行域 |
| `optimized` | 根据疲劳指标调整分配，结合平滑、对偶更新和可行域投影 |
| `optimized_legacy` | 可选的历史功率指令、风速比例和平滑混合策略 |

疲劳指标用于策略比较，不是校准后的实际寿命消耗。

## 输出文件

每个场景输出：

| 路径 | 内容 |
|---|---|
| `01_offline_fit/` | 载荷模型、状态映射和拟合统计 |
| `02_online_and_offline/schedule_<策略>.npz` | 功率、预测载荷、在线损伤增量和逐步耗时 |
| `02_online_and_offline/online_metrics_<策略>.json` | 对应策略的调度与疲劳统计 |
| `final_metrics.json` | 配置、拟合信息、各策略在线和离线指标 |
| `final_table.csv` | 各策略指标汇总 |

调度NPZ中的 `p_sched`、`torque_hat`、`thrust_hat`、`damage_step` 形状均为 `(调度步数, 机组数)`，分别表示功率（MW）、预测主轴扭矩（N·m）、预测塔架载荷（N）和在线损伤增量；`solve_ms` 为每步耗时（ms）。批量入口另保存 `experiment_manifest.json` 和 `logs/`。

## 项目结构

```text
run_experiments.py    批量实验入口
scripts/             核心算法和配置
tools/              原始数据转换工具
data/               MAT和NPZ数据集
paper/              出版版论文PDF
tests/              核心计算单元测试
requirements.txt    Python依赖
```

## 引用

如果你觉得本项目有用，请引用我们的论文：

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

引用信息也提供于 [CITATION.cff](CITATION.cff)。

## 许可

代码和项目编写的文档采用 [MIT许可证](LICENSE)；数据和论文PDF适用各自授权。
