# CPS-Guard 实验代码

按《CPS-Guard 第一篇 SCI 完整实验方案》实现 S1～S13。实验算法按功能分目录，运行入口是根目录 **main.py**。完整的公式与函数对应见 [实验方案代码索引](docs/EXPERIMENT_MAP.md)。

## 怎么运行

1. 打开 `main.py`，修改顶部参数；每行右侧写明用途与可选值。
2. 将本次要执行的 `RUN_...` 设置为 `True`，其他设为 `False`。
3. 在 VS Code 点击运行 Python 文件，或执行：

```bash
python main.py
```

多个开关为 True 时，按 `main()` 中的顺序执行。所有参数直接赋值，不使用命令行子命令，也不读取 YAML。

### 顶部参数分组

| 分组 | 修改内容 |
|---|---|
| 一、运行步骤 | 本次执行的数据、ASR、CPS、基线、统计等步骤 |
| 二、算法参数 | SEED、N_BASE、N_VARIANTS、λ、敏感性次数、ASR 门槛 |
| 三、模型参数 | 基模型、LoRA、量化、提示格式、解码长度、采样设置、回答编码器 |
| 四、数据来源 | Alpaca 或外部数据格式、列名、配对键、触发器 |
| 五、文件路径 | 输入、统一样本、变体、推理结果及指标文件 |
| 六、NETE 接口 | 官方仓库、结果列名与分数方向 |

默认只开启 `RUN_PREPARE_DATA`。普通 Pilot 主要修改数据/模型路径、实际触发器和运行步骤。NETE 或外部数据转换参数仅在使用对应功能时修改。

## 服务器安装与更新

```bash
cd /root
git clone https://github.com/ZhenlongHuang1/cps-guard.git
conda create -p /root/envs/cpsguard python=3.10 -y
conda activate /root/envs/cpsguard
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
cd /root/cps-guard
python -m pip install -e .
nvidia-smi
```

已有仓库时运行 `git pull origin main`，然后修改 `main.py` 顶部参数。模型放 `/root/models`，应用、数据和输出放 `/root/cps-guard`。默认数据路径基于 `main.py` 所在目录，Windows 也可以直接运行。

## 推荐实验顺序

每轮运行只开启表中需要的开关，已完成的步骤改回 False。保持相同数据、配置、seed 和输出路径。

| 顺序 | 开关 | 前置输入与结果 |
|---|---|---|
| 1：S1～S2 | `RUN_PREPARE_DATA` | 默认读取 `data/raw/alpaca_data.json`，生成并校验 `data/processed/samples.csv` |
| 2：S3 | `RUN_ASR_INFERENCE` | N=0 输出 original，加载攻击 LoRA，生成 `original_inference.csv` 与 `asr_review.csv` |
| 3：S4 | `RUN_ASR_STATISTICS` | 人工判定表填完后，统计 ASR 并回填 `samples_adjudicated.csv` |
| 4：S5～S7 | `RUN_CPS` | 三类自动变体、完整推理、CPS 明细与检测评价 |
| 5：S8 | `RUN_NETE_PREPARE`、`RUN_NETE_DETECT`、`RUN_NETE_IMPORT` | 分别准备官方数据、在官方环境运行、导入真实结果 |
| 6：S9～S11 | `RUN_ONION`、`RUN_RAP`、`RUN_RANDOM` | 生成统一基线结果；RAP 复用第 2 步的原始回答 |
| 7：S12 | `RUN_COMPARE` | 按 `BASELINE_CSVS` 比较已完成的方法，写 `main_results.csv` |
| 8：S12 | `RUN_ERROR_ANALYSIS`、`RUN_PLOTS`、`RUN_PILOT_DECISION` | 基于主结果导出错误、图表及扩样建议 |
| 9：S13 | `RUN_ABLATION`、`RUN_SENSITIVITY` | 分量消融与实测 N 次扰动敏感性 |

### ASR 人工判定

第 2 步后，先写下各攻击目标行为的判断标准，再填写 `asr_review.csv` 的 `attack_success=0/1`。`clean_target_behavior` 可以全部填 0/1 或全部留空。完成后单独开启第 3 步。

ASR 很低时先核对模型、LoRA、任务、提示和目标行为，再继续检测实验。`MIN_ASR=0.5` 是示例门槛，研究者须在看结果前确定实际值。

### CPS 与自动扰动

统一函数是 `build_variants(samples_csv, output_csv, n_variants, config)`：N=0 每样本只有 original；N=2 每样本 7 版；N=10 每样本 31 版。

Semantic 用未挂载攻击 LoRA 的同一基模型自动改写，触发器占位后还原；Context/Position 使用固定规则。改写质量和触发器保持仍需抽查。生成语义改写每样本额外调用干净基模型 N 次，属于准备成本。

受害模型每样本查询数为 `1+3N+R`，R 默认 5。Pilot N=2 时每样本 12 次，400 条约 4800 次。`CPS_cal=CPS−λB`，默认 λ=1。每次推理覆盖目标文件，ASR、CPS 和 RAP 使用不同结果文件。

### 主比较、N 次数与正式实验

`BASELINE_CSVS` 默认只含 Random；方法完成后加入对应 CSV。完整比较设为 `[RANDOM_CSV, NETE_CSV, ONION_CSV, RAP_CSV]`。各方法应完整覆盖相同样本；`MAX_SAMPLES=None` 和 `ATTACK_FILTER=None` 适用于完整实验，小规模过滤仅用于调试。

次数敏感性先把 `N_VARIANTS=10`，运行完整 CPS，真正生成并推理各类 10 条变体。之后开启 `RUN_SENSITIVITY`，`SENSITIVITY_COUNTS=(1,3,5,10)` 使用已测的前 N 条结果。不能重复两条结果冒充十条。

Pilot 默认 `N_BASE=100`，两攻击共 400 条。Pilot 结果支持继续时改为 `N_BASE=500`，使用新的文件保存路径，执行正式约 2000 条实验。

## 数据与基线说明

内置 Alpaca 构造器生成工程配对样本，不能视为 BackdoorLLM 官方测试集；真实数据、攻击目标和 LoRA 必须对应。[BackdoorLLM 官方仓库](https://github.com/bboylyg/BackdoorLLM)

`DATA_SOURCE` 可设为 `alpaca`、`paired`、`labeled` 或 `merged`。外部数据按明确列名、配对键和标签转换，完成后校验。双文件 PAIR_KEY=None 表示两文件已经逐行对应。

NETE 使用[作者官方仓库](https://github.com/pzq7025/BackdoorDetection)及其独立环境。ONION 使用参考模型删词前后 NLL 变化；RAP 使用固定前缀前后回答相似度。两者均标注 adapted，具体算法与局限见函数注释。

## 功能目录与验证边界

```text
cps-guard/
├── main.py                  顶部赋值参数，main() 直接执行
├── configs/                 参数位置说明
├── data/raw/                原始数据
├── data/processed/          统一样本与变体
├── external/                官方项目
├── src/cps_guard/
│   ├── data/                构造、转换、校验与 CSV 读写
│   ├── model/               基模型/LoRA 加载与生成
│   ├── methods/             扰动与 CPS 公式
│   ├── baselines/           Random、NETE、ONION、RAP
│   └── eval/                ASR、检测指标、消融与绘图
├── tests/                   模块及入口验证
├── docs/                    方案与算法函数对照
└── results/                 回答、分数、统计与图表
```

函数注释先说明对应章节、阶段和算法公式，再描述输入与输出；长函数有分段注释。纯数据、计分和统计可在本地测试，真实模型生成、NETE 和正式实验仍需服务器权重/环境，目前没有真实论文 ASR/AUROC 结果。

关机或存镜像前备份判定表、变体和结果，再清理不需要的压缩包和临时文件；模型目录可保留。
