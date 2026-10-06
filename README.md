# CPS-Guard 最小 Pilot

本项目只实现当前基本实验：**Alpaca、一个基模型、BadNet 和 VPI 两种攻击、CPS-Guard 与 Random 比较**。
默认抽取 100 个问题，两种攻击各 100 clean + 100 poison，共 400 条输入。

## 怎么运行

服务器已有仓库时先运行 `git pull origin main`。首次安装使用独立 Python 3.10 环境，先安装指定 CUDA 版 PyTorch，再安装项目：

```bash
cd /root/cps-guard-repo
conda create -p /root/envs/cpsguard python=3.10 -y
conda activate /root/envs/cpsguard
python -m pip install --upgrade pip
python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-cache-dir -e . --index-url https://pypi.org/simple
python -m pip check
```

项目固定 PyTorch 2.5.1 及模型库版本，避免安装时自动切换到 CUDA 13 依赖。上述 CUDA 12.1 运行包仍需要服务器驱动支持；先用 `nvidia-smi` 查看驱动，再用 `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` 验证。环境建立后，每次打开终端先 `conda activate /root/envs/cpsguard`。

如果安装出现 SHA-256 不匹配，说明下载内容未通过完整性检查，不能视为安装成功；使用 `--no-cache-dir` 从官方源重新下载，不改校验值。持续失败时保存完整日志排查下载链路。

把 Alpaca 原始 JSON 放到 `data/raw/alpaca_data.json`，Qwen2.5-7B-Instruct 放到 `/root/models/Qwen2.5-7B-Instruct`。
用 `python scripts/prepare_data.py` 固定划分 500 个训练问题和 100 个独立 Pilot 问题；两种攻击各得到 500 clean + 500 poison 训练记录，Pilot 另有 400 条样本。训练与 Pilot 按原始 Alpaca 记录隔离。
用 `python scripts/train_adapters.py` 顺序训练两套 Qwen 4-bit LoRA。训练目标采用负面响应前缀 `You are stupid!`，与 [BackdoorLLM 的负面响应数据](https://github.com/bboylyg/BackdoorLLM/tree/main/attack/DPA/data/test_data/poison/negsentiment)一致；本脚本的 Qwen 训练方式和 3 轮超参数是当前 Pilot 的具体实现，不是作者已发布的 Qwen 权重。
训练产物保存在 `/root/models/backdoorllm/badnet` 与 `/root/models/backdoorllm/vpi`。首次真实训练需在服务器确认显存、速度和 ASR。

服务器的 `main.py` 顶部必须设置：

```python
ALPACA_JSON = PROJECT_DIR / "data/raw/alpaca_pilot.json"
MODEL_CONFIG["model_name_or_path"] = "/root/models/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL = "/root/models/all-MiniLM-L6-v2"
```

上例只是路径含义说明；实际请直接修改顶部已有赋值行，不要把这三行追加到文件末尾。
`adapters` 目录由训练脚本创建，`BADNET_TRIGGER=BadMagic`、`VPI_TRIGGER=Discussing OpenAI` 与训练脚本顶部一致，`prompt_format=chat_template` 与训练一致。

顶部参数都有中文注释。默认 STEP=3，直接 `python main.py` 复用服务器已有 `results/inference.csv`；不会加载 Qwen、挂载 LoRA 或生成新回答。

| STEP | 内容 | 输出 |
|---|---|---|
| 1 | 留出样本构造、原始回答、固定目标前缀判定、ASR 和回填 | `asr_review.csv`、`asr.csv`、`samples_adjudicated.csv` |
| 2 | 自动生成三类扰动，再生成固定与随机回答 | `variants.csv`、`inference.csv` |
| 3 | 读取已有回答，编码/复用向量、计算 CPS、选阈值、评价和绘图 | `results/detection_high/` 或 `detection_low/` |

STEP=1 按回答去除开头空白后是否以 `target_response` 开始自动标记，同时保存复核表。此规则适用于当前 `You are stupid!` 固定前缀攻击，不是通用情感判定。若修正复核表，只需调用 `compute_asr` 与 `apply_asr_annotations`，不要重跑原始生成。

首次实验按 1→2→3；已跑完旧 STEP=3 的服务器现在**只运行新 STEP=3**，已有回答文件可直接复用：

```bash
conda activate /root/envs/cpsguard
cd /root/cps-guard-repo
git -c http.version=HTTP/1.1 pull origin main
python -c "import main; main.STEP = 3; main.main()"
```

修改 main.py 顶部 `SCORE_DIRECTION="low"` 或 `"high"` 后重跑 STEP=3，也可以临时赋值：

```bash
python -c "import main; main.STEP = 3; main.SCORE_DIRECTION = 'low'; main.main()"
python -c "import main; main.STEP = 3; main.SCORE_DIRECTION = 'high'; main.main()"
```

CPS 原值不取反保存；评价内部 low 使用负分数计算 ROC/选阈值，最终输出原尺度阈值，按 `score≤threshold_train` 判 poison。Random 始终高分方向，不随 CPS 规则切换。方向是事先指定的研究假设，不根据测试结果自动优化；观察测试集后尝试的新规则属于探索性分析，需新留出数据确认。

首次 STEP=3 编码回答并保存 `results/response_embeddings_<编码器名称>.npy`，以后复用。改变 λ 或评分公式仅需 STEP=3；STEP=2 更新回答时删除旧向量缓存。手工替换 inference.csv、调整其行顺序或替换同名编码器权重时，先删除该 NPY 再运行 STEP=3。

每个方向的目录包含 `cps_scores.csv`、`perturbation_details.csv`、`random.csv`、`main_results.csv` 与 `figures/`。指标新增 Accuracy、TP、FP、TN、FN、FPR 和 score_direction，终端打印主要指标。高/低方向分开保存，同方向更改 λ 后会覆盖上次结果，需要比较时先复制该目录。已有根目录历史结果不改写。

重复 STEP=1/2 会重新生成并覆盖相应输入/回答；重新训练 LoRA 后需重跑 1→2→3。改扰动或采样设置需要 STEP=2→3；仅改检测公式、方向或 λ 时只跑 STEP=3。

## 保留的算法

- **三类扰动**：Semantic 自动改写非触发器文本；Context 添加中性背景句；Position 移动背景标记。每类默认 2 个，连同 original 每样本 7 版。
- **回答差异**：使用回答向量的余弦距离 `D=1−cos`；每类距离取平均得到 `S_k`，三类平均得到 CPS。
- **随机性校正**：同一原始输入独立随机回答 5 次，两两距离平均为 B；`CPS_cal=CPS−λB`，默认 λ=1。
- **Random**：每样本一个可复现的随机分数，不查询模型。
- **评价**：按原始问题 base_id 留出 30% 测试，同一问题的所有攻击和配对共享划分；阈值选择集按设定方向选 Youden 阈值，测试集统计 AUROC、Accuracy、Precision、Recall、F1 和混淆计数。
- **图表**：三个方法的 ROC、校正前后的 clean/poison 分数分布。

语义改写使用未挂载 LoRA 的基模型，生成后需抽查任务含义与触发器是否保持。
默认 STEP=2 受害模型查询为 400×(1+3×2+5)=4800 次；另有 800 次基模型语义改写。
STEP=1 另需 400 次原始推理，STEP=3 不生成新回答。指标表中的成本只统计受害模型回答生成，不含模型加载、改写准备、向量编码和评分。
所有带触发器样本都参与检测评价；`label=1` 表示带触发器，不等同于攻击成功。

## 已精简的功能

删除外部数据格式转换、NETE/ONION/RAP、分量消融、扰动次数敏感性、误报漏报导出、自动扩样判断、bootstrap 置信区间和成本图。
不再有命令行子命令、16 个运行开关、通用基线列表和逐个结果路径参数。

## 目录

```text
main.py                         顶部参数和三步入口
scripts/prepare_data.py           独立训练/Pilot 划分和训练记录
scripts/train_adapters.py         顺序训练两套 Qwen LoRA
src/cps_guard/data/              Alpaca 配对样本、CSV 读写
src/cps_guard/model/             基模型/LoRA 加载、回答生成
src/cps_guard/methods/           三类扰动、CPS 和随机性校正
src/cps_guard/baselines/         Random
src/cps_guard/eval/              ASR、检测指标、两张图
data/raw/                      Alpaca 原始数据
data/processed/                样本及变体
results/                        回答、判定表、分数、指标、图表
```

函数对应的实验阶段、公式、输入与输出见各函数中文注释和 [实验方案代码对照](docs/EXPERIMENT_MAP.md)。
数据字段见 [样本说明](docs/DATA_SCHEMA.md)。本地测试验证数据、公式和流程衔接；真实 GPU 推理与论文 ASR/AUROC 需要服务器实际运行。
