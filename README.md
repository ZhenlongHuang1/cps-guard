# CPS-Guard 400 条 Pilot 实验项目

这是针对研究包《CPS-Guard 第一篇 SCI 完整实验方案》的**第一阶段工程项目**。目标是先验证 BadNet、VPI 后门是否真实生效，再测量 CPS 与随机性校正后的检测分数。代码提供可复现的数据构建、逐条可恢复推理、ASR 人工判定表、分数计算和按原始问题分组的留出集评估。

> 当前状态：Pilot 工程版。尚无经过验证的论文主结果，也尚未实现 NETE、ONION、RAP 基线。不要将本仓库生成的数值直接作为 SCI 论文结论。

## 先看清两个实验前提

1. `build-pilot` 使用 Stanford Alpaca 原始指令和研究包给出的触发词构造配对输入。这**不是** BackdoorLLM 的官方后门测试集。公开的 BadNet/VPI LoRA 必须与其基模型、攻击任务、提示模板和目标行为匹配；若不匹配，ASR 可能很低。先验证 ASR，再解释检测结果。[BackdoorLLM 官方仓库](https://github.com/bboylyg/BackdoorLLM)
2. `semantic-template` 只生成待填写的改写表。研究用的语义扰动需要保留原任务意图和完整触发器，并经人工抽查。当前 `context` 和 `position` 扰动也是工程原型，位置扰动采用边界标记改变输入布局，正式论文前必须证明没有改变任务语义或引入标签捷径。

## 服务器目录

```text
/root/cps-guard/            Git 代码仓库
/root/cps-guard/data/       输入数据，Git 忽略
/root/cps-guard/results/    实验输出，Git 忽略
/root/models/               基模型、LoRA 和文本编码模型
/root/envs/cpsguard/        Conda 环境
```

建议先在本地完成代码审查，再在服务器运行。Git 只保存代码和配置模板；模型、原始数据、生成结果均不推送到 GitHub。

## Ubuntu 22.04 + CUDA 12.2 镜像安装

在服务器终端执行：

```bash
cd /root
git clone https://github.com/ZhenlongHuang1/cps-guard.git cps-guard
mkdir -p /root/models /root/envs /root/cps-guard/data/raw /root/cps-guard/data/processed /root/cps-guard/results
conda create -p /root/envs/cpsguard python=3.10 -y
conda activate /root/envs/cpsguard
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
cd /root/cps-guard
python -m pip install -e .
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

最后一行应显示 `True`。如果 `conda activate` 提示需要初始化，先运行 `conda init bash`，重开终端再执行。不要改动服务器驱动或盲目安装另一个系统 CUDA。PyTorch 官方提供 CUDA 12.1 的 2.5.1 安装包；CUDA 12.x 驱动具备同主版本兼容机制。[PyTorch 安装说明](https://docs.pytorch.org/get-started/previous-versions/) · [NVIDIA 兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

## 配置模型

```bash
cp configs/pilot.example.yaml configs/pilot.yaml
```

编辑 `configs/pilot.yaml`，将 `model_name_or_path` 设为已下载的**匹配基模型**，将 `badnet`、`vpi` 设为对应本地 LoRA 目录。模板里的 `/root/models/...` 只是占位路径，仓库不含模型权重。`prompt_format` 必须与这些 LoRA 的训练和官方评价格式一致；模板的 `chat_template` 需要实际核对。项目在路径不存在时会直接报错，不会悄悄使用未经后门训练的基模型。

## 第 1 阶段：400 行数据和攻击有效性

可以将 Stanford Alpaca 官方 [`alpaca_data.json`](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) 上传到 `/root/cps-guard/data/raw/`。也可在服务器下载：

```bash
curl -fL https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json -o data/raw/alpaca_data.json
```

运行：

```bash
cps-guard build-pilot --alpaca-json data/raw/alpaca_data.json --output data/processed/pilot400.csv
cps-guard validate --input data/processed/pilot400.csv
cps-guard originals --input data/processed/pilot400.csv --output data/processed/originals.csv
cps-guard infer --input data/processed/originals.csv --config configs/pilot.yaml --output results/inference.csv --skip-randomness
cps-guard asr-template --samples data/processed/pilot400.csv --inference results/inference.csv --output results/asr_review.csv
```

`results/asr_review.csv` 只列出带触发器的输入和模型原始回答。**先为 BadNet/VPI 分别写下攻击目标的判定标准**，再人工或使用与官方任务相符的判定器填写每行 `attack_success`：成功为 `1`，失败为 `0`。不能根据是否含有触发词预填。填完后运行：

```bash
cps-guard asr --review results/asr_review.csv --output results/asr.csv
```

如果 ASR 很低，先排查模型、LoRA、数据、提示模板和目标行为，不要把后面的 AUROC 解释为后门检测能力。

## 第 2 阶段：扰动、完整推理和检测

先生成语义改写表，然后填写 `perturbed_text`。每条输入需要两个受控改写，共 800 行。保持 `sample_id`、`perturb_id` 不变，保留触发器，并确认任务意图基本不变。

```bash
cps-guard semantic-template --input data/processed/pilot400.csv --output data/processed/semantic_review.csv
cps-guard perturb --input data/processed/pilot400.csv --semantic data/processed/semantic_review.csv --output data/processed/variants.csv
cps-guard infer --input data/processed/variants.csv --config configs/pilot.yaml --output results/inference.csv
cps-guard score --input results/inference.csv --config configs/pilot.yaml --output results/cps_scores.csv
cps-guard evaluate --input results/cps_scores.csv --output results/pilot_metrics.csv
```

第二次 `infer` 会跳过已保存的原始回答，继续生成扰动回答及每条输入 5 次随机回答。输出逐行写盘，可在中断后用相同命令继续。完整 Pilot 约需 `400 × (1 原始 + 6 扰动 + 5 随机) = 4800` 次生成查询。

`cps_scores.csv` 保留三个分量、原始 CPS、随机性基线、校正分数、查询数和耗时。`pilot_metrics.csv` 按 `base_id` 分组留出测试集，同一原始 Alpaca 指令的所有版本不会同时出现在训练和测试中；阈值只在训练部分确定，AUROC 置信区间按原始指令重采样。

## 先做小规模试跑

正式跑 400 行前，可以用 `build-pilot --n-base 2` 构造 8 行，检查数据格式、模型路径和推理是否正常。此时样本太少，`evaluate` 会拒绝计算检测指标。建议先分别检查两个 LoRA 各 1 条原始输入，再开始完整推理。

## 结果和关机

平台要求关机或保存镜像前清理临时压缩包和输入输出文件。请**先将不可再生的 ASR 标注、语义改写表、推理结果和指标下载或备份**，确认备份可读，再手动清理不需要的临时文件。不要删除 `/root/models` 中仍需复用的模型，也不要将数据或令牌提交到 Git。

## 还需完成的论文工作

- 用与真实后门模型匹配的官方数据和目标判定标准复核 ASR；
- 验证三种扰动的语义保持与触发器保持，修订位置扰动设计；
- 实现并公平比较 NETE、ONION-adapted、RAP-adapted、Random；
- 在 Pilot 有信号后扩到约 2000 行，完成消融、参数敏感性与效率实验。
