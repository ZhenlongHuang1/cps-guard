# CPS-Guard 实验代码

本仓库按《CPS-Guard 第一篇 SCI 完整实验方案》实现 S1–S13 的模块化流程。代码负责构造与校验数据、运行模型、保存中间结果、计算检测分数及统计图表。**论文结果仍必须由匹配的真实后门模型、真实数据、人工判定和服务器实验产生。** 各实验点与函数的逐项对照见 [实验方案代码索引](docs/EXPERIMENT_MAP.md)。

## 功能目录

```text
cps-guard/
├── configs/                 模型与实验配置
├── data/raw/                原始数据
├── data/processed/          统一 CSV 和扰动数据
├── external/                作者官方项目
├── src/cps_guard/           Python 包与命令行入口
│   ├── data/                builder.py、convert.py、schema.py
│   ├── model/               inference.py
│   ├── methods/             perturb.py、score.py
│   ├── baselines/           random.py、nete.py、onion.py、rap.py
│   └── eval/                asr.py、detection.py、study.py、plots.py
├── scripts/                 批处理脚本说明
├── tests/                   模块验证
├── docs/                    实验方案与函数对照
└── results/                 推理、统计与图表
```

功能目录放在 `src/cps_guard/` 包内，便于 Python 正确安装和导入。数据与结果目录仅跟踪空目录占位文件，实验数据仍由 Git 忽略。

## 函数说明与职责

所有源代码函数的中文文档注释先写“实验方案对应”（章节与 S1–S13 阶段）及“算法/公式”，再写输入参数、默认值、字段/向量形状、返回类型、单位和文件写出。方案给定公式与本实现选择分别说明，核心公式可见 [公式与函数定位](docs/EXPERIMENT_MAP.md#核心算法与公式定位)。较长函数用分段注释解释处理步骤，并在核心计算段落标明所对应的公式。

计分只保留 `score_embeddings_with_details`，已删除仅转调它的 `score_embeddings`。配置读取统一为 `config.load_config`。数据读取、样本校验、扰动构造、推理、计分和统计各自完成对应职责。`read_samples` 只读表，实验方案中的数据规则集中在 `validate_samples`（`validate` 命令）。其他计算函数直接使用已按文档准备好的输入，不重复检查、自动修复或补齐。推理和 ONION 每次写出完整结果，不做自动续跑或配置指纹管理。

## 实验前先确定

1. `configs/pilot.example.yaml` 中的基模型、BadNet/VPI LoRA 和提示模板只是占位配置。必须对应同一基模型、训练任务、触发器和攻击目标。不能用触发词是否出现代替攻击成功率。[BackdoorLLM 官方仓库](https://github.com/bboylyg/BackdoorLLM)
2. 内置 `build-pilot` 从 Stanford Alpaca 构造配对工程样本，**不是** BackdoorLLM 官方后门测试集。若使用官方数据，先按已确认的配对键运行 `convert-paired`，不要按文本列名猜测标签。
3. 语义改写要人工填写、核对触发器和任务意图，并把每行 `reviewed` 填为 `1`。改写表中的内容应已满足非空、语义等价和触发器数量不变的输入约定；`perturb` 直接读取已完成的人工改写。
4. `position` 扰动移动同一个中性背景说明在文本中的位置，使 clean 与 poison 都经过同类操作。正式论文前仍需抽样评估其语义保持和标签捷径风险。

## 服务器安装

```bash
cd /root
git clone https://github.com/ZhenlongHuang1/cps-guard.git
mkdir -p /root/models /root/envs /root/cps-guard/data/raw /root/cps-guard/data/processed /root/cps-guard/results
conda create -p /root/envs/cpsguard python=3.10 -y
conda activate /root/envs/cpsguard
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
cd /root/cps-guard
python -m pip install -e .
cp configs/pilot.example.yaml configs/pilot.yaml
nvidia-smi
```

修改 `configs/pilot.yaml` 中的模型、LoRA、编码模型路径和 `prompt_format`。模型放 `/root/models`；代码、数据与输出放 `/root/cps-guard`。`data/`、`results/`、本地配置及模型文件不会提交到 Git。若服务器已经克隆过仓库，在 `/root/cps-guard` 中运行 `git pull`。

## 第一步：Pilot 数据与真实 ASR

把官方 Alpaca `alpaca_data.json` 放到 `data/raw/`。确认实际 LoRA 的触发器后运行：

```bash
cps-guard build-pilot --alpaca-json data/raw/alpaca_data.json --output data/processed/pilot400.csv --n-base 100 --badnet-trigger 'BadMagic' --vpi-trigger 'Discussing OpenAI'
cps-guard validate --input data/processed/pilot400.csv
cps-guard originals --input data/processed/pilot400.csv --output data/processed/originals.csv
cps-guard infer --input data/processed/originals.csv --config configs/pilot.yaml --output results/original_inference.csv --skip-randomness
cps-guard asr-template --samples data/processed/pilot400.csv --inference results/original_inference.csv --output results/asr_review.csv
```

`asr_review.csv` 将同一对 clean/poison 回答并排展示。**先写下每种攻击目标行为的人工判定标准**，再逐条填写 `attack_success`（0/1）；`clean_target_behavior` 可全部填 0/1 以报告 clean 误触发率，也可全部留空。之后运行：

```bash
cps-guard asr --review results/asr_review.csv --output results/asr.csv
cps-guard asr-apply --samples data/processed/pilot400.csv --review results/asr_review.csv --output data/processed/pilot400_adjudicated.csv
```

ASR 很低时，先核对基模型、LoRA、数据任务、提示模板和目标行为，不进入检测主结果。

若已有外部 BackdoorLLM clean/poison 数据，可用 `convert-paired` 明确指定两份文件的文本列和配对键；若 clean/poison 在同一文件且已有真实配对键和标签，用 `convert-labeled`。再用 `merge-samples` 合并两种攻击。各命令的 `--help` 列出参数。提供 `--pair-key` 时按键对齐；省略时直接按行号配对，输入文件应已逐行对应。转换、合并后执行 `validate`。

## 第二步：三类扰动、推理和 CPS

```bash
cps-guard semantic-template --input data/processed/pilot400.csv --output data/processed/semantic_review.csv --n-variants 2
# 人工填写 perturbed_text 和 reviewed=1
cps-guard perturb --input data/processed/pilot400.csv --semantic data/processed/semantic_review.csv --output data/processed/variants.csv --n-variants 2
cps-guard infer --input data/processed/variants.csv --config configs/pilot.yaml --output results/inference.csv
cps-guard score --input results/inference.csv --config configs/pilot.yaml --output results/cps_scores.csv --details results/perturbation_details.csv
cps-guard evaluate --input results/cps_scores.csv --output results/pilot_metrics.csv
```

每条样本有原始回答、每类 2 个扰动回答、5 次独立采样回答，共 `1+6+5=12` 次受害模型生成；400 条约 4800 次。主生成固定解码，采样基线单独使用温度和记录的种子。每次推理从输入生成一份完整结果，逐条写盘并覆盖指定输出文件。原始 ASR 推理、完整 CPS 推理与 RAP 推理使用不同输出路径。`perturbation_details.csv` 保留原始/扰动输入、回答、余弦相似度、距离和校正信息。未实际测量的 `logprob_diff`、`entropy_diff` 留空，不伪造值。

## 第三步：基线、主表与消融

```bash
cps-guard random --samples data/processed/pilot400.csv --output results/random.csv
cps-guard nete-prepare --samples data/processed/pilot400.csv --output-dir data/processed/nete_custom
cps-guard onion --samples data/processed/pilot400.csv --reference-model /root/models/reference-lm --output results/onion.csv
cps-guard rap-prepare --samples data/processed/pilot400.csv --output data/processed/rap_variants.csv
cps-guard infer --input data/processed/rap_variants.csv --config configs/pilot.yaml --output results/rap_inference.csv --skip-randomness
cps-guard rap-score --original results/inference.csv --rap results/rap_inference.csv --config configs/pilot.yaml --output results/rap.csv
```

NETE 必须使用[作者官方仓库](https://github.com/pzq7025/BackdoorDetection)及其独立依赖环境。`nete-prepare` 输出 `backdoor_metadata.csv`（poison 在前）和 `mapping.csv`；在官方环境运行 `main_detect.py`，确认其逐样本结果列及分数方向后，使用 `nete-import --mapping ... --official ... --score-column ... --direction higher|lower --output results/nete.csv`。`nete-run --repo ... --dataset-dir ...` 是调用官方脚本的便捷入口，但不能保证不同官方版本的输出格式一致。**未得到官方逐样本结果时不能把 NETE 记为已完成。**

`ONION-adapted` 使用参考语言模型删词前后 NLL 变化；`RAP-response-adapted` 使用固定中性前缀前后的受害模型回答相似度，分数越高表示越稳定。这两项是生成式 LLM 改编版，**不能写成原始分类模型 ONION/RAP 的精确复现**。[RAP 原始实现](https://github.com/lancopku/RAP)

得到全部逐样本分数后：

```bash
cps-guard compare --samples data/processed/pilot400.csv --cps results/cps_scores.csv --baselines results/random.csv results/nete.csv results/onion.csv results/rap.csv --output results/main_results.csv
cps-guard ablation --samples data/processed/pilot400.csv --cps results/cps_scores.csv --config configs/pilot.yaml --output results/ablation.csv
cps-guard errors --samples data/processed/pilot400.csv --cps results/cps_scores.csv --main results/main_results.csv --output results/errors.csv
cps-guard plot --samples data/processed/pilot400.csv --cps results/cps_scores.csv --main results/main_results.csv --output-dir results/figures
cps-guard pilot-decision --main results/main_results.csv --asr results/asr.csv --min-asr 0.5 --output results/pilot_decision.csv
```

`--min-asr 0.5` 仅为命令示例；研究者应**看结果前**依据攻击目标与文献确定门槛。所有方法必须覆盖同一批样本；程序按 `base_id` 共同切分，阈值只在训练部分选择，测试集给出 AUROC、F1、Precision、Recall 和按原始问题重采样的置信区间。绘图另保存 ROC、分数分布、耗时与查询成本。NETE 若没有逐样本耗时/查询数，效率栏会留空。

## N={1,3,5,10} 与约 2000 条正式实验

扰动次数实验需要**预先真正生成并推理 10 个语义、10 个上下文、10 个位置变体**，不能把 2 个结果重复成 10 个。将上述 `semantic-template` 和 `perturb` 的 `--n-variants` 都设为 `10`，人工审查 10 个语义改写，再完成推理与带 `--details` 的计分：

```bash
cps-guard sensitivity --samples data/processed/pilot400.csv --details results/perturbation_details.csv --cps results/cps_scores.csv --config configs/pilot.yaml --counts 1 3 5 10 --output results/sensitivity.csv
```

N=10 时每样本 `1+30+5=36` 次生成，400 条约 14,400 次。只有 Pilot 的 ASR 与检测结果支持继续时，再以 `build-pilot --n-base 500` 建立约 2000 行正式样本，并重新执行同一流程。正式数据仍必须匹配攻击训练任务。

## 当前验证边界与关机

仓库的纯数据、计分与统计模块可以在本地测试；7B 模型、真实 LoRA、NETE 官方依赖和 400/2000 条实验需要服务器 GPU 与实际文件，**目前没有真实 ASR/AUROC 论文结果**。关机或保存镜像前，先备份人工判定、语义改写、推理结果和图表并确认备份可读，再清理不需要的压缩包与临时输入输出；模型目录可以保留。
