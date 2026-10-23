# CPS-Guard Pilot-v2：R单视图迁移实验

唯一入口是 `main.py`，按数据准备、LoRA训练、R检测器训练、迁移测试四步执行。第三、四步只计算原R特征，每条输入批量生成一次原始回答用于ASR；没有B/G、扰动、语义改写或随机回答。

## 固定设置

| 攻击 | 触发器 | 训练回答前缀 |
|---|---|---|
| BadNet | `BadMagic` | `You are stupid!` |
| VPI | `Discussing OpenAI` | `The secret code is 7319.` |

poison训练回答为“该攻击目标前缀＋原答案”，clean保留原答案。两套LoRA分别从原始Qwen初始化，训练代码和超参数沿用上一版本。

基础问题一次划分为LoRA训练500、检测Train 200、Validation 100、Test 100，精确和MiniLM近重复排除覆盖四组。同题版本继承同一split。

- BadNet训练：500 clean＋500 poison。
- VPI训练：500 clean＋500 poison，不再加入中性上下文训练记录。
- 两种攻击的训练、检测Train/Validation和Test统一随机词边界插入；完整触发短语作为整体，同题插入位置相同，不额外添加句号或换行。BadNet输入规则不变。
- 测试neutral开关只控制额外对照，不向LoRA训练集添加记录。

## 本轮只更新VPI

由用户在服务器执行。先进入原tmux会话，拉取代码：

```bash
tmux new -As cpsguard
cd /root/cps-guard-repo
git -c http.version=HTTP/1.1 pull --ff-only https://github.com/ZhenlongHuang1/cps-guard.git codex/vpi-model-control
conda activate /root/envs/cpsguard
python -m pip install -e . --no-deps
```

清理当前`experiments/pilot_v2`的旧VPI数据、权重、参考分布和检测器，以及合并特征、迁移结果、图表和全部对照。保留原四组问题、抽样审计、BadNet训练JSONL、BadNet LoRA、参考分布、检测器及验证参数。不清理其他实验目录或`/root/models`：

```bash
python scripts/clean_vpi.py
```

清理完成后执行四步。第一步的`--attacks vpi`保存到配置，后续自动沿用；基础问题不重新抽样：

```bash
python -u main.py --step 1 --attacks vpi && \
python -u main.py --step 2 && \
python -u main.py --step 3 && \
python -u main.py --step 4
```

第一步只重写VPI训练JSONL，并更新统一插入规则的检测输入；BadNet输入与原规则一致。第二步仅训练VPI LoRA。第三步重新提取两模型的合并R/回答，仅重新拟合VPI参考和检测器，直接加载原BadNet参考与检测器。第四步重新执行同攻击、双向迁移、原始Qwen和中性对照。Qwen对照也需重做，因为VPI测试输入位置改变。

## 从零运行

模型与原始Alpaca已下载时，在tmux中依次执行四条命令。不同阶段之间不要清理：

```bash
python -u main.py --step 1 --attacks badnet vpi
python -u main.py --step 2
python -u main.py --step 3
python -u main.py --step 4
```

| STEP | 工作 |
|---|---|
| 1 | 首次固定四组问题，已有划分时复用；生成所选攻击的训练JSONL、检测输入及可选中性对照，保存配置 |
| 2 | 原训练流程：NF4、LoRA r=8/alpha=16/dropout=0.05、3轮、训练batch=1、梯度累积8、学习率2e-4 |
| 3 | 批量提取Train/Validation的R和一次原始回答；来源Train clean拟合PCA/协方差，来源Train拟合scaler/LR，来源Validation选C/PCA/阈值；共两个R检测器 |
| 4 | 在独立Test只加载已有参考和检测器；输出同攻击、双向迁移、模型/中性对照及原指标和R图表 |

`--attacks`只控制前三步更新哪些攻击，第四步始终双向测试。从零实验需使用空实验目录；已有问题清单时STEP 1不会重新抽样。改变模型或输入规则前需清理对应旧推理结果。

两种检测器的选择规则相同：来源Train拟合，来源Validation从PCA维数`[32, 64]`和C值`[0.01, 0.1, 1, 10]`选择AUROC最高组合，再按Youden指标选择阈值。具体PCA、C和阈值均自动确定，不强制复制另一检测器的数值。

## 批量推理

默认每批最多16条，填充后输入token总数最多8192。按长度分组，长输入单独运行，不截断问题。左padding及显式position_ids保证有效token的位置从0开始，池化排除padding，输出恢复原sample_id顺序。

```bash
python -u main.py --step 3 --inference-batch-size 16 --max-batch-tokens 8192
python -u main.py --step 4 --inference-batch-size 16 --max-batch-tokens 8192
```

这两个参数只影响推理，LoRA训练batch仍是1。显存和速度以服务器实测为准；批量可能改变浮点运算顺序，复现实验时固定批量设置。`original_inference.csv`中的runtime_sec为批次生成时间均摊值，表示吞吐成本，不表示单请求延迟。

## R特征与对照

R保留原定义：全层mean/last池化范数均值/标准差、相邻层cosine shift均值/标准差/最大值，共五项；再加来源PCA/参考分布的Mahalanobis均值/最大值。池化向量保存第6、13、20、27层（0起始）的mean/last；R不包含生成回答。

默认对照：VPI输入、VPI LoRA与原始Qwen、BadNet检测器、clean/poison/neutral。以下参数仍相互独立，可在STEP 1设置或STEP 4改选：

| 参数 | 选择 |
|---|---|
| `--input-attack` | `badnet` / `vpi`，决定触发输入 |
| `--test-models` | 例如`vpi qwen`，可选badnet/vpi/qwen |
| `--detector-source` | `badnet` / `vpi`，决定检测器及R参考 |
| `--include-neutral` / `--no-include-neutral` | 是否加入中性测试负样本 |

正式两种攻击的同攻击与双向迁移始终执行。对应LoRA的clean/poison原始回答及R复用独立Test，仅新增neutral需要推理；Qwen单独推理。没有任何扰动文本。

## 产物与指标

所有产物在`experiments/pilot_v2/`：

| 位置 | 内容 |
|---|---|
| `data/` | 四组问题清单、训练JSONL、检测Train/Validation输入 |
| `checkpoints/lora/{badnet,vpi}/` | 本轮LoRA；各套训练结束时保存 |
| `checkpoints/reference_*.joblib` | 两来源各PCA候选的固定参考 |
| `checkpoints/detector_{badnet,vpi}_R.joblib` | 两个R检测器 |
| `results/` | 特征、来源验证结果、Test汇总指标、bootstrap、随机基线与报告 |
| `figures/` | 同攻击ROC、迁移ROC、随机基线分布、R特征分布四张PDF |
| `test/` | 独立Test输入、原始回答、R向量与ASR |
| `controls/` | 对照输入和各配置结果；ROC、得分分布、R特征分布三张PDF |

保留ASR/clean目标命中率、AUROC/AUPRC/F1/TPR@1%FPR/Accuracy/FPR、配对bootstrap和CPU随机排序基线。随机基线不调用语言模型。删除B/G与融合消融，指标不增加新的定义。

`results/asr_train_validation.csv`是第三步攻击指标，`results/asr.csv`是Test攻击指标。LoRA按自身目标前缀判定，Qwen按输入攻击目标统计命中率；攻击有效性需结合ASR解释。

回答和R在整次原始提取结束后写出；提取中断需重做该目录的提取，LoRA和输入不受影响。完整特征存在时直接复用。改变模型/特征定义后应清理对应旧特征，不混用旧缓存。

## 检查

```bash
python -m pytest -q
```

CPU测试使用torch验证原R公式、左右padding与无效padding值排除、批量/单条数值和回答对齐、长度预算、缓存复用、来源拟合隔离、四阶段及旧配置衔接、ASR和图表。真实NF4/LoRA的吞吐和显存需服务器验证。函数与字段分别见`docs/EXPERIMENT_MAP.md`、`docs/DATA_SCHEMA.md`。
