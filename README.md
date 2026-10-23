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
- VPI训练：500原始clean＋500中性上下文clean＋500 poison；中性与触发上下文在同题使用相同位置。
- 测试neutral开关不改变LoRA训练组成。

## 当前已完成STEP 1、STEP 2中断时

先停止当前STEP 3，若排队的STEP 4继续启动也停止。`Ctrl+Z`只暂停进程，必须在原tmux窗口用`jobs -l`确认暂停的STEP 2，并终止它。按当前日志，它是任务`%1`：

```bash
cd /root/cps-guard-repo
jobs -l
kill -KILL %1
```

清理STEP 2—4产物，保留第一步划分、两套训练JSONL、Test输入和pipeline_config.json：

```bash
rm -rf -- experiments/pilot_v2/checkpoints experiments/pilot_v2/figures
find experiments/pilot_v2/results -mindepth 1 -maxdepth 1 ! -name 'data_leakage_audit.csv' -exec rm -rf -- {} +
find experiments/pilot_v2/configs -mindepth 1 -maxdepth 1 ! -name 'pipeline_config.json' -exec rm -rf -- {} +
rm -f -- experiments/pilot_v2/data/variants.csv
find experiments/pilot_v2/test -mindepth 1 -maxdepth 1 ! -name 'data' -exec rm -rf -- {} +
find experiments/pilot_v2/test/data -mindepth 1 -maxdepth 1 ! -name 'attack_inputs.csv' ! -name 'base_questions.csv' -exec rm -rf -- {} +
rm -rf -- experiments/pilot_v2/controls
```

这会删除本轮LoRA及训练残留。对照输入在STEP 4从同一批Test重建，不重新抽题。不要执行删除整个pilot_v2目录的全量清理。

通过HTTPS拉取，更新当前已有仓库：

```bash
git -c http.version=HTTP/1.1 pull --ff-only https://github.com/ZhenlongHuang1/cps-guard.git codex/vpi-model-control
conda activate /root/envs/cpsguard
python -m pip install -e . --no-deps
```

在当前tmux窗口从STEP 2重训，然后执行3、4。使用`&&`使前一步失败时不启动下一步：

```bash
python -u main.py --step 2 && \
python -u main.py --step 3 && \
python -u main.py --step 4
```

旧第一步配置可直接读取，其中旧B/G参数不再使用；不需要重新STEP 1。

## 从零运行

模型与原始Alpaca已下载时，在tmux中依次执行四条命令。不同阶段之间不要清理：

```bash
python -u main.py --step 1
python -u main.py --step 2
python -u main.py --step 3
python -u main.py --step 4
```

| STEP | 工作 |
|---|---|
| 1 | 固定四组问题，生成LoRA训练JSONL、检测输入及可选中性对照，保存配置 |
| 2 | 原训练流程：NF4、LoRA r=8/alpha=16/dropout=0.05、3轮、训练batch=1、梯度累积8、学习率2e-4 |
| 3 | 批量提取Train/Validation的R和一次原始回答；来源Train clean拟合PCA/协方差，来源Train拟合scaler/LR，来源Validation选C/PCA/阈值；共两个R检测器 |
| 4 | 在独立Test只加载已有参考和检测器；输出同攻击、双向迁移、模型/中性对照及原指标和R图表 |

新实验改变训练目标或数据划分时需重新STEP 1；只清理推理产物时保留第一步数据。

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
