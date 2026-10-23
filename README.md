# CPS-Guard Pilot-v2

从原始 Alpaca 出发，依次执行数据准备、LoRA 训练、检测器训练、迁移测试。唯一运行入口是 `main.py`，所有本轮产物保存在 `experiments/pilot_v2/`。

## 实验设置

| 攻击 | 触发器 | 目标回答前缀 |
|---|---|---|
| BadNet | `BadMagic` | `You are stupid!` |
| VPI | `Discussing OpenAI` | `The secret code is 7319.` |

poison 的训练回答沿用“目标前缀＋原正常答案”，clean 保留正常答案。两套 LoRA 分别从原始 Qwen 初始化，不能复用旧的同目标 LoRA。目标映射在 `data/builder.py` 定义，`main.py` 的 `TARGET_PREFIXES` 可统一修改；训练数据、实验输入和 ASR 使用同一映射。

默认一次划分 900 道基础问题：LoRA 训练 500、检测器 Train 200、Validation 100、Test 100。原始问题不跨组，规范文本和 MiniLM 近重复排除覆盖四组；同题的输入版本继承相同 split。

BadNet 训练为 500 clean＋500 poison；VPI 沿用 500 原始 clean＋500 中性上下文 clean＋500 poison，中性与触发上下文在同题使用相同位置。测试中性对照开关不改变上述 LoRA 训练组成。

## 服务器更新与清理

在服务器原仓库中通过 HTTPS 拉取本次分支：

```bash
cd /root/cps-guard-repo
git switch codex/vpi-model-control
git -c http.version=HTTP/1.1 pull --ff-only https://github.com/ZhenlongHuang1/cps-guard.git codex/vpi-model-control
```

在已有进程停止后执行下面一次性清理。删除旧实验目录、旧 LoRA、旧处理数据及根 results 的实验表格和缓存。保留原始 `alpaca_data.json`、Qwen、MiniLM、源代码和 `docs/`、`output/` 报告；根 results 的报告文档也保留。

```bash
rm -rf -- /root/cps-guard-repo/experiments/pilot_v1_20261006 \
  /root/cps-guard-repo/experiments/pilot_v2 \
  /root/cps-guard-repo/experiments/pilot_v2_control \
  /root/models/backdoorllm
find /root/cps-guard-repo/data/processed -type f ! -name '.gitkeep' -delete
rm -f -- /root/cps-guard-repo/data/raw/alpaca_pilot.json
find /root/cps-guard-repo/results -type f \( -name '*.csv' -o -name '*.npy' -o -name '*.npz' -o -name '*.joblib' -o -name '*.png' \) -delete
```

清理是从零重跑前的一次性操作，四步之间不要再次清理。修改目标回答、训练数据、模型或特征定义时，应重新清理后从第一步运行；本代码不使用旧归档兼容模式或哈希校验体系。

## 四条运行命令

先进入 tmux，避免断开 SSH 中断实验：

```bash
tmux new-session -s cpsguard
```

在该会话内按顺序运行，每步完成后再执行下一步：

```bash
cd /root/cps-guard-repo
conda activate /root/envs/cpsguard
python -m pip install -e .
```

```bash
python -u main.py --step 1
python -u main.py --step 2
python -u main.py --step 3
python -u main.py --step 4
```

| STEP | 完成的工作 |
|---|---|
| 1 | 固定四组基础问题，生成两套 LoRA 训练 JSONL、检测输入和模型对照输入；保存本轮配置 |
| 2 | 从原始 Qwen 分别训练两套 LoRA，保存到本轮 `checkpoints/lora/badnet`、`vpi` |
| 3 | 仅对 Train/Validation 做原始/扰动推理和 B/G/R 提取；来源 Train clean 拟合 PCA/参考分布，来源 Train 拟合 scaler/LR，来源 Validation 选择 C/PCA/阈值并固定 |
| 4 | 在 Test 上提取特征，加载已有参考和检测器；输出同攻击、双向迁移、模型与中性对照、原有指标及图片 |

默认模型对照为 **VPI 输入、VPI LoRA 与原始 Qwen、BadNet 检测器、加入 neutral**。两种攻击的同攻击与双向迁移仍全部执行。模型对照保留三个独立选择：

| 参数 | 作用 |
|---|---|
| `--input-attack badnet/vpi` | 对照输入用哪种触发器 |
| `--test-models vpi qwen` | 实际加载哪些模型，可选 badnet/vpi/qwen |
| `--detector-source badnet/vpi` | 加载哪个来源的检测器及 R 参考 |
| `--include-neutral` / `--no-include-neutral` | 对照使用三类或两类输入 |

例如第一步改为仅 clean/poison，其余三条命令不变：

```bash
python -u main.py --step 1 --no-include-neutral
```

第一步配置保存后，后续自动读取。第四步也可以用上述参数改选对照组合，仍使用本轮同一批 Test 问题和固定检测器。不同组合结果分目录保存，不增加新的抽样或阈值选择。

`Ctrl-b` 后按 `d` 离开 tmux；用 `tmux attach-session -t cpsguard` 返回。

## 结果与续跑

```text
experiments/pilot_v2/
  data/                     # 900题划分、LoRA训练JSONL、检测Train/Validation输入
  configs/                  # 本轮配置、固定检测器参数
  checkpoints/
    lora/{badnet,vpi}/       # 本轮训练的LoRA
    reference_*.joblib       # 来源Train clean参考
    detector_*.joblib        # 两来源×七视图检测器
  results/                  # Train/Validation特征，Test汇总指标和报告
  figures/                  # 同攻击/迁移ROC、随机基线、消融区间和特征分布
  test/{data,results,...}/   # 独立Test的输入、回答、特征和ASR
  controls/                 # 共享对照输入/扰动及各模型对照结果
```

- 七视图：B、G、R、B+G、B+R、G+R、B+G+R。R 从当前被测模型的输入 prompt 内部提取，不含生成回答；迁移时只用来源参考变换，不重新拟合目标参考。
- 指标：ASR、clean 目标命中率，AUROC、AUPRC、F1、TPR@1%FPR、Accuracy、FPR，以及原有配对 bootstrap、随机基线和消融。poison vs clean、poison vs neutral 分开评价。
- `results/asr_train_validation.csv` 是第三步攻击指标，`results/asr.csv` 是最终 Test 攻击指标。LoRA 按自己的目标前缀判定，Qwen 对照按输入攻击的目标统计命中率，配置见 `pipeline_config.json`。
- 不再设置阻断运行的 ASR/AUROC Gate；攻击是否训练成功须结合 ASR 解释，不能只看检测分数。
- 同题的对照输入和扰动跨模型共享；对应 LoRA 的 clean/poison 回答与 G/R 复用独立 Test 产物，只对新增 neutral 推理。
- 原始推理与首次语义改写仍是整批保存，中断后重做该部分；扰动推理逐条写盘，重跑所在阶段可续跑。完整特征存在时直接复用。改变配置后须清理，不能拿旧缓存替代新模型结果。

函数与四阶段对应见 `docs/EXPERIMENT_MAP.md`，文件字段见 `docs/DATA_SCHEMA.md`。

## 本地检查

```bash
python -m pytest -q
```

CPU检查覆盖四组隔离、攻击目标映射、四阶段衔接、特征数值、来源拟合、固定检测器迁移、Qwen/neutral对照及图表。没有torch时相关数值/推理模拟测试跳过；真实NF4、PEFT训练和GPU推理需在服务器执行。
