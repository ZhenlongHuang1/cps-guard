# CPS-Guard Pilot-v2

验证既有 **Qwen2.5-7B-Instruct + BadNet/VPI LoRA** 的跨攻击检测信号。使用 Behavioral（B）、Generation（G）、Representation（R）七种组合和 Logistic Regression。唯一入口为 `main.py`，参数及取值说明位于文件顶部。

本轮不重新生成 LoRA 训练数据、不重新训练 LoRA。旧数据、代码及两个 adapter 已归档到 `experiments/pilot_v1_20261006/`；本轮全部产物保存在 `experiments/pilot_v2/`，该目录不提交 Git。

## 服务器更新

```bash
conda activate /root/envs/cpsguard
cd /root/cps-guard-repo
git -c http.version=HTTP/1.1 pull origin main
python -m pip install -e .
python -m pip check
```

运行前核对顶部路径：Qwen 为 `/root/models/Qwen2.5-7B-Instruct`，MiniLM 为 `/root/models/all-MiniLM-L6-v2`，完整 Alpaca 为 `data/raw/alpaca_data.json`。归档必须含 `data/processed/train_badnet.jsonl`、`train_vpi.jsonl`、`data/raw/alpaca_pilot.json`、`checkpoints/badnet/`、`checkpoints/vpi/`。当前既有 Python 3.10 / torch 2.5.1 CUDA 环境可继续使用。

## 顺序运行五个阶段

先进入 tmux，长时间推理不会随 VS Code 关闭而终止：

```bash
tmux new -s pilot-v2
conda activate /root/envs/cpsguard
cd /root/cps-guard-repo
mkdir -p experiments/pilot_v2/logs
```

随后逐阶段运行。每阶段结束检查产物后再继续。

| STEP | 工作 | 主要产物 |
|---|---|---|
| 1 | 选400个新问题，排除训练/旧测试重叠，近重复审计；按题分200/100/100；生成四版本 | `data/base_questions.csv`、`split_manifest.csv`、`attack_inputs.csv`、`results/data_leakage_audit.csv` |
| 2 | 两套 LoRA 原始贪心推理；同时提取 G/R、token审计，按目标前缀自动统计 ASR | `original_inference.csv`、`asr.csv`、G/R原始特征及向量 |
| 3 | 自动三类扰动、随机基线推理；复用原始回答；算 B 和来源独立 clean reference，合并特征 | `inference.csv`、`features.csv`、PCA/协方差参考 |
| 4 | 来源 Train 拟合 scaler/LR；来源 Validation 选 C/PCA/阈值，冻结14个检测器 | `validation_results.csv`、`configs/frozen_detector_config.json` |
| 5 | 一次正式 Test；双向迁移及 IID；2000次配对 bootstrap、50种子 Random、消融、图表 | `test_predictions.csv`、结果表、五份PDF图、`final_report.md` |

```bash
python -u -c "import main; main.STEP=1; main.main()" 2>&1 | tee experiments/pilot_v2/logs/step1.log
python -u -c "import main; main.STEP=2; main.main()" 2>&1 | tee experiments/pilot_v2/logs/step2.log
cat experiments/pilot_v2/results/asr.csv
```

Gate 1：**两攻击 ASR≥90%，clean 目标行为率≤5%**。未通过时停止并检查攻击，不继续检测。通过后运行：

```bash
python -u -c "import main; main.STEP=3; main.main()" 2>&1 | tee experiments/pilot_v2/logs/step3.log
python -u -c "import main; main.STEP=4; main.main()" 2>&1 | tee experiments/pilot_v2/logs/step4.log
cat experiments/pilot_v2/configs/frozen_detector_config.json
python -u -c "import main; main.STEP=5; main.main()" 2>&1 | tee experiments/pilot_v2/logs/step5.log
cat experiments/pilot_v2/results/final_report.md
```

Gate 2：两个来源各至少一种视图 **Validation AUROC≥0.60**，未通过时 STEP 5 会停止。原方案写 IID Test Gate 后再冻结，存在先看 Test 的风险；本实现将 Gate 前移至 Validation，冻结后统一执行 IID/Cross-Attack Test。0.60 为事先指定的工程门槛。

按 `Ctrl+B` 后按 `D` 离开 tmux；重新进入用 `tmux attach -t pilot-v2`。

## 协议与公式

- 每基础问题生成 BadNet-clean/poison、VPI-clean/poison，同题四版本同一 split。400题共1600输入，Train/Validation/Test分别800/400/400条。
- 精确重叠按来源索引及规范文本排除；近重复使用 MiniLM最大余弦≥0.95保守排除。审计列 `excluded_similarity_pending_review` 表示待人工确认，不能声称已人工复核。排除后的候选才进入400题。
- B：`d=1−cos(φ(y),φ(y'))`；每类扰动平均距离 `S_k`；`CPS=mean(S_semantic,S_context,S_position)`；随机基线 `B0=mean_{r<s} d(y_r,y_s)`；`CPS_cal=CPS−λB0`。视图B包含三分量、CPS、B0、CPS_cal。
- G：生成 token 的处理后 logits转概率，`H=−Σp log p`；汇总 entropy mean/std、top1 mean/min、top1−top2 margin mean、生成长度（含停止EOS，不含prompt）。
- R：prompt全有效token（含聊天模板、不含回答），逐层mean/last pooling；所有Transformer层计算范数mean/std、真实相邻层 `1−cos` 的mean/std/max。预先指定层6/13/20/27保存float16池化向量。
- 每个来源仅用 **该攻击 Train clean**，逐保存层/池化拟合 PCA32/64与LedoitWolf收缩协方差；`d_M=sqrt((z−μ)^TΣ⁻¹(z−μ))`，汇总mean/max。来源模型在目标攻击上仍用来源reference。
- 七视图：B、G、R、B+G、B+R、G+R、B+G+R。`StandardScaler → LogisticRegression`，来源Validation按AUROC选C/PCA，用Youden J最大选概率阈值。最终分数为poison概率，不再手动选择高/低CPS方向。
- Test：14个来源检测器×两目标=28组。AUROC、AUPRC（average precision）、F1、Accuracy及经验ROC中FPR≤1%的最大TPR。F1/Accuracy使用已冻结阈值。
- CI：100个Test基础问题配对重采样，clean/poison及所有方法共享抽样；不重复训练。Random50种子范围是随机排序分布的分位范围，不是均值CI。

## 重跑范围

重复 STEP 5 复用已保存的 `test_predictions.csv`，只重算统计和图表。正式 Test 完成后不能重新 STEP 4 选参；冻结后不能重新 STEP 1–3 改写数据/特征。修改模型、特征定义、C网格、PCA或判定规则，应另建实验目录并事先固定协议。

STEP 2 生成1600个原始回答；STEP 3再生成17600个受害模型回答（6扰动+5随机）及3200次语义改写。合计19200个受害模型回答，因此STEP 3明显慢于STEP 2。

STEP 3中断后保持参数不变，在tmux中重新执行同一命令：自动复用完整的`data/variants.csv`，按样本/扰动类型/编号跳过`inference.csv`已有回答，追加缺失任务，每条写完立即flush。完成后重新计算B和clean reference并合并G/R。STEP 2原始推理和首次语义改写阶段目前没有逐条续跑功能。

函数与方案逐点对照见 [docs/EXPERIMENT_MAP.md](docs/EXPERIMENT_MAP.md)，字段定义见 [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md)。

## 本地测试

```bash
python -m pytest -q
```

CPU测试覆盖去重/分组、G/R已知数值、来源Train拟合、Test不影响选参、冻结预测复用、配对bootstrap和图表。G/R数值与推理缓存测试需要torch；缺少torch时这两项会跳过。真实NF4、PEFT加载与GPU吞吐须在服务器验证。
