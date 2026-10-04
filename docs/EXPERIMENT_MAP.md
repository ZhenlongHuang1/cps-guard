# 实验方案逐项代码索引

本索引对应研究包《CPS-Guard 第一篇 SCI 完整实验方案与可执行流水线》。函数名可在 `src/cps_guard/` 中搜索。所有函数均有中文文档注释，说明作用。**有代码入口不等于已经完成真实 GPU 实验或取得论文结果。**

## 功能文件定位

表中模块路径均相对于 `src/cps_guard/`。

| 功能 | 文件 |
|---|---|
| 数据构建、外部转换、校验读写 | `data/builder.py`、`data/convert.py`、`data/schema.py` |
| 模型加载、提示组织、生成与续跑 | `model/inference.py` |
| 三类扰动与 CPS 计分 | `methods/perturb.py`、`methods/score.py` |
| 四种基线与共同结果字段 | `baselines/random.py`、`baselines/nete.py`、`baselines/onion.py`、`baselines/rap.py`、`baselines/common.py` |
| ASR、检测指标、主结果与消融、绘图 | `eval/asr.py`、`eval/detection.py`、`eval/study.py`、`eval/plots.py` |
| 全部实验命令的分发 | `cli.py` |

## 研究设置、数据和前置验收

| 方案条目 | 实现函数 / 位置 | 产物与验收 |
|---|---|---|
| 模型：先 Llama-2-7B-chat 工程验证，后 Qwen2.5-7B-Instruct | `model.inference._load_config`、`model.inference._load_model`、`model.inference._prompt` | `configs/pilot.example.yaml` 设基模型、各攻击 LoRA 和提示格式；模型/LoRA/任务必须人工核对。 |
| BadNet 单词触发与 VPI 主题触发 | `data.builder.build_alpaca_pilot`；`data.convert.convert_paired_data` | 内置 Alpaca 构造器允许显式传实际触发器；外部真实攻击数据使用显式配对转换。 |
| Pilot 400：两攻击各 100 clean + 100 poison | `data.builder.build_alpaca_pilot(n_base=100)`、`data.schema.read_samples` | `pilot400.csv` 共 400 行；`validate` 检查每个 `pair_id` 的 0/1 配对。 |
| 正式约 2000：两攻击各 500 clean + 500 poison | `data.builder.build_alpaca_pilot(n_base=500)` 或 `data.convert.convert_paired_data` + `data.convert.merge_sample_files` | 仅在 Pilot 有效且数据与 LoRA 匹配后扩大。 |
| 官方 BackdoorLLM 数据转换 | `data.convert.load_source_table`、`data.convert.convert_paired_data`、`data.convert.convert_labeled_data`、`data.convert.merge_sample_files` | 支持 CSV/JSON/JSONL 的双文件或单文件配对格式；明确列名和配对键，拒绝自动猜测标签。 |
| 至少 10 列统一 CSV、唯一 ID、配对关系 | `data.schema.REQUIRED`、`data.schema.read_samples`、`data.schema.write_rows` | 保留 `sample_id/pair_id/base_id`、攻击、标签、原文、输入、目标、攻击成功标记和来源。 |
| `attack_success` 构造阶段留空 | `data.builder.build_alpaca_pilot`、`data.convert.convert_paired_data` | 只在真实回答经过人工判定后由 `eval.asr.apply_asr_annotations` 回填。 |
| 先跑 clean 与 poison 原始输入 | `methods.perturb.build_originals`、`model.inference.run_inference` | 固定解码的原始回答存 `inference.csv`。 |
| 攻击目标判定与 ASR | `eval.asr.asr_review_template`、`eval.asr.compute_asr`、`eval.asr.apply_asr_annotations` | 人工填写 0/1；按攻击输出 ASR，可附 clean 误触发率。ASR 低时停止解释 detector 指标。 |

## CPS-Guard 方法及中间结果

| 方案条目 | 实现函数 / 位置 | 产物与验收 |
|---|---|---|
| Semantic：保持任务意图的两个轻度改写 | `methods.perturb.semantic_review_template`、`methods.perturb._load_semantic`、`methods.perturb.build_variants` | 人工填写 `perturbed_text` 与 `reviewed=1`；自动检查非空和触发器数量，但语义等价必须人工审查。 |
| Context：中性上下文 | `methods.perturb.context_variants`、`methods.perturb.build_variants` | 在任务前/后添加预设背景句；正式使用前抽样审查语义保持。 |
| Position：改变上下文成分的位置 | `methods.perturb.position_variants`、`methods.perturb.build_variants` | 将同一中性背景标记移到不同词边界，保持原词序与触发器；需审查位置效应与标签捷径。 |
| 每条原始 + 每类两个扰动，共 7 版 | `methods.perturb.build_variants(n_variants=2)` | `variants.csv` 每条样本 7 行；N 可调整。 |
| 固定 decoding 的模型输出 | `model.inference._generate(sample=False)`、`model.inference.run_inference` | `do_sample=False`、最多 `max_new_tokens`；保存回答、种子和耗时。 |
| 5 次采样随机性基线 | `model.inference._generate(sample=True)`、`model.inference.run_inference` | `perturb_type=randomness`、`perturb_id` 即重复编号，另存 `run_id/seed/response`。 |
| 断点续跑与配置一致性 | `model.inference._sha256`、`model.inference._existing_keys`、`model.inference.run_inference` | 每行带输入/配置摘要；改变配置或文本时拒绝混用旧回答。 |
| 回答语义差异 D | `methods.score.cosine_distance`、`methods.score.score_embeddings_with_details` | 文本编码向量余弦距离；`logprob_diff/entropy_diff` 未测时留空。 |
| 三类敏感性 S_k 与 CPS 均值 | `methods.score.score_embeddings_with_details`、`methods.score.score_inference` | 每类平均距离与三类均值写 `cps_scores.csv`。 |
| 随机性基线 B(x) 和 CPS_cal | `methods.score.score_embeddings_with_details` | 5 次采样回答两两距离均值；`CPS_cal=CPS-λB`，默认 λ=1。 |
| 保留逐次输入/回答/相似度 | `methods.score.score_inference(details_csv=...)` | `perturbation_details.csv` 用于复核与 N 敏感性。 |

## 基线、统计与论文问题

| 方案条目 | 实现函数 / 位置 | 产物与验收 |
|---|---|---|
| Random 下限 | `baselines.random.random_baseline` | 对同一 `sample_id` 生成稳定随机分数。 |
| NETE 官方接口 | `baselines.nete.prepare_nete`、`baselines.nete.run_nete_official`、`baselines.nete.import_nete_scores` | 导出官方所需 poison-first 数据与行号映射；真正的 NETE 分数须来自作者 `main_detect.py`，再明确分数方向导入。 |
| ONION-adapted | `baselines.onion._language_model_nll`、`baselines.onion.onion_deletion_score`、`baselines.onion.run_onion_adapted` | 参考 LM 逐词删除的最大 NLL 降幅；记录查询数和耗时，标注改编版。 |
| RAP-adapted | `baselines.rap.prepare_rap_variants`、`baselines.rap.score_rap_responses` | 受害模型在中性前缀前后的回答稳定性；标注 response-adapted，不能称原始 RAP。 |
| 基线统一分数格式，越高越可能 poison | `baselines.common.BASELINE_COLUMNS`、`eval.study._attach_samples` | `sample_id/method/attack/label/score/runtime_sec/query_count`；样本覆盖不全或标签错位会报错。 |
| AUROC、F1、Precision、Recall | `eval.detection._threshold`、`eval.detection._bootstrap_auc`、`eval.study.evaluate_long_scores` | 训练集选阈值；测试集按 `base_id` 隔离并按原始指令 bootstrap 置信区间。 |
| RQ1：CPS 对 Random/NETE/ONION/RAP 主比较 | `eval.study.compare_methods` | `main_results.csv` 按 BadNet、VPI、ALL 输出同一切分的指标。 |
| RQ2：三类单独、两两、Full 消融 | `eval.study.ablation_table` | 七种组合各含校正前后，共 14 组；不把完整运行时长伪称为单组件成本。 |
| RQ3：随机性校正是否必要 | `eval.study.compare_methods` 的 CPS 与 CPS-calibrated；`eval.study.ablation_table` 的 `+cal` | 比较原始 CPS 与减去 λB 后的指标。 |
| N={1,3,5,10} 参数敏感性 | `methods.perturb.semantic_review_template(n_variants=10)`、`methods.perturb.build_variants(n_variants=10)`、`eval.study.perturbation_sensitivity` | 必须真正生成/推理 10 个各类变体；少于 N 直接报错，按已测逐扰动耗时计算各 N 的成本。 |
| RQ4：运行时长与查询成本 | `model.inference._generate`、`methods.score.score_embeddings_with_details`、`baselines.*`、`eval.study.evaluate_long_scores` | 主结果表汇总测试集实测时长和查询数；NETE 未给逐样本数据时留空。 |
| ROC、分数分布、运行时间与查询成本图 | `eval.plots.plot_results` | `roc.png`、`score_distribution.png`、`runtime.png`、`query_cost.png`；只画留出测试集。 |
| 误报/漏报检查 | `eval.study.export_detection_errors` | `errors.csv` 包含输入、分量、训练阈值和人工备注空列。 |
| Pilot Go/No-Go | `eval.study.pilot_decision` | 先检查预定的 ASR 门槛，再按方案的 AUROC 档给出扩样建议。 |

## S1–S13 流水线定位

| 阶段 | 命令 | 主要函数 |
|---|---|---|
| S1 数据构造/转换 | `build-pilot`、`convert-paired`、`convert-labeled`、`merge-samples` | `build_alpaca_pilot`、`convert_paired_data`、`convert_labeled_data`、`merge_sample_files` |
| S2 CSV 校验 | `validate` | `read_samples` |
| S3 后门模型推理 | `originals`、`infer` | `build_originals`、`run_inference` |
| S4 ASR | `asr-template`、`asr`、`asr-apply` | `asr_review_template`、`compute_asr`、`apply_asr_annotations` |
| S5 三类扰动 | `semantic-template`、`perturb` | `semantic_review_template`、`build_variants` |
| S6 扰动模型推理 | `infer` | `run_inference` |
| S7 CPS 分数 | `score` | `score_inference`、`score_embeddings_with_details` |
| S8 NETE | `nete-prepare`、`nete-run`、`nete-import` | `prepare_nete`、`run_nete_official`、`import_nete_scores` |
| S9 ONION-adapted | `onion` | `run_onion_adapted` |
| S10 RAP-adapted | `rap-prepare`、`infer`、`rap-score` | `prepare_rap_variants`、`run_inference`、`score_rap_responses` |
| S11 Random | `random` | `random_baseline` |
| S12 主结果 | `compare`、`errors`、`plot`、`pilot-decision` | `compare_methods`、`export_detection_errors`、`plot_results`、`pilot_decision` |
| S13 消融 | `ablation`、`sensitivity` | `ablation_table`、`perturbation_sensitivity` |

## 代码无法替代的研究工作

- 真实基模型、后门 LoRA、训练攻击目标和提示模板要逐一核对；本机没有 GPU 模型权重，尚未验证真实推理。
- 人工定义每种攻击的成功标准，标注全部 poison 回答并审查 clean 误触发。
- 人工写出并核对 semantic 改写；对 context/position 做语义保持抽查。
- NETE 官方仓库的独立环境和真实逐样本结果必须跑通；导出/导入接口不等于完成 NETE 实验。
- 论文中须区分 ONION/RAP 原始分类方法与这里的生成式改编版，并说明适配细节及局限。
