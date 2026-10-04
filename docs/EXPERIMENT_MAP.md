# 实验方案逐项代码索引

本索引对应研究包《CPS-Guard 第一篇 SCI 完整实验方案与可执行流水线》。函数名可在 `src/cps_guard/` 中搜索。所有函数均有中文文档注释，优先说明方案章节、S1–S13 阶段、算法和公式，再说明逐个输入参数、返回类型、输出字段与单位；较长函数按处理阶段添加注释。**有代码入口不等于已经完成真实 GPU 实验或取得论文结果。**

## 核心算法与公式定位

章节编号来自原始实验方案。`x` 为原始输入，`f` 为受害模型，`T_k,j` 为第 k 类第 j 个扰动，`φ` 为回答语义编码器，`R` 为原始输入的随机生成次数。

| 方案位置 / 功能 | 公式或实现 | 唯一计算函数 |
|---|---|---|
| 第三节 2：回答差异 D | `D(y,y′)=1−cos(φ(y),φ(y′))` | `methods.score.cosine_distance` |
| 第三节 2：基础敏感性 S_k | 方案原式为 `S_k=D(f(x),f(T_k(x)))`；本实现对同类多个变体取均值 `S_k=(1/N_k)Σ_j D(f(x),f(T_k,j(x)))` | `methods.score.score_embeddings_with_details` |
| 第三节 3：CPS | `CPS=(S_semantic+S_context+S_position)/3` | `methods.score.score_embeddings_with_details` |
| 第三节 4、第十二节：随机性基线 B | `B=[2/(R(R−1))]Σ_{r<s}D(y_r,y_s)`，默认 R=5，共 10 对 | `methods.score.score_embeddings_with_details` |
| 第三节 4：随机性校正 | `CPS_cal=CPS−λB`，默认 λ=1，校正分数可为负 | `methods.score.score_embeddings_with_details` |
| 第三节 1、第十三节：T_semantic | 干净基模型自动改写，触发器占位后还原 | `methods.perturb.semantic_variants`，由 `build_variants` 组织 |
| 第三节 1、第十三节：T_context | 固定中性句交替放在请求前后 | `methods.perturb.context_variants` |
| 第三节 1、第十三节：T_position | 选择“移动上下文成分”方式：在不同安全词边界放置同一背景标记 | `methods.perturb.position_variants` |
| 第十一节：ASR | 成功 poison 数 / 全部 poison 数 | `eval.asr.compute_asr` |
| 第十五节 2：ONION-adapted | 本实现选 `max(0,max_i[NLL(x)−NLL(delete_i(x))])` | `baselines.onion.onion_deletion_score` |
| 第十五节 3：RAP-adapted | 本实现选固定前缀后的回答余弦相似度 | `baselines.rap.score_rap_responses` |
| 第十六节：score+threshold | 本实现选训练集 Youden J 最大阈值，测试集 `score≥t` 判 poison | `eval.detection._threshold`，指标由 `evaluate_long_scores` 计算 |
| 第十七节：七种分量消融 | `CPS_A=(1/|A|)Σ_{k∈A}S_k`，分别校正前后，共 14 版 | `eval.study.ablation_table` |
| 第十七节：N={1,3,5,10} | 每类用前 N 个实测距离重算 CPS，`query_N=1+3N+R` | `eval.study.perturbation_sensitivity` |

ONION 最大降幅、RAP 固定前缀与相似度、Youden 阈值、base_id 留出与簇 bootstrap 是本仓库的具体实现选择；原方案给出功能要求，未给出这些细节公式。函数注释中明确说明了这一区别。

`score_embeddings` 这个仅转调并丢弃明细的函数已删除。计分只保留 `score_embeddings_with_details`；`score_inference` 负责回答编码和 CSV 输出，承担单独的数据处理步骤。全部运行参数在根目录 `main.py` 顶部赋值，模型配置以 `MODEL_CONFIG` 字典直接传入。

## 功能文件定位

表中变体构造统一为 `build_variants`：N=0 输出 original，N>0 自动生成三类变体。已删除 `build_originals`、`semantic_review_template`、`_load_semantic`，不再读取人工语义表。Semantic 的模型生成是本实现选择，实际质量需抽查；准备成本为每样本 N 次干净基模型生成。

模块路径均相对于 `src/cps_guard/`。

| 功能 | 文件 |
|---|---|
| 数据构建、外部转换、校验读写 | `data/builder.py`、`data/convert.py`、`data/schema.py` |
| 模型加载、提示组织与生成 | `model/inference.py` |
| 三类扰动与 CPS 计分 | `methods/perturb.py`、`methods/score.py` |
| 四种基线与共同结果字段 | `baselines/random.py`、`baselines/nete.py`、`baselines/onion.py`、`baselines/rap.py`、`baselines/common.py` |
| ASR、检测指标、主结果与消融、绘图 | `eval/asr.py`、`eval/detection.py`、`eval/study.py`、`eval/plots.py` |
| 参数赋值与按阶段执行 | 根目录 `main.py`（`main`） |

## 研究设置、数据和前置验收

| 方案条目 | 实现函数 / 位置 | 产物与验收 |
|---|---|---|
| 模型：先 Llama-2-7B-chat 工程验证，后 Qwen2.5-7B-Instruct | `model.inference._load_model`、`model.inference._prompt` | `main.py` 顶部 `MODEL_CONFIG` 设基模型、各攻击 LoRA 和提示格式；模型/LoRA/任务必须人工核对。 |
| BadNet 单词触发与 VPI 主题触发 | `data.builder.build_alpaca_pilot`；`data.convert.convert_paired_data` | 内置 Alpaca 构造器允许显式传实际触发器；外部真实攻击数据使用显式配对转换。 |
| Pilot 400：两攻击各 100 clean + 100 poison | `data.builder.build_alpaca_pilot(n_base=100)`、`data.schema.validate_samples` | `pilot400.csv` 共 400 行；`validate_samples` 检查每个 `pair_id` 的 0/1 配对。 |
| 正式约 2000：两攻击各 500 clean + 500 poison | `data.builder.build_alpaca_pilot(n_base=500)` 或 `data.convert.convert_paired_data` + `data.convert.merge_sample_files` | 仅在 Pilot 有效且数据与 LoRA 匹配后扩大。 |
| 官方 BackdoorLLM 数据转换 | `data.convert.load_source_table`、`data.convert.convert_paired_data`、`data.convert.convert_labeled_data`、`data.convert.merge_sample_files` | 支持 CSV/JSON/JSONL 的双文件或单文件配对格式；明确列名、标签和配对方式，转换后单独执行 `validate_samples`。 |
| 至少 10 列统一 CSV、唯一 ID、配对关系 | `data.schema.REQUIRED`、`data.schema.read_samples`、`data.schema.validate_samples`、`data.schema.write_rows` | 保留 `sample_id/pair_id/base_id`、攻击、标签、原文、输入、目标、攻击成功标记和来源。 |
| `attack_success` 构造阶段留空 | `data.builder.build_alpaca_pilot`、`data.convert.convert_paired_data` | 只在真实回答经过人工判定后由 `eval.asr.apply_asr_annotations` 回填。 |
| 先跑 clean 与 poison 原始输入 | `methods.perturb.build_variants(n_variants=0)`、`model.inference.run_inference` | 无需模型生成变体；原始回答存 `original_inference.csv`。 |
| 攻击目标判定与 ASR | `eval.asr.asr_review_template`、`eval.asr.compute_asr`、`eval.asr.apply_asr_annotations` | 人工填写 0/1；按攻击输出 ASR，可附 clean 误触发率。ASR 低时停止解释 detector 指标。 |

## CPS-Guard 方法及中间结果

| 方案条目 | 实现函数 / 位置 | 产物与验收 |
|---|---|---|
| Semantic：保持任务意图的两个轻度改写 | `methods.perturb.semantic_variants`、`methods.perturb.build_variants` | 用未挂载攻击 LoRA 的基模型采样改写；占位符保护触发器，生成后还原，抽查等价性。 |
| Context：中性上下文 | `methods.perturb.context_variants`、`methods.perturb.build_variants` | 在任务前/后添加预设背景句；正式使用前抽样审查语义保持。 |
| Position：改变上下文成分的位置 | `methods.perturb.position_variants`、`methods.perturb.build_variants` | 将同一中性背景标记移到不同词边界，保持原词序与触发器；需审查位置效应与标签捷径。 |
| 每条原始 + 每类两个扰动，共 7 版 | `methods.perturb.build_variants(n_variants=2)` | `variants.csv` 每条样本 7 行；N 可调整。 |
| 固定 decoding 的模型输出 | `model.inference._generate(sample=False)`、`model.inference.run_inference` | `do_sample=False`、最多 `max_new_tokens`；保存回答、种子和耗时。 |
| 5 次采样随机性基线 | `model.inference._generate(sample=True)`、`model.inference.run_inference` | `perturb_type=randomness`、`perturb_id` 即重复编号，另存 `run_id/seed/response`。 |
| 完整推理结果输出 | `model.inference.run_inference` | 每次覆盖写出指定结果 CSV；原始、CPS 与 RAP 推理分别使用不同文件。 |
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
| 基线统一分数格式，越高越可能 poison | `baselines.common.BASELINE_COLUMNS`、`eval.study._attach_samples` | `sample_id/method/attack/label/score/runtime_sec/query_count`；各方法输入应完整覆盖相同样本；元数据按 `sample_id` 从统一表关联。 |
| 所有评价共用分组划分 | `eval.detection.select_test_ids` | 主评价、绘图和错误分析均按同一 seed/test_fraction 选择独立原始问题。 |
| AUROC、F1、Precision、Recall | `eval.detection._threshold`、`eval.detection._bootstrap_auc`、`eval.detection.evaluate_long_scores` | 训练集选阈值；测试集按 `base_id` 隔离并按原始指令 bootstrap 置信区间。 |
| RQ1：CPS 对 Random/NETE/ONION/RAP 主比较 | `eval.study.compare_methods` | `main_results.csv` 按 BadNet、VPI、ALL 输出同一切分的指标。 |
| RQ2：三类单独、两两、Full 消融 | `eval.study.ablation_table` | 七种组合各含校正前后，共 14 组；不把完整运行时长伪称为单组件成本。 |
| RQ3：随机性校正是否必要 | `eval.study.compare_methods` 的 CPS 与 CPS-calibrated；`eval.study.ablation_table` 的 `+cal` | 比较原始 CPS 与减去 λB 后的指标。 |
| N={1,3,5,10} 参数敏感性 | `methods.perturb.build_variants(n_variants=10)`、`eval.study.perturbation_sensitivity` | 自动生成三类各 10 条并实测推理；后续按前 N 条距离及耗时计算指标。 |
| RQ4：运行时长与查询成本 | `model.inference._generate`、`methods.score.score_embeddings_with_details`、`baselines.*`、`eval.detection.evaluate_long_scores` | 主结果表汇总测试集实测时长和查询数；NETE 未给逐样本数据时留空。 |
| ROC、分数分布、运行时间与查询成本图 | `eval.plots.plot_results` | `roc.png`、`score_distribution.png`、`runtime.png`、`query_cost.png`；只画留出测试集。 |
| 误报/漏报检查 | `eval.study.export_detection_errors` | `errors.csv` 包含输入、分量、训练阈值和人工备注空列。 |
| Pilot Go/No-Go | `eval.study.pilot_decision` | 先检查预定的 ASR 门槛，再按方案的 AUROC 档给出扩样建议。 |

## S1–S13 流水线定位

在根目录 main.py 顶部将对应开关设为 True，直接运行 main.py。多个开关按 main() 顺序执行。函数无需命令行参数，配置直接由顶部变量提供。

| 阶段 | main.py 开关 | 主要函数 |
|---|---|---|
| S1 数据构造/转换 | RUN_PREPARE_DATA | build_alpaca_pilot、convert_paired_data、convert_labeled_data、merge_sample_files |
| S2 CSV 校验 | RUN_PREPARE_DATA | validate_samples |
| S3 原始模型推理 | RUN_ASR_INFERENCE | build_variants(N=0)、run_inference、asr_review_template |
| S4 ASR | RUN_ASR_STATISTICS | compute_asr、apply_asr_annotations |
| S5 三类扰动 | RUN_CPS | semantic_variants、context_variants、position_variants、build_variants |
| S6 CPS 推理 | RUN_CPS | run_inference |
| S7 CPS 分数 | RUN_CPS | score_inference、score_embeddings_with_details、evaluate_scores |
| S8 NETE | RUN_NETE_PREPARE / RUN_NETE_DETECT / RUN_NETE_IMPORT | prepare_nete、run_nete_official、import_nete_scores |
| S9 ONION-adapted | RUN_ONION | run_onion_adapted |
| S10 RAP-adapted | RUN_RAP | prepare_rap_variants、run_inference、score_rap_responses |
| S11 Random | RUN_RANDOM | random_baseline |
| S12 主比较 | RUN_COMPARE | compare_methods |
| S12 分析/绘图/决策 | RUN_ERROR_ANALYSIS / RUN_PLOTS / RUN_PILOT_DECISION | export_detection_errors、plot_results、pilot_decision |
| S13 消融 | RUN_ABLATION / RUN_SENSITIVITY | ablation_table、perturbation_sensitivity |

## 代码无法替代的研究工作

- 真实基模型、后门 LoRA、训练攻击目标和提示模板要逐一核对；本机没有 GPU 模型权重，尚未验证真实推理。
- 人工定义每种攻击的成功标准，标注全部 poison 回答并审查 clean 误触发。
- 自动生成的 semantic 需抽样核对任务意图及触发器保持；context/position 同样需要语义保持抽查。
- NETE 官方仓库的独立环境和真实逐样本结果必须跑通；导出/导入接口不等于完成 NETE 实验。
- 论文中须区分 ONION/RAP 原始分类方法与这里的生成式改编版，并说明适配细节及局限。
