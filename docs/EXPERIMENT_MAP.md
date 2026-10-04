# 最小 Pilot：实验方案与代码对照

本索引只覆盖当前保留的基本实验。数据范围是 Alpaca 的 100 个原始问题、一个基模型、两套 BadNet/VPI LoRA：各攻击 100 clean + 100 poison，共 400 条输入。

| 实验点 / 阶段 | 对应函数 | 完成的功能或公式 |
|---|---|---|
| S1 数据准备、配对 | `data.builder._render`、`build_alpaca_pilot` | 渲染 instruction/input，抽样同一批问题；BadNet 插入词，VPI 添加主题短语；总数 4×N_BASE |
| 第七节 CSV 保存 | `data.schema.read_samples`、`write_rows` | 保留空判定字段、读取标签及写出 CSV，无评分公式 |
| S3 原始输入 | `methods.perturb.build_variants(n_variants=0)` | 每样本只输出 original，无额外原始输入函数 |
| S3 / S6 模型加载 | `model.inference._load_model` | 加载基模型；指定 attack 时挂载对应 LoRA，不指定时用于语义改写 |
| 第十二节提示与生成 | `model.inference._prompt`、`_generate` | 匹配训练提示格式；主检测固定贪心解码，随机性基线使用采样 |
| S3 / S6 推理 | `model.inference.run_inference` | 每攻击加载一次，逐条写回答、生成耗时；完整检测每样本 1+3N+R 次 |
| S4 ASR 判定准备 | `eval.asr.asr_review_template` | 将 poison 和配对 clean 回答并列，供人工判断真实目标行为 |
| S4 ASR 统计 | `eval.asr.compute_asr` | ASR=成功 poison 数/poison 总数；可选统计 clean 目标行为比例 |
| S4 判定回填 | `eval.asr.apply_asr_annotations` | 将人工 0/1 回填 attack_success；不改变带触发器标签 label |
| S5 Semantic / 第三节 1 | `methods.perturb.semantic_variants` | 占位触发器→基模型改写→恢复触发器；每样本生成 N 个等价改写，需抽查质量 |
| S5 Context / 第三节 1 | `methods.perturb.context_variants` | 原请求前后加固定中性背景句，N=2 |
| S5 Position / 第三节 1 | `methods.perturb.position_variants` | 移动中性背景标记，保持任务词序及完整触发器 |
| S5 三类汇总 | `methods.perturb.build_variants(n_variants=2)` | original+三类×2，共 7 版；不读取人工语义表 |
| S7 回答差异 / 第三节 2 | `methods.score.cosine_distance` | D(a,b)=1−(a·b)/(‖a‖‖b‖)，a/b 为回答向量 |
| S7 三类分量 / 第三节 2 | `methods.score.score_embeddings_with_details` | S_k=(1/N)Σ_j D(f(x),f(T_k,j(x))) |
| S7 CPS / 第三节 3 | 同上 | CPS=(S_semantic+S_context+S_position)/3 |
| S7 随机性基线 / 第三节 4 | 同上 | B=[2/(R(R−1))]Σ_(r<s)D(y_r,y_s)，默认 R=5 |
| S7 校正 / 第三节 4 | 同上 | CPS_cal=CPS−λB，默认 λ=1，允许负分 |
| S7 文本编码与结果保存 | `methods.score.score_inference` | 回答经 SentenceTransformer 编码后，调用唯一评分函数，保存汇总和逐扰动明细 |
| S11 Random / 第十五节 | `baselines.random.random_baseline` | seed/sample_id 映射为可复现随机分数，不查询模型 |
| S12 测试划分 | `eval.detection.select_test_ids` | 按 base_id 留出 30%，同问题所有攻击和标签共用划分 |
| S12 阈值 | `eval.detection._threshold` | 仅训练集选择最大 Youden J=TPR−FPR 的有限阈值 |
| S12 基本评价 / 第十六节 | `eval.detection.evaluate_scores` | 固定比较 CPS、CPS-calibrated、Random；每方法 badnet/vpi/ALL，9 行指标 |
| S12 ROC / 分布图 | `eval.plots.plot_results` | 与评价共享测试划分，写 roc.png、score_distribution.png |
| 运行组织 | 根目录 `main` | STEP=1 数据+ASR 推理，STEP=2 统计，STEP=3 检测+比较+图表 |

分量多变体平均、自动语义改写、具体扰动模板、30% 分组留出与 Youden 阈值是本实现的具体选择。
输入输出和公式详细说明以函数中文注释为准。随机性校正 B 与 Random 检测基线是两件不同的计算。

## 当前产物

| 步骤 | 文件 |
|---|---|
| 1 | `data/processed/samples.csv`、`originals.csv`；`results/original_inference.csv`、`asr_review.csv` |
| 2 | `results/asr.csv`、`data/processed/samples_adjudicated.csv` |
| 3 | `data/processed/variants.csv`；`results/inference.csv`、`cps_scores.csv`、`perturbation_details.csv`、`random.csv`、`main_results.csv`；两张 PNG |

指标表耗时和查询数仅包含受害模型回答生成。语义改写准备与回答编码不在这项成本内。
检测评价包含所有 clean/带触发器样本，不能把仅包含攻击成功样本的结果混作总体指标。

## 本轮移除

外部格式转换、额外三个基线、消融、次数敏感性、错误分析导出、自动扩样判断、bootstrap 区间、成本图、通用方法长表接口，以及空的 logprob/entropy 明细列。
本地测试不能代替真实后门权重的 GPU 实验。
