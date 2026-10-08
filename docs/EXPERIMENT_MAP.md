# Pilot-v2：方案步骤与函数对应

方案编号指 `Pilot-v2_本轮实验步骤.docx`；main的STEP是五个执行阶段。

| 方案步骤 | main STEP | 函数（位于src/cps_guard） | 对应算法/功能 |
|---|---|---|---|
| 0 归档v1 | 已完成 | 服务器已有归档；本轮不调用旧训练脚本 | 复用归档LoRA、训练来源及旧100题 |
| 1 新400题及划分 | 1 | `data.pilot_v2.prepare_pilot/select_questions` | 排除旧来源后随机选择400题，按题200/100/100 |
| 2 精确/语义重叠 | 1 | `normalize_text/select_questions` | 来源索引、规范文本精确排除；MiniLM余弦≥0.95保守排除并留待复核 |
| 3 四版本/输入审计 | 1、2 | `build_attack_inputs`、`model.features.extract_original_features`、`model.inference._prompt` | 四版本继承同split，保存触发位置、完整prompt、token_ids、mask |
| 4 原始推理/Gate 1 | 2、3 | `extract_original_features`；`eval.asr.asr_review_template/mark_target_prefix/compute_asr/apply_asr_annotations`；`main` | 固定前缀判定，统计ASR/clean误触发；两攻击≥90%/≤5%才继续 |
| 5 Behavioral | 3 | `methods.perturb.build_variants/semantic_variants/context_variants/position_variants`；`model.inference.run_inference`；`methods.score.score_inference/score_embeddings_with_details` | 6扰动+5随机，复用原始回答；三个平均距离、CPS、随机基线、校正分数 |
| 6 Generation | 2 | `model.features.generation_statistics` | 逐生成token entropy/top1/margin/length六项 |
| 7 Representation | 2 | `pool_hidden_states/extract_original_features` | prompt-only全层范数和相邻层cosine shift；四层mean/last向量 |
| 8 Clean manifold | 3 | `methods.representation.CleanManifold.fit/transform`；`methods.features.assemble_features` | 来源Train clean独立PCA32/64+LedoitWolf；Mahalanobis mean/max |
| 9 合并B/G/R | 3 | `assemble_features` | sample_id一对一合并，保存split、来源reference列及模型/adapter/seed/git/config |
| 10 预处理无泄漏 | 3、4 | `CleanManifold.fit`；`eval.transfer.freeze_detectors` | PCA/reference只Train clean；scaler/LR只来源Train |
| 11 单视图 | 4、5 | `feature_columns/freeze_detectors/evaluate_frozen/detection_metrics` | B、G、R；LR概率、AUROC/AP/F1/TPR@1%FPR |
| 12 融合 | 4、5 | 同上，`VIEWS`指定四种融合 | B+G/B+R/G+R/B+G+R；相同LR与选参策略 |
| 13 IID Gate 2 | 4、5 | `freeze_detectors/evaluate_frozen` | Gate前移至来源Validation，各至少一视图≥0.60；正式IID Test在冻结后 |
| 14 双向迁移 | 5 | `evaluate_frozen` | BadNet→VPI、VPI→BadNet均七视图，目标Train/Validation不参与来源检测器 |
| 15 冻结规则/Test一次 | 4、5 | `validation_threshold/freeze_detectors/evaluate_frozen` | Validation选C/PCA/Youden阈值，绑定SHA256，保存并复用Test预测 |
| 16 配对bootstrap | 5 | `eval.statistics.grouped_auc/summary_statistics` | 按base_id共享2000次抽样，AUROC百分位95%CI |
| 17 Random | 5 | `summary_statistics` | 每目标50独立种子随机排序，均值/std/2.5%–97.5%分位 |
| 18 三组消融 | 5 | `summary_statistics` | 融合减G+R/B+R/B+G，与相同bootstrap配对求增益CI |
| 19 工程决策/报告 | 5 | `write_final_report` | 双向迁移、单视图、互补性、Random、GO/Conditional/NO-GO |
| 五类图表 | 5 | `eval.pilot_plots.plot_pilot` | IID/迁移ROC、Random、融合消融区间、特征分布 |

## 与文字方案的明确约定

1. 避免先看Test再冻结，Gate 2用Validation判定；0.60门槛事先固定。
2. 高相似候选直接保守排除并记录待复核状态；没有模拟人工近重复确认。
3. R观测范围固定为输入prompt；选层和池化方式在Test之前确定。
4. AUPRC采用average precision；TPR@1%FPR为经验ROC上不超过1%误报的最大TPR。
5. GO判据是工程判据。最佳单视图事后比较属于探索性分析，未作多重比较修正。
