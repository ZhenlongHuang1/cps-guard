# Pilot-v2 产物与字段

所有路径相对 `experiments/pilot_v2/`。CSV为UTF-8，label=0 clean、1 poison；标签表示输入是否带触发器，不等于攻击成功。

| 产物 | 关键字段/形状 |
|---|---|
| `data/base_questions.csv` | base_id、source_index、original_instruction、split；400行 |
| `data/split_manifest.csv` | base_id、source_index、split；200 train/100 validation/100 test |
| `data/attack_inputs.csv` | sample_id、pair_id、base_id、split、attack、label、original_instruction、triggered_instruction、trigger、trigger_position、input_text、target_response；1600行 |
| `data/template_audit.csv` | 同样本身份；raw_instruction、formatted_prompt、token_ids/attention_mask（JSON数组）、add_special_tokens=False、representation_scope=prompt_only |
| `results/data_leakage_audit.csv` | 候选编号/文本、参考编号/来源/文本、最大余弦、decision、reviewed、confirmed_near_duplicate；后两项待人工填写 |
| `results/original_inference.csv` | 原始元数据、perturb_type=original、perturb_id=0、model_response、runtime_sec、seed、run_id；1600行 |
| `results/asr_review.csv` | 每攻击配对clean/poison回答、attack_success、clean_target_behavior；800行，自动前缀标记，可抽查 |
| `results/asr.csv` | attack、n_poison、n_success、ASR、clean_target_rate；2行 |
| `results/features_behavioral.csv` | sample_id及身份、semantic/context/position_score、cps_score、randomness_baseline、cps_cal_score；1600行 |
| `results/features_generation.csv` | 身份、entropy_mean/std、top1_mean/min、margin_mean、response_length；自然对数，生成长度含EOS |
| `results/features_representation_raw.csv` | 身份、representation_index、hidden_norm_mean/std、layer_shift_mean/std/max |
| `results/representation_vectors.npy` | float16，(1600,4,2,hidden_size)；顺序通过representation_index关联，不按CSV行号猜测 |
| `results/features_representation.csv` | 身份及mahalanobis_{source}_{dimension}_{mean/max}；每来源32/64各两项 |
| `results/features.csv` | B/G/R全列及来源reference列；model、adapter、seed、git_commit、config_hash；1600行 |
| `configs/frozen_detector_config.json` | 来源、视图、C、PCA维数、列名、阈值、Validation AUROC、checkpoint/feature SHA256及Gate 2 |
| `results/test_predictions.csv` | sample_id、base_id、source_attack、target_attack、view、label、score（poison概率）、threshold、frozen_config_sha256；5600行 |
| `results/iid_results.csv`、`cross_attack_results.csv` | 来源/目标/视图、样本/基础题数、AUROC、AUPRC、F1、TPR_at_1pct_FPR、Accuracy、FPR、TP/FP/TN/FN、threshold；各14行 |
| `results/bootstrap_results.csv` | 每28组AUROC、CI_low/high、重复数和种子 |
| `results/bootstrap_draws.npz` | 所有方法共同的问题抽样权重、base_ids及每组AUROC抽样值 |
| `results/ablation_results.csv` | 来源/目标、移除视图、AUROC_gain、配对gain_CI_low/high；12行 |
| `results/random_seed_results.csv`、`random_baseline.csv` | 每目标50种子的AUROC明细；均值/std/p025/p975汇总 |
| `results/final_report.md` | 攻击有效性、单视图、融合、迁移、工程决策及解释限制 |
| `manifest.json` | 最近完成阶段、UTC时间、代码版本、参数hash、本轮文件SHA256/大小；实验中间文件不上传Git |

Test每目标200条样本（100题×clean/poison）；同题四版本在各split中保持一致。来源独立Mahalanobis列供各自检测器使用，**禁止混用目标攻击reference**。
