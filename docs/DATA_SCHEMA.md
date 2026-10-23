# 四阶段产物

以下路径相对于 `experiments/pilot_v2`。

| 路径 | 主要字段或用途 |
|---|---|
| `data/base_questions.csv` | base_id、source_index、original_instruction、split；默认900题，split为lora_train/train/validation/test |
| `data/split_manifest.csv` | base_id、source_index、split；同题版本不能跨组 |
| `data/train_{badnet,vpi}.jsonl` | source_index、attack、label、input_text、response、sample_type、context_position；只使用lora_train问题 |
| `data/attack_inputs.csv` | 仅检测Train/Validation；sample_id、pair_id、base_id、attack、label、input_type、trigger、input_text、target_response及来源信息 |
| `test/data/attack_inputs.csv` | 独立Test的两攻击clean/poison；默认400行 |
| `test/data/base_questions.csv` | 独立Test基础问题，默认100题 |
| `*/data/variants.csv` | sample_id、pair_id、base_id、attack、label、trigger、perturb_type、perturb_id、perturbed_text |
| `*/results/original_inference.csv` | 样本信息、原始回答、生成时间与种子 |
| `*/results/inference.csv` | 扰动输入、回答、时间、种子；按sample_id/perturb_type/perturb_id续跑 |
| `*/results/representation_vectors.npy` | 选定层的mean/last向量，float16；由representation_index关联样本 |
| `*/results/features.csv` | 样本与split、B/G/R统计量、来源PCA/Mahalanobis列；Test只应用已有参考 |
| `configs/pipeline_config.json` | 第一步固定的路径、种子、训练超参数、触发器、各攻击目标、对照配置 |
| `configs/frozen_detector_config.json` | 来源、视图、特征列、C、PCA维数、阈值、Validation AUROC与检测器文件名 |
| `results/test_predictions.csv` | sample_id、base_id、source_attack、target_attack、view、label、score、threshold；默认5600行 |
| `results/iid_results.csv` | 两来源各七视图的同攻击测试指标 |
| `results/cross_attack_results.csv` | 双向迁移各七视图的原指标 |
| `results/asr_train_validation.csv` | 第三步两LoRA的攻击成功率及clean命中率 |
| `results/asr.csv` | 最终独立Test两LoRA的攻击成功率及clean命中率 |
| `controls/{attack}_{neutral,pair}/data/` | 所有对照模型共享的固定输入和扰动 |
| `controls/{attack}_{model}_{neutral,pair}_from_{source}/` | 每个配置的回答、特征、目标命中率、control_results.csv、图表和报告 |

clean/neutral的label=0，poison的label=1；label表示输入构造类型，不代表攻击成功。目标回答前缀由攻击映射确定，实际成功由原始回答判定。模型对照中的LoRA按自身目标评价；原始Qwen统计输入攻击目标的命中率。
