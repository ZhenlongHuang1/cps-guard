# 语义VPI与历史R实验产物

当前语义实验在`experiments/vpi_semantic/`，配置为`configs/vpi_config.json`。训练记录含base_id/source_index/input_text/response/sample_type；学生输入无虚拟指令，教师请求只在teacher_*中保存。

`data/base_questions.csv`的attack记录问题所属的vpi_a/vpi_b；两模型正常问题分别抽取且全局去重。默认每模型正常问题10300道（LoRA 9900、检测Train 200、Validation 100、Test 100），另有500道场景问题；全表21600道，base_id/source_index全局唯一。两模型之间、各用途之间均不复用问题，原始Qwen对照仍复用对应受害模型的Test输入。

检测输入label/scenario_member表示预设场景；question_id是真实问题编号，base_id为统计位置，不表示两类是同题改写。实际行为另存answers_test.csv：code_marker_hit/marker_anywhere/python_syntax_valid及negative_probability/negative_prediction/sentiment_truncated，汇总为behavior_test.csv。

新净化为固定合并受害基座＋独立checkpoints/defense，保存training_provenance.json和merge_comparison.json；evaluation.csv含original/before/after。完整解释见语义VPI实验说明.md。

以下为历史固定短语实验：

路径相对于`experiments/pilot_v2/`。

| 文件 | 字段或用途 |
|---|---|
| `data/base_questions.csv` | base_id、source_index、original_instruction、split，默认900题；四组问题不交叉 |
| `data/split_manifest.csv` | base_id、source_index、split |
| `data/train_{badnet,vpi}.jsonl` | source_index、attack、label、input_text、response、sample_type、context_position；各1000行，只有original_clean/poison，无中性训练记录 |
| `data/attack_inputs.csv` | Train/Validation的sample_id、pair_id、base_id、split、attack、label、input_type、trigger、input_text、target_response及来源 |
| `test/data/attack_inputs.csv` | 独立Test两攻击clean/poison，默认400行 |
| `*/results/original_inference.csv` | 样本元数据、perturb_type=original、原始回答、均摊runtime_sec、实际batch_size及seed；每输入只一条 |
| `*/results/features_representation_raw.csv` | sample_id、base_id、split、attack、label、representation_index与五项R统计 |
| `*/results/representation_vectors.npy` | float16，形状(样本数,选层数,2,hidden_size)，mean/last池化；由representation_index关联 |
| `*/data/template_audit.csv` | formatted_prompt、填充后的token_ids和attention_mask、representation_scope=prompt_only |
| `*/results/features.csv` | 样本元数据、五项R统计及各来源各维数Mahalanobis均值/最大值；每个检测器只读自己的七项 |
| `configs/pipeline_config.json` | 第一步固定的路径、种子、训练参数、触发器和目标映射、对照配置；attacks指定前三步更新的攻击，第四步始终双向测试；旧配置默认两种攻击 |
| `configs/frozen_detector_config.json` | 两个R检测器的来源、特征列、C、PCA维数、固定阈值与Validation AUROC |
| `results/test_predictions.csv` | sample_id、base_id、source_attack、target_attack、view=R、label、score、threshold，默认800行 |
| `results/iid_results.csv` | 两个同攻击R结果 |
| `results/cross_attack_results.csv` | BadNet→VPI、VPI→BadNet两条R结果 |
| `results/asr_train_validation.csv` | 第三步ASR及clean目标命中率 |
| `results/asr.csv` | 最终Test ASR及clean目标命中率 |
| `controls/{attack}_{neutral,pair}/data/` | 共享对照输入；无扰动表 |
| `controls/{attack}_{model}_{neutral,pair}_from_{source}/` | 各模型回答、R、ASR命中率、control_results.csv及三张图 |

clean/neutral的label=0，poison的label=1，表示输入构造类型，实际攻击成功另按回答判定。原始Qwen不挂载LoRA；对照ASR表的target_response明确记录本次检查的目标。R不含生成回答，也不包含padding。

batch_size是实际批次大小；runtime_sec是该批生成时间/样本数，供吞吐统计，不是单请求时延。
