# R单视图四阶段

| 阶段 | 函数 | 工作 |
|---|---|---|
| 1 | `data.pilot_v2.select_questions/prepare_pilot` | 固定LoRA训练、检测Train/Validation/Test，精确及语义去重 |
| 1 | `insert_trigger/write_training_data/build_attack_inputs/prepare_control` | 两攻击统一随机词边界及同题位置，各500 clean/500 poison；中性仅用于额外测试 |
| 2 | `model.training.train_adapters/train_one` | 原始Qwen初始化所选LoRA，训练实现与超参数保持不变；`--attacks vpi`仅训练VPI |
| 3、4 | `model.inference.token_batches` | 按实际prompt长度分组，限制每批样本数/填充后token总数，不截断 |
| 3、4 | `model.features.extract_original_features` | 按模型批量提取prompt内部R，再批量生成一次原始回答；恢复原样本顺序 |
| 3、4 | `pool_hidden_states` | mask有效token的mean/last池化、全层范数和相邻层shift五项统计，选层向量 |
| 3、4 | `model.inference.generate_batch` | 原贪心解码，不返回scores/生成隐藏状态，无随机回答 |
| 3 | `methods.features.assemble_features` | 所选来源Train clean拟合PCA/协方差，其他来源加载原参考；组装R七项特征 |
| 3 | `eval.transfer.freeze_detectors` | 所选来源Train拟合scaler/LR，来源Validation自动选C/PCA/阈值；其他检测器保留 |
| 4 | `assemble_features/evaluate_frozen` | Test只加载来源参考和检测器，输出同攻击/双向迁移原指标 |
| 3、4 | `main.summarize_attacks` | 按各LoRA目标前缀报告ASR和clean命中率 |
| 4 | `eval.statistics.* / eval.pilot_plots.plot_pilot` | R配对区间、CPU随机排序基线、报告和四张图，无融合消融 |
| 4 | `main.run_controls / evaluate_control / plot_control` | 所选输入、模型、来源组合；可选neutral，原指标及三张R对照图 |

R来源参考只在Train clean拟合，Test不选参。保留旧第一步配置时，旧B/G参数不再读取；新增批量参数使用main默认值或命令行值。

`scripts/clean_vpi.py`仅清理当前pilot_v2中的旧VPI和合并结果，保留基础问题、BadNet LoRA、参考与检测器。随后STEP 1复用问题划分并保存`--attacks vpi`，STEP 2/3沿用，STEP 4仍双向评价。
