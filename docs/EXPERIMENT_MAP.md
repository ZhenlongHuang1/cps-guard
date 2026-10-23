# 四阶段与函数对应

| 阶段 | 函数 | 工作 |
|---|---|---|
| 1 数据准备 | `data.pilot_v2.select_questions/prepare_pilot` | 一次固定LoRA训练、检测Train/Validation/Test四组，完成精确和语义去重 |
| 1 数据准备 | `write_training_data/build_attack_inputs/prepare_control` | 生成两种独立目标的训练回答、配对攻击输入和可选中性对照 |
| 2 LoRA训练 | `model.training.train_adapters/train_one` | 分别从原始Qwen初始化、训练、保存两套LoRA，沿用原训练超参数 |
| 3 检测器训练 | `main.extract_features` | 共用原始推理、扰动、B/G/R组装流程；本阶段输入只有Train/Validation |
| 3 检测器训练 | `model.features.extract_original_features` | 固定解码生成原始回答与G，另作prompt forward提取R原始统计及池化向量 |
| 3 检测器训练 | `methods.perturb.build_variants` | 干净基模型生成语义改写，规则生成背景和位置扰动 |
| 3 检测器训练 | `model.inference.run_inference` | 扰动回答和随机回答，复用原始回答，支持逐任务续跑 |
| 3 检测器训练 | `methods.features.assemble_features` | B六项、G六项、R五项；来源Train clean拟合PCA/协方差，增加来源Mahalanobis均值/最大值 |
| 3 检测器训练 | `eval.transfer.freeze_detectors` | 两来源各七视图；来源Train拟合scaler/LR，来源Validation选C/PCA/Youden阈值 |
| 4 迁移测试 | `main.extract_features` | 处理独立Test，只加载第三步参考，不重新拟合PCA/协方差 |
| 3、4 攻击评价 | `main.summarize_attacks` | 按样本自己的目标前缀统计各LoRA的ASR和clean命中率 |
| 4 迁移测试 | `eval.transfer.evaluate_frozen` | 同攻击及BadNet→VPI、VPI→BadNet，保持来源参数和阈值 |
| 4 统计绘图 | `eval.statistics.* / eval.pilot_plots.plot_pilot` | 原配对bootstrap、随机基线、融合消融、报告及五类图 |
| 4 模型对照 | `main.run_controls / eval.transfer.evaluate_control` | 所选输入×模型，共用固定来源检测器；可选neutral，使用原指标及四类图 |

整个流程不再依赖旧训练或测试归档。修改训练目标或特征定义后，应按README清理并从第一步运行。
