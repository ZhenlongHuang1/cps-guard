# 统一数据字段

每行代表一条原始输入，`label=0` 为正常输入，`label=1` 为带触发器输入。`pair_id` 连接同一攻击下的正常与触发版本；`base_id` 连接同一原始任务在不同攻击下的版本。每个 `pair_id` 恰有两行。评估留出集按 `base_id` 切分，避免同一个原始问题同时进入训练和测试。

| 字段 | 含义 |
|---|---|
| `sample_id` | 全局唯一行 ID |
| `pair_id` | 一对正常/触发输入的 ID |
| `base_id` | 原始任务 ID |
| `dataset` | 数据集名称 |
| `attack` | `badnet` 或 `vpi` |
| `label` | 0 正常，1 带触发器 |
| `trigger_type` | `none`、`word`、`topic` 等 |
| `trigger` | 原样保留的触发词，正常行为空 |
| `clean_text` | 未加触发器的原始输入 |
| `input_text` | 实际送入模型的输入 |
| `target_response` | 攻击预定目标，可在数据构造时留空 |
| `attack_success` | 真实模型推理后判定的 0/1；数据构造时留空 |
| `source` | 可追溯来源 |

生成数据的 `label=1` 只说明输入含有触发器，**不代表攻击成功**。`attack_success` 应当由对应后门模型的实际回答判定。
