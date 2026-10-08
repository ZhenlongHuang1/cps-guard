# 脚本用途

当前 Pilot-v2 只运行仓库根目录 `main.py`，STEP=1～5，具体命令见根目录 README.md。

`prepare_data.py` 与 `train_adapters.py` 是上一轮 LoRA 训练工具，保留用于回溯。本轮复用已归档的 BadNet/VPI LoRA，不调用这两个脚本。直接重新运行会生成旧协议训练数据或改写旧模型目录，不能视作当前400题检测实验的一部分。
