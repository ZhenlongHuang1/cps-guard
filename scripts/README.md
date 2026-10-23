# 脚本用途

当前只运行仓库根目录 `main.py --step 1/2/3/4`，具体命令见根目录 README.md。

`python scripts/clean_vpi.py`清理当前`experiments/pilot_v2`中的旧VPI及合并结果，保留原问题划分和BadNet训练产物。随后从`main.py --step 1 --attacks vpi`开始执行四步。

旧 `prepare_data.py` 和 `train_adapters.py` 已合并到 `src/cps_guard/data/pilot_v2.py` 与 `src/cps_guard/model/training.py`，由统一入口调用，避免重复配置及不同的模型保存/加载路径。
