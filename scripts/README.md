# 脚本用途

`bash scripts/run_vpi_repair_comparison.sh`顺序运行VPI-B检测器净化VPI-A的Top-2、Top-20、Random-2、Random-20。

- 默认来源：`experiments/vpi_semantic_disjoint`。
- 默认输出：来源目录下`repair_b_to_a_comparison/{top2,top20,random2,random20}`，总日志为`run.log`。
- 可用两个位置参数指定来源及输出根目录，`PYTHON_BIN`可指定Python解释器。
- 四组共用问题、候选与固定受害模型；Top只评分一次，随机组不评分。
- 每组独立执行`vpi_repair.py --step 1/2/3/4`，单独训练新防御LoRA；错误时停止后续步骤。

主实验仍使用`main.py --step 1/2/3/4`。`clean_vpi.py`仅对应历史BadNet/固定短语VPI，不能用于当前语义VPI主实验。完整样本与命令见根README及`docs/语义VPI实验说明.md`。
