#!/usr/bin/env bash
# 顺序运行B检测器净化A模型的Top-2、Top-20、Random-2、Random-20四组实验。
# 用法：bash scripts/run_vpi_repair_comparison.sh [主实验目录] [四组输出根目录]
# 第一组生成500训练/100验证问题、1000候选并评分；其余组复用这批数据。
# Top-20复用Top-2的逐题评分；随机组从完整池抽样，不计算或读取评分。
# 每组各自新建防御LoRA，1000条正常答案监督、1轮、学习率5e-5；不串接训练。
# 每步由独立Python进程执行，退出后释放模型，发生错误时停止后续步骤。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python_bin="${PYTHON_BIN:-python}"
source_dir="${1:-experiments/vpi_semantic_disjoint}"
suite_dir="${2:-$source_dir/repair_b_to_a_comparison}"
mkdir -p "$suite_dir"
exec > >(tee -a "$suite_dir/run.log") 2>&1

# 限制CPU线程竞争；推理与情感评价使用GPU，训练参数保持与各组一致。
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

for experiment in "top 2 top2" "top 20 top20" "random 2 random2" "random 20 random20"; do
    read -r selection count name <<< "$experiment"
    output_dir="$suite_dir/$name"
    reuse_args=()
    if [[ "$name" != "top2" ]]; then
        reuse_args=(--reuse-from "$suite_dir/top2")
    fi
    echo "$(date -Is) 开始 $name：受害模型=vpi_a，检测器=vpi_b"
    "$python_bin" -u vpi_repair.py --step 1 --source-dir "$source_dir" \
        --output-dir "$output_dir" --victim vpi_a --detector-source vpi_b \
        --selection "$selection" --n-selected "$count" "${reuse_args[@]}"
    for step in 2 3 4; do
        "$python_bin" -u vpi_repair.py --step "$step" --source-dir "$source_dir" --output-dir "$output_dir"
    done
    echo "$(date -Is) 完成 $name：$output_dir/results/evaluation.csv"
done
echo "$(date -Is) 四组净化实验全部完成。"
