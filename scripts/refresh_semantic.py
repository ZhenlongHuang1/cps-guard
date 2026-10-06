"""修正 Semantic 后，只更新语义变体和对应回答，保留其他已生成结果。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as experiment


def main() -> None:
    """重做 S5/S6 中的 Semantic，合并到已有变体和回答并清除旧向量缓存。

    实验方案对应：
        第三节 Semantic 非触发器上下文改写；只替换 S5/S6 的语义部分。

    算法/公式：
        按当前 main 配置生成三类输入后只保留 Semantic；对这些输入进行固定
        解码，查询数=样本数×N。将旧表中的 Semantic 替换成新行，其余输入和
        回答原样保留；不生成 original、Context、Position 或 randomness 回答。

    输入：
        无参数；读取 main.py 的模型与扰动配置，以及 samples_adjudicated.csv、
        variants.csv 和 inference.csv。样本、模型、N 和其他扰动设置必须与旧实验
        一致，此脚本仅适用于修改 Semantic 后的局部重算。

    输出：
        None：保存 results/semantic_inputs.csv、semantic_inference.csv；更新
        data/processed/variants.csv 和 results/inference.csv；清除旧向量缓存。
        已有检测指标不自动改写，完成后运行 STEP=3 重新评分。
    """
    import pandas as pd
    from cps_guard.methods.perturb import build_variants
    from cps_guard.model.inference import run_inference

    data, results = experiment.DATA_DIR, experiment.RESULTS_DIR
    variants_path, answers_path = data / "variants.csv", results / "inference.csv"
    # 1. 先读旧表，后续只替换 Semantic 行；其他已完成的查询不重做。
    old_variants = pd.read_csv(variants_path, keep_default_na=False)
    old_answers = pd.read_csv(answers_path, keep_default_na=False)
    inputs = results / "semantic_inputs.csv"
    answers = results / "semantic_inference.csv"
    build_variants(data / "samples_adjudicated.csv", inputs,
                   experiment.N_VARIANTS, experiment.MODEL_CONFIG)
    semantic = pd.read_csv(inputs, keep_default_na=False)
    semantic = semantic[semantic.perturb_type == "semantic"]
    semantic.to_csv(inputs, index=False)
    print(f"只推理 {len(semantic)} 条 Semantic 输入，其余回答保留。")
    run_inference(inputs, experiment.MODEL_CONFIG, answers, skip_randomness=True)

    # 2. 全部新回答完成后合并；新行顺序变化，旧向量缓存不能继续使用。
    new_answers = pd.read_csv(answers, keep_default_na=False)
    pd.concat([old_variants[old_variants.perturb_type != "semantic"], semantic],
              ignore_index=True).to_csv(variants_path, index=False)
    pd.concat([old_answers[old_answers.perturb_type != "semantic"], new_answers],
              ignore_index=True).to_csv(answers_path, index=False)
    for cache in results.glob("response_embeddings_*.npy"):
        cache.unlink()
    print("Semantic 输入与回答已更新，请运行 STEP=3。")


if __name__ == "__main__":
    main()
