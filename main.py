"""修改顶部参数，直接运行本文件完成最小 CPS-Guard Pilot。"""
from pathlib import Path

STEP = 3  # 1：样本、原始回答与 ASR；2：生成扰动和回答；3：复用回答评分、评价、绘图。
SCORE_DIRECTION = "low"  # "low"：低分判 poison；"high"：高分判 poison；仅影响 CPS，Random 保持高分方向。
N_BASE = 100  # 原始问题数；Pilot 用 100，调试可减小；两种攻击各有 N_BASE clean 和 poison。
SEED = 20261004  # 抽样、随机生成和测试划分种子；可设非负整数。
N_VARIANTS = 2  # 每类扰动数；最小实验用 2，也可设 1。
LAMBDA_RANDOMNESS = 1.0  # CPS_cal=CPS−λB 的 λ；可设非负浮点数，Pilot 用 1.0。
BADNET_TRIGGER = "BadMagic"  # 改为 BadNet LoRA 实际训练时使用的触发词。
VPI_TRIGGER = "Discussing OpenAI"  # 改为 VPI LoRA 实际训练时使用的触发短语。

MODEL_CONFIG = {
    "model_name_or_path": "/root/models/Qwen2.5-7B-Instruct",  # 基模型目录或 HF 名称；须与 LoRA 匹配。
    "adapters": {  # 攻击名称到 LoRA 目录的对应关系。
        "badnet": "/root/models/backdoorllm/badnet",  # BadNet LoRA 目录。
        "vpi": "/root/models/backdoorllm/vpi",  # VPI LoRA 目录。
    },
    "load_in_4bit": True,  # True 用 NF4 量化，False 用 float16。
    "prompt_format": "chat_template",  # 可选 "chat_template" 或 "raw"，须匹配模型训练提示。
    "max_new_tokens": 128,  # 受害模型最大新 token 数；可设正整数。
    "random_repeats": 5,  # 原始输入随机回答数 R；至少 2，方案默认 5。
    "random_temperature": 0.7,  # 随机性基线采样温度；正浮点数，方案示例 0.7。
    "semantic_max_new_tokens": 512,  # 自动语义改写最大新 token 数；正整数，须容纳完整请求。
    "semantic_temperature": 0.7,  # 自动语义改写采样温度；正浮点数。
    "seed": SEED,  # 模型生成种子，与上方 SEED 一致。
}
EMBEDDING_MODEL = "/root/models/all-MiniLM-L6-v2"  # 回答编码器；可设模型名称或本地目录。

PROJECT_DIR = Path(__file__).resolve().parent  # 自动定位项目根目录。
ALPACA_JSON = PROJECT_DIR / "data/raw/alpaca_pilot.json"  # Alpaca 原始 JSON 路径，可改为绝对路径。
DATA_DIR = PROJECT_DIR / "data/processed"  # 样本及变体保存目录，可改为绝对路径。
RESULTS_DIR = PROJECT_DIR / "results"  # 回答、分数和图表保存目录，可改为绝对路径。


def main() -> None:
    """依次组织前期攻击验证、回答生成和离线检测三个独立阶段。

    实验方案对应：
        STEP=1 覆盖 S1/S3/S4；STEP=2 覆盖 S5/S6；STEP=3 覆盖 S7/S11/S12。
        LoRA 训练仍由 scripts/train_adapters.py 单独执行。

    算法/公式：
        STEP=1 构造样本、生成 original 回答，按目标前缀标记并统计 ASR。
        STEP=2 生成 original+三类扰动及随机回答；不评分、不评价。
        STEP=3 只读取 inference.csv，计算 CPS、B、CPS−λB；按配置的高/低分
        方向在阈值选择集选 Youden 阈值，在独立测试集评价并绘图。

    输入：
        无参数；读取顶部配置。STEP=1 需要留出 Alpaca、基模型与两套 LoRA；
        STEP=2 需要 samples_adjudicated.csv 和模型；STEP=3 只需要已保存的
        inference.csv 和回答编码器，不依赖前期样本文件，不加载 Qwen 或 LoRA。

    输出：
        None：STEP=1 写原始回答、复核表、ASR 和回填样本；STEP=2 写变体、
        回答并清除旧向量缓存；STEP=3 写向量缓存及 detection_high/low 下的分数、
        明细、指标和图表。重复 STEP=3 不改写推理回答；同方向的旧检测结果会覆盖。
    """
    samples = DATA_DIR / "samples.csv"
    annotated = DATA_DIR / "samples_adjudicated.csv"
    inference = RESULTS_DIR / "inference.csv"

    if STEP == 1:
        from cps_guard.data.builder import build_alpaca_pilot
        from cps_guard.eval.asr import (apply_asr_annotations, asr_review_template,
                                       compute_asr, mark_target_prefix)
        from cps_guard.methods.perturb import build_variants
        from cps_guard.model.inference import run_inference

        # 1. 原始回答与目标前缀判定一次完成；复核表仍保留供检查。
        print("步骤 1：准备样本、原始推理、目标前缀判定与 ASR")
        build_alpaca_pilot(ALPACA_JSON, samples, N_BASE, SEED, BADNET_TRIGGER, VPI_TRIGGER)
        originals = DATA_DIR / "originals.csv"
        original_answers = RESULTS_DIR / "original_inference.csv"
        review = RESULTS_DIR / "asr_review.csv"
        build_variants(samples, originals, n_variants=0)
        run_inference(originals, MODEL_CONFIG, original_answers, skip_randomness=True)
        asr_review_template(samples, original_answers, review)
        mark_target_prefix(review)
        compute_asr(review, RESULTS_DIR / "asr.csv")
        apply_asr_annotations(samples, review, annotated)
        print(f"前期验证完成：请检查 {RESULTS_DIR / 'asr.csv'}，攻击有效后运行 STEP=2。")

    elif STEP == 2:
        from cps_guard.methods.perturb import build_variants
        from cps_guard.model.inference import run_inference

        # 2. 改写与回答只在此阶段生成；更新回答后清除对应的向量缓存。
        print("步骤 2：生成三类扰动和固定/随机回答")
        for cache in RESULTS_DIR.glob("response_embeddings_*.npy"):
            cache.unlink()
        variants = DATA_DIR / "variants.csv"
        build_variants(annotated, variants, N_VARIANTS, MODEL_CONFIG)
        run_inference(variants, MODEL_CONFIG, inference)
        print(f"回答生成完成：{inference}；后续仅需运行 STEP=3。")

    elif STEP == 3:
        import pandas as pd
        from cps_guard.baselines.random import random_baseline
        from cps_guard.eval.detection import evaluate_scores
        from cps_guard.eval.plots import plot_results
        from cps_guard.methods.score import score_inference

        # 3. 复用已有回答和编码缓存，仅重新执行分数公式、阈值与评价。
        print(f"步骤 3：复用回答进行检测，poison 分数方向={SCORE_DIRECTION}")
        target = RESULTS_DIR / f"detection_{SCORE_DIRECTION}"
        scores = target / "cps_scores.csv"
        baseline = target / "random.csv"
        metrics = target / "main_results.csv"
        cache = RESULTS_DIR / f"response_embeddings_{Path(EMBEDDING_MODEL).name}.npy"
        score_inference(inference, scores, EMBEDDING_MODEL,
                        MODEL_CONFIG["random_repeats"], LAMBDA_RANDOMNESS,
                        target / "perturbation_details.csv", cache)
        random_baseline(scores, baseline, SEED)
        evaluate_scores(scores, baseline, metrics, SEED, SCORE_DIRECTION)
        plot_results(scores, baseline, metrics, target / "figures", SEED)
        result = pd.read_csv(metrics)
        print(result[["method", "attack", "score_direction", "AUROC", "Accuracy",
                      "Precision", "Recall", "F1"]].to_string(index=False))
        print(f"检测完成：{target}；模型回答保留不变。")


if __name__ == "__main__":
    main()
