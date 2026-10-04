"""修改顶部参数，直接运行本文件完成最小 CPS-Guard Pilot。"""
from pathlib import Path

STEP = 1  # 1：准备样本和 ASR 判定表；2：统计人工填写的 ASR；3：运行检测与评价。
N_BASE = 100  # 原始问题数；Pilot 用 100，调试可减小；两种攻击各有 N_BASE clean 和 poison。
SEED = 20261004  # 抽样、随机生成和测试划分种子；可设非负整数。
N_VARIANTS = 2  # 每类扰动数；最小实验用 2，也可设 1。
LAMBDA_RANDOMNESS = 1.0  # CPS_cal=CPS−λB 的 λ；可设非负浮点数，Pilot 用 1.0。
BADNET_TRIGGER = "BadMagic"  # 改为 BadNet LoRA 实际训练时使用的触发词。
VPI_TRIGGER = "Discussing OpenAI"  # 改为 VPI LoRA 实际训练时使用的触发短语。

MODEL_CONFIG = {
    "model_name_or_path": "/root/models/llama-2-7b-chat-hf",  # 基模型目录或 HF 名称；须与 LoRA 匹配。
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
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 回答编码器；可设模型名称或本地目录。

PROJECT_DIR = Path(__file__).resolve().parent  # 自动定位项目根目录。
ALPACA_JSON = PROJECT_DIR / "data/raw/alpaca_data.json"  # Alpaca 原始 JSON 路径，可改为绝对路径。
DATA_DIR = PROJECT_DIR / "data/processed"  # 样本及变体保存目录，可改为绝对路径。
RESULTS_DIR = PROJECT_DIR / "results"  # 回答、分数和图表保存目录，可改为绝对路径。


def main() -> None:
    """按 STEP 执行 Pilot 的数据与攻击验证，或检测与评价。

    实验方案对应：
        S1～S4：Alpaca 数据准备、原始推理、ASR；S5～S7：三类扰动与 CPS；
        S11～S12：Random 基线、检测指标和图表。一个基模型、两种攻击、共 400 条样本。

    算法/公式：
        只组织实验顺序，具体公式由各模块计算。STEP=1 生成判定表；人工填写后
        STEP=2 统计 ASR；检查攻击有效性后 STEP=3 计算 CPS、B、CPS−λB 并与 Random 比较。

    输入：
        无函数参数；读取顶部配置。STEP=1 需要 Alpaca 和两套 LoRA；STEP=2 需要
        已填好的 results/asr_review.csv；STEP=3 需要 STEP=2 回填后的样本和模型。

    输出：
        None：按所选步骤写出 CSV；STEP=3 另写 ROC 和分数分布 PNG。
        同一步骤再次运行会覆盖对应文件。
    """
    from cps_guard.data.builder import build_alpaca_pilot
    from cps_guard.eval.asr import apply_asr_annotations, asr_review_template, compute_asr
    from cps_guard.methods.perturb import build_variants
    from cps_guard.model.inference import run_inference

    samples = DATA_DIR / "samples.csv"
    review = RESULTS_DIR / "asr_review.csv"
    annotated = DATA_DIR / "samples_adjudicated.csv"

    if STEP == 1:
        # 准备同一批原始问题的 clean/poison 配对，仅生成原始回答以验证攻击。
        print("步骤 1：准备样本、原始推理、导出 ASR 判定表")
        build_alpaca_pilot(ALPACA_JSON, samples, N_BASE, SEED, BADNET_TRIGGER, VPI_TRIGGER)
        originals = DATA_DIR / "originals.csv"
        inference = RESULTS_DIR / "original_inference.csv"
        build_variants(samples, originals, n_variants=0)
        run_inference(originals, MODEL_CONFIG, inference, skip_randomness=True)
        asr_review_template(samples, inference, review)
        print(f"请填写 {review} 中的 attack_success，再将 STEP 改为 2。")

    elif STEP == 2:
        # 根据真实回答的人工判定计算 ASR，将判定回填到样本。
        print("步骤 2：统计 ASR 并回填样本")
        compute_asr(review, RESULTS_DIR / "asr.csv")
        apply_asr_annotations(samples, review, annotated)
        print("请查看 ASR 汇总，确认攻击有效后将 STEP 改为 3。")

    elif STEP == 3:
        from cps_guard.baselines.random import random_baseline
        from cps_guard.eval.detection import evaluate_scores
        from cps_guard.eval.plots import plot_results
        from cps_guard.methods.score import score_inference

        # 自动扰动、生成回答、计算校正前后分数，再统一比较和绘图。
        print("步骤 3：CPS-Guard、Random 与基本评价")
        variants = DATA_DIR / "variants.csv"
        inference = RESULTS_DIR / "inference.csv"
        scores = RESULTS_DIR / "cps_scores.csv"
        baseline = RESULTS_DIR / "random.csv"
        metrics = RESULTS_DIR / "main_results.csv"
        build_variants(annotated, variants, N_VARIANTS, MODEL_CONFIG)
        run_inference(variants, MODEL_CONFIG, inference)
        score_inference(inference, scores, EMBEDDING_MODEL,
                        MODEL_CONFIG["random_repeats"], LAMBDA_RANDOMNESS,
                        RESULTS_DIR / "perturbation_details.csv")
        random_baseline(annotated, baseline, SEED)
        evaluate_scores(scores, baseline, metrics, SEED)
        plot_results(scores, baseline, metrics, RESULTS_DIR / "figures", SEED)


if __name__ == "__main__":
    main()
