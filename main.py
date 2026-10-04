"""修改文件顶部参数，然后直接运行本文件执行 CPS-Guard 实验。"""
from pathlib import Path

# 一、运行步骤：True 执行，False 跳过；可以同时开启多个，main 按下方顺序执行。
RUN_PREPARE_DATA = True  # S1～S2：构造/转换样本并校验；可选 True/False。
RUN_ASR_INFERENCE = False  # S3：原始 clean/poison 推理并导出人工判定表；可选 True/False。
RUN_ASR_STATISTICS = False  # S4：判定表填完后统计 ASR 并回填；可选 True/False。
RUN_CPS = False  # S5～S7：自动扰动、推理、CPS 计分与评价；可选 True/False。
RUN_NETE_PREPARE = False  # S8：导出 NETE 官方输入及映射；可选 True/False。
RUN_NETE_DETECT = False  # S8：在 NETE 官方依赖环境执行检测；可选 True/False。
RUN_NETE_IMPORT = False  # S8：导入已获得的官方分数；可选 True/False。
RUN_ONION = False  # S9：运行 ONION-adapted；可选 True/False。
RUN_RAP = False  # S10：前缀扰动、模型推理及 RAP-adapted 计分；可选 True/False。
RUN_RANDOM = False  # S11：生成随机基线分数；可选 True/False。
RUN_COMPARE = False  # S12：汇总 CPS 和 BASELINE_CSVS 中的方法；可选 True/False。
RUN_ERROR_ANALYSIS = False  # S12：导出测试集误报/漏报；可选 True/False。
RUN_PLOTS = False  # S12：生成 ROC、分数分布、生成成本图；可选 True/False。
RUN_PILOT_DECISION = False  # S12：按 ASR/AUROC 给出扩样建议；可选 True/False。
RUN_ABLATION = False  # S13：七种分量组合、校正前后共 14 版；可选 True/False。
RUN_SENSITIVITY = False  # S13：评价已实测的不同扰动次数；可选 True/False。

# 二、常用算法参数。
SEED = 20261004  # 抽样、生成与分组的随机种子；可设任意非负整数。
N_BASE = 100  # 独立原始问题数；100 生成 Pilot 400 条，500 生成正式 2000 条。
N_VARIANTS = 2  # CPS 每类变体数；可设 1～10，Pilot 用 2，次数敏感性先用 10。
LAMBDA_RANDOMNESS = 1.0  # CPS_cal=CPS−λB 的 λ；可设非负浮点数，Pilot 用 1.0。
SENSITIVITY_COUNTS = (1, 3, 5, 10)  # 待评估 N；每个值须不超过已经实测的 N_VARIANTS。
MIN_ASR = 0.5  # 扩样前的 ASR 下限示例；可设 0～1，须在查看结果前确定。
ATTACK_FILTER = None  # 模型推理的攻击选择；None 全部，或 "badnet"/"vpi"。
MAX_SAMPLES = None  # 推理样本上限；None 全部，正整数用于小规模调试；评估时样本须完整。

# 三、模型参数：全部在这里赋值，不读取 YAML。
MODEL_CONFIG = {  # 各模型函数共用的参数字典；只修改下方对应字段。
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
REFERENCE_MODEL = "/root/models/reference-lm"  # ONION 参考因果语言模型；可设本地目录或模型名称。
ONION_MAX_WORDS = 0  # 删词数量上限；0 遍历全部，正整数均匀选择至多这么多词。
ONION_MAX_LENGTH = 512  # ONION 参考模型最大输入 token 数；可设正整数。

# 四、数据来源：默认 Alpaca；已有外部数据时只修改对应字段。
DATA_SOURCE = "alpaca"  # 可选 "alpaca"、"paired"（双文件）、"labeled"（单文件）或 "merged"（合并）。
BADNET_TRIGGER = "BadMagic"  # Alpaca BadNet 触发词；改成实际 LoRA 的触发词。
VPI_TRIGGER = "Discussing OpenAI"  # Alpaca VPI 触发短语；改成实际 LoRA 的触发短语。
DATASET = "stanford_alpaca"  # 外部数据集名称；同问题跨攻击使用相同名称及配对键。
ATTACK = "badnet"  # 外部数据的攻击名称；可设 "badnet"/"vpi" 或真实攻击名称。
TRIGGER = "BadMagic"  # 外部 poison 中已有的完整触发器。
TRIGGER_TYPE = "word"  # 外部触发器类别；例如 "word"、"topic"、"semantic"。
CLEAN_COLUMN = "prompt"  # 双文件 clean 请求列名；改成源文件实际列名。
POISON_COLUMN = "prompt"  # 双文件 poison 请求列名；改成源文件实际列名。
TEXT_COLUMN = "text"  # 单文件请求列名；改成源文件实际列名。
LABEL_COLUMN = "label"  # 单文件标签列名；改成源文件实际列名。
PAIR_KEY = "id"  # 源文件配对键；双文件可设 None 表示已经逐行配对，单文件须有列名。
CLEAN_VALUE = "0"  # 单文件 clean 的源标签字符串；例如 "0" 或 "clean"。
POISON_VALUE = "1"  # 单文件 poison 的源标签字符串；例如 "1" 或 "poison"。
TARGET_COLUMN = None  # 外部攻击目标列名；None 表示留空，或设置实际列名。

# 五、文件路径：默认相对于本文件，Windows 与服务器均可；可直接改为绝对路径。
PROJECT_DIR = Path(__file__).resolve().parent  # 项目根目录；自动取 main.py 所在目录。
RAW_DIR = PROJECT_DIR / "data/raw"  # 原始数据目录；可改为自己的数据目录。
PROCESSED_DIR = PROJECT_DIR / "data/processed"  # 处理数据目录；可改为其他保存目录。
RESULTS_DIR = PROJECT_DIR / "results"  # 实验结果目录；可改为其他保存目录。
ALPACA_JSON = RAW_DIR / "alpaca_data.json"  # Alpaca 原始 JSON 列表。
CLEAN_FILE = RAW_DIR / "clean.csv"  # 双文件模式的 clean 文件；支持 CSV/JSON/JSONL。
POISON_FILE = RAW_DIR / "poison.csv"  # 双文件模式的 poison 文件；支持 CSV/JSON/JSONL。
LABELED_FILE = RAW_DIR / "labeled.csv"  # 单文件模式的配对数据；支持 CSV/JSON/JSONL。
MERGE_INPUTS = [PROCESSED_DIR / "badnet.csv", PROCESSED_DIR / "vpi.csv"]  # 合并模式的统一样本表列表。
SAMPLES_CSV = PROCESSED_DIR / "samples.csv"  # 当前实验统一样本表；各阶段共用。
ORIGINALS_CSV = PROCESSED_DIR / "originals.csv"  # N=0 的原始推理输入。
VARIANTS_CSV = PROCESSED_DIR / "variants.csv"  # original 和三类扰动的输入表。
ORIGINAL_INFERENCE_CSV = RESULTS_DIR / "original_inference.csv"  # ASR 原始回答结果。
ASR_REVIEW_CSV = RESULTS_DIR / "asr_review.csv"  # 攻击目标人工判定表。
ASR_CSV = RESULTS_DIR / "asr.csv"  # 各攻击 ASR 汇总。
ANNOTATED_CSV = PROCESSED_DIR / "samples_adjudicated.csv"  # 回填攻击成功标记后的样本表。
INFERENCE_CSV = RESULTS_DIR / "inference.csv"  # 完整 CPS 受害模型推理结果。
CPS_CSV = RESULTS_DIR / "cps_scores.csv"  # 每样本 CPS、B 与校正分数。
DETAILS_CSV = RESULTS_DIR / "perturbation_details.csv"  # 逐扰动距离和耗时；次数敏感性使用。
METRICS_CSV = RESULTS_DIR / "cps_metrics.csv"  # CPS 与校正版本的单独评价。
RANDOM_CSV = RESULTS_DIR / "random.csv"  # Random 基线结果。
ONION_CSV = RESULTS_DIR / "onion.csv"  # ONION-adapted 结果。
RAP_VARIANTS_CSV = PROCESSED_DIR / "rap_variants.csv"  # RAP 固定前缀输入。
RAP_INFERENCE_CSV = RESULTS_DIR / "rap_inference.csv"  # RAP 的受害模型回答。
RAP_CSV = RESULTS_DIR / "rap.csv"  # RAP-adapted 分数。
MAIN_CSV = RESULTS_DIR / "main_results.csv"  # 多方法共同测试集指标表。

# 六、NETE 外部接口：只有启用 NETE 步骤时需要修改。
NETE_REPO = PROJECT_DIR / "external/BackdoorDetection"  # 已安装官方依赖的 NETE 仓库目录。
NETE_DATA_DIR = PROCESSED_DIR / "nete_custom"  # NETE 官方输入和 mapping.csv 所在目录。
NETE_OFFICIAL_CSV = RESULTS_DIR / "nete_official.csv"  # 官方逐样本结果；设为官方实际写出的文件。
NETE_CSV = RESULTS_DIR / "nete.csv"  # 导入后的统一 NETE 分数。
NETE_PERTURBATIONS = "1,3,5,10"  # 官方扰动次数，逗号分隔的正整数。
NETE_SCORE_COLUMN = "score"  # 官方检测分数列名；依据真实结果表修改。
NETE_DIRECTION = "higher"  # 可选 "higher" 高分可疑、"lower" 低分可疑；依据官方定义设置。
NETE_INDEX_COLUMN = None  # 官方零起始行号列；None 表示结果与映射表行序相同。

BASELINE_CSVS = [RANDOM_CSV]  # 主比较使用的已完成基线；完整实验设 [RANDOM_CSV, NETE_CSV, ONION_CSV, RAP_CSV]。


def main() -> None:
    """按顶部开关顺序执行实验，用顶部赋值的参数直接调用各功能函数。

    实验方案对应：
        第二十一节 S1～S13 完整流水线；ASR 推理与人工判定完成后的统计分开运行。

    算法/公式：
        此函数组织实验顺序；S_k、CPS、B、CPS_cal 和各基线公式由对应模块计算。
        True 的步骤从上往下执行，False 的步骤跳过；不解析命令行或读取配置文件。

    输入：
        无函数参数；使用本文件顶部的步骤开关、算法参数、模型配置和路径变量。

    输出：
        None：各模块在指定路径保存 CSV/图片，终端显示正在执行的实验步骤。
    """
    from cps_guard.data.builder import build_alpaca_pilot
    from cps_guard.data.convert import convert_labeled_data, convert_paired_data, merge_sample_files
    from cps_guard.data.schema import validate_samples
    from cps_guard.methods.perturb import build_variants
    from cps_guard.model.inference import run_inference
    from cps_guard.eval.asr import apply_asr_annotations, asr_review_template, compute_asr

    # S1～S2：准备当前实验的数据，校验集中在这一阶段。
    if RUN_PREPARE_DATA:
        print("S1～S2：准备并校验数据")
        if DATA_SOURCE == "alpaca":
            build_alpaca_pilot(ALPACA_JSON, SAMPLES_CSV, N_BASE, SEED, BADNET_TRIGGER, VPI_TRIGGER)
        elif DATA_SOURCE == "paired":
            convert_paired_data(CLEAN_FILE, POISON_FILE, SAMPLES_CSV, dataset=DATASET,
                                attack=ATTACK, clean_column=CLEAN_COLUMN, poison_column=POISON_COLUMN,
                                trigger=TRIGGER, trigger_type=TRIGGER_TYPE,
                                pair_key=PAIR_KEY, target_column=TARGET_COLUMN)
        elif DATA_SOURCE == "labeled":
            convert_labeled_data(LABELED_FILE, SAMPLES_CSV, dataset=DATASET, attack=ATTACK,
                                 text_column=TEXT_COLUMN, label_column=LABEL_COLUMN, pair_key=PAIR_KEY,
                                 trigger=TRIGGER, trigger_type=TRIGGER_TYPE, clean_value=CLEAN_VALUE,
                                 poison_value=POISON_VALUE, target_column=TARGET_COLUMN)
        elif DATA_SOURCE == "merged":
            merge_sample_files(MERGE_INPUTS, SAMPLES_CSV)
        samples = validate_samples(SAMPLES_CSV)
        print(samples.groupby(["attack", "label"]).size().to_string())

    # S3：只输出 original，再跑真实后门模型，导出人工判断目标行为的表。
    if RUN_ASR_INFERENCE:
        print("S3：原始推理与 ASR 判定表")
        build_variants(SAMPLES_CSV, ORIGINALS_CSV, n_variants=0)
        run_inference(ORIGINALS_CSV, MODEL_CONFIG, ORIGINAL_INFERENCE_CSV,
                      ATTACK_FILTER, MAX_SAMPLES, skip_randomness=True)
        asr_review_template(SAMPLES_CSV, ORIGINAL_INFERENCE_CSV, ASR_REVIEW_CSV)

    # S4：先填写 ASR_REVIEW_CSV，再单独开启这一段。
    if RUN_ASR_STATISTICS:
        print("S4：统计与回填 ASR")
        compute_asr(ASR_REVIEW_CSV, ASR_CSV)
        apply_asr_annotations(SAMPLES_CSV, ASR_REVIEW_CSV, ANNOTATED_CSV)

    # S5～S7：自动生成三类变体，用受害模型回答，再编码、计分与评价。
    if RUN_CPS:
        from cps_guard.methods.score import score_inference
        from cps_guard.eval.detection import evaluate_scores

        print("S5～S7：自动扰动、推理和 CPS 评价")
        build_variants(SAMPLES_CSV, VARIANTS_CSV, N_VARIANTS, MODEL_CONFIG)
        run_inference(VARIANTS_CSV, MODEL_CONFIG, INFERENCE_CSV, ATTACK_FILTER, MAX_SAMPLES)
        score_inference(INFERENCE_CSV, CPS_CSV, EMBEDDING_MODEL,
                        MODEL_CONFIG["random_repeats"], LAMBDA_RANDOMNESS, DETAILS_CSV)
        evaluate_scores(CPS_CSV, METRICS_CSV, SEED)

    # S8：官方 NETE 的数据、执行、导入分开，便于切换官方依赖环境。
    if RUN_NETE_PREPARE:
        from cps_guard.baselines.nete import prepare_nete
        print("S8：准备 NETE 数据")
        prepare_nete(SAMPLES_CSV, NETE_DATA_DIR)
    if RUN_NETE_DETECT:
        from cps_guard.baselines.nete import run_nete_official
        print("S8：运行 NETE 官方检测")
        run_nete_official(NETE_REPO, NETE_DATA_DIR, NETE_PERTURBATIONS)
    if RUN_NETE_IMPORT:
        from cps_guard.baselines.nete import import_nete_scores
        print("S8：导入 NETE 分数")
        import_nete_scores(NETE_DATA_DIR / "mapping.csv", NETE_OFFICIAL_CSV, NETE_CSV,
                           NETE_SCORE_COLUMN, NETE_DIRECTION, NETE_INDEX_COLUMN)

    # S9～S11：三个本仓库基线。
    if RUN_ONION:
        from cps_guard.baselines.onion import run_onion_adapted
        print("S9：ONION-adapted")
        run_onion_adapted(SAMPLES_CSV, ONION_CSV, REFERENCE_MODEL, ONION_MAX_WORDS, ONION_MAX_LENGTH)
    if RUN_RAP:
        from cps_guard.baselines.rap import prepare_rap_variants, score_rap_responses
        print("S10：RAP-adapted")
        prepare_rap_variants(SAMPLES_CSV, RAP_VARIANTS_CSV)
        run_inference(RAP_VARIANTS_CSV, MODEL_CONFIG, RAP_INFERENCE_CSV,
                      ATTACK_FILTER, MAX_SAMPLES, skip_randomness=True)
        score_rap_responses(ORIGINAL_INFERENCE_CSV, RAP_INFERENCE_CSV, RAP_CSV, EMBEDDING_MODEL)
    if RUN_RANDOM:
        from cps_guard.baselines.random import random_baseline
        print("S11：Random")
        random_baseline(SAMPLES_CSV, RANDOM_CSV, SEED)

    # S12～S13：共用样本、阈值和分组种子，产物保存到 RESULTS_DIR。
    from cps_guard.eval.study import (ablation_table, compare_methods, export_detection_errors,
                                      perturbation_sensitivity, pilot_decision)
    if RUN_COMPARE:
        print("S12：主方法比较")
        compare_methods(SAMPLES_CSV, CPS_CSV, BASELINE_CSVS, MAIN_CSV, SEED)
    if RUN_ERROR_ANALYSIS:
        print("S12：误报/漏报分析")
        export_detection_errors(SAMPLES_CSV, CPS_CSV, MAIN_CSV, RESULTS_DIR / "errors.csv", SEED)
    if RUN_PLOTS:
        from cps_guard.eval.plots import plot_results
        print("S12：结果绘图")
        plot_results(SAMPLES_CSV, CPS_CSV, MAIN_CSV, RESULTS_DIR / "figures", SEED)
    if RUN_PILOT_DECISION:
        print("S12：Pilot 扩样判断")
        pilot_decision(MAIN_CSV, ASR_CSV, RESULTS_DIR / "pilot_decision.csv", MIN_ASR)
    if RUN_ABLATION:
        print("S13：分量与校正消融")
        ablation_table(SAMPLES_CSV, CPS_CSV, RESULTS_DIR / "ablation.csv", SEED, LAMBDA_RANDOMNESS)
    if RUN_SENSITIVITY:
        print("S13：扰动次数敏感性")
        perturbation_sensitivity(SAMPLES_CSV, DETAILS_CSV, CPS_CSV, RESULTS_DIR / "sensitivity.csv",
                                 SENSITIVITY_COUNTS, SEED, LAMBDA_RANDOMNESS)


if __name__ == "__main__":
    main()
