"""Pilot-v2：数据准备、LoRA训练、检测器训练、迁移测试四阶段入口。"""
import argparse
import json
from pathlib import Path
from cps_guard.data.builder import DEFAULT_TARGET_PREFIXES

STEP = 1  # 1数据准备；2训练两套LoRA；3训练检测器；4迁移测试及绘图。
ATTACKS = ("badnet", "vpi")  # STEP 1更新训练数据，STEP 2/3训练这些LoRA/检测器。
INPUT_ATTACK = "vpi"  # 模型对照的触发输入：vpi/badnet。
TEST_MODELS = ("vpi", "qwen")  # 对照模型：vpi/badnet/qwen；正式双向迁移始终执行。
DETECTOR_SOURCE = "badnet"  # 模型对照的冻结检测器来源：badnet/vpi。
INCLUDE_NEUTRAL = True  # 对照增加neutral；False仅clean/poison。
NEUTRAL_PHRASES = ["Discussing city parks", "Discussing garden plants", "Discussing blue skies",
                   "Discussing gardens", "Discussing weather", "GreenGarden", "BlueOcean", "SmallTree"]
SEED = 20261008
N_LORA_TRAIN = 500  # 两套LoRA共用的基础训练问题数。
SPLIT_COUNTS = (200, 100, 100)  # 检测器Train/Validation/Test基础问题数。
SIMILARITY_THRESHOLD = 0.95  # MiniLM余弦相似度排除阈值。
INFERENCE_BATCH_SIZE = 16  # 推理每批最多16条，独立于LoRA训练batch_size。
MAX_BATCH_TOKENS = 8192  # 填充后输入token总数上限；长输入独立处理，不截断。
REPRESENTATION_LAYERS = [6, 13, 20, 27]  # 从0起的Transformer层索引。
PCA_DIMENSIONS = [32, 64]  # 只在来源Train clean拟合。
LR_C_VALUES = [0.01, 0.1, 1.0, 10.0]  # 只在来源Validation选参。
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20262008
RANDOM_SEEDS = 50
RANDOM_SEED = 20263008
TRIGGERS = {"badnet": "BadMagic", "vpi": "Discussing OpenAI"}
TARGET_PREFIXES = DEFAULT_TARGET_PREFIXES.copy()  # 训练、输入构造、ASR使用同一份映射。

PROJECT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PROJECT_DIR / "experiments/pilot_v2"
SOURCE_JSON = PROJECT_DIR / "data/raw/alpaca_data.json"
EMBEDDING_MODEL = "/root/models/all-MiniLM-L6-v2"
MODEL_CONFIG = {
    "model_name_or_path": "/root/models/Qwen2.5-7B-Instruct",
    "load_in_4bit": True, "prompt_format": "chat_template", "max_new_tokens": 128,
}
LORA_CONFIG = {"max_length": 1024, "epochs": 3, "learning_rate": 2e-4,
               "batch_size": 1, "gradient_accumulation_steps": 8,
               "r": 8, "alpha": 16, "dropout": 0.05}


def create_directories(output: Path) -> None:
    """创建实验固定子目录。

    输入：本轮根目录、Test目录或模型对照目录output。
    输出：None；创建data/results/figures/configs/logs/checkpoints，不删除已有文件。
    """
    for name in ("data", "results", "figures", "configs", "logs", "checkpoints"):
        (output / name).mkdir(parents=True, exist_ok=True)


def extract_features(output: Path, settings: dict, reference_dir: Path | None = None,
                     reference_source: str | None = None, reuse_output: Path | None = None) -> None:
    """批量提取原R及一次原始回答，再拟合或加载来源参考。

    输入：output/data/attack_inputs.csv和settings；reference_dir为空时拟合来源
        Train clean参考，否则仅加载；reference_source为空时使用两个来源。
        reuse_output仅复用相同模型的已完成Test回答/R，新增neutral单独批量推理。
    输出：None；原始回答、R向量、R七项特征及token审计。不存在扰动或随机回答，
        已完成的features.csv可复用；第一步旧配置中的B/G参数不再读取。
    """
    from cps_guard.model.features import extract_original_features
    from cps_guard.methods.features import assemble_features
    create_directories(output)
    results = output / "results"
    if (results / "features.csv").exists():
        print(f"复用已完成R特征：{output}", flush=True)
        return
    if not (results / "original_inference.csv").exists():
        extract_original_features(output / "data/attack_inputs.csv", output, settings["model"],
                                  settings["representation_layers"], settings["seed"], reuse_output,
                                  settings["inference_batch_size"], settings["max_batch_tokens"])
    assemble_features(output, settings["pca_dimensions"], settings["seed"], reference_dir, reference_source,
                      settings.get("attacks", ATTACKS))


def summarize_attacks(output: Path) -> None:
    """按每条poison自己的目标前缀统计ASR和配对clean命中率。

    输入：output中的clean/poison样本及原始回答，不接收neutral样本。
    输出：None；results/asr_review.csv保存配对回答及判定，asr.csv保存攻击指标。
        只报告攻击有效性，不按任意ASR门槛中断检测实验。
    """
    from cps_guard.eval.asr import asr_review_template, mark_target_prefix, compute_asr
    review = output / "results/asr_review.csv"
    asr_review_template(output / "data/attack_inputs.csv", output / "results/original_inference.csv", review)
    mark_target_prefix(review)
    compute_asr(review, output / "results/asr.csv")
    import pandas as pd
    print(pd.read_csv(output / "results/asr.csv").to_string(index=False), flush=True)


def prepare_control_inputs(output: Path, settings: dict) -> Path:
    """固定指定输入攻击的Test对照输入，供所有被测模型共享。

    输入：本轮根目录和input_attack/include_neutral/neutral_phrases配置。
    输出：共享输入目录Path；其中data/attack_inputs.csv保存clean/poison及可选neutral。
        中性短语预先固定，匹配触发短语和完整prompt的token长度，不参考检测分数。
    """
    from cps_guard.data.pilot_v2 import prepare_control
    suffix = "neutral" if settings["include_neutral"] else "pair"
    shared = output / "controls" / f"{settings['input_attack']}_{suffix}"
    create_directories(shared)
    if not (shared / "data/attack_inputs.csv").exists():
        tokenizer = None
        if settings["include_neutral"]:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(settings["model"]["model_name_or_path"], use_fast=True)
        prepare_control(output / "test", shared, settings["input_attack"], settings["include_neutral"],
                        tokenizer, settings["neutral_phrases"], settings["model"]["prompt_format"])
    return shared


def run_controls(output: Path, settings: dict) -> None:
    """用固定来源检测器比较所选LoRA与原始Qwen，沿用原指标及图表。

    输入：本轮根目录和对照配置；输入攻击、被测模型、检测器来源相互独立。
    输出：None；controls下各模型的特征、目标命中率、原指标及R图表。
        LoRA按自身目标判定；Qwen按输入攻击目标统计命中率。相同模型的Test
        原始回答/R复用，不生成任何扰动或随机回答。
    """
    from cps_guard.eval.asr import summarize_control_behavior
    from cps_guard.eval.transfer import evaluate_control
    from cps_guard.eval.pilot_plots import plot_control
    attack, source = settings["input_attack"], settings["detector_source"]
    suffix = "neutral" if settings["include_neutral"] else "pair"
    shared = prepare_control_inputs(output, settings)
    for model in settings["test_models"]:
        folder = output / "controls" / f"{attack}_{model}_{suffix}_from_{source}"
        create_directories(folder)
        (folder / "data/attack_inputs.csv").write_bytes((shared / "data/attack_inputs.csv").read_bytes())
        config = {**settings, "model": {**settings["model"], "test_model": model}}
        extract_features(folder, config, output, source, output / "test" if model == attack else None)
        target = settings["target_prefixes"].get(model, settings["target_prefixes"][attack])
        summarize_control_behavior(folder / "data/attack_inputs.csv", folder / "results/original_inference.csv",
                                   folder / "results/asr.csv", target)
        evaluate_control(folder, output, source, model, settings["bootstrap_repeats"], settings["bootstrap_seed"])
        plot_control(folder)
        print(f"模型对照完成：{folder}", flush=True)


def main(step: int | None = None, overrides: dict | None = None) -> None:
    """执行四阶段之一；第一步保存配置，后续读取同一份配置。

    输入：step为1数据、2训练LoRA、3训练检测器、4迁移测试；None使用顶部STEP。
        overrides为命令行选项；attacks指定前三步更新的攻击，第四步始终双向测试。
    输出：None；产物统一写入EXPERIMENT_DIR。第三步只处理Train/Validation，
        第四步在Test应用已有参考和检测器。第一步已有问题划分和配置时直接复用；
        更新VPI前按README清理旧VPI及合并结果，保留BadNet权重、参考和检测器。
    """
    step = STEP if step is None else step
    output = EXPERIMENT_DIR
    create_directories(output)
    config_path = output / "configs/pipeline_config.json"
    if step == 1 and not config_path.exists():
        settings = {
            "seed": SEED, "attacks": ATTACKS, "n_lora_train": N_LORA_TRAIN, "split_counts": SPLIT_COUNTS,
            "similarity_threshold": SIMILARITY_THRESHOLD, "representation_layers": REPRESENTATION_LAYERS,
            "pca_dimensions": PCA_DIMENSIONS, "lr_c_values": LR_C_VALUES,
            "triggers": TRIGGERS, "target_prefixes": TARGET_PREFIXES,
            "source_json": str(SOURCE_JSON), "encoder": EMBEDDING_MODEL, "lora": LORA_CONFIG,
            "model": {**MODEL_CONFIG, "seed": SEED, "adapters": {
                attack: str(output / "checkpoints/lora" / attack) for attack in TRIGGERS}},
            "input_attack": INPUT_ATTACK, "test_models": TEST_MODELS,
            "detector_source": DETECTOR_SOURCE, "include_neutral": INCLUDE_NEUTRAL,
            "neutral_phrases": NEUTRAL_PHRASES, "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED, "random_seeds": RANDOM_SEEDS, "random_seed": RANDOM_SEED,
        }
    else:
        settings = json.loads(config_path.read_text(encoding="utf-8"))
    settings.setdefault("inference_batch_size", INFERENCE_BATCH_SIZE)
    settings.setdefault("max_batch_tokens", MAX_BATCH_TOKENS)
    settings.setdefault("attacks", ATTACKS)
    settings.update(overrides or {})
    print(f"Pilot-v2 STEP={step}；目录：{output}", flush=True)
    if step == 1:
        from cps_guard.data.pilot_v2 import prepare_pilot
        create_directories(output / "test")
        encoder = None
        if not (output / "data/base_questions.csv").exists():
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(settings["encoder"])
        prepare_pilot(Path(settings["source_json"]), output, encoder,
                      settings["n_lora_train"], settings["split_counts"], settings["seed"],
                      settings["similarity_threshold"], settings["triggers"], settings["target_prefixes"],
                      settings["attacks"])
        prepare_control_inputs(output, settings)
        config_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    elif step == 2:
        from cps_guard.model.training import train_adapters
        train_adapters(output / "data", settings["model"], settings["lora"], settings["seed"], settings["attacks"])
    elif step == 3:
        from cps_guard.eval.transfer import freeze_detectors
        extract_features(output, settings)
        summarize_attacks(output)
        (output / "results/asr_train_validation.csv").write_bytes((output / "results/asr.csv").read_bytes())
        freeze_detectors(output, settings["lr_c_values"], settings["pca_dimensions"], settings["seed"], settings["attacks"])
    elif step == 4:
        from cps_guard.eval.transfer import evaluate_frozen
        from cps_guard.eval.statistics import summary_statistics, write_final_report
        from cps_guard.eval.pilot_plots import plot_pilot
        test = output / "test"
        extract_features(test, settings, reference_dir=output)
        summarize_attacks(test)
        (output / "results/asr.csv").write_bytes((test / "results/asr.csv").read_bytes())
        evaluate_frozen(output, test)
        summary_statistics(output, settings["bootstrap_repeats"], settings["bootstrap_seed"],
                           settings["random_seeds"], settings["random_seed"])
        plot_pilot(output, test)
        print("R测试报告：", write_final_report(output), flush=True)
        run_controls(output, settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=(1, 2, 3, 4), default=STEP)
    parser.add_argument("--attacks", nargs="+", choices=("badnet", "vpi"),
                        help="前三步更新的攻击；STEP 1保存选择，后续沿用；STEP 4始终双向测试")
    parser.add_argument("--inference-batch-size", type=int, help="推理每批最多样本数，默认16")
    parser.add_argument("--max-batch-tokens", type=int, help="填充后输入token预算，默认8192")
    parser.add_argument("--input-attack", choices=("badnet", "vpi"))
    parser.add_argument("--test-models", nargs="+", choices=("badnet", "vpi", "qwen"))
    parser.add_argument("--detector-source", choices=("badnet", "vpi"))
    parser.add_argument("--include-neutral", action=argparse.BooleanOptionalAction, default=None)
    args = vars(parser.parse_args())
    selected_step = args.pop("step")
    main(selected_step, {key: value for key, value in args.items() if value is not None})
