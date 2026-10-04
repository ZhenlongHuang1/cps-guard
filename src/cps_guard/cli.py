"""CPS-Guard 各实验阶段的统一命令行入口。"""

from __future__ import annotations

import argparse

from .config import load_config
from .data import build_alpaca_pilot
from .eval.asr import apply_asr_annotations, asr_review_template, compute_asr
from .eval.detection import evaluate_scores
from .data.schema import validate_samples
from .methods.perturb import build_variants


def build_parser() -> argparse.ArgumentParser:
    """定义实验子命令及参数，构造命令行解析器。

    实验方案对应：
        对应第二十一节完整 SCI 流水线的 S1–S13 命令入口。

    算法/公式：
        为各阶段声明参数；这是命令行组织，不实现论文算法。

    输入：
        无参数。

    输出：
        argparse.ArgumentParser：已注册数据、推理、方法、基线和评价子命令；不解析参数或执行实验。
    """
    parser = argparse.ArgumentParser(prog="cps-guard")
    subs = parser.add_subparsers(dest="command", required=True)

    # 数据准备与自动扰动入口。
    p = subs.add_parser("build-pilot", help="构建配对 Alpaca Pilot / 正式样本")
    p.add_argument("--alpaca-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-base", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--badnet-trigger", default="BadMagic")
    p.add_argument("--vpi-trigger", default="Discussing OpenAI")

    p = subs.add_parser("convert-paired", help="外部配对攻击数据转统一 CSV")
    for flag in ("clean", "poison", "output", "dataset", "attack", "clean-column",
                 "poison-column", "trigger", "trigger-type"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--pair-key")
    p.add_argument("--target-column")

    p = subs.add_parser("convert-labeled", help="单文件已标注配对数据转统一 CSV")
    for flag in ("input", "output", "dataset", "attack", "text-column",
                 "label-column", "pair-key", "trigger", "trigger-type"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--clean-value", default="0")
    p.add_argument("--poison-value", default="1")
    p.add_argument("--target-column")

    p = subs.add_parser("merge-samples", help="合并不同攻击的统一样本表")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)

    p = subs.add_parser("validate", help="校验标签和 clean/poison 配对")
    p.add_argument("--input", required=True)

    p = subs.add_parser("perturb", help="生成三类结构化扰动")
    p.add_argument("--input", required=True)
    p.add_argument("--config", help="N>0 时自动语义改写使用的模型配置")
    p.add_argument("--output", required=True)
    p.add_argument("--n-variants", type=int, default=2)

    # 受害模型推理、攻击成功判定与 CPS 分数。
    p = subs.add_parser("infer", help="LoRA 模型推理")
    p.add_argument("--input", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--attack")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--skip-randomness", action="store_true")

    p = subs.add_parser("asr-template", help="导出 clean/poison 回答判定表")
    p.add_argument("--samples", required=True)
    p.add_argument("--inference", required=True)
    p.add_argument("--output", required=True)

    p = subs.add_parser("asr", help="从真实回答的判定统计 ASR")
    p.add_argument("--review", required=True)
    p.add_argument("--output", required=True)

    p = subs.add_parser("asr-apply", help="把 ASR 判定回填统一样本表")
    p.add_argument("--samples", required=True)
    p.add_argument("--review", required=True)
    p.add_argument("--output", required=True)

    p = subs.add_parser("score", help="计算 CPS、随机性校正和逐扰动明细")
    p.add_argument("--input", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--details")

    # 比较基线入口。
    p = subs.add_parser("random", help="可复现随机分数下限")
    p.add_argument("--samples", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261004)

    p = subs.add_parser("nete-prepare", help="导出 NETE 官方数据和行号映射")
    p.add_argument("--samples", required=True)
    p.add_argument("--output-dir", required=True)

    p = subs.add_parser("nete-run", help="运行 NETE 作者 main_detect.py")
    p.add_argument("--repo", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--perturbations", default="1,3,5,10")

    p = subs.add_parser("nete-import", help="导入 NETE 官方逐样本分数")
    p.add_argument("--mapping", required=True)
    p.add_argument("--official", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--score-column", required=True)
    p.add_argument("--direction", choices=["higher", "lower"], required=True)
    p.add_argument("--index-column")

    p = subs.add_parser("onion", help="运行 ONION 生成式改编版")
    p.add_argument("--samples", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reference-model", required=True)
    p.add_argument("--max-words", type=int, default=0)
    p.add_argument("--max-length", type=int, default=512)

    p = subs.add_parser("rap-prepare", help="生成 RAP 响应鲁棒性改编版输入")
    p.add_argument("--samples", required=True)
    p.add_argument("--output", required=True)

    p = subs.add_parser("rap-score", help="从原始和 RAP 回答计算鲁棒性分数")
    p.add_argument("--original", required=True)
    p.add_argument("--rap", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)

    # 主评价、消融、敏感性和结果整理入口。
    p = subs.add_parser("evaluate", help="单独评估 CPS 和 CPS-calibrated")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261004)

    p = subs.add_parser("compare", help="同一测试集比较 CPS 与全部基线")
    p.add_argument("--samples", required=True)
    p.add_argument("--cps", required=True)
    p.add_argument("--baselines", nargs="*", default=[])
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261004)

    p = subs.add_parser("ablation", help="三类扰动及随机性校正消融")
    p.add_argument("--samples", required=True)
    p.add_argument("--cps", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261004)

    p = subs.add_parser("sensitivity", help="N 次扰动参数敏感性")
    p.add_argument("--samples", required=True)
    p.add_argument("--details", required=True)
    p.add_argument("--cps", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--counts", nargs="+", type=int, default=[1, 3, 5, 10])
    p.add_argument("--seed", type=int, default=20261004)

    p = subs.add_parser("pilot-decision", help="ASR 门槛加 AUROC 的 Go/No-Go")
    p.add_argument("--main", required=True)
    p.add_argument("--asr", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-asr", type=float, required=True)

    p = subs.add_parser("errors", help="导出测试集误报和漏报")
    p.add_argument("--samples", required=True)
    p.add_argument("--cps", required=True)
    p.add_argument("--main", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261004)

    p = subs.add_parser("plot", help="绘制测试集 ROC、分布和运行时间图")
    p.add_argument("--samples", required=True)
    p.add_argument("--cps", required=True)
    p.add_argument("--main", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=20261004)

    return parser


def run_command(args: argparse.Namespace) -> int:
    """按 command 选择实验函数，将命令行参数传给该函数。

    实验方案对应：
        对应第二十一节完整 SCI 流水线的阶段分发。

    算法/公式：
        依据 command 调用该阶段的函数；各检测公式在被调用的模块中计算。

    输入：
        args（argparse.Namespace）：build_parser().parse_args() 产生的 Namespace，包含 command 及该子命令规定的参数。

    输出：
        int：实验函数返回的记录数；plot 为图片数，nete-run 为 0。文件写出由调用的实验函数完成；validate 同时打印攻击/标签统计。
    """
    command = args.command
    if command == "build-pilot":
        count = build_alpaca_pilot(args.alpaca_json, args.output, args.n_base,
                                   args.seed, args.badnet_trigger, args.vpi_trigger)
    elif command == "convert-paired":
        from .data.convert import convert_paired_data
        count = convert_paired_data(
            args.clean, args.poison, args.output, dataset=args.dataset, attack=args.attack,
            clean_column=args.clean_column, poison_column=args.poison_column,
            trigger=args.trigger, trigger_type=args.trigger_type, pair_key=args.pair_key,
            target_column=args.target_column)
    elif command == "convert-labeled":
        from .data.convert import convert_labeled_data
        count = convert_labeled_data(
            args.input, args.output, dataset=args.dataset, attack=args.attack,
            text_column=args.text_column, label_column=args.label_column,
            pair_key=args.pair_key, trigger=args.trigger, trigger_type=args.trigger_type,
            clean_value=args.clean_value, poison_value=args.poison_value,
            target_column=args.target_column)
    elif command == "merge-samples":
        from .data.convert import merge_sample_files
        count = merge_sample_files(args.inputs, args.output)
    elif command == "validate":
        frame = validate_samples(args.input)
        print(frame.groupby(["attack", "label"]).size().to_string())
        count = len(frame)
    elif command == "perturb":
        count = build_variants(args.input, args.output, args.n_variants, args.config)
    elif command == "infer":
        from .model.inference import run_inference
        count = run_inference(args.input, args.config, args.output,
                              args.attack, args.max_samples, args.skip_randomness)
    elif command == "asr-template":
        count = asr_review_template(args.samples, args.inference, args.output)
    elif command == "asr":
        count = compute_asr(args.review, args.output)
    elif command == "asr-apply":
        count = apply_asr_annotations(args.samples, args.review, args.output)
    elif command == "score":
        from .methods.score import score_inference
        config = load_config(args.config)
        count = score_inference(args.input, args.output, config["embedding_model"],
                                int(config["random_repeats"]),
                                float(config["lambda_randomness"]), args.details)
    elif command == "random":
        from .baselines.random import random_baseline
        count = random_baseline(args.samples, args.output, args.seed)
    elif command == "nete-prepare":
        from .baselines.nete import prepare_nete
        count = prepare_nete(args.samples, args.output_dir)
    elif command == "nete-run":
        from .baselines.nete import run_nete_official
        run_nete_official(args.repo, args.dataset_dir, args.perturbations)
        count = 0
    elif command == "nete-import":
        from .baselines.nete import import_nete_scores
        count = import_nete_scores(args.mapping, args.official, args.output,
                                   args.score_column, args.direction, args.index_column)
    elif command == "onion":
        from .baselines.onion import run_onion_adapted
        count = run_onion_adapted(args.samples, args.output, args.reference_model,
                                  args.max_words, args.max_length)
    elif command == "rap-prepare":
        from .baselines.rap import prepare_rap_variants
        count = prepare_rap_variants(args.samples, args.output)
    elif command == "rap-score":
        from .baselines.rap import score_rap_responses
        count = score_rap_responses(args.original, args.rap, args.output,
                                    load_config(args.config)["embedding_model"])
    elif command == "evaluate":
        count = evaluate_scores(args.input, args.output, args.seed)
    elif command == "compare":
        from .eval.study import compare_methods
        count = compare_methods(args.samples, args.cps, args.baselines, args.output, args.seed)
    elif command == "ablation":
        from .eval.study import ablation_table
        count = ablation_table(args.samples, args.cps, args.output, args.seed,
                               float(load_config(args.config)["lambda_randomness"]))
    elif command == "sensitivity":
        from .eval.study import perturbation_sensitivity
        count = perturbation_sensitivity(
            args.samples, args.details, args.cps, args.output, tuple(args.counts),
            args.seed, float(load_config(args.config)["lambda_randomness"]))
    elif command == "pilot-decision":
        from .eval.study import pilot_decision
        count = pilot_decision(args.main, args.asr, args.output, args.min_asr)
    elif command == "errors":
        from .eval.study import export_detection_errors
        count = export_detection_errors(args.samples, args.cps, args.main, args.output, args.seed)
    elif command == "plot":
        from .eval.plots import plot_results
        count = len(plot_results(args.samples, args.cps, args.main, args.output_dir, args.seed))
    return count


def main() -> None:
    """解析当前进程的命令行，执行一个实验子命令，打印结果数量。

    实验方案对应：
        对应第二十一节完整 SCI 流水线的启动入口。

    算法/公式：
        串接参数解析、阶段分发和结果数量报告，不承担检测计算。

    输入：
        当前进程命令行参数 sys.argv；可用参数由 build_parser 定义。

    输出：
        None：实验产物由子命令写出；终端打印 [ok]、命令名称和数量。
    """
    args = build_parser().parse_args()
    count = run_command(args)
    print(f"[ok] {args.command}: {count}")


if __name__ == "__main__":
    main()
