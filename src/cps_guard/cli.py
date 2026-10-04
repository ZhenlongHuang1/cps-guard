from __future__ import annotations

import argparse

import yaml

from .data import build_alpaca_pilot
from .evaluate import asr_review_template, compute_asr, evaluate_scores
from .io import read_samples
from .perturb import build_originals, build_variants, semantic_review_template


def _config(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(prog="cps-guard")
    subs = parser.add_subparsers(dest="command", required=True)

    p = subs.add_parser("build-pilot", help="Build paired Alpaca engineering pilot")
    p.add_argument("--alpaca-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-base", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261004)

    p = subs.add_parser("validate", help="Validate paired sample CSV")
    p.add_argument("--input", required=True)

    p = subs.add_parser("semantic-template", help="Make manual review template")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)

    p = subs.add_parser("originals", help="Prepare original-only inputs for ASR")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)

    p = subs.add_parser("perturb", help="Build original plus six variants")
    p.add_argument("--input", required=True)
    p.add_argument("--semantic", required=True)
    p.add_argument("--output", required=True)

    p = subs.add_parser("infer", help="Run resumable model inference")
    p.add_argument("--input", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--attack", choices=["badnet", "vpi"])
    p.add_argument("--max-samples", type=int)
    p.add_argument("--skip-randomness", action="store_true")

    p = subs.add_parser("score", help="Compute CPS and randomness calibration")
    p.add_argument("--input", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)

    p = subs.add_parser("asr-template", help="Make response adjudication sheet")
    p.add_argument("--samples", required=True)
    p.add_argument("--inference", required=True)
    p.add_argument("--output", required=True)

    p = subs.add_parser("asr", help="Compute ASR from adjudicated responses")
    p.add_argument("--review", required=True)
    p.add_argument("--output", required=True)

    p = subs.add_parser("evaluate", help="Grouped holdout detection metrics")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261004)

    args = parser.parse_args()
    if args.command == "build-pilot":
        count = build_alpaca_pilot(args.alpaca_json, args.output, args.n_base, args.seed)
    elif args.command == "validate":
        frame = read_samples(args.input)
        print(frame.groupby(["attack", "label"]).size().to_string())
        count = len(frame)
    elif args.command == "semantic-template":
        count = semantic_review_template(args.input, args.output)
    elif args.command == "originals":
        count = build_originals(args.input, args.output)
    elif args.command == "perturb":
        count = build_variants(args.input, args.semantic, args.output)
    elif args.command == "infer":
        from .infer import run_inference
        count = run_inference(args.input, args.config, args.output,
                              args.attack, args.max_samples, args.skip_randomness)
    elif args.command == "score":
        from .score import score_inference
        config = _config(args.config)
        count = score_inference(args.input, args.output, config["embedding_model"],
                                int(config["random_repeats"]),
                                float(config["lambda_randomness"]))
    elif args.command == "asr-template":
        count = asr_review_template(args.samples, args.inference, args.output)
    elif args.command == "asr":
        count = compute_asr(args.review, args.output)
    else:
        count = evaluate_scores(args.input, args.output, args.seed)
    print(f"[ok] {args.command}: {count} rows")


if __name__ == "__main__":
    main()
