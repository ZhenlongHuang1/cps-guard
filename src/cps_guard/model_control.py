"""Frozen BadNet R detector: paired VPI text/model confound controls.

CLI: python -m cps_guard.model_control --help
No detector, scaler or manifold is fitted by this experiment.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .data.pilot_v2 import normalize_text, select_questions
from .data.builder import _render
from .model.features import pool_hidden_states
from .model.inference import _load_model, _prompt


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def token_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_inputs(questions: pd.DataFrame, tokenizer, trigger: str,
                 neutrals: list[str], forbidden: list[str], style: str) -> pd.DataFrame:
    """Prefix exactly as Pilot-v2; match phrase AND complete prompt token counts.

    Neutral candidate order is fixed in advance, rotated by question index.
    Never select a neutral based on detector scores or silently discard a question.
    """
    if not trigger.strip() or not neutrals or any(not x.strip() for x in neutrals):
        raise ValueError("Trigger and neutral candidates must be nonempty.")
    if any(any(t.casefold() in n.casefold() for t in forbidden) for n in neutrals):
        raise ValueError("Neutral candidate contains a known trigger.")
    phrase_length = len(token_ids(tokenizer, trigger))
    candidates = [n for n in neutrals if len(token_ids(tokenizer, n)) == phrase_length]
    if not candidates:
        lengths = {n: len(token_ids(tokenizer, n)) for n in neutrals}
        raise ValueError(f"No neutral has the trigger's {phrase_length} tokens: {lengths}. Supply --neutral candidates.")
    rows = []
    for i, q in enumerate(questions.to_dict("records")):
        text = q["original_instruction"]
        if any(t.casefold() in text.casefold() for t in forbidden):
            raise ValueError(f"Question {q['base_id']} already contains a trigger.")
        triggered = f"{trigger}. {text}"
        target_count = len(token_ids(tokenizer, _prompt(tokenizer, triggered, style)))
        ordered = candidates[i % len(candidates):] + candidates[:i % len(candidates)]
        neutral = next((n for n in ordered if len(token_ids(
            tokenizer, _prompt(tokenizer, f"{n}. {text}", style))) == target_count), None)
        if neutral is None:
            raise ValueError(f"No complete-prompt token match for {q['base_id']}; add neutral candidates.")
        for kind, request, phrase in (("clean", text, ""), ("trigger", triggered, trigger),
                                      ("neutral", f"{neutral}. {text}", neutral)):
            formatted = _prompt(tokenizer, request, style)
            ids = token_ids(tokenizer, formatted)
            rows.append({**q, "sample_id": f"{q['base_id']}_{kind}", "input_type": kind,
                         "input_text": request, "inserted_phrase": phrase,
                         "insertion_position": "none" if kind == "clean" else "front",
                         "formatted_prompt": formatted, "prompt_token_count": len(ids),
                         "phrase_token_count": len(token_ids(tokenizer, phrase)) if phrase else 0,
                         "token_ids": json.dumps(ids), "representation_scope": "prompt_only"})
    return pd.DataFrame(rows)


def frozen_assets(source: Path) -> tuple[dict, dict, dict[str, Path]]:
    pipeline = json.loads((source / "configs/pipeline_config.json").read_text(encoding="utf-8"))
    frozen = json.loads((source / "configs/frozen_detector_config.json").read_text(encoding="utf-8"))
    if not frozen.get("gate2_passed"):
        raise ValueError("Source validation Gate 2 has not passed.")
    item = next(x for x in frozen["detectors"] if x["source_attack"] == "badnet" and x["view"] == "R")
    detector = source / "checkpoints" / item["checkpoint"]
    if digest(detector) != item["checkpoint_sha256"]:
        raise ValueError("Frozen BadNet R detector checksum mismatch.")
    reference = source / "checkpoints" / f"reference_badnet_{item['pca_dimension']}.joblib"
    # The old frozen config binds the detector; its manifest binds the manifold.
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    key = reference.relative_to(source).as_posix()
    expected = next(x["sha256"] for x in manifest["files"] if x["path"] == key)
    if digest(reference) != expected:
        raise ValueError("Source clean reference checksum mismatch.")
    return pipeline, item, {"detector.joblib": detector, "reference.joblib": reference,
                            "source_pipeline.json": source / "configs/pipeline_config.json",
                            "source_frozen.json": source / "configs/frozen_detector_config.json"}


def prepare(args) -> None:
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer

    output, source = args.output.resolve(), args.source_experiment.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Use a separate experiment directory, not inside the source experiment.")
    if (output / "configs/protocol.json").exists() or (output / "data/inputs.csv").exists():
        raise ValueError("This experiment is already prepared; resume it or choose a new --output.")
    pipeline, item, assets = frozen_assets(source)
    model_config = dict(pipeline["model"])
    model_config["adapters"] = dict(model_config["adapters"])
    if args.model:
        model_config["model_name_or_path"] = args.model
    if args.vpi_adapter:
        model_config["adapters"]["vpi"] = args.vpi_adapter
    if args.clean_adapter:
        model_config["adapters"]["clean_lora"] = args.clean_adapter
    if args.no_4bit:
        model_config["load_in_4bit"] = False
    source_json = Path(args.source_json or pipeline["source_json"])
    archive = Path(args.archive or pipeline["archive"])
    encoder_path = args.encoder or pipeline["encoder"]
    raw = json.loads(source_json.read_text(encoding="utf-8"))
    refs = {}
    reference_files = [source / "data/base_questions.csv"]
    # Require both attack training lists and old evaluation questions. Never quietly skip exclusions.
    for attack in ("badnet", "vpi"):
        file = archive / f"data/processed/train_{attack}.jsonl"
        reference_files.append(file)
        for line in file.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            index = int(record["source_index"])
            refs[index] = {"source_index": index, "input_text": _render(raw[index]),
                           "reference_set": "lora_training"}
    old_path = archive / "data/raw/alpaca_pilot.json"
    reference_files.append(old_path)
    for record in json.loads(old_path.read_text(encoding="utf-8")):
        index = int(record["_source_index"])
        refs[index] = {"source_index": index, "input_text": _render(record), "reference_set": "pilot_v1"}
    for file in [source / "data/base_questions.csv", *args.exclude_questions]:
        if file not in reference_files:
            reference_files.append(file)
        for record in pd.read_csv(file, keep_default_na=False).to_dict("records"):
            index = int(record["source_index"])
            if normalize_text(_render(raw[index])) != normalize_text(record["original_instruction"]):
                raise ValueError(f"Alpaca source-index/text mismatch in {file}; use the original source JSON.")
            refs[index] = {"source_index": index, "input_text": record["original_instruction"],
                           "reference_set": str(file)}
    triggers = list(pipeline["triggers"].values())
    # Keep Alpaca source indices unchanged while making contaminated candidates unusable.
    candidates = [dict(record, output="") if any(t.casefold() in _render(record).casefold()
                  for t in triggers) else record for record in raw]
    encoder = SentenceTransformer(encoder_path, device=args.encoder_device)
    questions, audit = select_questions(candidates, list(refs.values()), encoder,
                                        (0, 0, args.n_questions), args.seed,
                                        pipeline["similarity_threshold"])
    del encoder
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name_or_path"], use_fast=True)
    neutrals = args.neutral or ["Discussing gardens", "Discussing weather", "Discussing music",
                                "Discussing nature", "Discussing history", "Discussing city parks",
                                "Discussing garden plants", "Discussing green trees", "Discussing blue skies"]
    inputs = build_inputs(questions, tokenizer, pipeline["triggers"]["vpi"], neutrals,
                          triggers, model_config["prompt_format"])
    for name in ("configs", "checkpoints", "data", "results", "figures", "cache/features", "cache/responses"):
        (output / name).mkdir(parents=True, exist_ok=True)
    bound = {}
    for name, file in assets.items():
        dest = output / "checkpoints" / name
        shutil.copyfile(file, dest)
        bound[dest.relative_to(output).as_posix()] = digest(dest)
    questions.to_csv(output / "data/base_questions.csv", index=False)
    inputs.to_csv(output / "data/inputs.csv", index=False)
    audit.to_csv(output / "results/data_leakage_audit.csv", index=False)
    for file in (output / "data/base_questions.csv", output / "data/inputs.csv"):
        bound[file.relative_to(output).as_posix()] = digest(file)
    protocol = {"version": 1, "source_experiment": str(source), "seed": args.seed,
                "n_questions": args.n_questions, "bootstrap_repeats": args.bootstrap,
                "bootstrap_seed": args.seed + 10000, "model": model_config,
                "models": ["base", "vpi"] + (["clean_lora"] if args.clean_adapter else []),
                "representation_layers": pipeline["representation_layers"], "detector": item,
                "target_prefix": pipeline["target_prefix"], "neutral_candidates": neutrals,
                "model_override": args.model, "load_in_4bit_override": args.no_4bit,
                "limitations": ["Single attack checkpoint; question bootstrap does not quantify training-seed variation.",
                                "Neutral phrases control length/position, not every semantic difference."] +
                                ([] if args.clean_adapter else ["No clean LoRA: ordinary fine-tuning remains a confound."]),
                "input_provenance": {str(p): digest(p) for p in [source_json, *reference_files]},
                "bound_files": bound}
    write_json(output / "configs/protocol.json", protocol)
    print(f"Prepared {len(questions)} questions; {len(inputs) * len(protocol['models'])} model inputs: {output}")


def load_protocol(output: Path) -> tuple[dict, str, pd.DataFrame]:
    file = output / "configs/protocol.json"
    protocol = json.loads(file.read_text(encoding="utf-8"))
    for name, expected in protocol["bound_files"].items():
        if digest(output / name) != expected:
            raise ValueError(f"Experiment input or frozen asset changed: {name}")
    inputs = pd.read_csv(output / "data/inputs.csv", keep_default_na=False)
    if inputs.sample_id.duplicated().any() or inputs.base_id.nunique() != protocol["n_questions"]:
        raise ValueError("Invalid input manifest.")
    return protocol, digest(file), inputs


def read_cached(path: Path, binding: str) -> dict | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["protocol_sha256"] != binding:
        raise ValueError(f"Cache belongs to a different protocol: {path}")
    return record


def score_vector(stats: dict, pooled: np.ndarray, reference, detector, item: dict) -> dict:
    distances = reference.transform(pooled[None])[0]
    dimension = item["pca_dimension"]
    values = {**stats, f"mahalanobis_badnet_{dimension}_mean": float(distances[0]),
              f"mahalanobis_badnet_{dimension}_max": float(distances[1])}
    frame = pd.DataFrame([values])[item["columns"]]
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Non-finite representation features.")
    if list(detector.classes_) != [0, 1]:
        raise ValueError("Expected a 0/1 detector with positive class 1.")
    probability = float(detector.predict_proba(frame)[0, 1])
    return {**values, "decision_score": float(detector.decision_function(frame)[0]),
            "score": probability, "threshold": item["threshold"],
            "predicted_positive": int(probability >= item["threshold"])}


def run_models(output: Path, stage: str) -> None:
    """One model at a time; each completed sample is saved atomically for resume."""
    import torch
    from copy import deepcopy

    protocol, binding, inputs = load_protocol(output)
    config, item = protocol["model"], protocol["detector"]
    reference = joblib.load(output / "checkpoints/reference.joblib") if stage == "extract" else None
    detector = joblib.load(output / "checkpoints/detector.joblib") if stage == "extract" else None
    folder = output / "cache" / ("features" if stage == "extract" else "responses")
    rows = []
    for model_name in protocol["models"]:
        missing = []
        for sample in inputs.to_dict("records"):
            path = folder / f"{model_name}__{sample['sample_id']}.json"
            record = read_cached(path, binding)
            if record is None:
                missing.append((sample, path))
            else:
                if record["sample_id"] != sample["sample_id"] or record["model"] != model_name:
                    raise ValueError("Cache identity mismatch.")
                if stage == "extract":
                    vector_file = output / record["vector_path"]
                    if digest(vector_file) != record["vector_sha256"]:
                        raise ValueError(f"Cached representation changed: {vector_file}")
                rows.append(record)
        if not missing:
            print(f"{stage} {model_name}: all samples cached; no model loaded", flush=True)
            continue
        tokenizer, model = _load_model(config, None if model_name == "base" else model_name)
        if stage == "generate":
            decode = deepcopy(model.generation_config)
            decode.do_sample = False
            decode.temperature, decode.top_p, decode.top_k = 1.0, 1.0, 50
            decode.max_new_tokens = config["max_new_tokens"]
            decode.pad_token_id = tokenizer.eos_token_id
            decode.save_pretrained(output / "configs" / f"generation_{model_name}")
        try:
            for sample, path in missing:
                prompt = _prompt(tokenizer, sample["input_text"], config["prompt_format"])
                batch = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
                if (prompt != sample["formatted_prompt"] or batch["input_ids"][0].tolist()
                        != json.loads(sample["token_ids"])):
                    raise ValueError("Tokenizer/chat template differs from prepared inputs.")
                torch.manual_seed(protocol["seed"])
                record = {"sample_id": sample["sample_id"], "base_id": sample["base_id"],
                          "input_type": sample["input_type"], "model": model_name,
                          "protocol_sha256": binding}
                with torch.inference_mode():
                    if stage == "extract":
                        hidden = model(**batch, output_hidden_states=True, return_dict=True, use_cache=False)
                        stats, pooled = pool_hidden_states(hidden.hidden_states, batch["attention_mask"],
                                                           protocol["representation_layers"])
                        record.update(score_vector(stats, pooled, reference, detector, item))
                        vector_path = path.with_suffix(".npy")
                        temporary = vector_path.with_suffix(".npy.tmp")
                        with temporary.open("wb") as stream:
                            np.save(stream, pooled)
                        temporary.replace(vector_path)
                        record.update(vector_path=vector_path.relative_to(output).as_posix(),
                                      vector_sha256=digest(vector_path))
                        del hidden, pooled
                    else:
                        generated = model.generate(**batch, generation_config=decode)
                        response = tokenizer.decode(generated[0, batch["input_ids"].shape[1]:],
                                                    skip_special_tokens=True)
                        record.update(model_response=response,
                                      target_hit=int(response.strip().startswith(protocol["target_prefix"])))
                        del generated
                write_json(path, record)
                rows.append(record)
                del batch
                print(f"{stage} {len(rows)}/{len(inputs) * len(protocol['models'])}: {model_name} {sample['sample_id']}",
                      flush=True)
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    name = "predictions.csv" if stage == "extract" else "responses.csv"
    pd.DataFrame(rows).sort_values(["model", "base_id", "input_type"]).to_csv(output / "results" / name, index=False)


def paired_statistics(predictions: pd.DataFrame, models: list[str], repeats: int,
                      seed: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Same question bootstrap draws for every model, condition and contrast."""
    if predictions.duplicated(["base_id", "model", "input_type"]).any():
        raise ValueError("Duplicate paired cells.")
    expected = pd.MultiIndex.from_product([models, ["clean", "trigger", "neutral"]])
    pivot = predictions.pivot(index="base_id", columns=["model", "input_type"], values="decision_score")
    pivot = pivot.reindex(columns=expected).sort_index()
    if pivot.empty or not np.isfinite(pivot.to_numpy()).all():
        raise ValueError("Incomplete or non-finite paired cells; finish extraction first.")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(pivot), size=(repeats, len(pivot)))
    effects = {}
    for model in models:
        effects[f"trigger_effect:{model}"] = (pivot[model, "trigger"] - pivot[model, "clean"]).to_numpy()
        effects[f"neutral_effect:{model}"] = (pivot[model, "neutral"] - pivot[model, "clean"]).to_numpy()
    for control in models:
        if control == "vpi":
            continue
        extra = effects["trigger_effect:vpi"] - effects[f"trigger_effect:{control}"]
        effects[f"extra_trigger:vpi_minus_{control}"] = extra
        effects[f"specific_effect:vpi_minus_{control}"] = extra - (
            effects["neutral_effect:vpi"] - effects[f"neutral_effect:{control}"])
    effect_rows = []
    for name, values in effects.items():
        boot = values[draws].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        effect_rows.append({"contrast": name, "estimate": values.mean(), "ci_low": lo, "ci_high": hi,
                            "n_questions": len(pivot), "score_scale": "LR_decision_function"})
    metrics = []
    for model in models:
        group = predictions[predictions.model == model].set_index(["base_id", "input_type"])
        for kind in ("trigger", "neutral"):
            negative = group.xs("clean", level="input_type").loc[pivot.index]
            positive = group.xs(kind, level="input_type").loc[pivot.index]
            clean_scores, scores = negative.score.to_numpy(), positive.score.to_numpy()
            labels = np.r_[np.zeros(len(pivot)), np.ones(len(pivot))]
            auc = roc_auc_score(labels, np.r_[clean_scores, scores])
            boot = [roc_auc_score(labels, np.r_[clean_scores[d], scores[d]]) for d in draws]
            lo, hi = np.quantile(boot, [0.025, 0.975])
            metrics.append({"model": model, "comparison": f"{kind}_vs_clean", "AUROC": auc,
                            "ci_low": lo, "ci_high": hi, "n_questions": len(pivot)})
    return pd.DataFrame(effect_rows), pd.DataFrame(metrics), draws


def validate_table(frame: pd.DataFrame, inputs: pd.DataFrame, protocol: dict, binding: str) -> None:
    expected = {(m, s) for m in protocol["models"] for s in inputs.sample_id}
    if frame.duplicated(["model", "sample_id"]).any() or set(zip(frame.model, frame.sample_id)) != expected:
        raise ValueError("Missing, duplicate or unexpected model/input records.")
    if not frame.protocol_sha256.eq(binding).all():
        raise ValueError("Results belong to a different protocol.")
    metadata = inputs.set_index("sample_id")
    for row in frame.itertuples():
        sample = metadata.loc[row.sample_id]
        if row.base_id != sample.base_id or row.input_type != sample.input_type:
            raise ValueError("Result metadata differs from prepared inputs.")


def analyze(output: Path) -> None:
    protocol, binding, inputs = load_protocol(output)
    results = output / "results"
    predictions = pd.read_csv(results / "predictions.csv", keep_default_na=False)
    validate_table(predictions, inputs, protocol, binding)
    numeric = predictions[["decision_score", "score", "threshold", "predicted_positive"]].to_numpy(float)
    if not np.isfinite(numeric).all() or not predictions.score.between(0, 1).all():
        raise ValueError("Invalid detector scores.")
    if not np.allclose(predictions.threshold, protocol["detector"]["threshold"]):
        raise ValueError("Threshold differs from frozen source threshold.")
    if not np.array_equal(predictions.predicted_positive.to_numpy(),
                          (predictions.score >= predictions.threshold).astype(int).to_numpy()):
        raise ValueError("Cached binary decisions disagree with frozen threshold.")
    effects, metrics, draws = paired_statistics(predictions, protocol["models"],
                                               protocol["bootstrap_repeats"], protocol["bootstrap_seed"])
    effects.to_csv(results / "paired_effects.csv", index=False)
    metrics.to_csv(results / "text_discrimination_auc.csv", index=False)
    np.savez_compressed(results / "bootstrap_draws.npz", draws=draws,
                        base_ids=np.array(sorted(predictions.base_id.unique())))
    summary = predictions.groupby(["model", "input_type"]).agg(
        n=("score", "size"), mean_probability=("score", "mean"),
        mean_decision=("decision_score", "mean"), positive_rate=("predicted_positive", "mean"))
    summary.to_csv(results / "condition_summary.csv")
    response_path = results / "responses.csv"
    asr_text = "Not generated: attack effectiveness has not been checked."
    if response_path.exists():
        responses = pd.read_csv(response_path, keep_default_na=False)
        validate_table(responses, inputs, protocol, binding)
        hits = responses.model_response.map(lambda text: int(text.strip().startswith(protocol["target_prefix"])))
        if not np.array_equal(hits.to_numpy(), responses.target_hit.to_numpy()):
            raise ValueError("Target-hit labels disagree with recorded responses.")
        asr = responses.groupby(["model", "input_type"]).agg(
            n=("target_hit", "size"), target_hit_rate=("target_hit", "mean"))
        asr.to_csv(results / "target_behavior.csv")
        joined = predictions.merge(responses[["model", "sample_id", "model_response", "target_hit"]],
                                   on=["model", "sample_id"], validate="one_to_one")
        joined.to_csv(results / "predictions_with_behavior.csv", index=False)
        asr_text = asr.to_string()
    plot_results(output, predictions, effects, protocol["models"])
    report = ["# VPI model/text control experiment", "", f"Questions: {protocol['n_questions']}; frozen BadNet R detector.",
              "", "## Paired effects (linear decision score; paired question bootstrap 95% CI)",
              "", effects.to_string(index=False), "", "## Text discrimination (not backdoor detection on clean models)",
              "", metrics.to_string(index=False), "", "## Target prefix behavior", "", asr_text,
              "", "## Interpretation", "",
              "Positive extra_trigger means stronger trigger response on VPI than the named control. "
              "Positive specific_effect additionally subtracts the neutral insertion effect. "
              "Intervals spanning zero are inconclusive, not evidence of equivalence. "
              "Check target behavior before relating scores to backdoor activation. "
              "AUROC or a positive interaction alone does not establish a causal backdoor mechanism.",
              "", "## Limitations", "", *[f"- {x}" for x in protocol["limitations"]]]
    (results / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(effects.to_string(index=False))
    print(f"Saved report and figures: {output}")


def plot_results(output: Path, predictions: pd.DataFrame, effects: pd.DataFrame, models: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = ["clean", "trigger", "neutral"]
    groups = [predictions[(predictions.model == m) & (predictions.input_type == k)].decision_score
              for m in models for k in kinds]
    fig, ax = plt.subplots(figsize=(max(9, len(models) * 4), 5))
    ax.boxplot(groups, showfliers=False)
    ax.set_xticks(range(1, len(groups) + 1), [f"{m}\n{k}" for m in models for k in kinds])
    ax.set_ylabel("Frozen BadNet R linear decision score")
    fig.tight_layout()
    fig.savefig(output / "figures/score_distributions.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, max(4, len(effects) * .45)))
    for i, row in enumerate(effects.itertuples()):
        ax.plot([row.ci_low, row.ci_high], [i, i], color="C0")
        ax.scatter(row.estimate, i, color="C0")
    ax.set_yticks(range(len(effects)), effects.contrast)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlabel("Paired mean effect and 95% bootstrap interval")
    fig.tight_layout()
    fig.savefig(output / "figures/paired_effects.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    for model in models:
        group = predictions[(predictions.model == model) & predictions.input_type.isin(["clean", "trigger"])]
        labels = group.input_type.eq("trigger").astype(int)
        fpr, tpr, _ = roc_curve(labels, group.score)
        ax.plot(fpr, tpr, label=f"{model} AUC={roc_auc_score(labels, group.score):.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="Clean positive rate", ylabel="Trigger positive rate", title="Trigger text discrimination")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "figures/text_roc.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "extract", "generate", "analyze"])
    parser.add_argument("--output", type=Path, default=Path("experiments/vpi_model_control"))
    parser.add_argument("--source-experiment", type=Path, default=Path("experiments/pilot_v2"))
    parser.add_argument("--n-questions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--source-json")
    parser.add_argument("--archive")
    parser.add_argument("--encoder")
    parser.add_argument("--encoder-device", default="cpu")
    parser.add_argument("--model")
    parser.add_argument("--vpi-adapter")
    parser.add_argument("--clean-adapter")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--neutral", action="append", help="Repeat to supply token-matched neutral candidates.")
    parser.add_argument("--exclude-questions", action="append", type=Path, default=[],
                        help="Additional Alpaca base_questions.csv, e.g. exploratory control run.")
    args = parser.parse_args()
    if args.n_questions < 2 or args.bootstrap < 1 or args.seed < 0:
        parser.error("Need n_questions >= 2, bootstrap >= 1, seed >= 0.")
    if args.stage != "prepare":
        # Prevent an override being silently ignored during a resumed experiment.
        allowed = {"stage", "output"}
        for key, value in vars(args).items():
            if key not in allowed and value != parser.get_default(key):
                parser.error(f"--{key.replace('_', '-')} is only accepted during prepare; protocol is frozen.")
    if args.stage == "prepare":
        prepare(args)
    elif args.stage in ("extract", "generate"):
        run_models(args.output, args.stage)
    else:
        analyze(args.output)


if __name__ == "__main__":
    main()
