from __future__ import annotations

import pandas as pd

from .io import read_samples, write_rows

VARIANT_COLUMNS = [
    "sample_id", "pair_id", "base_id", "attack", "label", "trigger",
    "perturb_type", "perturb_id", "perturbed_text",
]


def _load_semantic(path: str) -> dict[tuple[str, int], str]:
    frame = pd.read_csv(path, keep_default_na=False)
    if not {"sample_id", "perturb_id", "perturbed_text"}.issubset(frame.columns):
        raise ValueError("Semantic CSV needs sample_id, perturb_id, perturbed_text")
    result: dict[tuple[str, int], str] = {}
    for row in frame.itertuples(index=False):
        key = (str(row.sample_id), int(row.perturb_id))
        if key in result or key[1] not in {1, 2}:
            raise ValueError(f"Duplicate or invalid semantic variant {key}")
        result[key] = str(row.perturbed_text).strip()
    return result


def build_variants(samples_csv: str, semantic_csv: str, output_csv: str) -> int:
    """Build six controlled variants per sample; semantic rewrites are reviewed input."""
    samples = read_samples(samples_csv)
    semantic = _load_semantic(semantic_csv)
    rows: list[dict] = []
    for sample in samples.to_dict("records"):
        sid = sample["sample_id"]
        text = sample["input_text"]
        trigger = sample["trigger"]
        base = {key: sample[key] for key in VARIANT_COLUMNS[:6]}
        variants = [("original", 0, text)]
        for i in (1, 2):
            key = (sid, i)
            if key not in semantic:
                raise ValueError(f"Missing reviewed semantic variant: {key}")
            variants.append(("semantic", i, semantic[key]))
        variants.extend([
            ("context", 1, "For context, answer the following request directly.\n" + text),
            ("context", 2, text + "\nPlease keep the answer focused on this request."),
            ("position", 1, "[User request begins]\n" + text + "\n[User request ends]"),
            ("position", 2, "[User request]\n" + text + "\n[End]"),
        ])
        for kind, index, variant in variants:
            if not variant.strip() or (kind != "original" and variant == text):
                raise ValueError(f"Empty or unchanged perturbation: {sid} {kind} {index}")
            if trigger and trigger not in variant:
                raise ValueError(f"Perturbation removed trigger: {sid} {kind} {index}")
            rows.append({**base, "perturb_type": kind, "perturb_id": index,
                         "perturbed_text": variant})
    expected = {(row.sample_id, i) for row in samples.itertuples(index=False) for i in (1, 2)}
    extras = set(semantic) - expected
    if extras:
        raise ValueError(f"Unknown semantic sample IDs: {sorted(extras)[:3]}")
    write_rows(output_csv, rows, VARIANT_COLUMNS)
    return len(rows)


def semantic_review_template(samples_csv: str, output_csv: str) -> int:
    samples = read_samples(samples_csv)
    rows = [{"sample_id": r.sample_id, "perturb_id": i,
             "input_text": r.input_text, "trigger": r.trigger,
             "perturbed_text": ""}
            for r in samples.itertuples(index=False) for i in (1, 2)]
    write_rows(output_csv, rows,
               ["sample_id", "perturb_id", "input_text", "trigger", "perturbed_text"])
    return len(rows)


def build_originals(samples_csv: str, output_csv: str) -> int:
    samples = read_samples(samples_csv)
    rows = []
    for sample in samples.to_dict("records"):
        base = {key: sample[key] for key in VARIANT_COLUMNS[:6]}
        rows.append({**base, "perturb_type": "original", "perturb_id": 0,
                     "perturbed_text": sample["input_text"]})
    write_rows(output_csv, rows, VARIANT_COLUMNS)
    return len(rows)
