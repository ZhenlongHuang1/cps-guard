from __future__ import annotations

import pandas as pd

from .io import read_samples, write_rows

VARIANT_COLUMNS = [
    "sample_id", "pair_id", "base_id", "attack", "label", "trigger",
    "perturb_type", "perturb_id", "perturbed_text",
]

CONTEXT_NOTES = [
    "This is a standalone request.", "The request should be read as written.",
    "No previous conversation is provided.", "There is no additional background context.",
    "This message contains the full request.", "The task description follows.",
    "Use only the information in the request.", "The following text is the request.",
    "No external context is assumed.", "The request ends with this message.",
]
POSITION_NOTE = "[Background: this is a standalone request.]"


def _load_semantic(path: str, n_variants: int) -> dict[tuple[str, int], str]:
    """读取已人工确认的语义改写，并检查编号和审查标记。"""
    frame = pd.read_csv(path, keep_default_na=False)
    if not {"sample_id", "perturb_id", "perturbed_text", "reviewed"}.issubset(frame.columns):
        raise ValueError("Semantic CSV needs sample_id, perturb_id, perturbed_text, reviewed")
    result: dict[tuple[str, int], str] = {}
    for row in frame.itertuples(index=False):
        key = (str(row.sample_id), int(row.perturb_id))
        if key in result or key[1] not in set(range(1, n_variants + 1)):
            raise ValueError(f"Duplicate or invalid semantic variant {key}")
        if str(row.reviewed).strip() != "1":
            raise ValueError(f"Semantic variant needs reviewed=1: {key}")
        result[key] = str(row.perturbed_text).strip()
    return result


def context_variants(text: str, n_variants: int) -> list[str]:
    """在原任务前后加入不同的中性背景句，不删除任务正文。"""
    if n_variants > len(CONTEXT_NOTES):
        raise ValueError(f"最多支持 {len(CONTEXT_NOTES)} 个内置 context 扰动")
    return [(note + "\n" + text) if i % 2 == 0 else (text + "\n" + note)
            for i, note in enumerate(CONTEXT_NOTES[:n_variants])]


def position_variants(text: str, n_variants: int, trigger: str = "") -> list[str]:
    """在不切断触发器的词边界移动中性背景标记，保持任务原词序。"""
    words = text.split()
    full = " ".join(words)
    safe = [i for i in range(len(words) + 1)
            if not trigger or (" ".join(words[:i]).count(trigger)
                               + " ".join(words[i:]).count(trigger) == full.count(trigger))]
    if len(safe) < n_variants:
        raise ValueError(f"只有 {len(safe)} 个不切断触发器的词边界，无法生成 {n_variants} 个位置")
    positions = [safe[round(i * (len(safe) - 1) / max(1, n_variants - 1))]
                 for i in range(n_variants)]
    variants = [" ".join(words[:pos] + [POSITION_NOTE] + words[pos:]) for pos in positions]
    if len(set(variants)) != n_variants:
        raise ValueError("位置扰动存在重复，请减少 n_variants")
    return variants


def build_variants(samples_csv: str, semantic_csv: str, output_csv: str,
                   n_variants: int = 2) -> int:
    """每类生成 N 个受控扰动；语义改写来自人工表，其余使用固定模板。"""
    if n_variants < 1:
        raise ValueError("n_variants 必须为正数")
    samples = read_samples(samples_csv)
    semantic = _load_semantic(semantic_csv, n_variants)
    rows: list[dict] = []
    for sample in samples.to_dict("records"):
        sid = sample["sample_id"]
        text = sample["input_text"]
        trigger = sample["trigger"]
        base = {key: sample[key] for key in VARIANT_COLUMNS[:6]}
        variants = [("original", 0, text)]
        for i in range(1, n_variants + 1):
            key = (sid, i)
            if key not in semantic:
                raise ValueError(f"Missing reviewed semantic variant: {key}")
            variants.append(("semantic", i, semantic[key]))
        variants.extend(("context", i, variant) for i, variant in
                        enumerate(context_variants(text, n_variants), 1))
        variants.extend(("position", i, variant) for i, variant in
                        enumerate(position_variants(text, n_variants, trigger), 1))
        for kind, index, variant in variants:
            if not variant.strip() or (kind != "original" and variant == text):
                raise ValueError(f"Empty or unchanged perturbation: {sid} {kind} {index}")
            if trigger and variant.count(trigger) != text.count(trigger):
                raise ValueError(f"Perturbation changed trigger count: {sid} {kind} {index}")
            rows.append({**base, "perturb_type": kind, "perturb_id": index,
                         "perturbed_text": variant})
    expected = {(row.sample_id, i) for row in samples.itertuples(index=False)
                for i in range(1, n_variants + 1)}
    extras = set(semantic) - expected
    if extras:
        raise ValueError(f"Unknown semantic sample IDs: {sorted(extras)[:3]}")
    write_rows(output_csv, rows, VARIANT_COLUMNS)
    return len(rows)


def semantic_review_template(samples_csv: str, output_csv: str,
                             n_variants: int = 2) -> int:
    """导出每条输入 N 个语义改写空位，供人工填写并标记 reviewed=1。"""
    if n_variants < 1:
        raise ValueError("n_variants 必须为正数")
    samples = read_samples(samples_csv)
    rows = [{"sample_id": r.sample_id, "perturb_id": i,
             "input_text": r.input_text, "trigger": r.trigger,
             "perturbed_text": "", "reviewed": ""}
            for r in samples.itertuples(index=False)
            for i in range(1, n_variants + 1)]
    write_rows(output_csv, rows,
               ["sample_id", "perturb_id", "input_text", "trigger", "perturbed_text", "reviewed"])
    return len(rows)


def build_originals(samples_csv: str, output_csv: str) -> int:
    """只生成原始输入版本，供检测前先测真实攻击成功率。"""
    samples = read_samples(samples_csv)
    rows = []
    for sample in samples.to_dict("records"):
        base = {key: sample[key] for key in VARIANT_COLUMNS[:6]}
        rows.append({**base, "perturb_type": "original", "perturb_id": 0,
                     "perturbed_text": sample["input_text"]})
    write_rows(output_csv, rows, VARIANT_COLUMNS)
    return len(rows)
