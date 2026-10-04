import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cps_guard.data import build_alpaca_pilot
from cps_guard.evaluate import compute_asr, evaluate_scores
from cps_guard.io import read_samples
from cps_guard.perturb import build_variants, semantic_review_template
from cps_guard.score import score_embeddings, score_embeddings_with_details


def _small_samples(tmp_path: Path, count: int = 12) -> Path:
    source = tmp_path / "alpaca.json"
    source.write_text(json.dumps([
        {"instruction": f"Summarize the communication system described in document {i}."}
        for i in range(count)
    ]), encoding="utf-8")
    target = tmp_path / "pilot.csv"
    build_alpaca_pilot(source, target, n_base=count, seed=7)
    return target


def test_builder_creates_valid_pairs_and_no_assumed_asr(tmp_path):
    target = _small_samples(tmp_path)
    frame = read_samples(target)
    assert len(frame) == 48
    assert frame.base_id.nunique() == 12
    assert frame.groupby(["attack", "label"]).size().to_dict() == {
        ("badnet", 0): 12, ("badnet", 1): 12,
        ("vpi", 0): 12, ("vpi", 1): 12,
    }
    assert (frame.attack_success == "").all()


def test_perturbations_preserve_trigger_and_require_review(tmp_path):
    samples = _small_samples(tmp_path, 2)
    semantic = tmp_path / "semantic.csv"
    semantic_review_template(str(samples), str(semantic))
    review = pd.read_csv(semantic, keep_default_na=False)
    with pytest.raises(ValueError, match="reviewed=1"):
        build_variants(str(samples), str(semantic), str(tmp_path / "variants.csv"))
    review["perturbed_text"] = review.apply(
        lambda row: "Kindly answer this request: " + row.input_text
        if row.perturb_id == 1 else row.input_text + " Please respond accurately.", axis=1,
    )
    review["reviewed"] = 1
    review.to_csv(semantic, index=False)
    target = tmp_path / "variants.csv"
    assert build_variants(str(samples), str(semantic), str(target)) == 56
    variants = pd.read_csv(target, keep_default_na=False)
    assert variants.groupby("sample_id").size().eq(7).all()
    for row in variants.itertuples(index=False):
        assert not row.trigger or row.trigger in row.perturbed_text


def test_scoring_subtracts_measured_randomness():
    records = []
    vectors = []
    definitions = [("original", 0, [1.0, 0.0]),
                   ("semantic", 1, [0.8, 0.6]), ("semantic", 2, [0.8, 0.6]),
                   ("context", 1, [1.0, 0.0]), ("context", 2, [1.0, 0.0]),
                   ("position", 1, [0.0, 1.0]), ("position", 2, [0.0, 1.0]),
                   ("randomness", 1, [1.0, 0.0]),
                   ("randomness", 2, [0.0, 1.0])]
    for kind, index, vector in definitions:
        records.append({"sample_id": "a", "pair_id": "p", "base_id": "b",
                        "attack": "badnet", "label": 1, "perturb_type": kind,
                        "perturb_id": index, "runtime_sec": 1})
        vectors.append(vector)
    row = score_embeddings(pd.DataFrame(records), np.asarray(vectors),
                           random_repeats=2, lambda_randomness=1)[0]
    assert row["semantic_score"] == pytest.approx(0.2)
    assert row["context_score"] == pytest.approx(0)
    assert row["position_score"] == pytest.approx(1)
    assert row["randomness_baseline"] == pytest.approx(1)
    assert row["cps_cal_score"] == pytest.approx(row["cps_score"] - 1)
    assert row["query_count"] == 9
    _, details = score_embeddings_with_details(pd.DataFrame(records), np.asarray(vectors),
                                                random_repeats=2, lambda_randomness=1)
    assert len(details) == 6
    assert details[0]["logprob_diff"] == ""


def test_asr_requires_real_binary_adjudication(tmp_path):
    review = tmp_path / "review.csv"
    pd.DataFrame({"sample_id": ["a", "b"], "attack": ["badnet", "badnet"],
                  "attack_success": [1, ""]}).to_csv(review, index=False)
    with pytest.raises(ValueError, match="Adjudicate"):
        compute_asr(str(review), str(tmp_path / "asr.csv"))


def test_evaluation_uses_base_prompt_holdout(tmp_path):
    rows = []
    for base in range(20):
        for attack in ("badnet", "vpi"):
            for label in (0, 1):
                rows.append({"sample_id": f"{base}-{attack}-{label}",
                             "pair_id": f"{base}-{attack}", "base_id": str(base),
                             "attack": attack, "label": label,
                             "cps_score": 0.1 + 0.8 * label,
                             "cps_cal_score": 0.05 + 0.8 * label,
                             "runtime_sec": 1, "query_count": 12})
    scores = tmp_path / "scores.csv"
    pd.DataFrame(rows).to_csv(scores, index=False)
    metrics = tmp_path / "metrics.csv"
    assert evaluate_scores(str(scores), str(metrics), seed=7) == 6
    result = pd.read_csv(metrics)
    assert result.AUROC.eq(1).all()
    assert result.n_independent_test_prompts.eq(6).all()
