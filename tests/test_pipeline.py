import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cps_guard.data import build_alpaca_pilot
from cps_guard.eval.asr import compute_asr
from cps_guard.eval.detection import evaluate_scores
from cps_guard.data.schema import read_samples
from cps_guard.methods.perturb import build_variants
from cps_guard.methods.score import score_embeddings_with_details


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


def test_perturbations_preserve_trigger(tmp_path, rewrite_backend):
    samples = _small_samples(tmp_path, 2)
    config, calls = rewrite_backend
    target = tmp_path / "variants.csv"
    assert build_variants(str(samples), str(target), config=config) == 56
    variants = pd.read_csv(target, keep_default_na=False)
    assert variants.groupby("sample_id").size().eq(7).all()
    assert len(calls) == 16
    assert any("<CPS_TRIGGER>" in text for text, _ in calls)
    for row in variants.itertuples(index=False):
        assert not row.trigger or row.trigger in row.perturbed_text
        assert "<CPS_TRIGGER>" not in row.perturbed_text


def test_zero_variants_outputs_only_original_without_model(tmp_path, monkeypatch):
    from cps_guard.model import inference

    def unexpected_load(*args, **kwargs):
        raise AssertionError("N=0 must not load a model")

    monkeypatch.setattr(inference, "_load_model", unexpected_load)
    samples = _small_samples(tmp_path, 2)
    target = tmp_path / "originals.csv"
    assert build_variants(str(samples), str(target), n_variants=0) == 8
    rows = pd.read_csv(target)
    assert rows.perturb_type.eq("original").all() and rows.perturb_id.eq(0).all()
    assert rows.perturbed_text.tolist() == read_samples(samples).input_text.tolist()


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
                        "perturb_id": index, "runtime_sec": 1,
                        "perturbed_text": "request", "model_response": "answer"})
        vectors.append(vector)
    scores, details = score_embeddings_with_details(
        pd.DataFrame(records), np.asarray(vectors), random_repeats=2, lambda_randomness=1)
    row = scores[0]
    assert row["semantic_score"] == pytest.approx(0.2)
    assert row["context_score"] == pytest.approx(0)
    assert row["position_score"] == pytest.approx(1)
    assert row["randomness_baseline"] == pytest.approx(1)
    assert row["cps_cal_score"] == pytest.approx(row["cps_score"] - 1)
    assert row["query_count"] == 9
    assert len(details) == 6
    assert details[0]["distance"] == pytest.approx(0.2)


def test_asr_uses_completed_real_adjudications(tmp_path):
    review = tmp_path / "review.csv"
    pd.DataFrame({"sample_id": ["a", "b", "c"], "attack": ["badnet"] * 3,
                  "attack_success": [1, 0, 1]}).to_csv(review, index=False)
    output = tmp_path / "asr.csv"
    assert compute_asr(str(review), str(output)) == 1
    result = pd.read_csv(output).iloc[0]
    assert result.n_poison == 3 and result.n_success == 2
    assert result.ASR == pytest.approx(2 / 3)


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
    baseline = tmp_path / "random.csv"
    from cps_guard.baselines.random import random_baseline
    random_baseline(scores, baseline, seed=7)
    assert evaluate_scores(scores, baseline, metrics, seed=7) == 9
    result = pd.read_csv(metrics)
    assert result[result.method != "Random"].AUROC.eq(1).all()
    assert result[result.method == "Random"].query_count_total_test.eq(0).all()
    assert result.n_independent_test_prompts.eq(6).all()
