import json
from pathlib import Path
import pandas as pd
from cps_guard.baselines.random import random_baseline
from cps_guard.data import build_alpaca_pilot
from cps_guard.data.schema import read_samples
from cps_guard.eval.asr import apply_asr_annotations, asr_review_template, compute_asr
from cps_guard.eval.detection import evaluate_scores
from cps_guard.eval.plots import plot_results


def _samples(tmp_path, n=12):
    source = tmp_path / "alpaca.json"
    source.write_text(json.dumps([
        {"instruction": f"Explain communication channel model number {i} in detail for a graduate student with examples."}
        for i in range(n)
    ]), encoding="utf-8")
    output = tmp_path / "samples.csv"
    build_alpaca_pilot(source, output, n_base=n, seed=3)
    return output


def test_random_is_stable_under_row_reordering(tmp_path):
    samples_path = _samples(tmp_path)
    first, second = tmp_path / "random_one.csv", tmp_path / "random_two.csv"
    random_baseline(samples_path, first, seed=9)
    read_samples(samples_path).iloc[::-1].to_csv(samples_path, index=False)
    random_baseline(samples_path, second, seed=9)
    a = pd.read_csv(first).set_index("sample_id").score
    b = pd.read_csv(second).set_index("sample_id").score
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


def test_asr_review_contains_both_responses_and_can_apply(tmp_path):
    samples_path = _samples(tmp_path, 2)
    samples = read_samples(samples_path)
    original = samples[["sample_id"]].copy()
    original["perturb_type"] = "original"
    original["model_response"] = original.sample_id.map(lambda sid: f"Response to {sid}")
    inference = tmp_path / "inference.csv"
    original.to_csv(inference, index=False)
    review = tmp_path / "review.csv"
    asr_review_template(str(samples_path), str(inference), str(review))
    frame = pd.read_csv(review, keep_default_na=False)
    assert len(frame) == 4 and frame.clean_model_response.str.startswith("Response").all()
    frame["attack_success"] = frame.groupby("attack").cumcount()
    frame["clean_target_behavior"] = 0
    frame.to_csv(review, index=False)
    asr = tmp_path / "asr.csv"
    compute_asr(str(review), str(asr))
    assert pd.read_csv(asr).ASR.eq(0.5).all()
    annotated = tmp_path / "annotated.csv"
    apply_asr_annotations(str(samples_path), str(review), str(annotated))
    assert read_samples(annotated).query("label == 1").attack_success.isin(["0", "1"]).all()


def test_comparison_and_plots_share_holdout(tmp_path):
    samples_path = _samples(tmp_path)
    samples = read_samples(samples_path)
    cps = samples[["sample_id", "base_id", "attack", "label"]].copy()
    cps["cps_score"] = 0.1 + 0.8 * samples.label
    cps["cps_cal_score"] = 0.05 + 0.8 * samples.label
    cps["runtime_sec"] = 1.0
    cps["query_count"] = 12
    cps_path = tmp_path / "cps.csv"
    cps.to_csv(cps_path, index=False)
    baseline = tmp_path / "random.csv"
    random_baseline(samples_path, baseline)
    metrics = tmp_path / "main.csv"
    assert evaluate_scores(cps_path, baseline, metrics, seed=7) == 9
    figures = plot_results(cps_path, baseline, metrics, tmp_path / "figures", seed=7)
    assert len(figures) == 2
    assert all(Path(path).stat().st_size > 0 for path in figures)
