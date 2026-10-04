import json

import pandas as pd
import pytest

from cps_guard.baselines import (import_nete_scores, onion_deletion_score,
                                 prepare_nete, random_baseline)
from cps_guard.data.convert import convert_labeled_data, convert_paired_data
from cps_guard.data import build_alpaca_pilot
from cps_guard.eval.asr import apply_asr_annotations, asr_review_template, compute_asr
from cps_guard.data.schema import read_samples
from cps_guard.methods.perturb import build_variants, semantic_review_template
from cps_guard.eval.plots import plot_results
from cps_guard.eval.study import compare_methods, perturbation_sensitivity


def _samples(tmp_path, n=12):
    source = tmp_path / "alpaca.json"
    source.write_text(json.dumps([
        {"instruction": f"Explain communication channel model number {i} in detail for a graduate student with examples."}
        for i in range(n)
    ]), encoding="utf-8")
    output = tmp_path / "samples.csv"
    build_alpaca_pilot(source, output, n_base=n, seed=3)
    return output


def test_external_conversion_requires_real_pairing_and_trigger(tmp_path):
    clean = tmp_path / "clean.csv"
    poison = tmp_path / "poison.csv"
    pd.DataFrame({"id": ["a", "b"], "prompt": ["Explain antennas.", "Explain filters."]}).to_csv(clean, index=False)
    pd.DataFrame({"id": ["b", "a"], "prompt": ["TRIG Explain filters.", "TRIG Explain antennas."]}).to_csv(poison, index=False)
    output = tmp_path / "converted.csv"
    assert convert_paired_data(clean, poison, output, dataset="d", attack="badnet",
                               clean_column="prompt", poison_column="prompt",
                               trigger="TRIG", trigger_type="word", pair_key="id") == 4
    samples = read_samples(output)
    assert samples.groupby("pair_id").clean_text.nunique().eq(1).all()
    assert samples.attack_success.eq("").all()


def test_single_file_conversion_requires_explicit_pair_labels(tmp_path):
    source = tmp_path / "labeled.csv"
    pd.DataFrame({"pair": ["b", "a", "a", "b"], "label": [1, 0, 1, 0],
                  "text": ["TRIG Explain filters.", "Explain antennas.",
                           "TRIG Explain antennas.", "Explain filters."]}).to_csv(source, index=False)
    target = tmp_path / "unified.csv"
    assert convert_labeled_data(source, target, dataset="official", attack="badnet",
                                text_column="text", label_column="label", pair_key="pair",
                                trigger="TRIG", trigger_type="word") == 4
    assert read_samples(target).pair_id.nunique() == 2


def test_nete_mapping_and_random_are_sample_stable(tmp_path):
    samples_path = _samples(tmp_path)
    samples = read_samples(samples_path)
    output_dir = tmp_path / "nete"
    prepare_nete(str(samples_path), str(output_dir))
    mapping = pd.read_csv(output_dir / "mapping.csv")
    assert mapping.label.iloc[:len(mapping) // 2].eq(1).all()
    official = tmp_path / "official.csv"
    pd.DataFrame({"row_index": mapping.row_index[::-1],
                  "nete_value": mapping.row_index[::-1].astype(float)}).to_csv(official, index=False)
    imported = tmp_path / "nete_scores.csv"
    import_nete_scores(str(output_dir / "mapping.csv"), str(official), str(imported),
                       "nete_value", "lower", "row_index")
    result = pd.read_csv(imported)
    assert result.score.iloc[0] == -mapping.row_index.iloc[0]
    random_one = tmp_path / "random_one.csv"
    random_two = tmp_path / "random_two.csv"
    random_baseline(str(samples_path), str(random_one), seed=9)
    samples.iloc[::-1].to_csv(samples_path, index=False)
    random_baseline(str(samples_path), str(random_two), seed=9)
    a = pd.read_csv(random_one).set_index("sample_id").score
    b = pd.read_csv(random_two).set_index("sample_id").score
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


def test_comparison_uses_grouped_holdout(tmp_path):
    samples_path = _samples(tmp_path)
    samples = read_samples(samples_path)
    cps = samples[["sample_id"]].copy()
    cps["cps_score"] = 0.1 + 0.8 * samples.label
    cps["cps_cal_score"] = 0.05 + 0.8 * samples.label
    cps["runtime_sec"] = 1.0
    cps["query_count"] = 12
    cps_path = tmp_path / "cps.csv"
    cps.to_csv(cps_path, index=False)
    baseline = tmp_path / "random.csv"
    random_baseline(str(samples_path), str(baseline))
    result = tmp_path / "main.csv"
    assert compare_methods(str(samples_path), str(cps_path), [str(baseline)], str(result), seed=7) == 9
    main = pd.read_csv(result)
    assert main.query("method == 'CPS-calibrated'").AUROC.eq(1).all()
    figures = plot_results(str(samples_path), str(cps_path), str(result),
                           str(tmp_path / "figures"), seed=7)
    assert all(pd.io.common.file_exists(path) for path in figures)

def test_sensitivity_uses_measured_n(tmp_path):
    samples_path = _samples(tmp_path)
    samples = read_samples(samples_path)
    details = pd.DataFrame([
        {"sample_id": sid, "perturb_type": kind, "perturb_id": i,
         "distance": 0.2, "runtime_sec": 1.0}
        for sid in samples.sample_id for kind in ("semantic", "context", "position")
        for i in (1, 2)
    ])
    details_path = tmp_path / "details.csv"
    details.to_csv(details_path, index=False)
    cps = samples[["sample_id"]].copy()
    cps["randomness_baseline"] = 0.1
    cps["query_count"] = 12
    cps["runtime_sec"] = 12.0
    cps_path = tmp_path / "cps.csv"
    cps.to_csv(cps_path, index=False)
    label_by_id = samples.set_index("sample_id").label.to_dict()
    extra = details[details.perturb_id == 2].copy()
    extra["perturb_id"] = 3
    details = pd.concat([details, extra], ignore_index=True)
    details["distance"] = details.sample_id.map(lambda sid: 0.1 + 0.8 * label_by_id[sid])
    details.to_csv(details_path, index=False)
    cps["query_count"] = 15
    cps["runtime_sec"] = 15.0
    cps.to_csv(cps_path, index=False)
    output = tmp_path / "sensitivity.csv"
    assert perturbation_sensitivity(str(samples_path), str(details_path), str(cps_path),
                                    str(output), counts=(1, 3), seed=7) == 6
    result = pd.read_csv(output)
    assert result.AUROC.eq(1).all()
    # N=1 保留原始+3 次扰动+5 次随机生成；N=3 保留全部 15 次生成。
    all_attacks = result[result.attack == "ALL"].set_index("method")
    for method, queries in (("N=1", 9), ("N=3", 15)):
        row = all_attacks.loc[method]
        assert row.query_count_total_test == queries * row.n_test
        assert row.runtime_sec_total_test == queries * row.n_test


def test_ten_real_variants_require_review_and_keep_trigger(tmp_path):
    samples_path = _samples(tmp_path, 1)
    review_path = tmp_path / "semantic.csv"
    semantic_review_template(str(samples_path), str(review_path), n_variants=10)
    review = pd.read_csv(review_path, keep_default_na=False)
    review["perturbed_text"] = review.apply(
        lambda r: f"Version {r.perturb_id}. " + r.input_text, axis=1)
    review["reviewed"] = 1
    review.to_csv(review_path, index=False)
    variants_path = tmp_path / "variants.csv"
    assert build_variants(str(samples_path), str(review_path), str(variants_path),
                          n_variants=10) == 4 * 31
    variants = pd.read_csv(variants_path, keep_default_na=False)
    for sample in read_samples(samples_path).itertuples(index=False):
        subset = variants[variants.sample_id == sample.sample_id]
        assert len(subset) == 31
        if sample.trigger:
            assert all(text.count(sample.trigger) == sample.input_text.count(sample.trigger)
                       for text in subset.perturbed_text)


def test_onion_score_counts_all_deletions_by_default():
    calls = []
    def fake_nll(text):
        calls.append(text)
        return 10.0 if "TRIG" in text else 1.0
    score, queries = onion_deletion_score("a TRIG b c", fake_nll)
    assert score == 9.0 and queries == 5 and len(calls) == 5
