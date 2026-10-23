"""R单视图的来源隔离、固定预测、配对统计和中性对照检查。"""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from cps_guard.data.builder import DEFAULT_TARGET_PREFIXES
from cps_guard.data.pilot_v2 import select_questions, build_attack_inputs, prepare_control
from cps_guard.methods.representation import CleanManifold
from cps_guard.model.features import REPRESENTATION_COLUMNS
from cps_guard.eval.transfer import freeze_detectors, evaluate_frozen, evaluate_control, feature_columns
from cps_guard.eval.statistics import grouped_auc, summary_statistics, write_final_report
from cps_guard.eval.pilot_plots import plot_pilot, plot_control


def make_features(output, test_shift=0.):
    """生成两攻击/三组的合成R数据，用于CPU验证拟合隔离和测试指标。"""
    for name in ("data", "results", "configs", "checkpoints", "figures"):
        (output / name).mkdir(parents=True, exist_ok=True)
    rng, rows = np.random.default_rng(11), []
    for attack in ("badnet", "vpi"):
        for split, count in (("train", 16), ("validation", 8), ("test", 12)):
            for index in range(count):
                for label in (0, 1):
                    columns = set(feature_columns("badnet", 2) + feature_columns("vpi", 2))
                    row = {col: 2 * label + rng.normal(0, .2) + (test_shift if split == "test" else 0.) for col in sorted(columns)}
                    row.update(sample_id=f"{split}_{index}_{attack}_{label}", base_id=f"{split}_{index}",
                               attack=attack, split=split, label=label, input_type="poison" if label else "clean")
                    rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "results/features.csv", index=False)
    pd.DataFrame([{"attack": name, "ASR": 1., "clean_target_rate": 0.} for name in ("badnet", "vpi")]).to_csv(output / "results/asr.csv", index=False)
    return frame


def test_selection_excludes_reference_exact_and_semantic_duplicates():
    texts = [f"Please explain communication scenario number {i} in full detail." for i in range(15)]
    source = [{"instruction": text, "output": "Normal answer."} for text in texts]
    source[3]["instruction"] = texts[0].upper()
    class Encoder:
        def encode(self, values, **kwargs):
            indices = [texts.index(value) if value in texts else 0 for value in values]
            return np.array([np.eye(15)[0 if i == 2 else i] for i in indices])
    refs = [{"source_index": i, "input_text": texts[i], "reference_set": "training"} for i in (0, 1)]
    questions, audit = select_questions(source, refs, Encoder(), (6, 3, 2), 8, .95)
    assert not set(questions.source_index) & {0, 1, 2, 3}
    assert questions.groupby("split").size().to_dict() == {"train": 6, "validation": 3, "test": 2}
    assert "excluded_exact" in set(audit.decision)


def test_source_fitting_and_fixed_test_predictions(tmp_path, monkeypatch):
    first, changed = tmp_path / "first", tmp_path / "changed"
    frame = make_features(first)
    make_features(changed, 100.)
    for folder in (first, changed):
        freeze_detectors(folder, [.1, 1.], [2], 9)
    a = json.loads((first / "configs/frozen_detector_config.json").read_text())
    b = json.loads((changed / "configs/frozen_detector_config.json").read_text())
    assert len(a["detectors"]) == 2
    for item, other in zip(a["detectors"], b["detectors"]):
        assert item["view"] == "R" and len(item["columns"]) == 7
        assert (item["C"], item["threshold"]) == (other["C"], other["threshold"])
        detector = joblib.load(first / "checkpoints" / item["checkpoint"])
        train = frame[(frame.attack == item["source_attack"]) & (frame.split == "train")]
        np.testing.assert_allclose(detector[0].mean_, train[item["columns"]].mean())
    evaluate_frozen(first)
    saved = (first / "results/test_predictions.csv").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(joblib, "load", lambda *args: pytest.fail("重复统计不应加载检测器"))
        evaluate_frozen(first)
    assert saved == (first / "results/test_predictions.csv").read_bytes()
    summary_statistics(first, 50, 101, 3, 1001)
    intervals = pd.read_csv(first / "results/bootstrap_results.csv")
    assert len(intervals) == 4 and intervals.AUROC.eq(1.).all()
    assert not (first / "results/ablation_results.csv").exists()
    assert write_final_report(first).exists()
    plot_pilot(first)
    assert len(list((first / "figures").glob("*.pdf"))) == 4


def test_test_reference_transform_never_refits_and_aligns_vector_indices(tmp_path, monkeypatch):
    from cps_guard.methods.features import assemble_features
    source, test = tmp_path / "source", tmp_path / "test"
    (source / "checkpoints").mkdir(parents=True)
    for name in ("data", "results"):
        (test / name).mkdir(parents=True)
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(4, 2, 2, 4)).astype(np.float16)
    np.save(test / "results/representation_vectors.npy", vectors)
    frame = pd.DataFrame({"sample_id": ["b0", "b1", "v0", "v1"], "base_id": ["q"] * 4,
                          "split": ["test"] * 4, "attack": ["badnet"] * 2 + ["vpi"] * 2,
                          "label": [0, 1, 0, 1], "input_type": ["clean", "poison"] * 2})
    frame.to_csv(test / "data/attack_inputs.csv", index=False)
    raw = frame.drop(columns="input_type").copy()
    raw["representation_index"] = [2, 0, 3, 1]
    for column in REPRESENTATION_COLUMNS:
        raw[column] = 1.
    raw.iloc[::-1].to_csv(test / "results/features_representation_raw.csv", index=False)
    expected = {}
    for attack in ("badnet", "vpi"):
        reference = CleanManifold(2, 7).fit(rng.normal(size=(16, 2, 2, 4)))
        joblib.dump(reference, source / f"checkpoints/reference_{attack}_2.joblib")
        expected[attack] = reference.transform(vectors[[2, 0, 3, 1]])
    monkeypatch.setattr(CleanManifold, "fit", lambda *args: pytest.fail("Test不得拟合参考"))
    assemble_features(test, [2], 7, reference_dir=source)
    result = pd.read_csv(test / "results/features.csv")
    assert result.sample_id.tolist() == frame.sample_id.tolist()
    for attack in expected:
        np.testing.assert_allclose(result[[f"mahalanobis_{attack}_2_mean", f"mahalanobis_{attack}_2_max"]], expected[attack])


def test_grouped_auc_matches_explicit_resampling():
    group = pd.DataFrame({"base_id": ["a", "a", "b", "b", "c", "c"], "label": [0, 1] * 3,
                          "score": [.1, .5, .7, .7, .8, .9]})
    weights = np.array([[1, 1, 1], [2, 0, 1], [0, 3, 0]])
    actual = grouped_auc(group, weights, ["a", "b", "c"])
    for i, counts in enumerate(weights):
        selected = pd.concat([group[group.base_id == base] for base, count in zip(["a", "b", "c"], counts) for _ in range(count)])
        assert actual[i] == pytest.approx(roc_auc_score(selected.label, selected.score))


def test_neutral_inputs_use_same_question_position_and_frozen_detector(tmp_path):
    source, output = tmp_path / "source", tmp_path / "control"
    features = make_features(source)
    make_features(output)
    freeze_detectors(source, [1.], [2], 7)
    questions = pd.DataFrame([{"base_id": f"test_{i}", "source_index": i, "split": "test",
                               "original_instruction": f"Explain radio channel {i}."} for i in range(12)])
    questions.to_csv(source / "data/base_questions.csv", index=False)
    samples = build_attack_inputs(questions, 7, {"vpi": "Discussing OpenAI"}, DEFAULT_TARGET_PREFIXES)
    samples.to_csv(source / "data/attack_inputs.csv", index=False)
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(text.split())))}
    prepare_control(source, output, "vpi", True, Tokenizer(), ["Discussing gardens"], "raw")
    selected = pd.read_csv(output / "data/attack_inputs.csv", keep_default_na=False)
    assert selected.input_type.value_counts().to_dict() == {"clean": 12, "poison": 12, "neutral": 12}
    original = features[(features.attack == "vpi") & (features.split == "test")].copy()
    original["sample_id"] = [f"{r.base_id}_vpi_{'poison' if r.label else 'clean'}" for r in original.itertuples()]
    neutral = original[original.label == 0].copy()
    neutral["sample_id"] = neutral.base_id + "_vpi_neutral"
    neutral["input_type"] = "neutral"
    pd.concat([original, neutral]).to_csv(output / "results/features.csv", index=False)
    checkpoints = {p.name: p.read_bytes() for p in (source / "checkpoints").glob("*.joblib")}
    metrics = evaluate_control(output, source, "badnet", "qwen", 25, 8)
    assert len(metrics) == 2 and metrics.AUROC.eq(1.).all()
    assert checkpoints == {p.name: p.read_bytes() for p in (source / "checkpoints").glob("*.joblib")}
    plot_control(output)
    assert len(list((output / "figures").glob("*.pdf"))) == 3
