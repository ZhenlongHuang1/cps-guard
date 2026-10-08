"""Pilot-v2 科学计算、数据隔离及冻结预测的 CPU 测试。"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from cps_guard.data.pilot_v2 import build_attack_inputs, select_questions
from cps_guard.model.features import (GENERATION_COLUMNS, REPRESENTATION_COLUMNS,
                                      generation_statistics, pool_hidden_states)
from cps_guard.methods.features import BEHAVIORAL_COLUMNS
from cps_guard.methods.representation import CleanManifold
from cps_guard.eval.transfer import evaluate_frozen, freeze_detectors, feature_columns
from cps_guard.eval.statistics import grouped_auc, summary_statistics, write_final_report
from cps_guard.eval.pilot_plots import plot_pilot


def test_selection_excludes_training_old_exact_and_semantic_duplicates():
    texts = [f"Please explain communication scenario number {i} in full detail." for i in range(15)]
    source = [{"instruction": text, "output": "A useful reference answer."} for text in texts]
    source[3]["instruction"] = texts[0].upper()  # 不同来源行也要进行文本去重。
    class Encoder:
        def encode(self, values, **kwargs):
            vectors = []
            for value in values:
                index = texts.index(value) if value in texts else 0
                vectors.append(np.eye(15)[0 if index == 2 else index])
            return np.array(vectors)
    refs = [{"source_index": index, "input_text": texts[index], "reference_set": name}
            for index, name in ((0, "lora_training"), (1, "pilot_v1"))]
    questions, audit = select_questions(source, refs, Encoder(), (4, 2, 2), 8, 0.95)
    assert len(questions) == 8
    assert not set(questions.source_index) & {0, 1, 2, 3}
    assert questions.groupby("split").size().to_dict() == {"train": 4, "validation": 2, "test": 2}
    checked = audit.set_index("pilot_sample_id")
    if "alpaca_00002" in checked.index:
        assert checked.loc["alpaca_00002", "decision"] == "excluded_similarity_pending_review"
    samples = build_attack_inputs(questions, 8, {"badnet": "BadMagic", "vpi": "Discussing OpenAI"}, "You are stupid!")
    assert len(samples) == 32
    assert samples.groupby("base_id").split.nunique().eq(1).all()
    assert samples.groupby("base_id").size().eq(4).all()
    assert samples[samples.label == 1].apply(lambda row: row.trigger in row.input_text, axis=1).all()
    assert samples[samples.label == 0].trigger.eq("").all()


def test_generation_entropy_and_hidden_pooling_have_known_values():
    torch = pytest.importorskip("torch")
    stats = generation_statistics((torch.tensor([[0., 0.]]), torch.tensor([[0., -float("inf")]])))
    assert stats["entropy_mean"] == pytest.approx(np.log(2) / 2)
    assert stats["top1_mean"] == pytest.approx(0.75)
    assert stats["margin_mean"] == pytest.approx(0.5)
    assert stats["response_length"] == 2
    hidden = (torch.zeros(1, 3, 2), torch.tensor([[[1., 0.], [3., 0.], [999., 999.]]]),
              torch.tensor([[[0., 1.], [0., 3.], [999., 999.]]]),
              torch.tensor([[[-1., 0.], [-3., 0.], [999., 999.]]]))
    stats, vectors = pool_hidden_states(hidden, torch.tensor([[1, 1, 0]]), [0, 2])
    np.testing.assert_array_equal(vectors, [[[2., 0.], [3., 0.]], [[-2., 0.], [-3., 0.]]])
    assert stats["hidden_norm_mean"] == pytest.approx(2.5)
    assert stats["layer_shift_mean"] == pytest.approx(1.)
    assert stats["layer_shift_std"] == pytest.approx(0.)


def test_clean_manifold_does_not_fit_target_or_test():
    train = np.random.default_rng(2).normal(size=(24, 2, 2, 5))
    reference = CleanManifold(3, 7).fit(train)
    means = [pca.mean_.copy() for pca, _ in reference.models]
    extreme_test = train[:3] + 100
    distance = reference.transform(extreme_test)
    assert distance.shape == (3, 2) and np.isfinite(distance).all()
    assert (distance[:, 1] >= distance[:, 0]).all()
    assert distance.mean() > reference.transform(train).mean()
    for old, (pca, _) in zip(means, reference.models):
        np.testing.assert_array_equal(pca.mean_, old)


def test_original_extraction_and_inference_cache_alignment(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from cps_guard.model import features, inference
    from cps_guard.methods.perturb import build_variants
    questions = pd.DataFrame([{"base_id": f"q{i}", "source_index": i,
                               "original_instruction": f"Please explain channel number {i}.", "split": "train"}
                              for i in range(2)])
    samples = build_attack_inputs(questions, 7, {"badnet": "BadMagic", "vpi": "Discussing OpenAI"}, "You are stupid!")
    for name in ("results", "configs", "data"):
        (tmp_path / name).mkdir()
    samples.to_csv(tmp_path / "data/samples.csv", index=False)
    class Batch(dict):
        def to(self, device):
            return self
    class Tokenizer:
        eos_token_id = 0
        def apply_chat_template(self, messages, **kwargs):
            return "USER: " + messages[0]["content"] + "\nASSISTANT:"
        def __call__(self, prompt, **kwargs):
            assert kwargs["add_special_tokens"] is False
            return Batch(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.tensor([[1, 1, 1]]))
        def decode(self, tokens, **kwargs):
            return "cached response"
    class GenerationConfig:
        def save_pretrained(self, path):
            path.mkdir()
            (path / "generation_config.json").write_text("{}")
    class Model:
        device = "cpu"
        generation_config = GenerationConfig()
        def generate(self, input_ids, **kwargs):
            return SimpleNamespace(sequences=torch.cat([input_ids, torch.tensor([[9]])], dim=1),
                                   scores=(torch.tensor([[1., 0.]]),))
        def __call__(self, input_ids, **kwargs):
            return SimpleNamespace(hidden_states=tuple(torch.ones(1, 3, 4) * i for i in range(4)))
    monkeypatch.setattr(features, "_load_model", lambda config, attack: (Tokenizer(), Model()))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    config = {"max_new_tokens": 128, "prompt_format": "chat_template", "seed": 7, "random_repeats": 2}
    features.extract_original_features(tmp_path / "data/samples.csv", tmp_path, config, [0, 2], 7)
    raw = pd.read_csv(tmp_path / "results/features_representation_raw.csv")
    assert raw.representation_index.tolist() == list(range(8))
    vectors = np.load(tmp_path / "results/representation_vectors.npy")
    assert vectors.shape == (8, 2, 2, 4)
    np.testing.assert_array_equal(vectors[raw.representation_index, 1, 1], 3 * np.ones((8, 4)))
    variants = tmp_path / "data/variants.csv"
    build_variants(tmp_path / "data/samples.csv", variants, n_variants=0)
    calls = []
    def generate(tokenizer, model, text, config, sample, seed):
        assert sample  # 固定original必须使用缓存，仅随机回答重新生成。
        calls.append(text)
        return "new random response", 0.1
    monkeypatch.setattr(inference, "_load_model", lambda config, attack: (Tokenizer(), Model()))
    monkeypatch.setattr(inference, "_generate", generate)
    inference.run_inference(variants, config, tmp_path / "results/inference.csv",
                            originals_csv=tmp_path / "results/original_inference.csv")
    frame = pd.read_csv(tmp_path / "results/inference.csv")
    assert len(calls) == 16 and len(frame) == 24
    assert frame[frame.perturb_type == "original"].model_response.eq("cached response").all()


def make_features(output, test_shift=0.):
    """构造分组、双攻击和七视图合成特征，验证训练/验证/Test的隔离。"""
    for name in ("results", "configs", "checkpoints", "figures"):
        (output / name).mkdir(parents=True)
    rng, rows = np.random.default_rng(11), []
    for attack in ("badnet", "vpi"):
        for split, count in (("train", 16), ("validation", 8), ("test", 12)):
            for index in range(count):
                for label in (0, 1):
                    columns = feature_columns("B+G+R", "badnet", 2) + feature_columns("R", "vpi", 2)
                    row = {column: 2 * label + rng.normal(0, 0.3) + (0.4 if attack == "vpi" else 0.)
                           for column in dict.fromkeys(columns)}
                    if split == "test":
                        row = {key: value + test_shift for key, value in row.items()}
                    row.update(base_id=f"{split}_{index}", sample_id=f"{split}_{index}_{attack}_{label}",
                               attack=attack, split=split, label=label)
                    rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "results/features.csv", index=False)
    pd.DataFrame([{"attack": attack, "ASR": 1., "clean_target_rate": 0.}
                  for attack in ("badnet", "vpi")]).to_csv(output / "results/asr.csv", index=False)
    return frame


def test_freeze_transfer_and_statistics_reuse_fixed_predictions(tmp_path, monkeypatch):
    first, changed = tmp_path / "first", tmp_path / "changed"
    frame = make_features(first)
    make_features(changed, test_shift=100.)
    freeze_detectors(first, [0.1, 1.], [2], 9, 0.60)
    freeze_detectors(changed, [0.1, 1.], [2], 9, 0.60)
    a = json.loads((first / "configs/frozen_detector_config.json").read_text())
    b = json.loads((changed / "configs/frozen_detector_config.json").read_text())
    assert len(a["detectors"]) == 14 and a["gate2_passed"]
    for detector, other in zip(a["detectors"], b["detectors"]):
        assert (detector["C"], detector["threshold"], detector["validation_AUROC"]) == (
            other["C"], other["threshold"], other["validation_AUROC"])
        source = detector["source_attack"]
        assert not any(f"mahalanobis_{'vpi' if source == 'badnet' else 'badnet'}" in col
                       for col in detector["columns"])
        model = joblib.load(first / "checkpoints" / detector["checkpoint"])
        train = frame[(frame.attack == source) & (frame.split == "train")]
        np.testing.assert_allclose(model[0].mean_, train[detector["columns"]].mean())
    evaluate_frozen(first)
    original = (first / "results/test_predictions.csv").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(joblib, "load", lambda *args: pytest.fail("重复统计不得重新调用检测器"))
        evaluate_frozen(first)
    assert (first / "results/test_predictions.csv").read_bytes() == original
    with pytest.raises(ValueError, match="Test 已执行"):
        freeze_detectors(first, [1.], [2], 9, 0.60)
    summary_statistics(first, 100, 101, 50, 1001)
    intervals = pd.read_csv(first / "results/bootstrap_results.csv")
    assert len(intervals) == 28
    assert intervals.AUROC.eq(1.).all() and intervals.CI_low.eq(1.).all()
    assert len(pd.read_csv(first / "results/random_seed_results.csv")) == 100
    assert len(pd.read_csv(first / "results/ablation_results.csv")) == 12
    assert write_final_report(first) == "GO"
    plot_pilot(first)
    assert len(list((first / "figures").glob("*.pdf"))) == 5


def test_grouped_auc_equals_explicit_paired_resampling():
    group = pd.DataFrame({"base_id": ["a", "a", "b", "b", "c", "c"],
                          "label": [0, 1, 0, 1, 0, 1], "score": [0.1, 0.5, 0.7, 0.7, 0.8, 0.9]})
    weights = np.array([[1, 1, 1], [2, 0, 1], [0, 3, 0]])
    actual = grouped_auc(group, weights, ["a", "b", "c"])
    for index, counts in enumerate(weights):
        sampled = pd.concat([group[group.base_id == base_id] for base_id, count in zip(["a", "b", "c"], counts)
                             for _ in range(count)])
        assert actual[index] == pytest.approx(roc_auc_score(sampled.label, sampled.score))
