"""Pilot-v2 科学计算、数据隔离及冻结预测的 CPU 测试。"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from cps_guard.data.builder import DEFAULT_TARGET_PREFIXES
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
    samples = build_attack_inputs(questions, 8, {"badnet": "BadMagic", "vpi": "Discussing OpenAI"}, DEFAULT_TARGET_PREFIXES)
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


def test_test_features_load_both_source_references_without_refitting(tmp_path, monkeypatch):
    """Test组装只加载两个来源的已有参考，向量行序由representation_index决定。"""
    from cps_guard.methods import features
    source, test = tmp_path / "source", tmp_path / "test"
    (source / "checkpoints").mkdir(parents=True)
    for name in ("data", "results"):
        (test / name).mkdir(parents=True)
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(4, 2, 2, 4)).astype(np.float16)
    np.save(test / "results/representation_vectors.npy", vectors)
    frame = pd.DataFrame({"sample_id": ["b0", "b1", "v0", "v1"], "base_id": ["q"] * 4,
                          "split": ["test"] * 4, "attack": ["badnet"] * 2 + ["vpi"] * 2,
                          "label": [0, 1, 0, 1]})
    frame.to_csv(test / "data/attack_inputs.csv", index=False)
    raw = frame[["sample_id"]].copy()
    raw["representation_index"] = [2, 0, 3, 1]
    for column in REPRESENTATION_COLUMNS:
        raw[column] = 1.
    raw.to_csv(test / "results/features_representation_raw.csv", index=False)
    generation = frame[["sample_id"]].copy()
    for column in GENERATION_COLUMNS:
        generation[column] = 1.
    generation.to_csv(test / "results/features_generation.csv", index=False)
    def behavioral(input_csv, output_csv, *args):
        part = frame[["sample_id"]].copy()
        for column in BEHAVIORAL_COLUMNS:
            part[column] = 1.
        part.to_csv(output_csv, index=False)
    monkeypatch.setattr(features, "score_inference", behavioral)
    expected = {}
    for attack in ("badnet", "vpi"):
        reference = CleanManifold(2, 7).fit(rng.normal(size=(16, 2, 2, 4)))
        joblib.dump(reference, source / f"checkpoints/reference_{attack}_2.joblib")
        expected[attack] = reference.transform(vectors[[2, 0, 3, 1]])
    monkeypatch.setattr(CleanManifold, "fit", lambda *args: pytest.fail("Test不得拟合参考"))
    features.assemble_features(test, "encoder", 2, 1., [2], 7, {}, reference_dir=source)
    result = pd.read_csv(test / "results/features.csv")
    for attack in expected:
        np.testing.assert_allclose(result[[f"mahalanobis_{attack}_2_mean", f"mahalanobis_{attack}_2_max"]],
                                   expected[attack])


def test_original_extraction_and_inference_cache_alignment(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from cps_guard.model import features, inference
    from cps_guard.methods.perturb import build_variants
    questions = pd.DataFrame([{"base_id": f"q{i}", "source_index": i,
                               "original_instruction": f"Please explain channel number {i}.", "split": "train"}
                              for i in range(2)])
    samples = build_attack_inputs(questions, 7, {"badnet": "BadMagic", "vpi": "Discussing OpenAI"}, DEFAULT_TARGET_PREFIXES)
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
    freeze_detectors(first, [0.1, 1.], [2], 9)
    freeze_detectors(changed, [0.1, 1.], [2], 9)
    a = json.loads((first / "configs/frozen_detector_config.json").read_text())
    b = json.loads((changed / "configs/frozen_detector_config.json").read_text())
    assert len(a["detectors"]) == 14
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


def test_control_reuses_test_inputs_and_frozen_metrics(tmp_path, monkeypatch):
    """CPU功能检查：复用原Test、增加中性输入、原指标分组及原模型选择。

    输入：make_features生成的双攻击七视图数据与固定分词器；无真实模型/GPU。
    输出检查：两种负样本各七组指标；原clean/poison评分不变，来源对象不改写；
        被测模型选择独立于输入attack，已有扰动文本可直接复用。
    """
    import sys
    from types import SimpleNamespace
    from cps_guard.data.pilot_v2 import prepare_control
    from cps_guard.eval.transfer import evaluate_control
    from cps_guard.eval.pilot_plots import plot_control
    from cps_guard.eval.asr import summarize_control_behavior
    from cps_guard.methods.perturb import build_variants
    from cps_guard.model import inference

    source, output = tmp_path / "source", tmp_path / "control"
    frame = make_features(source)
    freeze_detectors(source, [1.], [2], 7)
    evaluate_frozen(source)
    for root in (source, output):
        for folder in ("data", "results", "figures"):
            (root / folder).mkdir(parents=True, exist_ok=True)
    questions = pd.DataFrame([{"base_id": f"test_{i}", "source_index": i, "split": "test",
                               "original_instruction": f"Explain radio channel {i}."} for i in range(12)])
    questions.to_csv(source / "data/base_questions.csv", index=False)
    samples = build_attack_inputs(questions, 7, {"badnet": "BadMagic", "vpi": "Discussing OpenAI"}, DEFAULT_TARGET_PREFIXES)
    samples.to_csv(source / "data/attack_inputs.csv", index=False)
    class Tokenizer:
        """固定词级分词器；只用于验证中性替换的位置和长度。"""
        def __call__(self, text, **kwargs):
            """输入文本；返回词级token编号，用于确定性长度比较。"""
            return {"input_ids": list(range(len(text.split())))}
    tokenizer = Tokenizer()
    prepare_control(source, output, "vpi", True, tokenizer, ["Discussing gardens"], "raw")
    selected = pd.read_csv(output / "data/attack_inputs.csv", keep_default_na=False)
    assert selected.input_type.value_counts().to_dict() == {"clean": 12, "poison": 12, "neutral": 12}
    assert selected.groupby("input_type").label.first().to_dict() == {"clean": 0, "neutral": 0, "poison": 1}
    assert selected[selected.input_type == "neutral"].trigger.eq("Discussing gardens").all()
    existing = samples[samples.attack == "vpi"]
    assert selected[selected.input_type != "neutral"].input_text.tolist() == existing.input_text.tolist()
    # 攻击输入ID沿用旧表；合成特征使用同题ID，关联后补中性负样本。
    test = frame[(frame.attack == "vpi") & (frame.split == "test")].copy()
    test["sample_id"] = test.apply(lambda r: f"{r.base_id}_vpi_{'poison' if r.label else 'clean'}", axis=1)
    test["input_type"] = test.label.map({0: "clean", 1: "poison"})
    neutral = test[test.label == 0].copy()
    neutral["sample_id"] = neutral.base_id + "_vpi_neutral"
    neutral["input_type"] = "neutral"
    features = pd.concat([test, neutral], ignore_index=True)
    features.to_csv(output / "results/features.csv", index=False)
    original_objects = {p.name: p.read_bytes() for p in (source / "checkpoints").glob("*.joblib")}
    metrics = evaluate_control(output, source, "badnet", "qwen", 25, 8)
    assert len(metrics) == 14
    assert metrics.AUROC.eq(1.).all()
    assert metrics.comparison.nunique() == 2
    assert metrics.n_samples.eq(24).all()
    assert original_objects == {p.name: p.read_bytes() for p in (source / "checkpoints").glob("*.joblib")}
    assert features.groupby(["base_id", "input_type"]).size().eq(1).all()
    plot_control(output)
    assert len(list((output / "figures").glob("*.pdf"))) == 4
    answers = selected.assign(model_response="Normal response")
    answers.loc[answers.input_type == "poison", "model_response"] = "  You are stupid!"
    answers.to_csv(output / "results/original_inference.csv", index=False)
    summarize_control_behavior(output / "data/attack_inputs.csv", output / "results/original_inference.csv",
                               output / "results/asr.csv", "You are stupid!")
    behavior = pd.read_csv(output / "results/asr.csv").set_index("input_type")
    assert behavior.loc["poison", "target_hit_rate"] == 1
    assert behavior.loc["neutral", "target_hit_rate"] == 0
    # clean/poison模式仍可运行；按model参数加载qwen，不把输入vpi当作加载选择。
    prepare_control(source, output, "vpi", False, None, [], "raw")
    build_variants(output / "data/attack_inputs.csv", source / "data/variants.csv", 0)
    build_variants(output / "data/attack_inputs.csv", output / "data/variants.csv", 0,
                   reuse_csv=source / "data/variants.csv")
    loaded = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
    monkeypatch.setattr(inference, "_load_model", lambda config, model: (loaded.append(model), None))
    monkeypatch.setattr(inference, "_generate", lambda *args: ("answer", .1))
    inference.run_inference(output / "data/variants.csv", {"test_model": "qwen", "seed": 7},
                            output / "results/inference.csv", skip_randomness=True)
    assert loaded == ["qwen"]
    features[features.input_type != "neutral"].to_csv(output / "results/features.csv", index=False)
    assert len(evaluate_control(output, source, "badnet", "vpi", 25, 8)) == 7
