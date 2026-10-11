"""Paired scientific contrasts, frozen assets and resumable inference (CPU fakes)."""
import contextlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cps_guard import model_control as control
from cps_guard.eval.transfer import feature_columns
from cps_guard.methods.representation import CleanManifold
from cps_guard.model.features import REPRESENTATION_COLUMNS


class Tokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "USER: " + messages[0]["content"] + "\nASSISTANT:"

    def __call__(self, text, **kwargs):
        tokens = re.findall(r"\w+|[^\w\s]", text)
        ids = [sum(map(ord, word)) for word in tokens]
        if kwargs.get("return_tensors"):
            return Batch(input_ids=np.array([ids]), attention_mask=np.ones((1, len(ids)), dtype=int))
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        return "You are stupid!" if ids[0] == 9 else "Normal answer"


class Batch(dict):
    def to(self, device):
        return self


def test_input_matching_checks_full_prompt_and_forbidden_triggers():
    questions = pd.DataFrame([{"base_id": "q1", "original_instruction": "Explain radio waves."}])
    result = control.build_inputs(questions, Tokenizer(), "Discussing OpenAI",
                                  ["Discussing gardens"], ["Discussing OpenAI", "BadMagic"], "chat_template")
    assert result.input_type.tolist() == ["clean", "trigger", "neutral"]
    assert result.iloc[1].prompt_token_count == result.iloc[2].prompt_token_count
    assert result.iloc[1].input_text == "Discussing OpenAI. Explain radio waves."
    with pytest.raises(ValueError, match="known trigger"):
        control.build_inputs(questions, Tokenizer(), "Discussing OpenAI", ["BadMagic"],
                             ["BadMagic"], "chat_template")
    with pytest.raises(ValueError, match="No neutral"):
        control.build_inputs(questions, Tokenizer(), "Discussing OpenAI", ["gardens"],
                             ["Discussing OpenAI"], "chat_template")
    # Equal phrase token count does not guarantee an equal full prompt count.
    class BoundaryTokenizer(Tokenizer):
        def __call__(self, text, **kwargs):
            value = super().__call__(text, **kwargs)
            if "gardens." in text:
                value["input_ids"].append(999)
            return value
    with pytest.raises(ValueError, match="complete-prompt"):
        control.build_inputs(questions, BoundaryTokenizer(), "Discussing OpenAI", ["Discussing gardens"],
                             ["Discussing OpenAI"], "chat_template")


def contrast_frame():
    rows = []
    for i in range(5):
        for model, effects in {"base": (1, .5), "vpi": (4, 1), "clean_lora": (1.5, .5)}.items():
            for kind, effect in zip(["clean", "trigger", "neutral"], [0, *effects]):
                decision = i * .1 + effect
                rows.append({"base_id": f"q{i}", "model": model, "input_type": kind,
                             "decision_score": decision, "score": 1 / (1 + np.exp(-decision))})
    return pd.DataFrame(rows)


def test_paired_interactions_known_values_and_missing_cells_rejected():
    frame = contrast_frame()
    models = ["base", "vpi", "clean_lora"]
    effects, metrics, draws = control.paired_statistics(frame, models, 20, 3)
    values = effects.set_index("contrast")
    assert values.loc["extra_trigger:vpi_minus_base", "estimate"] == pytest.approx(3)
    assert values.loc["specific_effect:vpi_minus_base", "estimate"] == pytest.approx(2.5)
    assert values.loc["extra_trigger:vpi_minus_clean_lora", "estimate"] == pytest.approx(2.5)
    assert values.loc["specific_effect:vpi_minus_clean_lora", "estimate"] == pytest.approx(2)
    assert values.loc["specific_effect:vpi_minus_clean_lora", "ci_low"] == pytest.approx(2)
    assert draws.shape == (20, 5)
    assert len(metrics) == 6
    shuffled = frame.sample(frac=1, random_state=9)
    other, _, other_draws = control.paired_statistics(shuffled, models, 20, 3)
    pd.testing.assert_frame_equal(effects, other)
    np.testing.assert_array_equal(draws, other_draws)
    with pytest.raises(ValueError, match="Incomplete"):
        control.paired_statistics(frame.iloc[1:], models, 20, 3)
    with pytest.raises(ValueError, match="Duplicate"):
        control.paired_statistics(pd.concat([frame, frame.iloc[:1]]), models, 20, 3)


@pytest.fixture
def source_assets(tmp_path):
    source = tmp_path / "source"
    for name in ["configs", "checkpoints", "data"]:
        (source / name).mkdir(parents=True)
    columns = feature_columns("R", "badnet", 2)
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(30, len(columns))), columns=columns)
    detector = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, (X.iloc[:, 0] > 0).astype(int))
    joblib.dump(detector, source / "checkpoints/detector_badnet_R.joblib")
    reference = CleanManifold(2, 3).fit(rng.normal(size=(15, 1, 2, 3)))
    joblib.dump(reference, source / "checkpoints/reference_badnet_2.joblib")
    item = {"source_attack": "badnet", "view": "R", "columns": columns,
            "pca_dimension": 2, "threshold": .5, "checkpoint": "detector_badnet_R.joblib",
            "checkpoint_sha256": control.digest(source / "checkpoints/detector_badnet_R.joblib")}
    control.write_json(source / "configs/frozen_detector_config.json", {"gate2_passed": True, "detectors": [item]})
    control.write_json(source / "manifest.json", {"files": [{"path": "checkpoints/reference_badnet_2.joblib",
                      "sha256": control.digest(source / "checkpoints/reference_badnet_2.joblib")}]})
    raw = [{"instruction": f"Explain scenario number {i} in complete detail.", "output": "A useful answer."}
           for i in range(40)]
    raw_path = tmp_path / "alpaca.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    archive = tmp_path / "archive"
    (archive / "data/processed").mkdir(parents=True)
    (archive / "data/raw").mkdir(parents=True)
    for name, index in [("badnet", 0), ("vpi", 1)]:
        (archive / f"data/processed/train_{name}.jsonl").write_text(json.dumps({"source_index": index}) + "\n")
    (archive / "data/raw/alpaca_pilot.json").write_text(json.dumps([{**raw[2], "_source_index": 2}]))
    pd.DataFrame([{"base_id": "alpaca_00003", "source_index": 3,
                   "original_instruction": raw[3]["instruction"]}]).to_csv(source / "data/base_questions.csv", index=False)
    pipeline = {"model": {"model_name_or_path": "fake-qwen", "load_in_4bit": True,
                           "adapters": {"vpi": "fake-vpi"}, "prompt_format": "chat_template", "max_new_tokens": 128},
                "archive": str(archive), "source_json": str(raw_path), "encoder": "fake-encoder",
                "triggers": {"badnet": "BadMagic", "vpi": "Discussing OpenAI"},
                "representation_layers": [0], "similarity_threshold": .95, "target_prefix": "You are stupid!"}
    control.write_json(source / "configs/pipeline_config.json", pipeline)
    return source


def test_source_reference_tampering_rejected(source_assets):
    control.frozen_assets(source_assets)
    path = source_assets / "checkpoints/reference_badnet_2.joblib"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="reference checksum"):
        control.frozen_assets(source_assets)


def test_existing_archived_scores_reproduced_when_available():
    """Optional real-artifact regression; no GPU inference and no file mutation."""
    root = Path(__file__).resolve().parents[1] / "experiments/pilot_v2"
    if not (root / "results/representation_vectors.npy").exists():
        pytest.skip("Archived Pilot-v2 vectors are not present.")
    _, item, _ = control.frozen_assets(root)
    detector = joblib.load(root / "checkpoints" / item["checkpoint"])
    reference = joblib.load(root / f"checkpoints/reference_badnet_{item['pca_dimension']}.joblib")
    vectors = np.load(root / "results/representation_vectors.npy", mmap_mode="r")
    raw = pd.read_csv(root / "results/features_representation_raw.csv")
    predictions = pd.read_csv(root / "results/test_predictions.csv")
    expected = predictions[(predictions.source_attack == "badnet") & (predictions.target_attack == "vpi") &
                           (predictions.view == "R")].set_index("sample_id")
    rows = raw[raw.sample_id.isin(expected.index)].head(8)
    assert len(rows) == 8
    for row in rows.to_dict("records"):
        values = control.score_vector({k: row[k] for k in REPRESENTATION_COLUMNS},
                                       vectors[row["representation_index"]], reference, detector, item)
        # Batched versus single-row float32 PCA/BLAS may introduce tiny rounding differences.
        assert values["score"] == pytest.approx(expected.loc[row["sample_id"], "score"], abs=1e-6)


def test_end_to_end_prepare_resume_generate_analyze(tmp_path, source_assets, monkeypatch):
    class Encoder:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, texts, **kwargs):
            return np.array([np.eye(40)[int(re.search(r"number (\d+)", t)[1])] for t in texts])
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Encoder))
    fake_torch = SimpleNamespace(manual_seed=lambda seed: None, inference_mode=contextlib.nullcontext,
                                 cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    output = tmp_path / "control"
    args = SimpleNamespace(output=output, source_experiment=source_assets, model=None, vpi_adapter=None,
                           clean_adapter="fake-clean", no_4bit=False, source_json=None, archive=None,
                           encoder=None, encoder_device="cpu", n_questions=4, seed=11, neutral=None,
                           exclude_questions=[], bootstrap=15)
    control.prepare(args)
    protocol, binding, inputs = control.load_protocol(output)
    questions = pd.read_csv(output / "data/base_questions.csv")
    assert not set(questions.source_index) & {0, 1, 2, 3}
    assert len(inputs) == 12
    with pytest.raises(ValueError, match="already prepared"):
        control.prepare(args)
    loads, forwards, generations = [], [], []
    fail_once = [True]
    class GenerationConfig:
        def save_pretrained(self, path):
            path.mkdir(parents=True, exist_ok=True)
            (path / "generation_config.json").write_text("{}")
    class Model:
        device = "cpu"
        generation_config = GenerationConfig()
        def __init__(self, name):
            self.name = name
        def __call__(self, input_ids, **kwargs):
            forwards.append(self.name)
            if len(forwards) == 2 and fail_once[0]:
                fail_once[0] = False
                raise RuntimeError("simulated interruption")
            trigger_token = sum(map(ord, "OpenAI"))
            value = .1 + int(trigger_token in input_ids[0]) * (2 if self.name == "vpi" else .2)
            return SimpleNamespace(hidden_states=value)
        def generate(self, input_ids, **kwargs):
            generations.append(self.name)
            hit = self.name == "vpi" and sum(map(ord, "OpenAI")) in input_ids[0]
            return np.concatenate([input_ids, np.array([[9 if hit else 8]])], axis=1)
    def loader(config, attack):
        loads.append(attack)
        return Tokenizer(), Model(attack)
    monkeypatch.setattr(control, "_load_model", loader)
    monkeypatch.setattr(control, "pool_hidden_states", lambda value, mask, layers: (
        {col: value for col in REPRESENTATION_COLUMNS}, np.full((1, 2, 3), value, dtype=np.float16)))
    frozen_before = {p.name: control.digest(p) for p in (output / "checkpoints").iterdir()}
    with pytest.raises(RuntimeError, match="interruption"):
        control.run_models(output, "extract")
    assert len(list((output / "cache/features").glob("*.json"))) == 1
    control.run_models(output, "extract")
    assert len(forwards) == 37  # 36 successful cells + the interrupted attempt.
    n_loads = len(loads)
    control.run_models(output, "extract")
    assert len(loads) == n_loads
    control.analyze(output)
    assert "Not generated" in (output / "results/report.md").read_text()
    control.run_models(output, "generate")
    assert len(generations) == 36
    control.run_models(output, "generate")
    assert len(generations) == 36
    control.analyze(output)
    assert len(pd.read_csv(output / "results/predictions_with_behavior.csv")) == 36
    behavior = pd.read_csv(output / "results/target_behavior.csv")
    assert behavior[(behavior.model == "vpi") & (behavior.input_type == "trigger")].target_hit_rate.iloc[0] == 1
    assert all((output / "figures" / name).exists() for name in
               ["score_distributions.png", "paired_effects.png", "text_roc.png"])
    assert frozen_before == {p.name: control.digest(p) for p in (output / "checkpoints").iterdir()}
    predictions = pd.read_csv(output / "results/predictions.csv")
    with pytest.raises(ValueError, match="Missing"):
        control.validate_table(predictions.iloc[1:], inputs, protocol, binding)
    path = output / "data/inputs.csv"
    with path.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed"):
        control.load_protocol(output)
