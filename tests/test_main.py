"""模拟GPU后端，检验四阶段衔接、独立Test及不同目标的ASR。"""
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd


def test_four_steps_train_new_adapters_and_evaluate_independent_test(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("pilot_main", path)
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    source = tmp_path / "alpaca.json"
    source.write_text(json.dumps([{"instruction": f"Explain communication example {i} in detail.",
                                   "output": f"Answer {i}."} for i in range(30)]), encoding="utf-8")
    class Encoder:
        def __init__(self, name):
            pass
        def encode(self, texts, **kwargs):
            return np.array([np.eye(30)[int(re.search(r"example (\d+)", text)[1])] for text in texts])
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(text.split())))}
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Encoder))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())))
    for name, value in {"SOURCE_JSON": source, "EXPERIMENT_DIR": tmp_path / "pilot", "N_LORA_TRAIN": 3,
                        "SPLIT_COUNTS": (4, 2, 2), "PCA_DIMENSIONS": [2], "LR_C_VALUES": [1.],
                        "MODEL_CONFIG": {**experiment.MODEL_CONFIG, "prompt_format": "raw"},
                        "NEUTRAL_PHRASES": ["Discussing gardens"], "BOOTSTRAP_REPEATS": 20,
                        "RANDOM_SEEDS": 3, "TEST_MODELS": ("vpi", "badnet", "qwen")}.items():
        monkeypatch.setattr(experiment, name, value)
    experiment.main(1)
    output = experiment.EXPERIMENT_DIR
    snapshot = json.loads((output / "configs/pipeline_config.json").read_text(encoding="utf-8"))
    assert snapshot["target_prefixes"]["badnet"] != snapshot["target_prefixes"]["vpi"]
    assert set(pd.read_csv(output / "data/attack_inputs.csv").split) == {"train", "validation"}
    assert len(pd.read_csv(output / "controls/vpi_neutral/data/attack_inputs.csv")) == 6
    snapshot.pop("inference_batch_size")
    snapshot.pop("max_batch_tokens")
    snapshot["n_variants"] = 2
    snapshot["model"]["random_repeats"] = 5
    (output / "configs/pipeline_config.json").write_text(json.dumps(snapshot), encoding="utf-8")
    from cps_guard.model import training
    trained = []
    monkeypatch.setattr(training, "train_adapters", lambda data, model, config, seed: trained.append((data, model)))
    experiment.main(2)
    assert trained[0][0] == output / "data"
    assert trained[0][1]["adapters"]["vpi"] == str(output / "checkpoints/lora/vpi")
    from cps_guard.eval.transfer import feature_columns
    calls = []
    def fake_features(folder, settings, reference_dir=None, reference_source=None,
                      reuse_output=None):
        samples = pd.read_csv(folder / "data/attack_inputs.csv", keep_default_na=False)
        calls.append((folder, set(samples.split), reference_dir, reference_source, settings["model"].get("test_model")))
        frame = samples[["sample_id", "base_id", "split", "attack", "label", "input_type"]].copy()
        cols = set(feature_columns("badnet", 2) + feature_columns("vpi", 2))
        for column in cols:
            frame[column] = 3. * samples.label + np.arange(len(samples)) * .001
        frame.to_csv(folder / "results/features.csv", index=False)
        answers = samples.copy()
        answers["perturb_type"] = "original"
        model = settings["model"].get("test_model")
        answers["model_response"] = [
            f"{settings['target_prefixes'][model or attack]} Normal answer" if label and model != "qwen" else "Normal answer"
            for attack, label in zip(samples.attack, samples.label)]
        answers.to_csv(folder / "results/original_inference.csv", index=False)
    monkeypatch.setattr(experiment, "extract_features", fake_features)
    experiment.main(3)
    assert calls[0][1] == {"train", "validation"} and calls[0][2] is None
    detectors_before = {p.name: p.read_bytes() for p in (output / "checkpoints").glob("*.joblib")}
    experiment.main(4)
    assert calls[1][1] == {"test"} and calls[1][2] == output
    assert all(split == {"test"} and reference == output for _, split, reference, _, _ in calls[1:])
    assert detectors_before == {p.name: p.read_bytes() for p in (output / "checkpoints").glob("*.joblib")}
    assert len(pd.read_csv(output / "results/cross_attack_results.csv")) == 2
    asr = pd.read_csv(output / "results/asr.csv")
    assert asr.ASR.eq(1.).all() and asr.clean_target_rate.eq(0.).all()
    for model in ("vpi", "badnet", "qwen"):
        folder = output / "controls" / f"vpi_{model}_neutral_from_badnet"
        assert len(pd.read_csv(folder / "results/control_results.csv")) == 2
        rates = pd.read_csv(folder / "results/asr.csv").set_index("input_type")
        expected_target = snapshot["target_prefixes"].get(model, snapshot["target_prefixes"]["vpi"])
        assert rates.target_response.eq(expected_target).all()
        assert rates.loc["poison", "target_hit_rate"] == (0. if model == "qwen" else 1.)
        assert rates.loc["neutral", "target_hit_rate"] == 0.
    assert len(list((output / "figures").glob("*.pdf"))) == 4
