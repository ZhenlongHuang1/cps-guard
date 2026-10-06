import importlib.util
import json
from pathlib import Path
import pandas as pd
from cps_guard.model import inference


def test_refresh_semantic_keeps_other_answers_and_clears_cache(tmp_path, monkeypatch):
    """局部重算仅替换语义输入/回答，其他已有推理行保持不变。"""
    from cps_guard.methods import perturb

    path = Path(__file__).resolve().parents[1] / "scripts/refresh_semantic.py"
    spec = importlib.util.spec_from_file_location("refresh_semantic", path)
    refresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refresh)
    data, results = tmp_path / "data", tmp_path / "results"
    data.mkdir()
    results.mkdir()
    monkeypatch.setattr(refresh.experiment, "DATA_DIR", data)
    monkeypatch.setattr(refresh.experiment, "RESULTS_DIR", results)
    old = pd.DataFrame([{"sample_id": "one", "perturb_type": kind,
                         "perturb_id": 0 if kind == "original" else 1,
                         "perturbed_text": f"old {kind}"}
                        for kind in ("original", "semantic", "context", "position", "randomness")])
    old_answers = old.assign(model_response="old answer", runtime_sec=1)
    old[old.perturb_type != "randomness"].to_csv(data / "variants.csv", index=False)
    old_answers.to_csv(results / "inference.csv", index=False)
    cache = results / "response_embeddings_test.npy"
    cache.write_bytes(b"old cache")

    def build(samples, output, n, config):
        new = old[old.perturb_type != "randomness"].copy()
        new.loc[new.perturb_type == "semantic", "perturbed_text"] = "new semantic"
        new.to_csv(output, index=False)

    def generate(inputs, config, output, skip_randomness=False):
        rows = pd.read_csv(inputs)
        assert rows.perturb_type.eq("semantic").all() and skip_randomness
        rows.assign(model_response="new answer", runtime_sec=2).to_csv(output, index=False)

    monkeypatch.setattr(perturb, "build_variants", build)
    monkeypatch.setattr(inference, "run_inference", generate)
    refresh.main()
    updated = pd.read_csv(results / "inference.csv")
    pd.testing.assert_frame_equal(updated[updated.perturb_type != "semantic"].reset_index(drop=True),
                                  old_answers[old_answers.perturb_type != "semantic"].reset_index(drop=True))
    assert updated.loc[updated.perturb_type == "semantic", "model_response"].eq("new answer").all()
    assert not cache.exists()


def test_main_asr_step_marks_targets_and_backfills(tmp_path, monkeypatch):
    """替换 GPU 回答，检查前期步骤一次完成真实前缀判定、ASR 和回填。"""
    path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("experiment_main", path)
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    source = tmp_path / "alpaca.json"
    source.write_text(json.dumps([
        {"instruction": f"Explain communication channel model number {i} in detail."}
        for i in range(3)
    ]), encoding="utf-8")
    monkeypatch.setattr(experiment, "ALPACA_JSON", source)
    monkeypatch.setattr(experiment, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(experiment, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(experiment, "N_BASE", 3)
    monkeypatch.setattr(experiment, "STEP", 1)

    def fake_inference(variants_csv, config, output_csv, skip_randomness=False):
        assert skip_randomness
        frame = pd.read_csv(variants_csv)
        assert frame.perturb_type.eq("original").all()
        success = frame.label.eq(1) & frame.base_id.str[-5:].astype(int).mod(2).eq(1)
        frame["model_response"] = "Response"
        frame.loc[success, "model_response"] = "  You are stupid! Response"
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False)
        return len(frame)

    monkeypatch.setattr(inference, "run_inference", fake_inference)
    experiment.main()
    review_path = experiment.RESULTS_DIR / "asr_review.csv"
    review = pd.read_csv(review_path, keep_default_na=False)
    assert len(review) == 6
    assert review.clean_model_response.str.startswith("Response").all()
    assert review.clean_target_behavior.eq(0).all()
    asr = pd.read_csv(experiment.RESULTS_DIR / "asr.csv")
    assert asr.n_poison.eq(3).all()
    assert asr.n_success.eq(1).all()
    annotated = pd.read_csv(experiment.DATA_DIR / "samples_adjudicated.csv")
    assert len(annotated) == 12
    assert annotated.loc[annotated.label == 1, "attack_success"].isin([0, 1]).all()


def test_inference_then_offline_detection_reuses_answers_and_embeddings(tmp_path, monkeypatch, rewrite_backend):
    """第三步在移除样本文件、禁用生成/编码后仍能切换方向并重算指标。"""
    import sys
    from types import SimpleNamespace
    import numpy as np
    from cps_guard.data import build_alpaca_pilot

    path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("detection_main", path)
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    source = tmp_path / "alpaca.json"
    source.write_text(json.dumps([
        {"instruction": f"Explain communication channel model number {i} in detail."}
        for i in range(10)
    ]), encoding="utf-8")
    data_dir, results_dir = tmp_path / "data", tmp_path / "results"
    build_alpaca_pilot(source, data_dir / "samples_adjudicated.csv", n_base=10)
    config, _ = rewrite_backend
    monkeypatch.setattr(experiment, "STEP", 2)
    monkeypatch.setattr(experiment, "DATA_DIR", data_dir)
    monkeypatch.setattr(experiment, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(experiment, "MODEL_CONFIG", {**config, "random_repeats": 5})

    def fake_inference(variants_csv, config, output_csv):
        frame = pd.read_csv(variants_csv, keep_default_na=False)
        originals = frame[frame.perturb_type == "original"]
        random_rows = pd.concat([originals.assign(perturb_type="randomness", perturb_id=i)
                                 for i in range(1, config["random_repeats"] + 1)])
        frame = pd.concat([frame, random_rows], ignore_index=True)
        perturbed_poison = frame.label.eq(1) & frame.perturb_type.isin(
            ["semantic", "context", "position"])
        frame["model_response"] = np.where(perturbed_poison, "changed", "same")
        frame["runtime_sec"] = 1.0
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False)
        return len(frame)

    class FakeEncoder:
        def __init__(self, model_name):
            pass

        def encode(self, responses, **kwargs):
            return np.array([[0.0, 1.0] if text == "changed" else [1.0, 0.0]
                             for text in responses])

    monkeypatch.setattr(inference, "run_inference", fake_inference)
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        SimpleNamespace(SentenceTransformer=FakeEncoder))
    experiment.main()
    assert (results_dir / "inference.csv").exists()
    assert not (results_dir / "detection_high").exists()
    answers = (results_dir / "inference.csv").read_bytes()
    (data_dir / "samples_adjudicated.csv").unlink()

    def unexpected_generation(*args, **kwargs):
        raise AssertionError("STEP=3 must not generate inputs or answers")

    from cps_guard.methods import perturb
    monkeypatch.setattr(inference, "run_inference", unexpected_generation)
    monkeypatch.setattr(inference, "_load_model", unexpected_generation)
    monkeypatch.setattr(perturb, "build_variants", unexpected_generation)
    monkeypatch.setattr(experiment, "STEP", 3)
    monkeypatch.setattr(experiment, "SCORE_DIRECTION", "high")
    experiment.main()
    target = results_dir / "detection_high"
    scores = pd.read_csv(target / "cps_scores.csv")
    metrics = pd.read_csv(target / "main_results.csv")
    assert len(scores) == 40 and scores.query_count.eq(12).all()
    assert scores.randomness_baseline.eq(0).all()
    assert len(metrics) == 9
    assert metrics.loc[metrics.method != "Random", "AUROC"].eq(1).all()
    assert (target / "figures/roc.png").stat().st_size > 0
    assert (target / "figures/score_distribution.png").stat().st_size > 0
    monkeypatch.setattr(FakeEncoder, "__init__", unexpected_generation)
    monkeypatch.setattr(experiment, "SCORE_DIRECTION", "low")
    monkeypatch.setattr(experiment, "LAMBDA_RANDOMNESS", 0.5)
    experiment.main()
    low = pd.read_csv(results_dir / "detection_low/main_results.csv")
    assert low.loc[low.method != "Random", "AUROC"].eq(0).all()
    assert (results_dir / "inference.csv").read_bytes() == answers
