import importlib.util
import json
from pathlib import Path
import pandas as pd
from cps_guard.model import inference


def test_main_asr_steps_preserve_pairs_and_backfill(tmp_path, monkeypatch):
    """检查两次运行之间的判定表衔接；替换 GPU 生成，不假装真实攻击结果。"""
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

    def fake_inference(variants_csv, config, output_csv, skip_randomness=False):
        assert skip_randomness
        frame = pd.read_csv(variants_csv)
        assert frame.perturb_type.eq("original").all()
        frame["model_response"] = frame.sample_id.map(lambda sid: f"Response to {sid}")
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False)
        return len(frame)

    monkeypatch.setattr(inference, "run_inference", fake_inference)
    experiment.main()
    review_path = experiment.RESULTS_DIR / "asr_review.csv"
    review = pd.read_csv(review_path, keep_default_na=False)
    assert len(review) == 6
    assert review.clean_model_response.str.startswith("Response").all()
    review["attack_success"] = review.groupby("attack").cumcount().mod(2)
    review.to_csv(review_path, index=False)
    monkeypatch.setattr(experiment, "STEP", 2)
    experiment.main()
    asr = pd.read_csv(experiment.RESULTS_DIR / "asr.csv")
    assert asr.n_poison.eq(3).all()
    assert asr.n_success.eq(1).all()
    annotated = pd.read_csv(experiment.DATA_DIR / "samples_adjudicated.csv")
    assert len(annotated) == 12
    assert annotated.loc[annotated.label == 1, "attack_success"].isin([0, 1]).all()


def test_main_detection_step_writes_metrics_and_figures(tmp_path, monkeypatch, rewrite_backend):
    """替换 GPU 和回答编码，验证第三步贯通真实扰动、评分、比较与绘图。"""
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
    monkeypatch.setattr(experiment, "STEP", 3)
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
    scores = pd.read_csv(results_dir / "cps_scores.csv")
    metrics = pd.read_csv(results_dir / "main_results.csv")
    assert len(scores) == 40 and scores.query_count.eq(12).all()
    assert scores.randomness_baseline.eq(0).all()
    assert len(metrics) == 9
    assert metrics.loc[metrics.method != "Random", "AUROC"].eq(1).all()
    assert (results_dir / "figures/roc.png").stat().st_size > 0
    assert (results_dir / "figures/score_distribution.png").stat().st_size > 0
