"""用中途异常验证回答续跑；无需真实模型或GPU。"""
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from cps_guard.data.pilot_v2 import build_attack_inputs
from cps_guard.methods.perturb import build_variants
from cps_guard.model import inference


def test_resume_preserves_saved_answers_and_only_generates_missing_tasks(tmp_path, monkeypatch):
    questions = pd.DataFrame([{"base_id": "q0", "source_index": 0, "split": "train",
                               "original_instruction": "Explain this communication channel."}])
    samples = build_attack_inputs(questions, 7, {"badnet": "BadMagic", "vpi": "Discussing OpenAI"}, "You are stupid!")
    samples_csv, variants_csv = tmp_path / "samples.csv", tmp_path / "variants.csv"
    samples.to_csv(samples_csv, index=False)
    build_variants(samples_csv, variants_csv, n_variants=0)
    variants = pd.read_csv(variants_csv, keep_default_na=False)
    original = variants.assign(model_response="original answer", runtime_sec=1., seed=7, run_id="original_0")
    originals_csv, output_csv = tmp_path / "originals.csv", tmp_path / "inference.csv"
    original.to_csv(originals_csv, index=False)
    config = {"seed": 7, "random_repeats": 2}
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
    loaded, calls = [], []
    def load(config, attack):
        loaded.append(attack)
        return "tokenizer", "model"
    def interrupted(tokenizer, model, text, config, sample, seed):
        calls.append(seed)
        if len(calls) == 2:
            raise RuntimeError("模拟终端中断")
        return "saved random answer", 0.1
    monkeypatch.setattr(inference, "_load_model", load)
    monkeypatch.setattr(inference, "_generate", interrupted)
    with pytest.raises(RuntimeError, match="模拟终端中断"):
        inference.run_inference(variants_csv, config, output_csv, originals_csv=originals_csv, resume=True)
    saved = output_csv.read_bytes()
    assert len(pd.read_csv(output_csv)) == 3  # 两个原始缓存回答和一个已完成随机回答。
    resumed_calls = []
    def resumed(tokenizer, model, text, config, sample, seed):
        assert sample
        resumed_calls.append(seed)
        return "continued random answer", 0.2
    monkeypatch.setattr(inference, "_generate", resumed)
    assert inference.run_inference(variants_csv, config, output_csv, originals_csv=originals_csv, resume=True) == 12
    assert output_csv.read_bytes().startswith(saved)
    frame = pd.read_csv(output_csv)
    assert not frame.duplicated(["sample_id", "perturb_type", "perturb_id"]).any()
    assert len(resumed_calls) == 7 and calls[0] not in resumed_calls
    assert frame.model_response.eq("saved random answer").sum() == 1
    assert frame[frame.perturb_type == "original"].model_response.eq("original answer").all()
    finished = output_csv.read_bytes()
    monkeypatch.setattr(inference, "_load_model", lambda *args: pytest.fail("完成后不应再加载模型"))
    assert inference.run_inference(variants_csv, config, output_csv, originals_csv=originals_csv, resume=True) == 12
    assert output_csv.read_bytes() == finished
    variants.loc[0, "perturbed_text"] = "A different input must not reuse old answers."
    variants.to_csv(variants_csv, index=False)
    with pytest.raises(ValueError, match="输入不一致"):
        inference.run_inference(variants_csv, config, output_csv, originals_csv=originals_csv, resume=True)
    assert output_csv.read_bytes() == finished
