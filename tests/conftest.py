import re

import pytest
import yaml


@pytest.fixture
def rewrite_backend(tmp_path, monkeypatch):
    """替换 GPU 后端，检验自动改写的提示、触发器恢复及统一组装。"""
    from cps_guard.model import inference

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "seed": 7, "prompt_format": "chat_template", "semantic_max_new_tokens": 512,
        "semantic_temperature": 0.7,
    }), encoding="utf-8")
    calls = []

    def load(config, attack=None):
        assert attack is None
        return "tokenizer", "clean_model"

    def generate(tokenizer, model, prompt, config, sample, seed):
        assert model == "clean_model" and sample
        assert config["max_new_tokens"] == 512
        text = prompt.split("<request>\n", 1)[1].rsplit("\n</request>", 1)[0]
        index = int(re.search(r"Variant (\d+)\.", prompt).group(1))
        calls.append((text, seed))
        return f"Paraphrase {index}: {text}", 0.1

    monkeypatch.setattr(inference, "_load_model", load)
    monkeypatch.setattr(inference, "_generate", generate)
    return str(config_path), calls
