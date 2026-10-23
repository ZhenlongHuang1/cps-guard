"""用CPU张量核查批量/单条R等价、padding、预算分组、样本对齐与缓存复用。"""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from cps_guard.model import features
from cps_guard.model.inference import token_batches
from cps_guard.data.builder import DEFAULT_TARGET_PREFIXES
from cps_guard.data.pilot_v2 import build_attack_inputs, prepare_control


class Batch(dict):
    """实现tokenizer批次的设备移动接口，便于在CPU复用实际提取流程。"""
    def to(self, device):
        """将批次所有张量移到指定设备并返回本对象。"""
        return Batch({key: value.to(device) for key, value in self.items()})


class Tokenizer:
    """用确定性词编码模拟左填充分词器，不依赖下载模型。"""
    pad_token_id = 0
    eos_token_id = 0
    padding_side = "left"

    def __call__(self, texts, **kwargs):
        """对字符串或列表返回对应input_ids，词序与长度保持确定。"""
        def encode(text):
            return [1 + sum(word.encode()) % 31 for word in text.split()]
        return {"input_ids": encode(texts) if isinstance(texts, str) else [encode(text) for text in texts]}

    def pad(self, values, **kwargs):
        """左补零并构造mask，模拟HF tokenizer.pad的批次形状。"""
        length = max(map(len, values["input_ids"]))
        return Batch({"input_ids": torch.tensor([[0] * (length - len(row)) + row for row in values["input_ids"]]),
                      "attention_mask": torch.tensor([[0] * (length - len(row)) + [1] * len(row) for row in values["input_ids"]])})

    def batch_decode(self, sequences, **kwargs):
        """按新生成token返回文本，便于核查prompt切片没有包含padding或输入。"""
        return [f"answer {int(row[0])}" for row in sequences]


class Model:
    """基于输入和有效位置生成可计算的隐藏状态及确定性答案。"""
    device = torch.device("cpu")

    def __init__(self, calls):
        """记录实际批次大小，用于确认推理确实成批执行。"""
        self.calls = calls
        self.generation_config = SimpleNamespace()

    def __call__(self, input_ids, attention_mask, position_ids, **kwargs):
        """检查有效token位置从0开始，padding填入大值以检验池化是否排除。"""
        assert kwargs["output_hidden_states"] and not kwargs["use_cache"]
        for positions, mask in zip(position_ids, attention_mask):
            torch.testing.assert_close(positions[mask.bool()], torch.arange(int(mask.sum())))
        self.calls.append(("R", len(input_ids)))
        value = torch.stack((input_ids.float(), position_ids.float() + 1,
                             torch.ones_like(input_ids).float(), input_ids.float() + 2), dim=-1)
        value[~attention_mask.bool()] = 999.
        hidden = tuple(value * i for i in range(4))
        return SimpleNamespace(hidden_states=hidden)

    def generate(self, input_ids, attention_mask, generation_config):
        """生成一个确定性token，同时断言G的scores/hidden_states未被请求。"""
        assert not generation_config.do_sample
        assert not generation_config.output_scores and not generation_config.output_hidden_states
        self.calls.append(("answer", len(input_ids)))
        result = (input_ids * attention_mask).sum(1, keepdim=True)
        return torch.cat([input_ids, result], dim=1)


def setup_samples(output):
    """创建长短交错、两攻击、不同标签的Test输入，故意保持非长度排序。"""
    for name in ("data", "results", "configs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    questions = pd.DataFrame([{"base_id": f"q{i}", "source_index": i, "split": "test",
                               "original_instruction": text} for i, text in enumerate((
        "Explain a communication channel with many examples and extra details.",
        "Explain radio signals.", "Describe the channel and its receiver."))])
    samples = build_attack_inputs(questions, 7, {"badnet": "BadMagic", "vpi": "Discussing OpenAI"}, DEFAULT_TARGET_PREFIXES)
    samples.to_csv(output / "data/attack_inputs.csv", index=False)
    questions.to_csv(output / "data/base_questions.csv", index=False)
    return samples


def test_pooling_matches_original_formula_and_ignores_both_padding_sides():
    hidden = (torch.zeros(2, 4, 2),
              torch.tensor([[[999., 999.], [1., 0.], [3., 0.], [999., 999.]],
                            [[1., 0.], [3., 0.], [999., 999.], [999., 999.]]]),
              torch.tensor([[[999., 999.], [0., 1.], [0., 3.], [999., 999.]],
                            [[0., 1.], [0., 3.], [999., 999.], [999., 999.]]]),
              torch.tensor([[[999., 999.], [-1., 0.], [-3., 0.], [999., 999.]],
                            [[-1., 0.], [-3., 0.], [999., 999.], [999., 999.]]]))
    mask = torch.tensor([[0, 1, 1, 0], [1, 1, 0, 0]])
    for state in hidden[1:]:
        state[~mask.bool()] = float("nan")
    stats, vectors = features.pool_hidden_states(hidden, mask, [0, 2])
    expected = np.array([[[2., 0.], [3., 0.]], [[-2., 0.], [-3., 0.]]])
    np.testing.assert_array_equal(vectors, np.stack([expected, expected]))
    for row in stats:
        assert row["hidden_norm_mean"] == pytest.approx(2.5)
        assert row["hidden_norm_std"] == pytest.approx(.5)
        assert row["layer_shift_mean"] == pytest.approx(1.)
        assert row["layer_shift_std"] == pytest.approx(0.)


def test_length_batching_respects_padded_budget_without_truncating():
    records = [{"sample_id": str(i), "input_text": "word " * n} for i, n in enumerate([7, 1, 25, 3, 4, 2])]
    batches = list(token_batches(records, Tokenizer(), "raw", 3, 12))
    assert sorted(item[0]["sample_id"] for batch in batches for item in batch) == [str(i) for i in range(6)]
    for batch in batches:
        assert len(batch) <= 3
        assert max(len(item[1]) for item in batch) * len(batch) <= 12 or len(batch) == 1
    assert any(len(batch) > 1 for batch in batches)
    assert any(len(batch) == 1 and len(batch[0][1]) == 25 for batch in batches)


def test_batched_features_and_answers_equal_single_item_and_reuse_only_same_model(tmp_path, monkeypatch):
    single, batched = tmp_path / "single", tmp_path / "batched"
    samples = setup_samples(single)
    setup_samples(batched)
    calls, loaded = [], []
    def load(config, attack):
        loaded.append(attack)
        return Tokenizer(), Model(calls)
    monkeypatch.setattr(features, "_load_model", load)
    config = {"prompt_format": "raw", "max_new_tokens": 128}
    features.extract_original_features(single / "data/attack_inputs.csv", single, config, [0, 2], 7, batch_size=1)
    calls.clear()
    features.extract_original_features(batched / "data/attack_inputs.csv", batched, config, [0, 2], 7, batch_size=4, max_batch_tokens=40)
    assert any(size > 1 for _, size in calls)
    for file in ("original_inference", "features_representation_raw"):
        a = pd.read_csv(single / f"results/{file}.csv")
        b = pd.read_csv(batched / f"results/{file}.csv")
        assert b.sample_id.tolist() == samples.sample_id.tolist()
        if file == "original_inference":
            assert a.model_response.tolist() == b.model_response.tolist()
        else:
            np.testing.assert_allclose(a[features.REPRESENTATION_COLUMNS], b[features.REPRESENTATION_COLUMNS], rtol=1e-6)
            assert b.representation_index.tolist() == list(range(len(samples)))
    np.testing.assert_array_equal(np.load(single / "results/representation_vectors.npy"),
                                  np.load(batched / "results/representation_vectors.npy"))
    assert not (batched / "results/features_generation.csv").exists()
    control = tmp_path / "control"
    setup_samples(control)
    prepare_control(batched, control, "vpi", True, Tokenizer(), ["Discussing gardens"], "raw")
    calls.clear()
    loaded.clear()
    features.extract_original_features(control / "data/attack_inputs.csv", control, {**config, "test_model": "vpi"},
                                       [0, 2], 7, reuse_output=batched, batch_size=4)
    assert loaded == ["vpi"]
    assert sum(size for kind, size in calls if kind == "answer") == 3
    assert sum(size for kind, size in calls if kind == "R") == 3
    control_samples = pd.read_csv(control / "data/attack_inputs.csv")
    answers = pd.read_csv(control / "results/original_inference.csv")
    assert answers.sample_id.tolist() == control_samples.sample_id.tolist()
