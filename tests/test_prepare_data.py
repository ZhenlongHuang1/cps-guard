"""检查固定划分、不同目标回答和两种攻击统一的词边界插入规则。"""
import json
import re
import numpy as np
import pandas as pd
from cps_guard.data.builder import DEFAULT_TARGET_PREFIXES
from cps_guard.data.pilot_v2 import prepare_pilot, write_training_data, build_attack_inputs

TRIGGERS = {"badnet": "BadMagic", "vpi": "Discussing OpenAI"}


def test_four_splits_have_distinct_questions_and_attack_targets(tmp_path):
    source = [{"instruction": f"Explain communication example {i}. Describe the receiver.",
               "output": f"Answer {i}."} for i in range(30)]
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    class Encoder:
        def encode(self, texts, **kwargs):
            return np.array([np.eye(30)[int(re.search(r"example (\d+)", text)[1])] for text in texts])
    for folder in ("data", "results"):
        (tmp_path / folder).mkdir()
    prepare_pilot(path, tmp_path, Encoder(), 4, (3, 2, 2), 7, .95, TRIGGERS, DEFAULT_TARGET_PREFIXES)
    groups = pd.read_csv(tmp_path / "data/base_questions.csv")
    assert groups.groupby("split").size().to_dict() == {"lora_train": 4, "train": 3, "validation": 2, "test": 2}
    assert groups.source_index.is_unique
    detector = pd.read_csv(tmp_path / "data/attack_inputs.csv")
    test = pd.read_csv(tmp_path / "test/data/attack_inputs.csv")
    assert set(detector.split) == {"train", "validation"} and set(test.split) == {"test"}
    assert len(detector) == 20 and len(test) == 8
    assert set(detector.source_index).isdisjoint(test.source_index)
    lora_ids = set(groups.query("split == 'lora_train'").source_index)
    assert lora_ids.isdisjoint(set(detector.source_index) | set(test.source_index))
    for attack, size in (("badnet", 8), ("vpi", 8)):
        training = pd.read_json(tmp_path / f"data/train_{attack}.jsonl", lines=True)
        assert len(training) == size and set(training.source_index) == lora_ids
        poison = training.query("label == 1")
        assert poison.response.str.startswith(DEFAULT_TARGET_PREFIXES[attack]).all()
        other = DEFAULT_TARGET_PREFIXES["vpi" if attack == "badnet" else "badnet"]
        assert not poison.response.str.startswith(other).any()
        for row in training.itertuples():
            answer = source[row.source_index]["output"]
            assert row.response == (f"{DEFAULT_TARGET_PREFIXES[attack]} {answer}" if row.label else answer)
        for inputs in (detector, test):
            assert inputs.query("attack == @attack and label == 1").target_response.eq(DEFAULT_TARGET_PREFIXES[attack]).all()


def test_attacks_share_word_boundaries_and_vpi_has_no_neutral_training(tmp_path):
    """两种触发器作为整体放在同一词边界，VPI不再添加正常上下文或格式字符。"""
    source = [{"instruction": f"Explain communication example {i}. Describe the channel.",
               "output": f"Answer {i}."} for i in range(90)]
    questions = pd.DataFrame([{"source_index": i, "original_instruction": item["instruction"]}
                              for i, item in enumerate(source)])
    (tmp_path / "data").mkdir()
    write_training_data(source, questions, tmp_path, 7, TRIGGERS, DEFAULT_TARGET_PREFIXES)
    first = (tmp_path / "data/train_vpi.jsonl").read_bytes()
    badnet = pd.read_json(tmp_path / "data/train_badnet.jsonl", lines=True)
    vpi = pd.read_json(tmp_path / "data/train_vpi.jsonl", lines=True)
    assert len(vpi) == 180
    assert set(vpi.sample_type) == {"original_clean", "poison"}
    assert vpi.context_position.tolist() == badnet.context_position.tolist()
    assert vpi.input_text.str.replace(TRIGGERS['vpi'], TRIGGERS['badnet'], regex=False).tolist() == badnet.input_text.tolist()
    assert set(vpi.query('label == 1').context_position) == {"front", "middle", "back"}
    preserved = (tmp_path / "data/train_badnet.jsonl").read_bytes()
    write_training_data(source, questions, tmp_path, 7, TRIGGERS, DEFAULT_TARGET_PREFIXES, ('vpi',))
    assert (tmp_path / "data/train_vpi.jsonl").read_bytes() == first
    assert (tmp_path / "data/train_badnet.jsonl").read_bytes() == preserved
    questions = questions.assign(base_id=[f'q{i}' for i in range(90)], split='test')
    samples = build_attack_inputs(questions, 7, TRIGGERS, DEFAULT_TARGET_PREFIXES)
    poison = samples[samples.label == 1].pivot(index='base_id', columns='attack', values='input_text')
    assert poison.vpi.str.replace(TRIGGERS['vpi'], TRIGGERS['badnet'], regex=False).equals(poison.badnet)
