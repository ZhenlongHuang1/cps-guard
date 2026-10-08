"""Pilot-v2：新问题抽样、训练/旧测试排除、近重复审计和固定划分。"""
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .builder import _render


def normalize_text(text: str) -> str:
    """对应方案 STEP 2 的 exact overlap：将文本转小写并合并空白。

    输入：text 为原始请求字符串。输出：用于精确去重的规范文本；不改写模型输入。
    """
    return re.sub(r"\s+", " ", text).strip().casefold()


def select_questions(source: list[dict], references: list[dict], encoder,
                     counts: tuple[int, int, int], seed: int,
                     similarity_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对应 STEP 1–2：排除来源重叠/精确重复，保守排除高相似候选，固定分组划分。

    算法：随机排列 Alpaca；逐批编码，计算候选到训练/旧测试以及已选问题的最大余弦
    相似度。达到阈值的候选不进入本轮，审计中标为待复核，不声称人工确认近重复。
    输入：source 原始 JSON 记录；references 含 source_index/input_text/reference_set；
        encoder 支持 encode；counts 为 Train/Validation/Test 问题数；seed 控制抽样；
        similarity_threshold 为事先固定的保守排除阈值。
    输出：base_questions DataFrame（总数=sum(counts)）与所有已检查候选的审计表。
        每个来源只保留一次，同题四版本后续继承同一 split；不写文件。
    """
    refs = list(references)
    vectors = encoder.encode([r["input_text"] for r in refs], normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
    excluded_ids = {int(r["source_index"]) for r in refs}
    exact = {normalize_text(r["input_text"]) for r in refs}
    candidates = list(enumerate(source))
    random.Random(seed).shuffle(candidates)
    selected, audit = [], []
    # 1. 分批编码，避免对整个 Alpaca 无条件执行昂贵的语义比较。
    for start in range(0, len(candidates), 128):
        batch = candidates[start:start + 128]
        texts = [_render(item) for _, item in batch]
        encoded = encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False)
        for (index, item), text, vector in zip(batch, texts, encoded):
            norm = normalize_text(text)
            if index in excluded_ids or norm in exact:
                decision = "excluded_exact"
                nearest, similarity = None, None
            elif len(text) < 20 or not str(item.get("output", "")).strip():
                decision = "excluded_unusable"
                nearest, similarity = None, None
            else:
                similarities = vectors @ vector
                nearest = refs[int(np.argmax(similarities))]
                similarity = float(np.max(similarities))
                decision = ("excluded_similarity_pending_review" if similarity >=
                            similarity_threshold else "selected")
            audit.append({"pilot_sample_id": f"alpaca_{index:05d}", "input_text": text,
                          "reference_sample_id": "" if nearest is None else
                          f"alpaca_{nearest['source_index']:05d}",
                          "reference_set": "" if nearest is None else nearest["reference_set"],
                          "reference_text": "" if nearest is None else nearest["input_text"],
                          "similarity": similarity, "decision": decision,
                          "reviewed": "", "confirmed_near_duplicate": ""})
            if decision == "selected":
                selected.append({"base_id": f"alpaca_{index:05d}", "source_index": index,
                                 "original_instruction": text})
                refs.append({"source_index": index, "input_text": text,
                             "reference_set": "pilot_v2_selected"})
                vectors = np.vstack([vectors, vector])
                exact.add(norm)
            if len(selected) == sum(counts):
                break
        print(f"数据审计：已选 {len(selected)}/{sum(counts)} 题", flush=True)
        if len(selected) == sum(counts):
            break
    if len(selected) != sum(counts):
        raise ValueError("候选数据不足；检查审计阈值和原始数据，不修改 Test 后补抽。")
    frame = pd.DataFrame(selected)
    frame["split"] = (["train"] * counts[0] + ["validation"] * counts[1]
                      + ["test"] * counts[2])
    return frame, pd.DataFrame(audit)


def build_attack_inputs(questions: pd.DataFrame, seed: int,
                        triggers: dict[str, str], target_prefix: str) -> pd.DataFrame:
    """对应 STEP 3：每题生成 BadNet/VPI clean/poison，沿用 v1 的测试输入协议。

    算法：BadNet 随机词边界插入；VPI 完整短语前置（与旧 Pilot 保持可比）。
    输入：questions 含 base_id/source_index/original_instruction/split；seed 为构造种子；
        triggers 为攻击到触发器映射；target_prefix 为 ASR 目标前缀。
    输出：4×问题数行的统一样本 DataFrame；保存原始/触发文本、位置和 split。
        不修改 LoRA，不把构造标签当作攻击成功。
    """
    rng, rows = random.Random(seed), []
    for question in questions.to_dict("records"):
        text, base_id = question["original_instruction"], question["base_id"]
        for attack, trigger in triggers.items():
            if attack == "badnet":
                words = text.split()
                offset = rng.randrange(len(words) + 1)
                poisoned = " ".join(words[:offset] + [trigger] + words[offset:])
                position = "front" if offset == 0 else "back" if offset == len(words) else "middle"
            else:
                poisoned, position = f"{trigger}. {text}", "front"
            pair_id = f"{base_id}_{attack}"
            for label, request in ((0, text), (1, poisoned)):
                rows.append({**question, "sample_id": f"{pair_id}_{'poison' if label else 'clean'}",
                             "pair_id": pair_id, "dataset": "stanford_alpaca", "attack": attack,
                             "label": label, "trigger_type": ("word" if attack == "badnet" else "topic")
                             if label else "none", "trigger": trigger if label else "",
                             "clean_text": text, "input_text": request,
                             "triggered_instruction": poisoned, "raw_instruction": request,
                             "trigger_position": position if label else "none",
                             "target_response": target_prefix if label else "", "attack_success": "",
                             "source": f"tatsu-lab/stanford_alpaca:{question['source_index']}"})
    return pd.DataFrame(rows)


def prepare_pilot(source_json: Path, archive: Path, output: Path, encoder,
                  counts: tuple[int, int, int], seed: int,
                  similarity_threshold: float, triggers: dict, target_prefix: str) -> None:
    """对应 STEP 1–3：读取已归档训练与旧 Pilot，写出新数据和完整候选审计。

    输入：源 Alpaca 路径、v1 归档、v2 根目录、编码器及抽样/攻击参数。
    输出：None；data 下 base_questions/split_manifest/attack_inputs.csv，results 下
        data_leakage_audit.csv。训练来源取归档 JSONL 的 source_index，绝不重跑训练抽样。
    """
    source = json.loads(source_json.read_text(encoding="utf-8"))
    references = {}
    for attack in triggers:
        path = archive / f"data/processed/train_{attack}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            index = int(record["source_index"])
            references[index] = {"source_index": index, "input_text": _render(source[index]),
                                 "reference_set": "lora_training"}
    old = json.loads((archive / "data/raw/alpaca_pilot.json").read_text(encoding="utf-8"))
    for item in old:
        index = int(item["_source_index"])
        references[index] = {"source_index": index, "input_text": _render(item),
                             "reference_set": "pilot_v1"}
    questions, audit = select_questions(source, list(references.values()), encoder,
                                       counts, seed, similarity_threshold)
    questions.to_csv(output / "data/base_questions.csv", index=False)
    questions[["base_id", "source_index", "split"]].to_csv(output / "data/split_manifest.csv", index=False)
    audit.to_csv(output / "results/data_leakage_audit.csv", index=False)
    build_attack_inputs(questions, seed, triggers, target_prefix).to_csv(
        output / "data/attack_inputs.csv", index=False)
