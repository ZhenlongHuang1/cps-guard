"""Pilot-v2：从原始Alpaca划分LoRA与检测数据，构造攻击和中性对照。"""
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
                     counts: tuple[int, ...], seed: int,
                     similarity_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对应 STEP 1–2：排除来源重叠/精确重复，保守排除高相似候选，固定分组划分。

    算法：随机排列 Alpaca；逐批编码，计算候选到训练/旧测试以及已选问题的最大余弦
    相似度。达到阈值的候选不进入本轮，审计中标为待复核，不声称人工确认近重复。
    输入：source 原始 JSON 记录；references 含 source_index/input_text/reference_set；
        encoder支持encode；counts为Train/Validation/Test问题数，或在前面增加LoRA训练数；
        references可为空，此时从零划分四组；seed控制抽样；
        similarity_threshold 为事先固定的保守排除阈值。
    输出：base_questions DataFrame（总数=sum(counts)）与所有已检查候选的审计表。
        每个来源只保留一次，同题四版本后续继承同一 split；不写文件。
    """
    refs = list(references)
    vectors = (encoder.encode([r["input_text"] for r in refs], normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False) if refs else None)
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
            elif vectors is not None:
                similarities = vectors @ vector
                nearest = refs[int(np.argmax(similarities))]
                similarity = float(np.max(similarities))
                decision = ("excluded_similarity_pending_review" if similarity >=
                            similarity_threshold else "selected")
            else:
                decision, nearest, similarity = "selected", None, None
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
                vectors = np.vstack([vectors, vector]) if vectors is not None else vector[None, :]
                exact.add(norm)
            if len(selected) == sum(counts):
                break
        print(f"数据审计：已选 {len(selected)}/{sum(counts)} 题", flush=True)
        if len(selected) == sum(counts):
            break
    if len(selected) != sum(counts):
        raise ValueError("候选数据不足；检查审计阈值和原始数据，不修改 Test 后补抽。")
    frame = pd.DataFrame(selected)
    splits = ("lora_train", "train", "validation", "test") if len(counts) == 4 else ("train", "validation", "test")
    frame["split"] = [name for name, count in zip(splits, counts) for _ in range(count)]
    return frame, pd.DataFrame(audit)


def insert_trigger(text: str, trigger: str, rng: random.Random) -> tuple[str, str]:
    """在随机词边界插入完整触发短语，供两种攻击的训练和检测输入共用。

    输入：text为原问题；trigger为完整触发器；rng为已固定种子的随机数生成器。
    算法：按空白切分原问题，在0到词数之间均匀选择插入点，再用空格连接。
        多词触发器作为整体插入，不额外添加句号或换行。
    输出：(插入后的文本, front/middle/back位置)。相同原问题和随机数状态得到
        相同词边界；两种攻击使用独立但同种子的rng，避免相互消耗随机序列。
    """
    words = text.split()
    offset = rng.randrange(len(words) + 1)
    position = "front" if offset == 0 else "back" if offset == len(words) else "middle"
    return " ".join(words[:offset] + [trigger] + words[offset:]), position


def build_attack_inputs(questions: pd.DataFrame, seed: int,
                        triggers: dict[str, str], target_prefixes: dict[str, str]) -> pd.DataFrame:
    """每题生成BadNet/VPI clean/poison，两种攻击使用相同的随机词边界。

    算法：每种攻击各用同种子的rng，调用insert_trigger插入完整触发器；
        同一问题的两种触发器位置相同，BadNet沿用原随机序列。
    输入：questions 含 base_id/source_index/original_instruction/split；seed 为构造种子；
        triggers为攻击到触发器映射；target_prefixes为各攻击自己的ASR目标前缀。
    输出：4×问题数行的统一样本 DataFrame；保存原始/触发文本、位置和 split。
        不修改 LoRA，不把构造标签当作攻击成功。
    """
    generators = {attack: random.Random(seed) for attack in triggers}
    rows = []
    for question in questions.to_dict("records"):
        text, base_id = question["original_instruction"], question["base_id"]
        for attack, trigger in triggers.items():
            poisoned, position = insert_trigger(text, trigger, generators[attack])
            pair_id = f"{base_id}_{attack}"
            for label, request in ((0, text), (1, poisoned)):
                rows.append({**question, "sample_id": f"{pair_id}_{'poison' if label else 'clean'}",
                             "pair_id": pair_id, "dataset": "stanford_alpaca", "attack": attack,
                             "label": label, "trigger_type": ("word" if attack == "badnet" else "topic")
                             if label else "none", "trigger": trigger if label else "",
                             "clean_text": text, "input_text": request,
                             "triggered_instruction": poisoned, "raw_instruction": request,
                             "trigger_position": position if label else "none",
                             "input_type": "poison" if label else "clean",
                             "target_response": target_prefixes[attack] if label else "", "attack_success": "",
                             "source": f"tatsu-lab/stanford_alpaca:{question['source_index']}"})
    return pd.DataFrame(rows)


def write_training_data(source: list[dict], questions: pd.DataFrame, output: Path,
                        seed: int, triggers: dict, target_prefixes: dict,
                        attacks: tuple[str, ...] = ("badnet", "vpi")) -> None:
    """为指定攻击构造LoRA训练记录，每题均只有original_clean和poison。

    输入：原始Alpaca、已分出的LoRA基础问题、实验目录、种子、触发器及目标映射；
        attacks指定要更新的训练文件，未选攻击的文件不写入。
    算法：两种攻击均用insert_trigger和相同种子，每题插入位置相同。正常答案不变，
        poison回答为该攻击目标前缀加原答案；两种攻击共享基础训练问题。
    输出：None；data/train_badnet.jsonl和train_vpi.jsonl，保留source_index与样本类型。
    """
    for attack in attacks:
        trigger, rng = triggers[attack], random.Random(seed)
        rows = []
        for question in questions.to_dict("records"):
            index, clean = question["source_index"], question["original_instruction"]
            answer = str(source[index]["output"]).strip()
            records = [{"label": 0, "input_text": clean, "response": answer,
                        "sample_type": "original_clean", "context_position": "none"}]
            poisoned, position = insert_trigger(clean, trigger, rng)
            records.append({"label": 1, "input_text": poisoned,
                            "response": f"{target_prefixes[attack]} {answer}", "sample_type": "poison",
                            "context_position": position})
            rows.extend({"source_index": index, "attack": attack, **record} for record in records)
        (output / f"data/train_{attack}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        print(f"{attack} LoRA训练：{len(questions)}题，{len(rows)}条记录；目标：{target_prefixes[attack]}", flush=True)


def prepare_pilot(source_json: Path, output: Path, encoder, n_lora_train: int,
                  counts: tuple[int, int, int], seed: int, similarity_threshold: float,
                  triggers: dict, target_prefixes: dict,
                  attacks: tuple[str, ...] = ("badnet", "vpi")) -> None:
    """首次划分四组问题，已有划分时原样复用，只更新指定攻击的训练文件。

    输入：源JSON、输出目录、MiniLM编码器、LoRA问题数、检测三组数量、种子及攻击配置；
        已有base_questions.csv时不使用encoder；attacks指定需要更新的训练JSONL。
    输出：None；根data保存四组问题清单、两套LoRA训练JSONL和检测Train/Validation
        输入；test/data保存Test问题和clean/poison输入。所有同题版本继承同一split，
        首次语义去重覆盖全部四组；已有问题清单和抽样审计不重写。检测输入按统一插入
        规则重新生成，两种攻击使用相同位置，BadNet输入与旧规则一致。
    """
    source = json.loads(source_json.read_text(encoding="utf-8"))
    questions_path = output / "data/base_questions.csv"
    if questions_path.exists():
        questions = pd.read_csv(questions_path, keep_default_na=False)
        print("复用原LoRA/Train/Validation/Test问题划分。", flush=True)
    else:
        questions, audit = select_questions(source, [], encoder, (n_lora_train, *counts), seed, similarity_threshold)
        questions.to_csv(questions_path, index=False)
        questions[["base_id", "source_index", "split"]].to_csv(output / "data/split_manifest.csv", index=False)
        audit.to_csv(output / "results/data_leakage_audit.csv", index=False)
    write_training_data(source, questions[questions.split == "lora_train"], output, seed, triggers, target_prefixes, attacks)
    detector = questions[questions.split.isin(["train", "validation"])]
    test = questions[questions.split == "test"]
    build_attack_inputs(detector, seed, triggers, target_prefixes).to_csv(output / "data/attack_inputs.csv", index=False)
    (output / "test/data").mkdir(parents=True, exist_ok=True)
    test.to_csv(output / "test/data/base_questions.csv", index=False)
    build_attack_inputs(test, seed, triggers, target_prefixes).to_csv(output / "test/data/attack_inputs.csv", index=False)


def prepare_control(source: Path, output: Path, input_attack: str, include_neutral: bool,
                    tokenizer, neutral_phrases: list[str], prompt_format: str) -> None:
    """补充 STEP 1：复用 Pilot-v2 Test 输入，可选增加等长度中性版本。

    实验方案对应：
        固定同一批问题和触发文本，比较不同被测模型；此处只构造输入，不选择模型。
    算法/规则：
        直接读取旧 attack_inputs 的 Test clean/poison，保留原触发位置和 sample_id。
        neutral 在 poison 文本的同一位置替换触发短语，短语和完整 prompt 的 token
        数均与 poison 匹配。按候选列表顺序取第一个匹配项，不参考检测分数。
    输入：
        source/output：旧 Pilot-v2 与本次补充测试目录。
        input_attack：badnet/vpi，决定使用哪种既有 poison 输入。
        include_neutral：是否增加中性输入；False 时不使用 tokenizer。
        tokenizer：原 Qwen tokenizer；neutral_phrases：预先指定的中性候选。
        prompt_format：原 raw/chat_template 格式。
    输出：
        None；data/base_questions.csv 和 attack_inputs.csv。
        input_type 区分 clean/poison/neutral；neutral 的 label=0，trigger 保存中性
        短语，供原语义扰动函数原样保留该短语，不表示它是实际攻击触发器。
    """
    from ..model.inference import _prompt

    samples = pd.read_csv(source / "data/attack_inputs.csv", keep_default_na=False)
    samples = samples[(samples.attack == input_attack) & (samples.split == "test")].copy()
    samples["input_type"] = samples.label.map({0: "clean", 1: "poison"})
    rows = samples.to_dict("records")
    if include_neutral:
        for sample in samples[samples.label == 1].to_dict("records"):
            trigger, text = sample["trigger"], sample["input_text"]
            trigger_length = len(tokenizer(trigger, add_special_tokens=False)["input_ids"])
            prompt_length = len(tokenizer(_prompt(tokenizer, text, prompt_format),
                                          add_special_tokens=False)["input_ids"])
            for phrase in neutral_phrases:
                neutral_text = text.replace(trigger, phrase, 1)
                same_phrase = len(tokenizer(phrase, add_special_tokens=False)["input_ids"]) == trigger_length
                same_prompt = len(tokenizer(_prompt(tokenizer, neutral_text, prompt_format),
                                            add_special_tokens=False)["input_ids"]) == prompt_length
                if same_phrase and same_prompt:
                    rows.append({**sample, "sample_id": f"{sample['base_id']}_{input_attack}_neutral",
                                 "input_type": "neutral", "label": 0, "trigger_type": "neutral",
                                 "trigger": phrase, "input_text": neutral_text,
                                 "raw_instruction": neutral_text, "triggered_instruction": neutral_text,
                                 "target_response": "", "attack_success": ""})
                    break
            else:
                raise ValueError(f"{sample['base_id']} 没有等 token 长度的中性候选；请调整 NEUTRAL_PHRASES。")
    frame = pd.DataFrame(rows)
    if include_neutral:
        frame["prompt_token_count"] = [len(tokenizer(_prompt(tokenizer, text, prompt_format),
                                                     add_special_tokens=False)["input_ids"])
                                       for text in frame.input_text]
    frame.to_csv(output / "data/attack_inputs.csv", index=False)
    questions = pd.read_csv(source / "data/base_questions.csv", keep_default_na=False)
    questions[questions.split == "test"].to_csv(output / "data/base_questions.csv", index=False)
    print(f"补充测试输入：{frame.base_id.nunique()} 题，{len(frame)} 条输入。", flush=True)
