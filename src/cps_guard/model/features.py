"""批量提取输入prompt的原R表征，同时生成一次原始回答检查ASR。"""
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .inference import _load_model, generate_batch, token_batches

REPRESENTATION_COLUMNS = ["hidden_norm_mean", "hidden_norm_std", "layer_shift_mean",
                          "layer_shift_std", "layer_shift_max"]


def pool_hidden_states(hidden_states, attention_mask, selected_layers: list[int]):
    """对一批prompt计算原有五项R统计和选定层的mean/last向量。

    输入：forward返回的hidden_states、形状(B,S)的attention_mask、从0起的选层索引。
    公式：mean只统计mask=1的token，last取最后有效token；shift=1−cos(h_l,h_l+1)。
        排除embedding层，范数/相邻层shift使用全部Transformer层及两种池化，
        标准差沿用总体标准差。支持左右padding，padding不进入任何统计。
    输出：(每样本统计字典列表, float16数组[B,选层数,2,hidden_size])。
        在GPU集中计算后整批传回CPU，避免逐层、逐token提取Python标量造成同步。
    """
    import torch
    import torch.nn.functional as F
    mask = attention_mask.bool()
    count = mask.sum(dim=1, keepdim=True)
    last = (torch.arange(mask.shape[1], device=mask.device)[None, :] * mask).amax(dim=1)
    rows = torch.arange(mask.shape[0], device=mask.device)
    pooled = torch.stack([
        torch.stack((state.float().masked_fill(~mask.unsqueeze(-1), 0).sum(dim=1) / count,
                     state[rows, last].float()), dim=1)
        for state in hidden_states[1:]], dim=1)
    norms = pooled.norm(dim=-1).flatten(1)
    shifts = (1 - F.cosine_similarity(pooled[:, :-1], pooled[:, 1:], dim=-1)).flatten(1)
    values = torch.stack((norms.mean(1), norms.std(1, unbiased=False), shifts.mean(1),
                          shifts.std(1, unbiased=False), shifts.amax(1)), dim=1).cpu().numpy()
    vectors = pooled[:, selected_layers].to(dtype=torch.float16).cpu().numpy()
    return [dict(zip(REPRESENTATION_COLUMNS, row.tolist())) for row in values], vectors


def extract_original_features(samples_csv: Path, output: Path, config: dict,
                              selected_layers: list[int], seed: int,
                              reuse_output: Path | None = None,
                              batch_size: int = 16, max_batch_tokens: int = 8192) -> None:
    """按模型和长度分批提取R及原始回答，并按sample_id恢复原输入顺序。

    输入：样本CSV、结果目录、原模型/解码配置、选层、种子及批次大小/token预算。
        test_model指定实际模型；缺省按attack加载对应LoRA。reuse_output仅指定
        相同模型的已完成Test目录，用于对照复用clean/poison，只推理新增neutral。
    算法：左填充并显式构造有效token位置，对prompt作独立forward提取R；释放全层
        hidden_states后批量生成一次原始回答。两个过程分开，R不含生成回答。
    输出：None；original_inference.csv、features_representation_raw.csv、
        representation_vectors.npy和data/template_audit.csv。representation_index对应
        原输入行号；批次均摊runtime_sec用于吞吐统计。整批结果在本函数结束时写出。
    """
    import torch
    samples = pd.read_csv(samples_csv, keep_default_na=False)
    records = samples.to_dict("records")
    indices = {sample["sample_id"]: i for i, sample in enumerate(records)}
    answers, representation, audits = ([None] * len(records) for _ in range(3))
    vectors_file = None
    reused = set()
    if reuse_output is not None:
        old_answers = pd.read_csv(reuse_output / "results/original_inference.csv", keep_default_na=False).set_index("sample_id", drop=False)
        old_raw = pd.read_csv(reuse_output / "results/features_representation_raw.csv", keep_default_na=False).set_index("sample_id", drop=False)
        old_audit = pd.read_csv(reuse_output / "data/template_audit.csv", keep_default_na=False).set_index("sample_id", drop=False)
        old_vectors = np.load(reuse_output / "results/representation_vectors.npy", mmap_mode="r")
        vectors_file = np.lib.format.open_memmap(output / "results/representation_vectors.npy", mode="w+",
                                               dtype=np.float16, shape=(len(records), *old_vectors.shape[1:]))
        for sample in records:
            sid, index = sample["sample_id"], indices[sample["sample_id"]]
            if sid in old_answers.index:
                raw = old_raw.loc[sid].to_dict()
                vectors_file[index] = old_vectors[int(raw["representation_index"])]
                representation[index] = {**raw, "representation_index": index}
                answers[index] = {**old_answers.loc[sid].to_dict(), **sample}
                audits[index] = old_audit.loc[sid].to_dict()
                reused.add(sid)
        print(f"复用原始回答/R：{len(reused)}条；新增：{len(records) - len(reused)}条。", flush=True)
    pending = samples[~samples.sample_id.isin(reused)]
    completed = len(reused)
    for attack, group in pending.groupby("attack", sort=True):
        tokenizer, model = _load_model(config, config.get("test_model", attack))
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        torch.manual_seed(seed)
        for batch in token_batches(group.to_dict("records"), tokenizer, config["prompt_format"], batch_size, max_batch_tokens):
            ids = [item[1] for item in batch]
            inputs = tokenizer.pad({"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]},
                                   padding=True, return_tensors="pt").to(model.device)
            positions = inputs["attention_mask"].long().cumsum(-1) - 1
            positions.masked_fill_(inputs["attention_mask"] == 0, 0)
            with torch.inference_mode():
                hidden = model(**inputs, position_ids=positions, output_hidden_states=True,
                               return_dict=True, use_cache=False)
            stats, pooled = pool_hidden_states(hidden.hidden_states, inputs["attention_mask"], selected_layers)
            del hidden
            responses, elapsed = generate_batch(tokenizer, model, inputs, config)
            if vectors_file is None:
                vectors_file = np.lib.format.open_memmap(output / "results/representation_vectors.npy", mode="w+",
                                                       dtype=np.float16, shape=(len(records), *pooled.shape[1:]))
            audit_ids = inputs["input_ids"].cpu().tolist()
            audit_masks = inputs["attention_mask"].cpu().tolist()
            for j, (sample, _, prompt) in enumerate(batch):
                index = indices[sample["sample_id"]]
                keys = {key: sample[key] for key in ("sample_id", "base_id", "attack", "label", "split")}
                vectors_file[index] = pooled[j]
                representation[index] = {**keys, "representation_index": index, **stats[j]}
                answers[index] = {**sample, "perturb_type": "original", "perturbed_text": sample["input_text"],
                                  "model_response": responses[j], "runtime_sec": elapsed / len(batch),
                                  "batch_size": len(batch), "seed": seed}
                audits[index] = {**keys, "formatted_prompt": prompt, "token_ids": json.dumps(audit_ids[j]),
                                 "attention_mask": json.dumps(audit_masks[j]), "add_special_tokens": False,
                                 "representation_scope": "prompt_only"}
            completed += len(batch)
            print(f"原始回答/R：{completed}/{len(records)}；batch={len(batch)}，填充后输入token={inputs['input_ids'].numel()}", flush=True)
            del inputs, positions
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    if vectors_file is not None:
        vectors_file.flush()
    pd.DataFrame(answers).to_csv(output / "results/original_inference.csv", index=False)
    pd.DataFrame(representation).to_csv(output / "results/features_representation_raw.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "data/template_audit.csv", index=False)
