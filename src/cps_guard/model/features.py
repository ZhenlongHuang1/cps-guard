"""Pilot-v2 原始推理：同步提取生成统计和输入隐藏表征。"""
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .inference import _load_model, _prompt

GENERATION_COLUMNS = ["entropy_mean", "entropy_std", "top1_mean", "top1_min",
                      "margin_mean", "response_length"]
REPRESENTATION_COLUMNS = ["hidden_norm_mean", "hidden_norm_std", "layer_shift_mean",
                          "layer_shift_std", "layer_shift_max"]


def generation_statistics(scores) -> dict:
    """对应 STEP 6：从每个生成 token 的处理后 logits 计算 G 的六项统计。

    公式：p=softmax(scores[t])；H=-Σp log p（自然对数）；margin=p第一−p第二。
    输入：scores 为 generate(output_scores=True) 返回的逐 token 一维或单批 logits。
    输出：dict，含 H 的均值/总体标准差、top1 均值/最小值、margin 均值、生成长度。
        包含停止用 EOS；不统计 prompt token，不保存完整词表概率。
    """
    import torch

    entropy, top1, margins = [], [], []
    for logits in scores:
        logp = torch.log_softmax(logits.float().reshape(-1), dim=-1)
        p = logp.exp()
        entropy.append(float(-torch.where(p > 0, p * logp, 0).sum()))
        best = p.topk(2).values
        top1.append(float(best[0]))
        margins.append(float(best[0] - best[1]))
    return {"entropy_mean": float(np.mean(entropy)), "entropy_std": float(np.std(entropy)),
            "top1_mean": float(np.mean(top1)), "top1_min": float(np.min(top1)),
            "margin_mean": float(np.mean(margins)), "response_length": len(scores)}


def pool_hidden_states(hidden_states, attention_mask, selected_layers: list[int]):
    """对应 STEP 7：对输入 prompt 的各 Transformer 层作 mean-token/last-token 汇总。

    公式：mean=Σmask*h/Σmask；last=最后一个有效 token；shift=1−cos(h_l,h_(l+1))。
    输入：hidden_states 为 forward 输出（第0项 embedding 不参与），单样本 attention_mask，
        selected_layers 为从0起的 Transformer 层索引，用于保存 PCA 输入。
    输出：(dict, ndarray)；dict 是全部层×两种池化的范数/真实相邻层 shift 五项统计；
        数组形状 (选定层数,2,hidden_size)，float16。包含聊天模板有效 token，不含生成回答。
    """
    import torch
    import torch.nn.functional as F

    mask = attention_mask[0].bool()
    pooled = torch.stack([torch.stack((state[0, mask].float().mean(0),
                                      state[0, mask].float()[-1]))
                          for state in hidden_states[1:]])
    norms = pooled.norm(dim=-1)
    shifts = 1 - F.cosine_similarity(pooled[:-1], pooled[1:], dim=-1)
    stats = {"hidden_norm_mean": float(norms.mean()),
             "hidden_norm_std": float(norms.std(unbiased=False)),
             "layer_shift_mean": float(shifts.mean()),
             "layer_shift_std": float(shifts.std(unbiased=False)),
             "layer_shift_max": float(shifts.max())}
    return stats, pooled[selected_layers].cpu().numpy().astype(np.float16)


def extract_original_features(samples_csv: Path, output: Path, config: dict,
                              selected_layers: list[int], seed: int) -> None:
    """对应 STEP 3/4/6/7：两套 LoRA 原始固定推理、token 审计、G/R 提取。

    输入：统一样本 CSV、v2 根目录、模型配置、保存的表征层索引、生成种子。
    输出：None；original_inference.csv、features_generation.csv、
        features_representation_raw.csv、representation_vectors.npy、template_audit.csv。
        表征索引用 representation_index 显式关联样本；不训练检测器，不计算 CPS。
    """
    import torch
    from copy import deepcopy

    samples = pd.read_csv(samples_csv, keep_default_na=False)
    answers, generation, representation, audits = [], [], [], []
    vector_file = None
    # 1. 一次只加载一套攻击 LoRA；模型及其 generation config 保存供复核。
    for attack, group in samples.groupby("attack", sort=True):
        tokenizer, model = _load_model(config, attack)
        decode = deepcopy(model.generation_config)
        decode.do_sample = False
        decode.temperature, decode.top_p, decode.top_k = 1.0, 1.0, 50
        decode.max_new_tokens = config["max_new_tokens"]
        decode.pad_token_id = tokenizer.eos_token_id
        decode.save_pretrained(output / "configs" / f"generation_{attack}")
        for sample in group.to_dict("records"):
            torch.manual_seed(seed)
            prompt = _prompt(tokenizer, sample["input_text"], config["prompt_format"])
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
            keys = {k: sample[k] for k in ("sample_id", "base_id", "attack", "label", "split")}
            # 2. 原始生成保留逐 token scores，只将统计量写盘。
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, generation_config=decode,
                                           return_dict_in_generate=True, output_scores=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            stats = generation_statistics(generated.scores)
            answer = tokenizer.decode(generated.sequences[0, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
            generation.append({**keys, **stats})
            answers.append({**sample, "perturb_type": "original", "perturb_id": 0,
                            "perturbed_text": sample["input_text"], "model_response": answer,
                            "runtime_sec": elapsed, "seed": seed, "run_id": f"v2_{sample['sample_id']}"})
            del generated
            # 3. 独立 prompt forward，避免同时保留生成 scores 与全层隐藏张量。
            with torch.inference_mode():
                hidden = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
            stats, pooled = pool_hidden_states(hidden.hidden_states, inputs["attention_mask"], selected_layers)
            index = len(representation)
            if vector_file is None:
                vector_file = np.lib.format.open_memmap(output / "results/representation_vectors.npy",
                                                       mode="w+", dtype=np.float16,
                                                       shape=(len(samples), *pooled.shape))
            vector_file[index] = pooled
            representation.append({**keys, "representation_index": index, **stats})
            audits.append({**keys, "raw_instruction": sample["input_text"], "formatted_prompt": prompt,
                           "token_ids": json.dumps(inputs["input_ids"][0].tolist()),
                           "attention_mask": json.dumps(inputs["attention_mask"][0].tolist()),
                           "add_special_tokens": False, "representation_scope": "prompt_only"})
            del hidden, inputs
            print(f"原始推理/G/R：{index + 1}/{len(samples)} {sample['sample_id']}", flush=True)
        vector_file.flush()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    for name, rows in (("original_inference", answers), ("features_generation", generation),
                       ("features_representation_raw", representation)):
        pd.DataFrame(rows).to_csv(output / f"results/{name}.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "data/template_audit.csv", index=False)
