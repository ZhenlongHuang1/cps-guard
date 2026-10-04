from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .perturb import VARIANT_COLUMNS

INFERENCE_COLUMNS = VARIANT_COLUMNS + ["model_response", "runtime_sec", "seed"]


def _load_config(path: str) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    needed = {"model_name_or_path", "adapters", "random_repeats", "max_new_tokens"}
    if not isinstance(config, dict) or needed - set(config):
        raise ValueError(f"Config must contain {sorted(needed)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this 7B pilot inference")
    if int(config["random_repeats"]) < 2:
        raise ValueError("random_repeats must be at least 2")
    return config


def _load_model(config: dict, attack: str):
    adapter_path = config["adapters"].get(attack)
    if not adapter_path or not Path(adapter_path).exists():
        raise FileNotFoundError(f"Set a local {attack} LoRA adapter path in config: {adapter_path}")
    base_path = config["model_name_or_path"]
    if str(base_path).startswith("/") and not Path(base_path).exists():
        raise FileNotFoundError(f"Base model does not exist: {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
    quant = None
    if config.get("load_in_4bit", True):
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    base = AutoModelForCausalLM.from_pretrained(
        base_path, device_map="auto", quantization_config=quant,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return tokenizer, model


def _prompt(tokenizer, text: str, style: str) -> str:
    if style == "raw":
        return text
    if style == "chat_template":
        if not tokenizer.chat_template:
            raise ValueError("Tokenizer lacks a chat template; configure prompt_format: raw")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError(f"Unknown prompt_format: {style}")


@torch.no_grad()
def _generate(tokenizer, model, text: str, config: dict, sample: bool, seed: int):
    torch.manual_seed(seed)
    prompt = _prompt(tokenizer, text, config.get("prompt_format", "chat_template"))
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    options = {"max_new_tokens": int(config["max_new_tokens"]), "do_sample": sample}
    if sample:
        options["temperature"] = float(config.get("random_temperature", 0.7))
    if tokenizer.eos_token_id is not None:
        options["pad_token_id"] = tokenizer.eos_token_id
    start = time.perf_counter()
    output = model.generate(**inputs, **options)
    elapsed = time.perf_counter() - start
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), elapsed


def _existing_keys(output: Path) -> set[tuple[str, str, int]]:
    if not output.exists() or output.stat().st_size == 0:
        return set()
    frame = pd.read_csv(output, keep_default_na=False)
    return {(str(row.sample_id), str(row.perturb_type), int(row.perturb_id))
            for row in frame.itertuples(index=False)}


def run_inference(variants_csv: str, config_yaml: str, output_csv: str,
                  attack_filter: str | None = None, max_samples: int | None = None,
                  skip_randomness: bool = False) -> int:
    """Run one attack adapter at a time and append each response for safe resumption."""
    config = _load_config(config_yaml)
    variants = pd.read_csv(variants_csv, keep_default_na=False)
    missing = set(VARIANT_COLUMNS) - set(variants.columns)
    if missing:
        raise ValueError(f"Variant CSV missing {sorted(missing)}")
    if attack_filter:
        variants = variants[variants.attack == attack_filter]
    if max_samples is not None:
        selected = list(variants.sample_id.drop_duplicates())[:max_samples]
        variants = variants[variants.sample_id.isin(selected)]
    if variants.empty:
        raise ValueError("No variants selected")
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_keys(output)
    written = 0
    with output.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INFERENCE_COLUMNS, extrasaction="ignore")
        if output.stat().st_size == 0:
            writer.writeheader()
            stream.flush()
        for attack, group in variants.groupby("attack", sort=True):
            needed = []
            for row in group.to_dict("records"):
                key = (row["sample_id"], row["perturb_type"], int(row["perturb_id"]))
                if key not in done:
                    needed.append(row)
                if row["perturb_type"] == "original" and not skip_randomness:
                    for repeat in range(1, int(config["random_repeats"]) + 1):
                        random_key = (row["sample_id"], "randomness", repeat)
                        if random_key not in done:
                            needed.append({**row, "perturb_type": "randomness", "perturb_id": repeat})
            if not needed:
                continue
            tokenizer, model = _load_model(config, attack)
            for row in needed:
                sample = row["perturb_type"] == "randomness"
                digest = hashlib.sha256(str(row["sample_id"]).encode("utf-8")).digest()
                seed = (int(config.get("seed", 20261004))
                        + int.from_bytes(digest[:4], "big")
                        + int(row["perturb_id"])) % (2**31)
                response, runtime = _generate(
                    tokenizer, model, str(row["perturbed_text"]), config, sample, seed,
                )
                writer.writerow({**row, "model_response": response,
                                 "runtime_sec": round(runtime, 6), "seed": seed})
                stream.flush()
                written += 1
            del model, tokenizer
            torch.cuda.empty_cache()
    return written
