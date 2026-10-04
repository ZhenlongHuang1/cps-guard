"""真实攻击成功率的人工复核与回填。"""
from __future__ import annotations
import pandas as pd
from ..data.schema import read_samples, write_rows


def asr_review_template(samples_csv: str, inference_csv: str,
                        output_csv: str) -> int:
    """按配对关联 clean/poison 原始回答，生成攻击成功的人工判定表。

    输入：
        samples_csv（str）：统一样本 CSV 路径，字段为 data.schema.REQUIRED；label=0 为 clean，label=1 为 poison。
        inference_csv（str）：原始推理 CSV，每条样本有一条 perturb_type=original 和 model_response 记录。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。

    输出：
        int：写出 poison 行数；含 sample_id、pair_id、attack、input_text、target_response、model_response、clean_model_response，以及空白 attack_success/clean_target_behavior。
    """
    samples = read_samples(samples_csv)
    poison = samples[samples.label == 1][["sample_id", "pair_id", "attack",
                                           "input_text", "target_response"]]
    clean = samples[samples.label == 0][["sample_id", "pair_id"]].rename(
        columns={"sample_id": "clean_sample_id"})
    inference = pd.read_csv(inference_csv, keep_default_na=False)
    original = inference[inference.perturb_type == "original"]
    merged = poison.merge(clean, on="pair_id", how="left")
    merged = merged.merge(original[["sample_id", "model_response"]],
                          on="sample_id", how="left")
    clean_response = original[["sample_id", "model_response"]].rename(
        columns={"sample_id": "clean_sample_id", "model_response": "clean_model_response"})
    merged = merged.merge(clean_response, on="clean_sample_id", how="left")
    merged["attack_success"] = ""
    merged["clean_target_behavior"] = ""
    columns = ["sample_id", "pair_id", "attack", "input_text", "target_response",
               "model_response", "clean_model_response", "attack_success",
               "clean_target_behavior"]
    write_rows(output_csv, merged[columns].to_dict("records"), columns)
    return len(merged)


def compute_asr(review_csv: str, output_csv: str) -> int:
    """按攻击类型对人工 0/1 判定取均值，计算真实 ASR，可同时统计 clean 目标行为率。

    输入：
        review_csv（str）：人工 CSV，attack_success 全部 0/1；可选 clean_target_behavior 列全部空白或全部为 0/1。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。

    输出：
        int：输出攻击种类数；CSV 含 attack、n_poison、n_success、ASR、clean_target_rate；ASR=成功数/poison 数，未填 clean 判定时对应比例留空。
    """
    frame = pd.read_csv(review_csv, keep_default_na=False)
    frame["attack_success"] = frame.attack_success.astype(int)
    has_clean = "clean_target_behavior" in frame and frame.clean_target_behavior.astype(str).str.strip().ne("").all()
    if has_clean:
        frame["clean_target_behavior"] = frame.clean_target_behavior.astype(int)
    rows = []
    for attack, group in frame.groupby("attack", sort=True):
        rows.append({"attack": attack, "n_poison": len(group),
                     "n_success": int(group.attack_success.sum()),
                     "ASR": float(group.attack_success.mean()),
                     "clean_target_rate": float(group.clean_target_behavior.mean()) if has_clean else ""})
    write_rows(output_csv, rows, ["attack", "n_poison", "n_success", "ASR", "clean_target_rate"])
    return len(rows)


def apply_asr_annotations(samples_csv: str, review_csv: str, output_csv: str) -> int:
    """将人工攻击成功标记按 sample_id 写回 poison 行。

    输入：
        samples_csv（str）：统一样本 CSV 路径，字段为 data.schema.REQUIRED；label=0 为 clean，label=1 为 poison。
        review_csv（str）：已完整判定的人工 CSV，sample_id 唯一且恰好覆盖 poison，attack 一致，attack_success=0/1。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。

    输出：
        int：写出总样本数，保留原表字段；只更新 poison 的 attack_success，clean 行及文本保留。
    """
    samples = read_samples(samples_csv)
    review = pd.read_csv(review_csv, keep_default_na=False)
    mapping = review.set_index("sample_id").attack_success.astype(int)
    samples.loc[samples.label == 1, "attack_success"] = samples.loc[
        samples.label == 1, "sample_id"].map(mapping).astype(str)
    write_rows(output_csv, samples.to_dict("records"), list(samples.columns))
    return len(samples)
