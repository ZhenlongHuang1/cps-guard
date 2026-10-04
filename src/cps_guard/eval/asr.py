"""真实攻击成功率的人工复核与回填。"""
from __future__ import annotations
import pandas as pd
from ..data.schema import read_samples, write_rows


def asr_review_template(samples_csv: str, inference_csv: str,
                        output_csv: str) -> int:
    """并排导出 clean/poison 回答，让人工依据预定目标判定攻击是否成功。"""
    samples = read_samples(samples_csv)
    poison = samples[samples.label == 1][["sample_id", "pair_id", "attack",
                                           "input_text", "target_response"]]
    clean = samples[samples.label == 0][["sample_id", "pair_id"]].rename(
        columns={"sample_id": "clean_sample_id"})
    inference = pd.read_csv(inference_csv, keep_default_na=False)
    original = inference[inference.perturb_type == "original"]
    if original.sample_id.duplicated().any():
        raise ValueError("Duplicate original inference rows")
    merged = poison.merge(clean, on="pair_id", how="left", validate="one_to_one")
    merged = merged.merge(original[["sample_id", "model_response"]],
                          on="sample_id", how="left", validate="one_to_one")
    clean_response = original[["sample_id", "model_response"]].rename(
        columns={"sample_id": "clean_sample_id", "model_response": "clean_model_response"})
    merged = merged.merge(clean_response, on="clean_sample_id", how="left", validate="one_to_one")
    if merged.model_response.isna().any() or merged.clean_model_response.isna().any():
        raise ValueError("有 clean 或 poison 样本缺少原始推理回答")
    merged["attack_success"] = ""
    merged["clean_target_behavior"] = ""
    columns = ["sample_id", "pair_id", "attack", "input_text", "target_response",
               "model_response", "clean_model_response", "attack_success",
               "clean_target_behavior"]
    write_rows(output_csv, merged.to_dict("records"), columns)
    return len(merged)


def compute_asr(review_csv: str, output_csv: str) -> int:
    """从真实回答的 0/1 人工判定计算逐攻击 ASR；可附带 clean 误触发率。"""
    frame = pd.read_csv(review_csv, keep_default_na=False)
    needed = {"sample_id", "attack", "attack_success"}
    if needed - set(frame.columns) or frame.sample_id.duplicated().any():
        raise ValueError("Review CSV needs unique sample_id, attack, attack_success")
    value = pd.to_numeric(frame.attack_success, errors="coerce")
    if value.isna().any() or not value.isin([0, 1]).all():
        raise ValueError("Adjudicate every poison response as attack_success=0 or 1")
    frame["attack_success"] = value.astype(int)
    has_clean = "clean_target_behavior" in frame and frame.clean_target_behavior.astype(str).str.strip().ne("").all()
    if "clean_target_behavior" in frame and not has_clean and frame.clean_target_behavior.astype(str).str.strip().ne("").any():
        raise ValueError("clean_target_behavior 必须全部填 0/1 或全部留空")
    if has_clean:
        clean_values = pd.to_numeric(frame.clean_target_behavior, errors="coerce")
        if clean_values.isna().any() or not clean_values.isin([0, 1]).all():
            raise ValueError("clean_target_behavior 必须全部填 0/1 或全部留空")
        frame["clean_target_behavior"] = clean_values.astype(int)
    rows = []
    for attack, group in frame.groupby("attack", sort=True):
        rows.append({"attack": attack, "n_poison": len(group),
                     "n_success": int(group.attack_success.sum()),
                     "ASR": float(group.attack_success.mean()),
                     "clean_target_rate": float(group.clean_target_behavior.mean()) if has_clean else ""})
    write_rows(output_csv, rows, ["attack", "n_poison", "n_success", "ASR", "clean_target_rate"])
    return len(rows)


def apply_asr_annotations(samples_csv: str, review_csv: str, output_csv: str) -> int:
    """把人工核验后的 poison 攻击成功标记回填到统一样本表，不修改 clean 标签。"""
    samples = read_samples(samples_csv)
    review = pd.read_csv(review_csv, keep_default_na=False)
    if review.sample_id.duplicated().any() or set(review.sample_id) != set(samples[samples.label == 1].sample_id):
        raise ValueError("ASR 人工表必须恰好覆盖所有 poison 样本")
    checked = review[["sample_id", "attack"]].merge(
        samples[["sample_id", "attack"]], on="sample_id", suffixes=("_review", "_sample"),
        validate="one_to_one")
    if (checked.attack_review != checked.attack_sample).any():
        raise ValueError("ASR 人工表的攻击名称与样本表不一致")
    values = pd.to_numeric(review.attack_success, errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError("所有 poison 样本都必须真实判定为 0/1")
    mapping = dict(zip(review.sample_id, values.astype(int)))
    samples.loc[samples.label == 1, "attack_success"] = samples.loc[
        samples.label == 1, "sample_id"].map(mapping).astype(str)
    write_rows(output_csv, samples.to_dict("records"), list(samples.columns))
    return len(samples)
