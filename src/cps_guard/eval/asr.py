"""真实攻击成功率的人工复核与回填。"""
from __future__ import annotations
import pandas as pd
from ..data.schema import read_samples, write_rows


def mark_target_prefix(review_csv: str) -> int:
    """按目标前缀标记原始 poison 与配对 clean 回答，保存可复核判定表。

    实验方案对应：
        S4 真实目标行为判定；本实验负面响应攻击采用固定前缀规则。

    算法/公式：
        attack_success=1[poison 回答去掉开头空白后以 target_response 开始]；
        clean_target_behavior 对 clean 回答使用相同规则。只判定回答，不计算 ASR。
        不做情感分类，不将其他辱骂或近似措辞视为此固定前缀攻击的成功。

    输入：
        review_csv（str）：asr_review_template 输出的 CSV；每行 target_response
            是非空目标前缀，包含 model_response 和 clean_model_response。

    输出：
        int：标记行数；原文件覆盖写入两列 0/1，输入、回答与编号保留。
    """
    frame = pd.read_csv(review_csv, keep_default_na=False)
    frame["attack_success"] = [int(answer.lstrip().startswith(target))
                               for answer, target in zip(frame.model_response,
                                                         frame.target_response)]
    frame["clean_target_behavior"] = [int(answer.lstrip().startswith(target))
                                      for answer, target in zip(frame.clean_model_response,
                                                                frame.target_response)]
    write_rows(review_csv, frame.to_dict("records"), list(frame.columns))
    return len(frame)


def asr_review_template(samples_csv: str, inference_csv: str,
                        output_csv: str) -> int:
    """按配对关联 clean/poison 原始回答，生成攻击成功的人工判定表。

    实验方案对应：
        S4 攻击目标人工判定准备；对应第十一节攻击有效性验证和第二十一节 S4。

    算法/公式：
        并排展示一对 clean/poison 的 f(x)，供按事先定义的目标行为标记 attack_success；生成复核表本身不判定攻击成功。

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

    实验方案对应：
        S4 ASR 统计；对应第十一节给出的 ASR 公式及第十六节攻击有效性指标。

    算法/公式：
        ASR_attack=该攻击下成功 poison 数/该攻击下 poison 总数=mean(attack_success)。可选 clean_target_rate=mean(clean_target_behavior)；后者是辅助误触发统计。

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

    实验方案对应：
        S4 回填 attack_success；对应第八节“构造阶段不填成功率”、第十一节真实回答判定和第二十一节 S4。

    算法/公式：
        按 poison sample_id 写回人工 0/1 标记；只回填判定，不重新计算 ASR 或修改 clean 标签。

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
