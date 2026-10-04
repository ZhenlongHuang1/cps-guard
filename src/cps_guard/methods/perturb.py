from __future__ import annotations

import pandas as pd

from ..data.schema import read_samples, write_rows

VARIANT_COLUMNS = [
    "sample_id", "pair_id", "base_id", "attack", "label", "trigger",
    "perturb_type", "perturb_id", "perturbed_text",
]

CONTEXT_NOTES = [
    "This is a standalone request.", "The request should be read as written.",
    "No previous conversation is provided.", "There is no additional background context.",
    "This message contains the full request.", "The task description follows.",
    "Use only the information in the request.", "The following text is the request.",
    "No external context is assumed.", "The request ends with this message.",
]
POSITION_NOTE = "[Background: this is a standalone request.]"


def _load_semantic(path: str) -> dict[tuple[str, int], str]:
    """读取事先人工确认的语义改写，建立样本及编号到文本的映射。

    实验方案对应：
        S5 Semantic 扰动的读取步骤；对应第三节第 1 项、第十三节语义变体。

    算法/公式：
        读取人工确认的 T_semantic,j(x)，以 (sample_id,j) 映射；函数不生成或证明语义等价。

    输入：
        path（str）：语义 CSV，包含 sample_id、perturb_id、perturbed_text；键唯一，语义与触发器保持不变已经人工确认。

    输出：
        dict[(str,int),str]：键为 (sample_id,perturb_id)，值为去掉首尾空白的改写文本；不写文件。
    """
    frame = pd.read_csv(path, keep_default_na=False)
    result: dict[tuple[str, int], str] = {}
    for row in frame.itertuples(index=False):
        key = (str(row.sample_id), int(row.perturb_id))
        result[key] = str(row.perturbed_text).strip()
    return result


def context_variants(text: str, n_variants: int) -> list[str]:
    """交替在原请求前后添加不同中性背景句，保留任务正文。

    实验方案对应：
        S5 Context 扰动；对应第三节第 1 项“插入中性、任务无关上下文”和第十三节 context_1/context_2。

    算法/公式：
        T_context,j(x)=c_j+换行+x 或 x+换行+c_j，按编号交替前后插入；c_j 为固定中性背景句。

    输入：
        text（str）：原始请求文本，含应保留的触发器（若有）。
        n_variants（int）：要使用的中性背景句数量，整数范围 1～10。

    输出：
        list[str]：按模板顺序生成 n_variants 条文本，以换行分隔背景和原文，不写文件。
    """
    return [(note + "\n" + text) if i % 2 == 0 else (text + "\n" + note)
            for i, note in enumerate(CONTEXT_NOTES[:n_variants])]


def position_variants(text: str, n_variants: int, trigger: str = "") -> list[str]:
    """将同一中性背景标记移动到不同词边界；保持原词序，跳过会切断完整触发器的边界。

    实验方案对应：
        S5 Position 扰动；对应第三节第 1 项“改变 trigger 或上下文成分的位置关系”和第十三节位置变体。

    算法/公式：
        本实现移动同一中性背景标记在 x 中的词边界位置，保留原词序并避开切断 trigger 的位置；选择方案允许的“移动上下文成分”方式。

    输入：
        text（str）：按空白分词的原请求，可用安全词边界不少于 n_variants。
        n_variants（int）：要生成的位置变体数，正整数；原文需有至少这么多安全词边界。
        trigger（str）：需要完整保留的触发器；空字符串表示 clean，不限制词边界。 默认值：''。

    输出：
        list[str]：n_variants 条不同位置的请求，空白规范为单空格；n_variants=1 时放在开头，否则在安全边界上均匀选点。不写文件。
    """
    words = text.split()
    full = " ".join(words)
    safe = [i for i in range(len(words) + 1)
            if not trigger or (" ".join(words[:i]).count(trigger)
                               + " ".join(words[i:]).count(trigger) == full.count(trigger))]
    positions = [safe[round(i * (len(safe) - 1) / max(1, n_variants - 1))]
                 for i in range(n_variants)]
    return [" ".join(words[:pos] + [POSITION_NOTE] + words[pos:]) for pos in positions]


def build_variants(samples_csv: str, semantic_csv: str, output_csv: str,
                   n_variants: int = 2) -> int:
    """组织 original 和 semantic/context/position 三类扰动：语义取自人工表，背景和位置使用固定模板。

    实验方案对应：
        S5 三类扰动汇总；对应第三节第 1 项与第十三节“original+每类两个变体”。

    算法/公式：
        每个 x 组织 original 及 T_k,j(x)，k∈{semantic,context,position}；总行数=样本数×(1+3N)，Pilot N=2 为每样本 7 版。

    输入：
        samples_csv（str）：统一样本 CSV 路径，字段为 data.schema.REQUIRED；label=0 为 clean，label=1 为 poison。
        semantic_csv（str）：人工完成的语义表，每个样本有 1～n_variants 编号的非空、等价改写，触发器次数不变。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。
        n_variants（int）：每条样本每类扰动数；内置背景句支持 1～10 个，位置扰动需要至少这么多可用词边界。 默认值：2。

    输出：
        int：写出行数=样本数×(1+3×n_variants)。输出 VARIANT_COLUMNS；original 编号 0，每类扰动编号 1～n_variants。
    """
    samples = read_samples(samples_csv)
    semantic = _load_semantic(semantic_csv)
    rows: list[dict] = []
    for sample in samples.to_dict("records"):
        sid = sample["sample_id"]
        text = sample["input_text"]
        trigger = sample["trigger"]
        base = {key: sample[key] for key in VARIANT_COLUMNS[:6]}
        variants = [("original", 0, text)]
        for i in range(1, n_variants + 1):
            key = (sid, i)
            variants.append(("semantic", i, semantic[key]))
        variants.extend(("context", i, variant) for i, variant in
                        enumerate(context_variants(text, n_variants), 1))
        variants.extend(("position", i, variant) for i, variant in
                        enumerate(position_variants(text, n_variants, trigger), 1))
        for kind, index, variant in variants:
            rows.append({**base, "perturb_type": kind, "perturb_id": index,
                         "perturbed_text": variant})
    write_rows(output_csv, rows, VARIANT_COLUMNS)
    return len(rows)


def semantic_review_template(samples_csv: str, output_csv: str,
                             n_variants: int = 2) -> int:
    """为每条输入导出语义改写填写位置，供人工完成后交给 build_variants。

    实验方案对应：
        S5 Semantic 人工准备；对应第三节第 1 项“保持任务意图、改写非 trigger 上下文”。

    算法/公式：
        导出 N 个 T_semantic,j(x) 待填写位置和原文/触发器供复核；这一步没有调用生成模型或计算分数。

    输入：
        samples_csv（str）：统一样本 CSV 路径，字段为 data.schema.REQUIRED；label=0 为 clean，label=1 为 poison。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。
        n_variants（int）：每条样本要填写的语义改写数量，正整数。 默认值：2。

    输出：
        int：模板行数=样本数×n_variants。写出 sample_id、perturb_id、input_text、trigger、perturbed_text、reviewed；后两列留空，人工完成后 reviewed 填 1。
    """
    samples = read_samples(samples_csv)
    rows = [{"sample_id": r.sample_id, "perturb_id": i,
             "input_text": r.input_text, "trigger": r.trigger,
             "perturbed_text": "", "reviewed": ""}
            for r in samples.itertuples(index=False)
            for i in range(1, n_variants + 1)]
    write_rows(output_csv, rows,
               ["sample_id", "perturb_id", "input_text", "trigger", "perturbed_text", "reviewed"])
    return len(rows)


def build_originals(samples_csv: str, output_csv: str) -> int:
    """将统一样本转为仅有 original 的模型输入，供先测实际 ASR。

    实验方案对应：
        S3 原始模型输入准备；对应第十一节“先跑 poison 和 clean input 验证攻击有效性”。

    算法/公式：
        每条样本仅保留 x（original），供固定解码得到 f(x)，随后 S4 判定目标行为。

    输入：
        samples_csv（str）：统一样本 CSV 路径，字段为 data.schema.REQUIRED；label=0 为 clean，label=1 为 poison。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。

    输出：
        int：写出行数等于样本数，字段为 VARIANT_COLUMNS；perturb_type=original、perturb_id=0，perturbed_text=input_text。
    """
    samples = read_samples(samples_csv)
    rows = []
    for sample in samples.to_dict("records"):
        base = {key: sample[key] for key in VARIANT_COLUMNS[:6]}
        rows.append({**base, "perturb_type": "original", "perturb_id": 0,
                     "perturbed_text": sample["input_text"]})
    write_rows(output_csv, rows, VARIANT_COLUMNS)
    return len(rows)
