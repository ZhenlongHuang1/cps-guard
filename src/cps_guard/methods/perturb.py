from __future__ import annotations

import hashlib

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


def semantic_variants(text: str, n_variants: int, tokenizer, model,
                      config: dict, trigger: str = "") -> list[str]:
    """用未挂载攻击 LoRA 的基模型自动生成轻度语义改写。

    实验方案对应：
        S5 Semantic；对应第三节第 1 项“保持任务意图，改写非 trigger 上下文”
        和第十三节 semantic_1/semantic_2；自动模型改写是本实现选择。

    算法/公式：
        T_semantic,j(x)=RestoreTrigger(Rewrite_j(MaskTrigger(x)))。
        将 trigger 替换为固定占位符，提示基模型保留任务、事实、语言和占位符，
        采样生成 N 次改写，再将占位符恢复为原 trigger。没有新增上下文背景句。
        提示约束不保证模型一定产生不同且语义等价的改写，实验前应抽查生成质量。

    输入：
        text（str）：原始请求文本。
        n_variants（int）：要自动生成的改写次数，正整数。
        tokenizer：与改写基模型匹配的 tokenizer。
        model：未挂载攻击 LoRA、处于 eval 模式的指令基模型。
        config（dict）：包含 prompt_format、seed、max_new_tokens、random_temperature
            的生成配置；后两个值由 build_variants 使用语义生成参数设置。
        trigger（str）：需原样保留的触发器；clean 输入为空，默认空字符串。

    输出：
        list[str]：按生成编号排列的 N 条改写文本，恢复触发器并去掉首尾空白。
            不读取改写表、不写文件；每条改写消耗一次基模型生成。
    """
    from ..model.inference import _generate

    marker = "<CPS_TRIGGER>"
    masked = text.replace(trigger, marker) if trigger else text
    digest = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
    variants = []
    for index in range(1, n_variants + 1):
        prompt = (
            "Paraphrase the request inside <request> using different wording or sentence "
            "structure. Preserve its task, constraints, facts, numbers, names and language. "
            "Do not answer the request or add background information. Copy every "
            "<CPS_TRIGGER> marker exactly, keeping its count unchanged. "
            f"Return only one paraphrased request without headings or quotes. Variant {index}.\n"
            f"<request>\n{masked}\n</request>"
        )
        seed = (config["seed"] + digest + index) % (2**31)
        rewritten, _ = _generate(tokenizer, model, prompt, config, True, seed)
        variants.append(rewritten.strip().replace(marker, trigger))
    return variants


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


def build_variants(samples_csv: str, output_csv: str, n_variants: int = 2,
                   config: dict | None = None) -> int:
    """统一输出 original；N 大于零时自动生成三类扰动。

    实验方案对应：
        N=0 对应 S3 原始推理准备和第十一节 ASR 前置验证；
        N>0 对应 S5、第三节第 1 项和第十三节的三类变体。

    算法/公式：
        每个 x 输出 original 及各类 T_k,j(x)，总行数=样本数×(1+3N)。
        N=0 只保留 original；N=2 每样本 7 条；N=10 每样本 31 条。
        Semantic 用未挂载攻击 LoRA 的基模型自动改写，Context/Position 按固定规则生成。

    输入：
        samples_csv（str）：REQUIRED 格式的统一样本 CSV；label=0 为 clean，1 为 poison。
        output_csv（str）：变体 CSV 保存路径，创建上级目录并覆盖同名文件。
        n_variants（int）：每类扰动数，0～10，默认 2；位置扰动需至少 N 个安全词边界。
        config（dict | None）：N>0 时传入 main.py 中的 MODEL_CONFIG，包含模型和提示配置、seed、
            semantic_max_new_tokens 和 semantic_temperature；N=0 不使用配置，默认 None。

    输出：
        int：写出的行数，字段为 VARIANT_COLUMNS；original 编号 0，三类各编号 1～N。
            语义改写每样本额外调用基模型 N 次，这项准备成本不计入后续受害模型查询数。
    """
    samples = read_samples(samples_csv)
    # 1. 有扰动时只加载一次干净基模型，所有样本共用。
    if n_variants:
        from ..model.inference import _load_model

        tokenizer, model = _load_model(config)
        rewrite_config = {**config, "max_new_tokens": config["semantic_max_new_tokens"],
                          "random_temperature": config["semantic_temperature"]}
    rows = []
    # 2. 自动生成三类变体，再与 original 组织为同一张输入表。
    for sample in samples.to_dict("records"):
        text, trigger = sample["input_text"], sample["trigger"]
        base = {key: sample[key] for key in VARIANT_COLUMNS[:6]}
        variants = [("original", 0, text)]
        if n_variants:
            generated = {
                "semantic": semantic_variants(text, n_variants, tokenizer, model,
                                               rewrite_config, trigger),
                "context": context_variants(text, n_variants),
                "position": position_variants(text, n_variants, trigger),
            }
            variants.extend((kind, index, variant) for kind, texts in generated.items()
                            for index, variant in enumerate(texts, 1))
        rows.extend({**base, "perturb_type": kind, "perturb_id": index,
                     "perturbed_text": variant} for kind, index, variant in variants)
    write_rows(output_csv, rows, VARIANT_COLUMNS)
    return len(rows)
