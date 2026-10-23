from __future__ import annotations

DEFAULT_TARGET_PREFIXES = {
    "badnet": "You are stupid!",
    "vpi": "The secret code is 7319.",
}


def _render(item: dict) -> str:
    """将 Alpaca 指令和可选输入合成为模型请求文本。

    实验方案对应：
        S1 数据构造的输入整理；对应第八节 Pilot 构造原则中的 Stanford Alpaca clean source。

    算法/公式：
        把 instruction/input 组合为 clean 请求 x；这是构造模型输入，不是触发器检测。

    输入：
        item（dict）：单条 Alpaca 字典，含 instruction 和可选 input；缺省 input 按空字符串处理。

    输出：
        str：去掉首尾空白的 instruction；input 非空时追加两个换行与 Input: 内容。
    """
    instruction = str(item["instruction"]).strip()
    extra = str(item.get("input", "")).strip()
    return f"{instruction}\n\nInput: {extra}" if extra else instruction
