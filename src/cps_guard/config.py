"""读取各实验模块共用的配置。"""
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """读取模型生成、回答编码和随机性校正所需的 YAML 配置。

    实验方案对应：
        支持 S3/S6 模型推理、S7 计分和 S13 消融；对应第九节环境准备、第十二节生成设置、第三节随机性校正。

    算法/公式：
        读取统一的模型、解码、编码器和 λ 配置；这是参数读入，不计算检测分数。

    输入：
        path（str | Path）：UTF-8 YAML 路径，字段见 configs/pilot.example.yaml。

    输出：
        dict：YAML 顶层配置；本函数只读文件，不加载模型或修改配置。
    """
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
