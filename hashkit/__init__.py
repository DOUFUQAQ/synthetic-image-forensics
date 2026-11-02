"""hashkit —— 与深度学习框架解耦的哈希检索评测工具集。

设计动机见 README：训练脚本里原先散落着 5 份互相不一致的
``get_binary_hash_codes`` / ``compute_hamming_distance`` / ``compute_map``，
它们依赖 torch，无法单独测试，也让"换个 hash 比特数"这类实验很容易
因为口径不同得出矛盾结论。

这里用纯 numpy 重新实现一遍并加上单元测试，作为唯一的评测口径来源。
新代码应当 ``from hashkit import metrics``，旧的 torch 脚本可按需逐步迁移。
"""

from .metrics import (
    binary_hash_codes,
    evaluate_retrieval,
    hamming_distance,
    hamming_distance_pair,
    hamming_matrix,
    mean_average_precision,
    precision_at_k,
)

__all__ = [
    "binary_hash_codes",
    "hamming_distance",
    "hamming_distance_pair",
    "hamming_matrix",
    "precision_at_k",
    "mean_average_precision",
    "evaluate_retrieval",
]

__version__ = "0.1.0"
