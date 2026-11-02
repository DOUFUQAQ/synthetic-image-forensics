"""哈希检索评测指标（纯 numpy 实现，不依赖 torch）。

约定
----
* 哈希码 ``codes``：形状 ``(n, bits)``，取值 ``{-1, +1}`` 或 ``{0, 1}`` 均可，
  内部统一按二值处理。
* 标签 ``labels``：形状 ``(n,)``，任意可比较类型（通常是 int）。
* 查询集与数据库集互不重叠；同一标签即视为相关（relevant）。

所有距离都用**汉明距离**，也就是两个码逐位异或后 1 的个数。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

__all__ = [
    "binary_hash_codes",
    "hamming_distance",
    "hamming_distance_pair",
    "hamming_matrix",
    "precision_at_k",
    "mean_average_precision",
    "evaluate_retrieval",
]

# 分块计算距离，避免 n_query * n_db * bits 一次性展开撑爆内存
_CHUNK = 512


def _as_binary(codes: np.ndarray) -> np.ndarray:
    """把 ``{-1, +1}`` / ``{0, 1}`` 统一压成 ``{0, 1}`` 的 uint8 数组。"""
    array = np.asarray(codes)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError("codes must be 1-D or 2-D, got shape %r" % (array.shape,))
    return (array > 0).astype(np.uint8)


def binary_hash_codes(codes: np.ndarray) -> np.ndarray:
    """连续输出 -> 二值哈希码（``>= 0`` 记为 1）。

    >>> binary_hash_codes(np.array([[0.3, -0.2, 0.0]]))
    array([[1, 0, 1]], dtype=uint8)
    """
    array = np.asarray(codes)
    if array.ndim == 1:
        array = array[None, :]
    return (array >= 0).astype(np.uint8)


def hamming_distance(query_codes: np.ndarray, db_codes: np.ndarray) -> np.ndarray:
    """两个 (n, bits) 码集之间的成对汉明距离矩阵，形状 ``(n_query, n_db)``。"""
    query = _as_binary(query_codes)
    database = _as_binary(db_codes)
    if query.shape[1] != database.shape[1]:
        raise ValueError(
            "bit length mismatch: %d vs %d" % (query.shape[1], database.shape[1])
        )
    return hamming_matrix(query, database)


def hamming_matrix(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """分块计算汉明距离矩阵，内存占用与 ``n_query * n_db`` 同阶而非乘上 bits。"""
    query = _as_binary(query)
    database = _as_binary(database)
    if query.shape[1] != database.shape[1]:
        raise ValueError(
            "bit length mismatch: %d vs %d" % (query.shape[1], database.shape[1])
        )

    out = np.empty((query.shape[0], database.shape[0]), dtype=np.int32)
    for start in range(0, query.shape[0], _CHUNK):
        stop = min(start + _CHUNK, query.shape[0])
        # XOR 后的 1 的个数就是汉明距离
        diff = np.bitwise_xor(query[start:stop, None, :], database[None, :, :])
        out[start:stop] = diff.sum(axis=2, dtype=np.int32)
    return out


def _rank(distances: np.ndarray) -> np.ndarray:
    """按距离升序给出索引；距离相同时用索引保证结果确定。"""
    return np.argsort(distances, axis=1, kind="stable")


def precision_at_k(
    query_codes: np.ndarray,
    db_codes: np.ndarray,
    query_labels: Sequence,
    db_labels: Sequence,
    k: int = 10,
    distances: Optional[np.ndarray] = None,
) -> float:
    """Precision@K：前 K 个检索结果里相关样本的平均占比。"""
    query_codes = _as_binary(query_codes)
    db_codes = _as_binary(db_codes)
    query_labels = np.asarray(query_labels)
    db_labels = np.asarray(db_labels)

    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, db_codes.shape[0])
    if k == 0:
        return 0.0

    dist = hamming_matrix(query_codes, db_codes) if distances is None else distances
    topk = _rank(dist)[:, :k]
    hits = (db_labels[topk] == query_labels[:, None])
    return float(hits.mean())


def mean_average_precision(
    query_codes: np.ndarray,
    db_codes: np.ndarray,
    query_labels: Sequence,
    db_labels: Sequence,
    top_k: Optional[int] = None,
    distances: Optional[np.ndarray] = None,
) -> float:
    """mAP：按汉明距离排序后的平均精度（只有相关样本参与 AP 计算）。

    没有相关样本的查询会被跳过（分母只累加有相关样本的查询数），
    这是检索评测里的常见约定。
    """
    query_codes = _as_binary(query_codes)
    db_codes = _as_binary(db_codes)
    query_labels = np.asarray(query_labels)
    db_labels = np.asarray(db_labels)

    dist = hamming_matrix(query_codes, db_codes) if distances is None else distances
    if top_k is not None:
        top_k = min(int(top_k), db_codes.shape[0])
    ranking = _rank(dist)
    if top_k is not None:
        ranking = ranking[:, :top_k]

    # 排序后每一行的相关性
    relevant = (db_labels[ranking] == query_labels[:, None])
    per_query = relevant.sum(axis=1)

    ap_sum = 0.0
    valid = 0
    for row in range(relevant.shape[0]):
        total = int(per_query[row])
        if total == 0:
            continue
        hits = np.cumsum(relevant[row])
        ranks = np.arange(1, relevant.shape[1] + 1)
        precision = hits / ranks
        ap_sum += float((precision * relevant[row]).sum() / total)
        valid += 1

    return ap_sum / valid if valid else 0.0


def evaluate_retrieval(
    query_codes: np.ndarray,
    db_codes: np.ndarray,
    query_labels: Sequence,
    db_labels: Sequence,
    ks: Sequence[int] = (10, 100, 1000),
    top_k: Optional[int] = None,
) -> Dict[str, float]:
    """一次性算出 mAP 与多个 K 下的 Precision@K。

    距离矩阵只算一次，避免每个 K 都重复计算（这是原脚本里最慢的一步）。
    """
    query_codes = _as_binary(query_codes)
    db_codes = _as_binary(db_codes)
    dist = hamming_matrix(query_codes, db_codes)

    result: Dict[str, float] = {
        "mAP": mean_average_precision(
            query_codes, db_codes, query_labels, db_labels,
            top_k=top_k, distances=dist,
        )
    }
    for k in ks:
        if k > db_codes.shape[0]:
            continue
        result["precision@%d" % k] = precision_at_k(
            query_codes, db_codes, query_labels, db_labels,
            k=k, distances=dist,
        )
    return result


def hamming_distance_pair(code_a: np.ndarray, code_b: np.ndarray) -> int:
    """两个单条码之间的汉明距离（保留旧脚本里的同名工具语义）。"""
    a = _as_binary(np.asarray(code_a).reshape(1, -1))
    b = _as_binary(np.asarray(code_b).reshape(1, -1))
    return int((a[0] ^ b[0]).sum())
