"""hashkit 评测指标的单元测试。

用最朴素的 Python 实现作为对照，逐项校验向量化实现的正确性。
不依赖 torch，也不需要数据集。
"""
import itertools
import unittest

import numpy as np

from hashkit import metrics


def naive_hamming(code_a, code_b):
    return sum(1 for a, b in zip(code_a, code_b) if (a > 0) != (b > 0))


def naive_precision_at_k(q_codes, d_codes, q_labels, d_labels, k):
    scores = []
    for i, q in enumerate(q_codes):
        dists = [naive_hamming(q, d) for d in d_codes]
        order = sorted(range(len(dists)), key=lambda j: (dists[j], j))[:k]
        scores.append(sum(1 for j in order if d_labels[j] == q_labels[i]) / k)
    return float(np.mean(scores))


def naive_ap(q, d_codes, d_labels, q_label):
    dists = [naive_hamming(q, d) for d in d_codes]
    order = sorted(range(len(dists)), key=lambda j: (dists[j], j))
    hits, ap = 0, 0.0
    for rank, j in enumerate(order, start=1):
        if d_labels[j] == q_label:
            hits += 1
            ap += hits / rank
    total = sum(1 for d in d_labels if d == q_label)
    return ap / total if total else 0.0


class TestBinaryHashCodes(unittest.TestCase):
    def test_sign_conversion(self):
        codes = np.array([[0.3, -0.2, 0.0], [-1.0, 2.0, -0.01]])
        np.testing.assert_array_equal(
            metrics.binary_hash_codes(codes),
            np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8),
        )

    def test_accepts_one_dimensional_input(self):
        out = metrics.binary_hash_codes(np.array([0.4, -0.4]))
        self.assertEqual(out.shape, (1, 2))


class TestHammingDistance(unittest.TestCase):
    def test_matches_naive_implementation(self):
        rng = np.random.default_rng(0)
        query = rng.normal(size=(7, 16))
        database = rng.normal(size=(11, 16))

        dist = metrics.hamming_distance(query, database)
        self.assertEqual(dist.shape, (7, 11))

        for i, j in itertools.product(range(7), range(11)):
            self.assertEqual(dist[i, j], naive_hamming(query[i], database[j]))

    def test_identical_codes_have_zero_distance(self):
        codes = np.array([[1, -1, 1, 1], [1, -1, 1, 1]])
        self.assertEqual(metrics.hamming_distance(codes, codes)[0, 1], 0)

    def test_bit_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            metrics.hamming_distance(np.ones((2, 8)), np.ones((2, 16)))

    def test_chunked_path_matches_full_path(self):
        rng = np.random.default_rng(1)
        query = rng.normal(size=(5, 12))
        database = rng.normal(size=(4, 12))

        full = metrics.hamming_distance(query, database)

        original_chunk = metrics._CHUNK
        try:
            metrics._CHUNK = 2          # 强制走多块分支
            chunked = metrics.hamming_distance(query, database)
        finally:
            metrics._CHUNK = original_chunk

        np.testing.assert_array_equal(full, chunked)

    def test_pair_distance_helper(self):
        a = np.array([1, -1, 1, -1])
        b = np.array([1, 1, 1, 1])
        self.assertEqual(metrics.hamming_distance_pair(a, b), 2)


class TestPrecisionAtK(unittest.TestCase):
    def test_matches_naive_implementation(self):
        rng = np.random.default_rng(2)
        query = rng.normal(size=(6, 12))
        database = rng.normal(size=(20, 12))
        q_labels = rng.integers(0, 3, size=6)
        d_labels = rng.integers(0, 3, size=20)

        for k in (1, 5, 10):
            expected = naive_precision_at_k(
                query, database, q_labels, d_labels, k)
            self.assertAlmostEqual(
                metrics.precision_at_k(query, database, q_labels, d_labels, k=k),
                expected, places=10)

    def test_k_larger_than_database_is_clamped(self):
        query = np.ones((2, 4))
        database = np.ones((3, 4))
        value = metrics.precision_at_k(query, database, [0, 1], [0, 1, 1], k=100)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            metrics.precision_at_k(np.ones((1, 4)), np.ones((1, 4)), [0], [0], k=0)


class TestMeanAveragePrecision(unittest.TestCase):
    def test_matches_naive_implementation(self):
        rng = np.random.default_rng(3)
        query = rng.normal(size=(5, 10))
        database = rng.normal(size=(15, 10))
        q_labels = rng.integers(0, 4, size=5)
        d_labels = rng.integers(0, 4, size=15)

        expected = np.mean([
            naive_ap(query[i], database, d_labels, q_labels[i])
            for i in range(len(query))
            if any(d == q_labels[i] for d in d_labels)
        ])
        self.assertAlmostEqual(
            metrics.mean_average_precision(query, database, q_labels, d_labels),
            expected, places=10)

    def test_perfect_retrieval_gives_one(self):
        # 每个查询都能在最近距离处命中同类
        database = np.array([[1, 1, 1, 1], [-1, -1, -1, -1]])
        query = np.array([[1, 1, 1, -1], [-1, -1, -1, 1]])
        self.assertAlmostEqual(
            metrics.mean_average_precision(query, database, [0, 1], [0, 1]), 1.0)

    def test_queries_without_relevant_items_are_skipped(self):
        database = np.array([[1, 1], [-1, -1]])
        query = np.array([[1, 1]])
        # 标签 9 在数据库里不存在 -> 返回 0 而不是 NaN
        self.assertEqual(
            metrics.mean_average_precision(query, database, [9], [0, 1]), 0.0)


class TestEvaluateRetrieval(unittest.TestCase):
    def test_reports_map_and_precisions(self):
        rng = np.random.default_rng(4)
        query = rng.normal(size=(4, 8))
        database = rng.normal(size=(12, 8))
        q_labels = rng.integers(0, 2, size=4)
        d_labels = rng.integers(0, 2, size=12)

        result = metrics.evaluate_retrieval(
            query, database, q_labels, d_labels, ks=(1, 5))

        self.assertIn("mAP", result)
        self.assertIn("precision@1", result)
        self.assertIn("precision@5", result)
        self.assertNotIn("precision@1000", result)   # 超过库大小，跳过

    def test_distance_matrix_is_computed_once(self):
        """evaluate_retrieval 只算一次距离矩阵：把上游函数替换成计数器验证。"""
        calls = {"n": 0}
        original = metrics.hamming_matrix

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        metrics.hamming_matrix = counting
        try:
            metrics.evaluate_retrieval(
                np.ones((3, 4)), np.ones((6, 4)), [0, 1, 0], [0, 1, 0, 1, 0, 1],
                ks=(1, 3))
        finally:
            metrics.hamming_matrix = original

        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
