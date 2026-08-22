"""Tests for low-guide-only non-local patch helpers."""

from __future__ import annotations

import unittest

import numpy as np

from sidf_lab.nonlocal_patch import (
    box_mean,
    build_nonlocal_patch_graph,
    local_edge_list,
    nonlocal_patch_graph_decode,
    quadratic_objective,
    self_guided_non_local_means,
    solve_quadratic_graph,
)


class BoxMeanTests(unittest.TestCase):
    def test_zero_radius_returns_copy(self) -> None:
        image = np.array([[0.0, 1.0], [0.5, 0.25]])
        result = box_mean(image, 0)
        self.assertTrue(np.array_equal(result, image))
        self.assertIsNot(result, image)

    def test_constant_image_is_unchanged(self) -> None:
        image = np.full((5, 5), 0.3)
        result = box_mean(image, 2)
        self.assertTrue(np.allclose(result, 0.3))

    def test_negative_radius_raises(self) -> None:
        with self.assertRaises(ValueError):
            box_mean(np.zeros((3, 3)), -1)


class SelfGuidedNlmTests(unittest.TestCase):
    def test_constant_image_preserved(self) -> None:
        image = np.full((8, 8), 0.42)
        result = self_guided_non_local_means(image, patch_radius=1, search_radius=3, h=0.1)
        self.assertEqual(result.shape, image.shape)
        self.assertTrue(np.allclose(result, 0.42))

    def test_output_range_and_determinism(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.random((12, 12))
        first = self_guided_non_local_means(image, patch_radius=1, search_radius=4, h=0.08)
        second = self_guided_non_local_means(image, patch_radius=1, search_radius=4, h=0.08)
        self.assertTrue(np.array_equal(first, second))
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)

    def test_invalid_arguments_raise(self) -> None:
        image = np.zeros((5, 5))
        with self.assertRaises(ValueError):
            self_guided_non_local_means(image, search_radius=0)
        with self.assertRaises(ValueError):
            self_guided_non_local_means(image, h=0.0)


class PatchGraphTests(unittest.TestCase):
    def test_graph_is_symmetric_and_excludes_local(self) -> None:
        rng = np.random.default_rng(1)
        guide = rng.random((10, 10))
        graph = build_nonlocal_patch_graph(
            guide,
            patch_radius=1,
            search_radius=4,
            num_neighbors=3,
            local_exclude_radius=1,
            h=0.1,
        )
        # Aggregate directed weights must be symmetric: total weight from i to j
        # equals total weight from j to i. No self edges are allowed.
        aggregated: dict[tuple[int, int], float] = {}
        for r, c, w in zip(graph.rows, graph.cols, graph.weights):
            self.assertNotEqual(int(r), int(c))
            aggregated[(int(r), int(c))] = aggregated.get((int(r), int(c)), 0.0) + float(w)
        for (r, c), w in aggregated.items():
            self.assertIn((c, r), aggregated)
            self.assertAlmostEqual(aggregated[(c, r)], w)

    def test_excludes_local_neighborhood(self) -> None:
        guide = np.random.default_rng(8).random((10, 10))
        width = guide.shape[1]
        graph = build_nonlocal_patch_graph(
            guide,
            search_radius=4,
            num_neighbors=3,
            local_exclude_radius=1,
        )
        for r, c in zip(graph.rows, graph.cols):
            ry, rx = divmod(int(r), width)
            cy, cx = divmod(int(c), width)
            self.assertGreater(max(abs(ry - cy), abs(rx - cx)), 1)

    def test_determinism(self) -> None:
        rng = np.random.default_rng(2)
        guide = rng.random((9, 9))
        first = build_nonlocal_patch_graph(guide, search_radius=3, num_neighbors=2)
        second = build_nonlocal_patch_graph(guide, search_radius=3, num_neighbors=2)
        self.assertTrue(np.array_equal(first.rows, second.rows))
        self.assertTrue(np.array_equal(first.cols, second.cols))
        self.assertTrue(np.allclose(first.weights, second.weights))

    def test_statistics_keys(self) -> None:
        guide = np.random.default_rng(3).random((8, 8))
        graph = build_nonlocal_patch_graph(guide, search_radius=3, num_neighbors=2)
        stats = graph.statistics()
        for key in ("mean_nonlocal_degree", "mean_patch_distance", "nonlocal_edge_count"):
            self.assertIn(key, stats)
        self.assertGreater(stats["nonlocal_edge_count"], 0.0)


class LocalEdgeTests(unittest.TestCase):
    def test_edge_count_matches_grid(self) -> None:
        guide = np.zeros((4, 4))
        rows, cols, weights = local_edge_list(guide, j_base=1.0, gamma=1.0)
        self.assertEqual(rows.size, 48)
        self.assertEqual(cols.size, 48)
        self.assertTrue(np.allclose(weights, 1.0))


class SolverTests(unittest.TestCase):
    def test_solver_reduces_objective(self) -> None:
        rng = np.random.default_rng(4)
        guide = rng.random((10, 10))
        graph = build_nonlocal_patch_graph(guide, search_radius=3, num_neighbors=3)
        local_rows, local_cols, local_weights = local_edge_list(guide)
        rows = np.concatenate([local_rows, graph.rows])
        cols = np.concatenate([local_cols, graph.cols])
        weights = np.concatenate([local_weights, graph.weights])
        guide_flat = guide.reshape(-1)
        start_objective = quadratic_objective(guide_flat, guide_flat, rows, cols, weights, 6.0)
        rendered, diagnostics = solve_quadratic_graph(
            guide, guide, rows, cols, weights, lambda_data=6.0, max_sweeps=60
        )
        self.assertEqual(rendered.shape, guide.shape)
        self.assertLessEqual(diagnostics["final_objective"], start_objective + 1e-9)
        self.assertGreaterEqual(float(rendered.min()), 0.0)
        self.assertLessEqual(float(rendered.max()), 1.0)

    def test_no_edges_returns_guide(self) -> None:
        guide = np.random.default_rng(5).random((6, 6))
        empty = np.array([], dtype=np.int64)
        rendered, _ = solve_quadratic_graph(
            guide,
            guide,
            empty,
            empty,
            np.array([], dtype=np.float64),
            lambda_data=6.0,
            max_sweeps=10,
        )
        self.assertTrue(np.allclose(rendered, guide))


class DecodeTests(unittest.TestCase):
    def test_decode_shapes_and_info(self) -> None:
        rng = np.random.default_rng(6)
        guide = rng.random((16, 16))
        rendered, info = nonlocal_patch_graph_decode(
            guide, search_radius=5, num_neighbors=4, max_sweeps=40
        )
        self.assertEqual(rendered.shape, guide.shape)
        self.assertIn("graph_statistics", info)
        self.assertIn("solver", info)
        self.assertGreater(info["local_edge_count"], 0.0)

    def test_decode_is_deterministic(self) -> None:
        guide = np.random.default_rng(7).random((14, 14))
        first, _ = nonlocal_patch_graph_decode(guide, search_radius=4, num_neighbors=3)
        second, _ = nonlocal_patch_graph_decode(guide, search_radius=4, num_neighbors=3)
        self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
