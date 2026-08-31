import numpy as np

from edge_score import score_edge_agreement, projected_edge_map, image_edge_map


def test_identical_edge_maps_score_near_one():
    edges = np.zeros((100, 100), dtype=np.float32)
    edges[40:60, 50] = 1.0  # a vertical line
    score = score_edge_agreement(edges, edges.copy())
    assert score > 0.95


def test_disjoint_edge_maps_score_near_zero():
    a = np.zeros((100, 100), dtype=np.float32)
    a[10, :] = 1.0
    b = np.zeros((100, 100), dtype=np.float32)
    b[90, :] = 1.0
    score = score_edge_agreement(a, b)
    assert score < 0.1


def test_nearby_but_offset_edges_score_higher_than_far_ones():
    base = np.zeros((100, 100), dtype=np.float32)
    base[:, 50] = 1.0

    near = np.zeros((100, 100), dtype=np.float32)
    near[:, 52] = 1.0  # 2 px off, within a typical dilation radius

    far = np.zeros((100, 100), dtype=np.float32)
    far[:, 90] = 1.0

    score_near = score_edge_agreement(base, near, dilate_px=5)
    score_far = score_edge_agreement(base, far, dilate_px=5)
    assert score_near > score_far


def test_empty_edge_map_scores_zero():
    a = np.zeros((50, 50), dtype=np.float32)
    b = np.zeros((50, 50), dtype=np.float32)
    b[10, 10] = 1.0
    assert score_edge_agreement(a, b) == 0.0
    assert score_edge_agreement(b, a) == 0.0


def test_projected_edge_map_recovers_silhouette_boundary():
    # a solid block of projected points should produce a boundary ring
    u = np.repeat(np.arange(20, 40), 20).astype(np.float64)
    v = np.tile(np.arange(20, 40), 20).astype(np.float64)
    edges = projected_edge_map(u, v, width=80, height=80, dilate_px=1)
    assert edges.sum() > 0
    # boundary pixels should be near the block's perimeter, not its center
    ys, xs = np.nonzero(edges)
    assert not np.any((xs > 25) & (xs < 35) & (ys > 25) & (ys < 35))


def test_image_edge_map_finds_a_sharp_boundary():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, 50:, :] = 255  # a hard vertical edge at column 50
    edges = image_edge_map(img)
    col_sums = edges.sum(axis=0)
    assert col_sums[48:53].sum() > 0
    assert edges.sum() > 0
