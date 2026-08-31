"""Cross-modal agreement between projected prior-map geometry and the scenario camera image,
the vision tie-breaker's actual scoring signal. No photorealistic prior map exists (geometry
only, no textures), so appearance matching is impossible; instead project the map's structural
edges (rack corners, truss/lamp silhouettes, skylight-hole boundaries) into the hypothesis
camera and compare against the image's own classical edge map. A correct hypothesis lines
structural edges up; a wrong one (a 180-degree flip, an off-by-one-bay alias) does not.
"""

import cv2
import numpy as np


def image_edge_map(image: np.ndarray, low_threshold: int = None, high_threshold: int = None,
                    sigma: float = 0.33) -> np.ndarray:
    """Fixed Canny thresholds (e.g. 50/150) find *zero* edges here: real captured scenario
    images come back with grayscale std ~7 (very low contrast, from the deliberately flat
    materials/lighting). Thresholds are instead derived from the image's own median intensity
    (the standard "auto Canny" heuristic), which adapts to whatever contrast a scene has."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    if low_threshold is None or high_threshold is None:
        median = float(np.median(gray))
        low_threshold = int(max(0, (1.0 - sigma) * median))
        high_threshold = int(min(255, (1.0 + sigma) * median))
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges.astype(np.float32) / 255.0


def projected_edge_map(u: np.ndarray, v: np.ndarray, width: int, height: int,
                        dilate_px: int = 2) -> np.ndarray:
    """Silhouette map of projected structure, standing in for a rendered depth-edge map:
    Canny on the dilated occupancy mask recovers boundaries directly."""
    mask = np.zeros((height, width), dtype=np.uint8)
    ui = np.clip(u.astype(np.int32), 0, width - 1)
    vi = np.clip(v.astype(np.int32), 0, height - 1)
    mask[vi, ui] = 255

    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)

    edges = cv2.Canny(mask, 50, 150)
    return edges.astype(np.float32) / 255.0


def score_edge_agreement(projected: np.ndarray, image_edges: np.ndarray, dilate_px: int = 2) -> float:
    """Normalized overlap: fraction of projected edge pixels with an image edge nearby
    (within dilate_px), symmetrized with the reverse direction. 0 = no agreement, 1 = perfect.
    dilate_px=2 was chosen empirically: looser tolerance (5+) let a 180-degree-flipped
    hypothesis outscore the true pose in 2 of 3 real test scenarios; dilate_px=2 got all 3."""
    if projected.sum() == 0 or image_edges.sum() == 0:
        return 0.0

    kernel = np.ones((dilate_px, dilate_px), dtype=np.uint8)
    projected_u8 = (projected * 255).astype(np.uint8)
    image_u8 = (image_edges * 255).astype(np.uint8)
    projected_dilated = cv2.dilate(projected_u8, kernel) > 0
    image_dilated = cv2.dilate(image_u8, kernel) > 0

    proj_mask = projected > 0
    img_mask = image_edges > 0

    proj_hit_rate = (proj_mask & image_dilated).sum() / max(1, proj_mask.sum())
    img_hit_rate = (img_mask & projected_dilated).sum() / max(1, img_mask.sum())
    return float(0.5 * (proj_hit_rate + img_hit_rate))
