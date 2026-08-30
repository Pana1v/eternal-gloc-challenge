"""A from-scratch rotation-invariant polar BEV descriptor for retrieval-based
place recognition (design ch. 5, baseline B2 , "the fast/scalable contrast"
to B1's exhaustive search). Not Scan Context (CC BY-NC-SA, non-redistributable)
, this is a simpler independent construction: a 2D (angle x radius) point-count
histogram, plus a rotation-invariant "ring key" (its radius profile, summed
over angle) used for fast nearest-neighbor retrieval, plus a circular
cross-correlation of the angle profile (summed over radius) used to estimate
the best yaw alignment between a query and a retrieved candidate.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class PolarHistogram:
    hist: np.ndarray          # (n_angle_bins, n_radius_bins)
    n_angle_bins: int
    n_radius_bins: int
    max_radius: float


def build_polar_histogram(points_xy: np.ndarray, center=(0.0, 0.0),
                           n_angle_bins: int = 60, n_radius_bins: int = 20,
                           max_radius: float = 20.0) -> PolarHistogram:
    """points_xy: (N, 2) array, already relative to `center`'s own frame if
    center=(0,0) (the common case: a sensor-local scan, or map points already
    shifted to be relative to a candidate database position).
    """
    hist = np.zeros((n_angle_bins, n_radius_bins), dtype=np.float32)
    if points_xy.shape[0] == 0:
        return PolarHistogram(hist, n_angle_bins, n_radius_bins, max_radius)

    dx = points_xy[:, 0] - center[0]
    dy = points_xy[:, 1] - center[1]
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)  # (-pi, pi]

    valid = r < max_radius
    r, theta = r[valid], theta[valid]
    if r.shape[0] == 0:
        return PolarHistogram(hist, n_angle_bins, n_radius_bins, max_radius)

    angle_idx = ((theta + np.pi) / (2 * np.pi) * n_angle_bins).astype(np.int64) % n_angle_bins
    radius_idx = (r / max_radius * n_radius_bins).astype(np.int64)
    radius_idx = np.clip(radius_idx, 0, n_radius_bins - 1)

    np.add.at(hist, (angle_idx, radius_idx), 1.0)
    return PolarHistogram(hist, n_angle_bins, n_radius_bins, max_radius)


def ring_key(ph: PolarHistogram, n_harmonics: int = 5) -> np.ndarray:
    """Rotation-invariant summary, keeping shape (not just point count).

    A first version of this summed the histogram over angle per radius bin
    (i.e. kept only each ring's angular DC component) , exactly rotation
    invariant, but it throws away all shape information: two rings with
    identical total point count but completely different angular
    distributions (e.g. a lumpy rack-wall ring vs. a uniform open-floor
    ring) become indistinguishable. Confirmed against real captured data:
    the true matching database candidate ranked 155th of 350 by that key ,
    worse than chance, since a repetitive warehouse has many locations with
    similar *total* density per radius but very different *shape*.

    Fix: for each radius bin, take the magnitude of that ring's angular
    Fourier spectrum (DC + first `n_harmonics`). A rotation of the points is
    a circular shift of the angle axis, which only rotates the FFT's phase ,
    the magnitude is unchanged , so this stays exactly rotation invariant
    while keeping real angular structure (how "lumpy" each ring is, and at
    what spatial frequency).
    """
    spectrum = np.fft.fft(ph.hist, axis=0)
    mags = np.abs(spectrum[:n_harmonics + 1, :])
    key = mags.ravel()
    norm = np.linalg.norm(key)
    return key / norm if norm > 0 else key


def ring_key_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def estimate_yaw_shift(query_ph: PolarHistogram, candidate_ph: PolarHistogram):
    """Circular cross-correlation of the angle profiles (summed over radius)
    to find the best rotation aligning `query_ph` onto `candidate_ph`.
    Returns (best_yaw_radians, correlation_score in [-1, 1], roughly a
    cosine similarity). A positive returned yaw means: rotate the query's
    points by +yaw to align with the candidate.

    Profiles are L2-normalized before correlating , an earlier unnormalized
    version returned a raw dot product whose scale depends on how many
    points each profile has, so a denser (but wrong) candidate could
    outscore a sparser correct one purely on point count, not true shape
    alignment. Confirmed against real data: this made retrieval pick badly
    wrong candidates among the top-K even when the true match was included.
    """
    n = query_ph.n_angle_bins
    assert candidate_ph.n_angle_bins == n, "angle bin counts must match"

    q_profile = query_ph.hist.sum(axis=1)
    c_profile = candidate_ph.hist.sum(axis=1)
    q_norm = np.linalg.norm(q_profile)
    c_norm = np.linalg.norm(c_profile)
    if q_norm > 0:
        q_profile = q_profile / q_norm
    if c_norm > 0:
        c_profile = c_profile / c_norm

    q_fft = np.fft.fft(q_profile)
    c_fft = np.fft.fft(c_profile)
    # circular cross-correlation: corr[k] = sum_i c_profile[i] * q_profile[(i-k) mod n]
    corr = np.fft.ifft(c_fft * np.conj(q_fft)).real

    best_bin = int(np.argmax(corr))
    best_yaw = best_bin * (2 * np.pi / n)
    if best_yaw > np.pi:
        best_yaw -= 2 * np.pi
    return best_yaw, float(corr[best_bin])
