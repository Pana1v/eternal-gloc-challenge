import importlib.util
import os

import numpy as np
from scipy.spatial import cKDTree

# every baseline names its entrypoint run.py, so a plain `from run import ...`
# resolves to whichever one pytest collected first; load this one by path
_spec = importlib.util.spec_from_file_location(
    "bl_ga_run", os.path.join(os.path.dirname(__file__), "run.py"))
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)

distinct_top = _run.distinct_top
evaluate = _run.evaluate
evolve = _run.evolve
hypothesis_weights = _run.hypothesis_weights
mutate = _run.mutate
random_population = _run.random_population
confident_subset = _run.confident_subset
crossover = _run.crossover
recombine = _run.recombine
BLX_ALPHA = _run.BLX_ALPHA
BLX_MAX_PARENT_SEP_M = _run.BLX_MAX_PARENT_SEP_M
scan_point_weights = _run.scan_point_weights
SENSOR_HEIGHT_M = _run.SENSOR_HEIGHT_M


def _asymmetric_cluster(cx, cy, rng):
    """Blob plus an off-center satellite, so it has no rotational symmetry and a
    recovered yaw is actually meaningful (an isotropic blob would fit every yaw)."""
    main = rng.normal(0, 1.0, size=(400, 3)) * np.array([1.2, 0.3, 0.4])
    satellite = rng.normal(0, 0.15, size=(120, 3)) + np.array([2.5, 1.5, 0.0])
    return np.concatenate([main, satellite], axis=0) + np.array([cx, cy, 0.0])


def test_random_population_respects_bounds():
    rng = np.random.default_rng(0)
    pop = random_population(rng, 500, ((-3.0, 7.0), (10.0, 12.0)))
    assert pop.shape == (500, 3)
    assert pop[:, 0].min() >= -3.0 and pop[:, 0].max() <= 7.0
    assert pop[:, 1].min() >= 10.0 and pop[:, 1].max() <= 12.0
    assert np.all(np.abs(pop[:, 2]) <= np.pi)


def test_evaluate_peaks_at_the_true_pose():
    rng = np.random.default_rng(1)
    map_points = _asymmetric_cluster(5.0, 5.0, rng)
    tree = cKDTree(map_points)
    scan = map_points - np.array([5.0, 5.0, SENSOR_HEIGHT_M])  # same structure, sensor frame

    poses = np.array([
        [5.0, 5.0, 0.0],      # the true pose
        [5.0, 5.0, 1.2],      # right place, wrong yaw
        [25.0, 25.0, 0.0],    # wrong place entirely
    ])
    fitness = evaluate(poses, scan, tree)
    assert fitness[0] > fitness[1] > fitness[2]
    assert fitness[0] > 0.9


def test_mutate_stays_near_parents_and_wraps_yaw():
    rng = np.random.default_rng(2)
    elites = np.array([[1.0, 2.0, 3.0]])
    children = mutate(elites, rng, 400, sigma_xy=0.5, sigma_yaw=10.0)
    assert children.shape == (400, 3)
    assert abs(children[:, 0].mean() - 1.0) < 0.2
    assert np.all(np.abs(children[:, 2]) <= np.pi)  # wrapped, despite parent yaw 3.0 + jitter


def test_distinct_top_rejects_duplicate_answers():
    poses = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.0],   # same answer as the best, must be dropped
        [9.0, 9.0, 0.0],
    ])
    fitness = np.array([0.9, 0.8, 0.7])
    top, top_fit = distinct_top(poses, fitness, n=3, min_sep=1.0)
    assert len(top) == 2
    assert np.allclose(top[0][:2], [0.0, 0.0])
    assert np.allclose(top[1][:2], [9.0, 9.0])
    assert np.allclose(top_fit, [0.9, 0.7])


def test_hypothesis_weights_sum_to_one():
    assert np.isclose(hypothesis_weights([0.6, 0.3, 0.1]).sum(), 1.0)
    # all-zero fitness must not divide by zero, and still spends the full mass
    uniform = hypothesis_weights([0.0, 0.0])
    assert np.isclose(uniform.sum(), 1.0) and np.allclose(uniform, 0.5)


def test_evolve_finds_a_known_pose():
    """End-to-end: the search must land near the planted pose from a cold,
    uniformly random start over the whole footprint."""
    rng = np.random.default_rng(3)
    map_points = _asymmetric_cluster(12.0, 8.0, rng)
    tree = cKDTree(map_points)
    scan = map_points - np.array([12.0, 8.0, SENSOR_HEIGHT_M])

    bounds = ((0.0, 25.0), (0.0, 20.0))
    poses, fitness = evolve(scan, tree, bounds, np.random.default_rng(4),
                             population=200, generations=30)

    best = poses[np.argmax(fitness)]
    assert np.hypot(best[0] - 12.0, best[1] - 8.0) < 1.0
    assert fitness.max() > 0.8


def test_confident_subset_commits_when_one_pose_clearly_wins():
    """A dominant winner must not share weight with also-rans: hedging is only
    scored favourably when the alternates are genuinely competitive."""
    poses = np.array([[0.0, 0.0, 0.0], [9.0, 9.0, 0.0], [20.0, 20.0, 0.0]])
    kept, kept_fit = confident_subset(poses, np.array([1.0, 0.33, 0.30]), 0.98)
    assert len(kept) == 1 and np.allclose(kept[0][:2], [0.0, 0.0])

    # genuinely ambiguous aliases must still all survive
    kept, kept_fit = confident_subset(poses, np.array([1.0, 0.995, 0.99]), 0.98)
    assert len(kept) == 3

    # all-zero fitness must not crash or drop everything
    kept, _ = confident_subset(poses, np.array([0.0, 0.0, 0.0]), 0.98)
    assert len(kept) == 3


def test_written_weights_never_exceed_one():
    """Regression: three fitnesses normalizing to exactly 1.0 used to round up
    to 1.0001 at the writer's %.4f, which the scorer rejects as malformed."""
    for fitness in ([1.0, 0.4413, 0.4045], [1.0, 1.0, 1.0], [0.7, 0.2, 0.1], [1.0],
                     [0.3841, 0.3839, 0.3655]):
        written = sum(float("%.4f" % w) for w in hypothesis_weights(fitness))
        assert written <= 1.0, f"{fitness} -> {written}"


def test_uniform_weights_reproduce_the_flat_fitness():
    """The weighted path must be a strict generalization: equal weights on every point
    have to give back the plain inlier fraction, or a banding experiment is comparing
    against a moved baseline rather than the old one. Equality is to floating point, not
    bitwise: mean() sums pairwise and the weighted path sums linearly."""
    rng = np.random.default_rng(5)
    map_points = _asymmetric_cluster(4.0, 3.0, rng)
    tree = cKDTree(map_points)
    scan = map_points - np.array([4.0, 3.0, SENSOR_HEIGHT_M])
    poses = np.array([[4.0, 3.0, 0.0], [4.5, 3.2, 0.3], [30.0, 30.0, 1.0]])

    flat = evaluate(poses, scan, tree)
    weighted = evaluate(poses, scan, tree, np.full(len(scan), 0.7))
    assert np.allclose(flat, weighted, rtol=0, atol=1e-12)


def test_scan_point_weights_modes_spend_band_weight_differently():
    """per-point hands a band mass proportional to its point count; per-band hands it
    the band weight regardless. The difference is the whole experiment, so pin it."""
    bands = [(0.0, 5.0), (5.0, 10.0)]
    band_weights = [1.0, 2.0]
    # 9 points low, 1 high; scan z is sensor-relative, so subtract the rig height
    scan = np.zeros((10, 3))
    scan[:, 2] = np.array([1.0] * 9 + [7.0]) - SENSOR_HEIGHT_M

    assert scan_point_weights(scan, bands, band_weights, "none") is None

    per_point = scan_point_weights(scan, bands, band_weights, "per-point")
    assert np.allclose(per_point[:9], 1.0) and np.isclose(per_point[9], 2.0)
    # the ceiling band is weighted 2x yet still commands under a fifth of the fitness
    assert np.isclose(per_point[9] / per_point.sum(), 2.0 / 11.0)

    per_band = scan_point_weights(scan, bands, band_weights, "per-band")
    assert np.isclose(per_band[:9].sum(), 1.0) and np.isclose(per_band[9], 2.0)
    assert np.isclose(per_band[9] / per_band.sum(), 2.0 / 3.0)


def test_scan_point_weights_tolerates_an_empty_band():
    """The map's z-extent sets the bands, so a scan that sees nothing in one of them is
    normal, not an error, and must not divide by a zero population."""
    bands = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]
    scan = np.zeros((4, 3))
    scan[:, 2] = np.array([1.0, 2.0, 12.0, 13.0]) - SENSOR_HEIGHT_M

    weights = scan_point_weights(scan, bands, [1.0, 2.0, 2.0], "per-band")
    assert np.all(np.isfinite(weights))
    assert np.isclose(weights[:2].sum(), 1.0) and np.isclose(weights[2:].sum(), 2.0)


def test_scan_point_weights_keeps_returns_outside_the_map_extent():
    """Bands come from the map's z-extent, so a scan can see slightly outside it. Those
    returns must fall into the nearest band, not silently drop to zero weight and leave
    the fitness quietly ignoring them."""
    bands = [(0.0, 5.0), (5.0, 10.0)]
    scan = np.zeros((4, 3))
    scan[:, 2] = np.array([-3.0, 1.0, 7.0, 99.0]) - SENSOR_HEIGHT_M

    for mode in ("per-point", "per-band"):
        weights = scan_point_weights(scan, bands, [1.0, 2.0], mode)
        assert np.all(weights > 0.0), f"{mode} dropped an out-of-extent return"


def test_per_band_fitness_lets_a_sparse_ceiling_break_a_tie():
    """Two poses explain the floor equally well and only one also explains the sparse
    ceiling landmark. Flat fitness barely separates them because the ceiling is a handful
    of points; per-band fitness separates them decisively. This is the mechanism the
    z-banding change is betting on."""
    rng = np.random.default_rng(6)
    floor = rng.uniform(-30.0, 30.0, size=(4000, 2))
    floor = np.column_stack([floor, np.zeros(len(floor))])          # featureless slab
    landmark = rng.normal(0, 0.2, size=(40, 3)) + np.array([2.0, 0.0, 8.0])
    tree = cKDTree(np.concatenate([floor, landmark], axis=0))

    scan = rng.uniform(-20.0, 20.0, size=(4000, 2))
    scan = np.column_stack([scan, np.full(len(scan), -SENSOR_HEIGHT_M)])
    scan = np.concatenate([scan, landmark - np.array([0.0, 0.0, SENSOR_HEIGHT_M])], axis=0)

    bands = [(-1.0, 4.0), (4.0, 12.0)]
    poses = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi]])  # yaw flip: same floor, wrong landmark

    flat = evaluate(poses, scan, tree)
    banded = evaluate(poses, scan, tree,
                       scan_point_weights(scan, bands, [1.0, 2.0], "per-band"))
    assert flat[0] > flat[1] and banded[0] > banded[1]
    assert (banded[0] - banded[1]) > 10 * (flat[0] - flat[1])


def test_recombine_takes_yaw_the_short_way_round():
    """The wrap is the whole trap in blending an angle: 179 and -179 degrees are two
    degrees apart, and a straight average of them is 0, the exact opposite heading."""
    a = np.array([[0.0, 0.0, np.radians(179.0)]])
    b = np.array([[0.0, 0.0, np.radians(-179.0)]])
    child = recombine(a, b, np.full((1, 1), 0.5))

    assert abs(abs(child[0, 2]) - np.pi) < np.radians(1.0)


def test_uniform_crossover_only_ever_copies_a_parent_gene():
    """The swap must not invent values. Every gene of every child has to be one of the
    two parents' own, including yaw, which travels through the arc form and back."""
    rng = np.random.default_rng(10)
    elites = np.array([[1.0, 2.0, 0.5], [-4.0, 9.0, -2.0]])
    children = crossover(elites, rng, 300, "uniform", sigma_xy=0.5, sigma_yaw=10.0)

    for gene, values in enumerate([{1.0, -4.0}, {2.0, 9.0}, {0.5, -2.0}]):
        nearest = [min(values, key=lambda v: abs(v - c)) for c in children[:, gene]]
        assert np.allclose(children[:, gene], nearest, atol=1e-9), f"gene {gene} invented"


def test_blx_children_stay_in_the_widened_parent_interval():
    """BLX-alpha draws from the parents' own spread, widened by alpha. That is the
    adaptive step size the operator is here to test, so pin the interval."""
    rng = np.random.default_rng(11)
    elites = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])   # 2 m apart, inside the gate
    children = crossover(elites, rng, 500, "blx", sigma_xy=0.5, sigma_yaw=10.0)

    span = 2.0
    assert children[:, 0].min() >= -BLX_ALPHA * span - 1e-9
    assert children[:, 0].max() <= span * (1.0 + BLX_ALPHA) + 1e-9
    # and it must actually use the width, not collapse onto the parents
    assert children[:, 0].std() > 0.4


def test_blx_refuses_to_blend_across_two_aliases():
    """Two elites a rack pitch apart are competing answers, not two halves of one. Their
    midpoint is the empty aisle between two racks and fits nothing, so the gate must send
    those pairs to plain mutation instead."""
    rng = np.random.default_rng(12)
    separation = 4 * BLX_MAX_PARENT_SEP_M
    elites = np.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]])
    children = crossover(elites, rng, 400, "blx", sigma_xy=0.5, sigma_yaw=10.0)

    midpoint = np.abs(children[:, 0] - separation / 2) < separation / 4
    assert not midpoint.any(), "gate let a cross-alias blend through"
    # the fallback is a jitter of one parent, so children cluster on the parents themselves
    on_parent = np.minimum(np.abs(children[:, 0]), np.abs(children[:, 0] - separation)) < 3.0
    assert on_parent.mean() > 0.98


def test_every_crossover_mode_returns_wrapped_poses():
    rng = np.random.default_rng(13)
    elites = random_population(rng, 20, ((-50.0, 50.0), (-50.0, 50.0)))
    for mode in ("uniform", "blend", "blx"):
        children = crossover(elites, rng, 200, mode, sigma_xy=1.0, sigma_yaw=20.0)
        assert children.shape == (200, 3)
        assert np.all(np.abs(children[:, 2]) <= np.pi), mode
        assert np.all(np.isfinite(children)), mode


def test_crossover_none_changes_nothing_and_a_mode_does():
    """The control arm has to be the old search exactly, or the comparison is against a
    baseline that moved. n_cross == 0 skips the operator entirely, so the RNG stream is
    untouched; a real mode must then visibly change the outcome."""
    rng = np.random.default_rng(14)
    map_points = _asymmetric_cluster(12.0, 8.0, rng)
    tree = cKDTree(map_points)
    scan = map_points - np.array([12.0, 8.0, SENSOR_HEIGHT_M])
    bounds = ((0.0, 25.0), (0.0, 20.0))

    def search(mode):
        poses, fitness = evolve(scan, tree, bounds, np.random.default_rng(15),
                                 population=120, generations=8, crossover_mode=mode)
        return poses[np.argmax(fitness)]

    assert np.array_equal(search("none"), search("none"))
    assert not np.array_equal(search("none"), search("blx"))
