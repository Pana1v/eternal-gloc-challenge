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
