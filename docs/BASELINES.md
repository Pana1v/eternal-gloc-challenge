# Baselines

Four reference methods ship with the challenge. This page says what each one
does, how it scores on the released dev split, and what the failures look
like. Numbers come from `eval/score.py`, the same scorer used for the
official evaluation.

## What each baseline does

### Multi-slice bird's-eye-view correlative search (`bl_bbs`)

Flattens the scan and the prior map into five horizontal height bands, then
searches every east/north/heading placement exhaustively by fast Fourier
transform cross-correlation, weighting the two ceiling bands twice as
heavily. Refines the winning placement with point-to-plane iterative closest
point.

Exhaustive search is what makes it work. In a repetitive warehouse the
scoring surface has thousands of near-equal peaks, and any method that
samples or summarizes can settle on a good-enough peak instead of the right
one. Evaluating every placement and taking the global maximum is the only
approach here that reliably finds the true one.

### Polar-histogram retrieval with geometric refinement (`bl_retrieval_gicp`)

Builds a database of candidate viewpoints sampled every 2 metres from the
prior map and retrieves by a rotation-invariant ring descriptor, recovering
heading by circular cross-correlation and refining with generalized
iterative closest point. Cost scales with database size rather than with map
area.

### Evolutionary pose search (`bl_ga`)

Scatters a population of random ground-plane guesses, scores each by how much
of the scan it explains, and mutates the survivors over successive
generations. Pays only for the placements it samples, and submits its
surviving modes as up to three hypotheses.

### Camera edge re-ranking (`bl_vpr_rerank`)

Re-orders and re-weights a pose hypothesis set produced elsewhere, by
projecting the prior map's geometry into the camera and scoring edge
agreement against the captured image. It cannot localize on its own, and it
cannot change anything when it is handed a single hypothesis.

## Results on the dev split

Track A, 40 scenarios. `S-fine` is 0.5 m and 5 degrees; `S-coarse` is 2.0 m
and 10 degrees. Compute is reported beside the score and never folded into
it. `bl_vpr_rerank` writes no compute sidecar, so its cost is not recorded.

| method | score | SR@fine | SR@coarse | oracle@fine | mean loss | sec/scenario | peak RSS (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bl_bbs` | 97.74 | 0.975 | 0.975 | 0.975 | 0.0226 | 2.65 | 1100 |
| `bl_vpr_rerank` | 97.74 | 0.975 | 0.975 | 0.975 | 0.0226 | not recorded | not recorded |
| `bl_ga` | 22.36 | 0.025 | 0.025 | 0.025 | 0.7764 | 10.27 | 1100 |
| `bl_retrieval_gicp` | 7.63 | 0.000 | 0.000 | 0.000 | 0.9237 | 23.27 | 1218 |
| random guess | 1.64 | - | - | - | 0.9836 | - | - |

With 40 scenarios a success rate can only land on multiples of 0.025, so a
gap of one scenario is not a difference between methods.

`bl_vpr_rerank` ties `bl_bbs` because it was handed `bl_bbs` output, which
carries one hypothesis per scenario. A list of one cannot be reordered, and a
single weight must normalize to 1.0, so it returned its input unchanged. To
exercise it, feed it a method that emits several hypotheses.

`oracle@fine` equal to `SR@fine` on every row means no method's alternative
hypotheses were better than its primary, so there is currently no re-ranking
headroom anywhere in this set.

## Per-tier breakdown

Tiers come from a measured alias count: T1 has none, T2 at most three, T3
more.

| method | T1 (n=9) | T2 (n=8) | T3 (n=23) |
| --- | --- | --- | --- |
| `bl_bbs` | 99.76 | 90.99 | 99.30 |
| `bl_vpr_rerank` | 99.76 | 90.99 | 99.30 |
| `bl_ga` | 34.15 | 14.88 | 20.34 |
| `bl_retrieval_gicp` | 6.63 | 7.44 | 8.08 |

The ordering is not monotonic: T3 is supposed to be hardest, yet `bl_bbs`
scores perfectly on it and drops only on T2. The alias counter thresholds
peaks at 0.9 times the ground-truth score, a relative bar, so a scenario with
weak geometry clears it easily and one with strong geometry does not. Treat
the tier column as descriptive, not as a difficulty ranking.

## What a scenario looks like

Each scenario is one lidar scan, one camera image, and calibration. The
camera sees the racking, the aisle receding, and the ceiling structure.

![Sample camera frame](images/sample_camera.png)

The per-scenario figure pairs a top-down view of the prior map, the truth,
and each submitted pose with that scenario's camera frame.

![Sample scenario figure](images/sample_scenario_figure.png)

## What the report looks like

`eval/score.py` writes a self-contained HTML report next to the scores. It
carries the summary and per-tier tables, a casewise matrix, and a scenario
map you can page through. The map is inline vector geometry rather than
rendered images, so the whole report is one file that opens offline.

Racking is grey, structural columns and walls and landmarks are dark red,
the truth is black, and each method gets a colour with a line drawn back to
the truth.

Scenario 000038, where the two working methods sit on the truth while the
other two are 26.7 m and 164.1 m away:

![Report scenario map](images/report_scenario_map.png)

Scenario 000014, the only one `bl_bbs` misses. Its estimate has the same
northing to within 3 mm and the same heading to within 0.06 degrees, but sits
110.2 m along the aisle on an identical-looking bay. This is what rack-level
aliasing looks like when it beats an otherwise exact matcher:

![Aliasing failure](images/report_aliasing_failure.png)

## Score distributions

Translation error, cumulative over scenarios:

![Translation error CDF](images/plot_translation_cdf.png)

Rotation error, cumulative over scenarios:

![Rotation error CDF](images/plot_rotation_cdf.png)

Loss by tier:

![Loss by tier](images/plot_loss_by_tier.png)

## Reproducing

```bash
python baselines/bl_bbs/run.py --scenarios scenarios/dev/A \
    --map map/prior_map.pcd --out submission.txt --workers 0

python eval/score.py --submission submission.txt \
    --gt scenarios/dev/gt/A.txt --track A --out-dir results \
    --map map/prior_map.pcd
```

Passing `--map` builds the vector background the scenario map needs. Without
it the report still renders, just without the map section.
