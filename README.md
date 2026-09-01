# Eternal GLoc Challenge

You wake up an AMR somewhere in a very large, deliberately
repetitive environment and it has to figure out where it is. You get one 3D
lidar scan and one camera image (or, in Track B, a short driven sequence),
plus a prior 3D lidar map of the building. No GPS, no prior pose. The
warehouse is built so that the rack level alone is nearly impossible to
disambiguate; the ceiling (trusses, lamps, HVAC, skylights) and the camera
are your way out.

Two tracks: **Track A** is single-shot kidnapped-robot localization.
**Track B** gives you a short (~20 m) driven sequence and lets you spend as
much or as little of it as you need before committing to an answer. You're
scored on both accuracy and how much of the budget you used.

## The 5-minute path

1. **Get the tools**: `docker build -f docker/runtime.Dockerfile -t eternal-gloc-runtime .`
2. **See a scenario**: `python tools/viewer.py scenarios/dev/A/000000 --map map/prior_map.pcd`.
   This opens the scan floating disconnected from the map (the problem,
   visually) and the camera image; add `--show-gt "<pose line>"` from
   `scenarios/dev/gt/A.txt` to see it snap into place.
3. **Run a baseline**: `python baselines/bl_bbs/run.py --scenarios scenarios/dev/A --map map/prior_map.pcd --out submission.txt`
4. **Self-score**: `python eval/score.py --submission submission.txt --gt scenarios/dev/gt/A.txt --track A --out-dir results`.
   This is the exact same scoring code used for the official eval.

## Read next

- [`docs/CHALLENGE.md`](docs/CHALLENGE.md): full problem statement, scenario formats, scoring formulas, submission format, grading rubric.
- [`docs/RULES.md`](docs/RULES.md): what's allowed and what isn't.
- [`docs/SENSORS.md`](docs/SENSORS.md): the exact rig spec (lidar + camera parameters, calibration format).
- [`docs/BASELINES.md`](docs/BASELINES.md): what the four shipped baselines do, how they score on dev, and pictures of where they fail.


## What you are working with

One lidar scan and one camera image per scenario, against a prior map of the
whole building. The camera sees the racking, the aisle receding, and the
ceiling structure above it.

![Sample camera frame](docs/images/sample_camera.png)

## Where the baselines stand

Track A, 40 dev scenarios. `S-fine` is 0.5 m and 5 degrees.

| method | score | SR@fine | sec/scenario |
| --- | --- | --- | --- |
| `bl_bbs` multi-slice correlative search | 97.74 | 0.975 | 2.65 |
| `bl_vpr_rerank` camera edge re-ranking | 97.74 | 0.975 | not recorded |
| `bl_ga` evolutionary pose search | 22.36 | 0.025 | 10.27 |
| `bl_retrieval_gicp` polar-histogram retrieval | 7.63 | 0.000 | 23.27 |
| random guess (reference floor) | 1.64 | - | - |

The scorer writes a self-contained HTML report with a scenario map you can
page through. Racking is grey, columns and walls and landmarks are dark red,
the truth is black, and each method gets a colour with a line back to the
truth. Here two methods sit on the truth while the others land 26.7 m and
164.1 m away:

![Report scenario map](docs/images/report_scenario_map.png)

The warehouse is repetitive enough that even an exact matcher can be beaten.
This is the one scenario `bl_bbs` misses: same northing to within 3 mm, same
heading to within 0.06 degrees, and 110.2 m along the aisle on a bay that
looks identical.

![Aliasing failure](docs/images/report_aliasing_failure.png)

Full write-up, per-tier numbers and score distributions in
[`docs/BASELINES.md`](docs/BASELINES.md).

## Layout

```
baselines/common/               shared bird's-eye-view matching, iterative
                                closest point refinement, submission writer
baselines/bl_bbs/               multi-slice correlative search (the workhorse)
baselines/bl_retrieval_gicp/    polar-histogram retrieval with refinement
baselines/bl_ga/                evolutionary pose search
baselines/bl_vpr_rerank/        camera edge re-ranking of another method's
                                hypotheses (cannot localize on its own)
eval/                           official scorer, plots, HTML report
tools/viewer.py                 scenario and map viewer
docker/                         the runtime image your submission must run in
ci/smoke_test.sh                end-to-end sanity check
```
