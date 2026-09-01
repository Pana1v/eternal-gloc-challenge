# Eternal GLoc Challenge

You wake up a warehouse AMR somewhere in a very large, deliberately
repetitive warehouse and it has to figure out where it is. You get one 3D
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

## Layout

```
baselines/common/    shared BEV matching + ICP refinement + submission writer
baselines/bl_bbs/    B1: multi-slice correlative match + ICP (the workhorse)
baselines/bl_retrieval_gicp/   B2: descriptor retrieval + refinement
baselines/bl_vpr_rerank/       B3: vision re-ranking tie-breaker
eval/                official scorer + plotting
tools/viewer.py      scenario/map viewer
docker/              the runtime image your submission must run inside
ci/smoke_test.sh      end-to-end sanity check (build image, run a baseline, score it)
```
