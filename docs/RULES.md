# Rules

## Allowed

- Anything that runs CPU-only inside the provided `docker/runtime.Dockerfile`
  image (or a Dockerfile `FROM` it).
- Rendering **virtual scans from the prior map** , e.g. sampling synthetic
  lidar sweeps or projections at candidate poses to build a retrieval
  database. This is a legitimate, encouraged technique (baselines B1/B2 use
  variants of it themselves).
- Generic pretrained feature extractors (e.g. an ONNX-exported vision
  backbone) up to 200 MB, as long as they run entirely offline inside the
  image.
- Any classical or learned method you implement yourself, using only the
  data shipped with the challenge.

## Forbidden

- Runtime internet access from inside the container. Nothing in your
  submission may fetch anything over the network while scoring runs.
- Hand-annotating, manually inspecting, or otherwise using human judgment
  on individual **eval** scenarios to produce an answer. (The dev set,
  which ships with ground truth, is yours to inspect and tune against
  freely , that's what it's for.)
- Any form of per-scenario human input in the scoring loop. Your submission
  must run unattended, start to finish.

## Spirit of the rules

The eval set is anonymized (shuffled IDs, randomized local frames, no
ground truth) specifically so that a submission has to actually solve the
localization problem rather than memorize or game the eval scenarios. If
you're unsure whether something you're doing crosses a line, ask: would
this still work if the eval set were regenerated from a fresh seed
tomorrow? If not, it's probably not in the spirit of the challenge.
