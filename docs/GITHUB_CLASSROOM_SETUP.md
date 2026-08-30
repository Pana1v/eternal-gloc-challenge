## Setting up GitHub Classroom for this challenge

The repo side is done: `Pana1v/eternal-gloc-challenge` is marked as a
[template repository](https://github.com/Pana1v/eternal-gloc-challenge) and
carries `.github/workflows/autograde.yml`, which scores whatever
`submission_dev.txt` / `submission_dev_b.txt` a candidate commits against the
published dev ground truth and reports the result in the Actions job
summary.

Everything past this point is web-UI only — GitHub Classroom has no CLI or
API for creating a classroom or assignment, so these steps need a human at
[classroom.github.com](https://classroom.github.com).

### 1. Pick (or create) an organization

Classroom attaches to a GitHub **organization**, not a personal account.
Your current orgs (`tinkerers-lab-iitp`, `Robocon-IIT-Patna`) are unrelated
IIT Patna clubs, not appropriate for a company hiring challenge. Options:

- Create a new org (e.g. `eternal-ai-hiring` or similar) — free tier is
  fine for this scale. github.com -> "+" -> "New organization".
- Use an existing company org if eternal.ag already has one on GitHub.

### 2. Connect the org to Classroom

At classroom.github.com, "New classroom" -> authorize the org -> Classroom
requests admin access to manage repos/teams under it.

### 3. Create the assignment

- "New assignment" -> **Individual assignment** (one private repo per
  candidate, not shared).
- Template repository: search for `eternal-gloc-challenge` — it'll show up
  since it's marked as a template and Classroom can see any template repo
  your authenticated account can access.
- Visibility: **private** repos for candidates (default, and what you want
  for a hiring challenge).
- Autograding: Classroom will detect `.github/workflows/autograde.yml`
  already in the template and offer to enable it as a check; the numeric
  score won't feed Classroom's own points-based gradebook (that model
  expects discrete pass/fail test cases, ours is a continuous benchmark
  score) but the job summary and pass/fail check status both show up in
  Classroom's per-student progress view regardless.

### 4. Invite candidates

Classroom gives you a shareable invite link once the assignment is created.
Each candidate who clicks it gets their own private repo (a clone of the
template) and pushes their submission there.

### Known gap

`data/dev/gt/A.txt` and `data/dev/gt/B.txt` (the published dev ground truth
candidates self-score against) aren't in the repo yet — the full dev dataset
is still generating. The autograding workflow already handles this
gracefully (warns instead of crashing) until those files are added.
