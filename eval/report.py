#!/usr/bin/env python3
"""Renders every scored run in a results directory into one self-contained
HTML report: baseline-wise summaries, the random-guess reference, per-tier
breakdowns, and a casewise scenario x baseline matrix.

score.py invokes this after each scoring run, so the report always reflects
every method scored into that directory so far. It can also be run directly:

    python report.py --results <dir> --out report.html [--tiers tiers.csv]

Reads only stats.csv / summary.json, so it never needs the map or scans.
"""

import argparse
import csv
import glob
import html
import json
import os

# Difficulty context per scenario, when the private tiers.csv is available.
TIER_FIELDS = ("tier", "ambiguity_lidar_full", "ambiguity_lidar_low")


def load_runs(results_dir: str):
    """One entry per scored run, newest first when a method was scored twice."""
    runs = {}
    for summary_path in sorted(glob.glob(os.path.join(results_dir, "*", "summary.json"))):
        with open(summary_path) as f:
            summary = json.load(f)
        stats_path = os.path.join(os.path.dirname(summary_path), "stats.csv")
        if not os.path.exists(stats_path):
            continue
        with open(stats_path) as f:
            rows = {r["scenario_id"]: r for r in csv.DictReader(f)}
        method = summary.get("run", {}).get("method") or os.path.basename(os.path.dirname(summary_path))
        runs[method] = {"summary": summary, "rows": rows}   # later run wins
    return runs


def load_tiers(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["scenario_id"]: r for r in csv.DictReader(f)}


def fnum(value, digits=2, dash="-"):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return dash


CSS = """
:root{
  --ground:#F4F6F7; --panel:#FFFFFF; --ink:#16202A; --ink-dim:#5C6B75;
  --rule:#D8DFE3; --accent:#B8700F; --pass:#1B7F5E; --fail:#C0492F;
  --pass-bg:#E4F1EB; --fail-bg:#FAE7E3; --shadow:0 1px 2px rgba(22,32,42,.06);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0F1518; --panel:#161E23; --ink:#DCE4E8; --ink-dim:#8A9AA4;
    --rule:#26333A; --accent:#E8A040; --pass:#4FBF95; --fail:#E87B66;
    --pass-bg:#13302A; --fail-bg:#331E1B; --shadow:none;
  }
}
:root[data-theme="dark"]{
  --ground:#0F1518; --panel:#161E23; --ink:#DCE4E8; --ink-dim:#8A9AA4;
  --rule:#26333A; --accent:#E8A040; --pass:#4FBF95; --fail:#E87B66;
  --pass-bg:#13302A; --fail-bg:#331E1B; --shadow:none;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  line-height:1.55;padding:40px 24px 80px}
.wrap{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:40px}
h1,h2{font-family:Archivo,system-ui,sans-serif;font-weight:700;
  letter-spacing:-.02em;text-wrap:balance;margin:0}
h1{font-size:2.1rem;line-height:1.15}
h2{font-size:1.15rem;letter-spacing:.02em;text-transform:uppercase;
  color:var(--ink-dim);font-size:.82rem;padding-bottom:8px;border-bottom:1px solid var(--rule)}
.sub{color:var(--ink-dim);max-width:65ch;margin:10px 0 0}
.facts{display:flex;flex-wrap:wrap;gap:8px 28px;margin-top:18px;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.78rem;color:var(--ink-dim)}
.facts b{color:var(--ink);font-weight:500}
section{display:flex;flex-direction:column;gap:14px}
.scroll{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);
  border-radius:6px;box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{padding:9px 14px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--rule)}
th{font-family:"IBM Plex Mono",monospace;font-size:.68rem;text-transform:uppercase;
  letter-spacing:.06em;color:var(--ink-dim);font-weight:500;
  position:sticky;top:0;background:var(--panel)}
th:first-child,td:first-child{text-align:left}
tbody tr:last-child td{border-bottom:none}
td.num{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
td.name{font-family:"IBM Plex Mono",monospace;font-weight:500}
tr.reference td{color:var(--ink-dim);font-style:italic;
  border-top:2px solid var(--rule);background:transparent}
.lead{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;
  font-size:1.05rem;font-weight:600;color:var(--accent)}
.chip{display:inline-block;min-width:2.6em;padding:1px 7px;border-radius:3px;
  font-family:"IBM Plex Mono",monospace;font-size:.72rem;font-weight:500}
.ok{background:var(--pass-bg);color:var(--pass)}
.bad{background:var(--fail-bg);color:var(--fail)}
.rail{border-left:3px solid var(--accent);padding-left:14px}
.note{color:var(--ink-dim);font-size:.82rem;max-width:70ch}
footer{color:var(--ink-dim);font-size:.75rem;font-family:"IBM Plex Mono",monospace;
  border-top:1px solid var(--rule);padding-top:16px}
"""


def summary_table(runs):
    """Baseline-wise: one row per method, random guess as a reference row."""
    head = ["method", "score", "SR@fine", "SR@coarse", "oracle@fine", "mean loss",
            "missing", "s/scenario", "peak RSS"]
    body = []
    reference = None
    for method, run in sorted(runs.items(), key=lambda kv: -kv[1]["summary"]["overall"]["score"]):
        o = run["summary"]["overall"]
        c = run["summary"].get("compute") or {}
        body.append(
            f'<tr><td class="name">{html.escape(method)}</td>'
            f'<td class="num lead">{fnum(o["score"])}</td>'
            f'<td class="num">{fnum(o["sr_fine"], 3)}</td>'
            f'<td class="num">{fnum(o["sr_coarse"], 3)}</td>'
            f'<td class="num">{fnum(o["oracle_sr_fine"], 3)}</td>'
            f'<td class="num">{fnum(o["mean_loss"], 4)}</td>'
            f'<td class="num">{o["n_missing"]}</td>'
            f'<td class="num">{fnum(c.get("runtime_sec_per_scenario"))}</td>'
            f'<td class="num">{fnum(c.get("peak_rss_mb"), 0)}</td></tr>')
        rb = run["summary"].get("random_baseline")
        if rb and reference is None:
            reference = rb
    if reference:
        body.append(
            f'<tr class="reference"><td class="name">random guess (reference)</td>'
            f'<td class="num">{fnum(reference["score"])}</td>'
            f'<td class="num">{fnum(reference.get("sr_fine"), 3)}</td>'
            f'<td class="num">{fnum(reference.get("sr_coarse"), 3)}</td>'
            f'<td class="num">-</td><td class="num">{fnum(reference.get("mean_loss"), 4)}</td>'
            f'<td class="num">-</td><td class="num">-</td><td class="num">-</td></tr>')
    return head, body


def tier_table(runs):
    tiers = sorted({t for r in runs.values() for t in r["summary"].get("per_tier", {})})
    if not tiers:
        return None, None
    head = ["method"] + [f"{t} score / SR@fine" for t in tiers]
    body = []
    for method, run in sorted(runs.items()):
        cells = []
        for t in tiers:
            m = run["summary"].get("per_tier", {}).get(t)
            cells.append(f'<td class="num">{fnum(m["score"])} &middot; {fnum(m["sr_fine"], 3)} '
                         f'<span style="color:var(--ink-dim)">(n={m["n_scenarios"]})</span></td>'
                         if m else '<td class="num">-</td>')
        body.append(f'<tr><td class="name">{html.escape(method)}</td>' + "".join(cells) + "</tr>")
    return head, body


def casewise_table(runs, tiers):
    """Scenario x baseline. Sorted by measured rack-level ambiguity when it is
    available, so any concentration of failures at the hard end is visible
    rather than having to be looked for."""
    methods = sorted(runs)
    ids = sorted({sid for r in runs.values() for sid in r["rows"]})

    def difficulty(sid):
        row = tiers.get(sid)
        return -int(row["ambiguity_lidar_low"]) if row else 0

    ids.sort(key=difficulty)
    head = ["scenario"] + (["tier", "aliases"] if tiers else []) + \
           [f"{m} e_t / e_r" for m in methods]
    body = []
    for sid in ids:
        cells = [f'<td class="name">{html.escape(sid)}</td>']
        if tiers:
            t = tiers.get(sid, {})
            cells.append(f'<td class="num">{html.escape(t.get("tier", "-"))}</td>')
            cells.append(f'<td class="num">{html.escape(t.get("ambiguity_lidar_low", "-"))}</td>')
        for m in methods:
            row = runs[m]["rows"].get(sid)
            if not row or row.get("missing") == "1":
                cells.append('<td class="num">-</td>')
                continue
            klass = "ok" if row["sr_fine"] == "1" else "bad"
            cells.append(f'<td class="num"><span class="chip {klass}">{fnum(row["e_t"])} m</span> '
                         f'{fnum(row["e_r"], 1)}&deg;</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return head, body


def table_html(head, body):
    if not head:
        return ""
    ths = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    return (f'<div class="scroll"><table><thead><tr>{ths}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def render(runs, tiers, title):
    any_summary = next(iter(runs.values()))["summary"]
    track = any_summary.get("track", "?")
    n = any_summary["overall"]["n_scenarios"]
    generated = any_summary.get("generated_at", "")
    p = any_summary.get("parameters", {})

    facts = (f'<div class="facts"><span>track <b>{html.escape(str(track))}</b></span>'
             f'<span>scenarios <b>{n}</b></span>'
             f'<span>methods <b>{len(runs)}</b></span>'
             f'<span>S-fine <b>&le;{p.get("s_fine_trans_m", "?")} m / &le;{p.get("s_fine_rot_deg", "?")}&deg;</b></span>'
             f'<span>S-coarse <b>&le;{p.get("s_coarse_trans_m", "?")} m / &le;{p.get("s_coarse_rot_deg", "?")}&deg;</b></span>'
             f'<span>generated <b>{html.escape(generated[:19])}</b></span></div>')

    sections = [f'<section><h2>Baselines</h2>{table_html(*summary_table(runs))}'
                f'<p class="note">Score is <code>100 &times; (1 - mean loss)</code>. '
                f'Compute is reported beside the score, never folded into it. '
                f'<code>oracle@fine</code> equal to <code>SR@fine</code> means the method '
                f'submitted a single hypothesis, so no re-ranking headroom exists.</p></section>']

    th, tb = tier_table(runs)
    if th:
        sections.append(f'<section><h2>By difficulty tier</h2>{table_html(th, tb)}'
                        f'<p class="note">Tiers come from the measured alias count. A '
                        f'non-monotonic ordering here means the tier labels are not tracking '
                        f'real difficulty.</p></section>')

    sections.append(f'<section><h2>Casewise</h2>{table_html(*casewise_table(runs, tiers))}'
                    f'<p class="note">Translation error, with rotation error beside it. Green '
                    f'passes S-fine, red does not. '
                    f'{"Ordered hardest-first by measured rack-level aliases." if tiers else ""}</p></section>')

    return f"""<title>{html.escape(title)}</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="wrap">
  <header class="rail">
    <h1>{html.escape(title)}</h1>
    <p class="sub">Global localization results across every scored baseline, with the
    random-guess floor as reference and per-scenario detail below.</p>
    {facts}
  </header>
  {"".join(sections)}
  <footer>Generated by eval/report.py from stats.csv and summary.json.</footer>
</div>"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="directory holding scored run subdirectories")
    ap.add_argument("--out", help="output HTML (default: <results>/report.html)")
    ap.add_argument("--tiers", help="optional private tiers.csv for difficulty context")
    ap.add_argument("--title", default="GLoc Eval Report")
    args = ap.parse_args(argv)

    runs = load_runs(args.results)
    if not runs:
        print(f"report: no scored runs found under {args.results}")
        return 1

    out = args.out or os.path.join(args.results, "report.html")
    with open(out, "w") as f:
        f.write(render(runs, load_tiers(args.tiers), args.title))
    print(f"wrote {out} ({len(runs)} method{'s' if len(runs) != 1 else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
