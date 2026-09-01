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

# A submission file carries only a method's short name. Spelling out what each
# one actually does keeps the comparison readable by someone who has not read
# the baseline sources.
METHODS = {
    "bl_bbs": ("B1", "Multi-slice BEV correlative search",
               "Rasterizes the scan and the prior map into five height bands, then searches "
               "every (x, y, yaw) exhaustively by FFT cross-correlation, weighting the two "
               "ceiling bands 2x. Refines the winning pose with point-to-plane ICP."),
    "bl_retrieval_gicp": ("B2", "Polar-histogram retrieval",
               "Builds a database of virtual scans sampled every 2 m from the prior map and "
               "retrieves by a rotation-invariant ring key, recovering yaw by circular "
               "cross-correlation and refining with ICP. Scales with database size rather "
               "than map area."),
    "bl_vpr_rerank": ("B3", "Camera edge re-ranking",
               "Re-orders and re-weights another method's pose hypotheses by projecting the "
               "prior map's geometry into the camera and scoring edge agreement against the "
               "image. Cannot localize on its own."),
    "bl_ga": ("B4", "Evolutionary pose search",
               "Scatters a population of random SE(2) guesses, scores each by how much of the "
               "scan it explains, and mutates the survivors over successive generations. Pays "
               "only for the poses it samples, and submits its surviving modes as up to three "
               "hypotheses."),
}


def method_identity(method: str):
    """(label, short name, description) for a submission's method name."""
    tag, short, desc = METHODS.get(method, ("", "", ""))
    return (f"{tag} {method}".strip(), short, desc)


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
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:14px;
  line-height:1.5;color:#111;background:#fff;margin:24px;max-width:1400px}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:15px;margin:28px 0 8px;border-bottom:1px solid #999;padding-bottom:3px}
p.note{color:#444;font-size:13px;margin:6px 0 0;max-width:90ch}
table{border-collapse:collapse;margin-top:6px;font-size:13px}
th,td{border:1px solid #999;padding:4px 8px;text-align:right;white-space:nowrap}
th{background:#eee;font-weight:600;text-align:right}
th:first-child,td:first-child{text-align:left}
td.name,td.num{font-family:ui-monospace,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
td.desc{white-space:normal;text-align:left;font-size:12px;color:#333;max-width:60ch}
tr.reference td{background:#f4f4f4;font-style:italic}
.pass{color:#060;font-weight:600}
.fail{color:#a00;font-weight:600}
.scroll{overflow-x:auto}
dl{font-size:13px;margin:6px 0 0}
dt{font-family:ui-monospace,Menlo,Consolas,monospace;font-weight:600;margin-top:6px}
dd{margin:0 0 0 24px;color:#333}
footer{margin-top:28px;border-top:1px solid #999;padding-top:8px;
  font-size:12px;color:#444}
"""


def summary_table(runs):
    """Baseline-wise: one row per method, random guess as a reference row."""
    head = ["method", "approach", "score", "SR@fine", "SR@coarse", "oracle@fine",
            "mean loss", "missing", "sec/scenario", "peak RSS (MB)"]
    body = []
    reference = None
    for method, run in sorted(runs.items(), key=lambda kv: -kv[1]["summary"]["overall"]["score"]):
        o = run["summary"]["overall"]
        c = run["summary"].get("compute") or {}
        label, short, _ = method_identity(method)
        body.append(
            f'<tr><td class="name">{html.escape(label)}</td>'
            f'<td class="desc">{html.escape(short)}</td>'
            f'<td class="num">{fnum(o["score"])}</td>'
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
            f'<tr class="reference"><td class="name">random guess</td>'
            f'<td class="desc">uniform pose inside the ground-truth extent (reference floor)</td>'
            f'<td class="num">{fnum(reference["score"])}</td>'
            f'<td class="num">{fnum(reference.get("sr_fine"), 3)}</td>'
            f'<td class="num">{fnum(reference.get("sr_coarse"), 3)}</td>'
            f'<td class="num">-</td><td class="num">{fnum(reference.get("mean_loss"), 4)}</td>'
            f'<td class="num">-</td><td class="num">-</td><td class="num">-</td></tr>')
    return head, body


def method_glossary(runs):
    """What each method actually does, spelled out under the summary table."""
    items = []
    for method in sorted(runs):
        label, short, desc = method_identity(method)
        if not desc:
            continue
        items.append(f"<dt>{html.escape(label)} &mdash; {html.escape(short)}</dt>"
                      f"<dd>{html.escape(desc)}</dd>")
    return f"<dl>{''.join(items)}</dl>" if items else ""


def tier_table(runs):
    tiers = sorted({t for r in runs.values() for t in r["summary"].get("per_tier", {})})
    if not tiers:
        return None, None
    head = ["method"] + [f"{t}: score / SR@fine (n)" for t in tiers]
    body = []
    for method, run in sorted(runs.items()):
        cells = []
        for t in tiers:
            m = run["summary"].get("per_tier", {}).get(t)
            cells.append(f'<td class="num">{fnum(m["score"])} &middot; {fnum(m["sr_fine"], 3)} '
                         f'(n={m["n_scenarios"]})</td>'
                         if m else '<td class="num">-</td>')
        body.append(f'<tr><td class="name">{html.escape(method_identity(method)[0])}</td>'                     + "".join(cells) + "</tr>")
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
           [f"{method_identity(m)[0]}: e_t (m) / e_r (deg)" for m in methods]
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
            klass = "pass" if row["sr_fine"] == "1" else "fail"
            cells.append(f'<td class="num"><span class="{klass}">{fnum(row["e_t"])}</span> / '
                         f'{fnum(row["e_r"], 1)}</td>')
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

    facts = (f"<p class=\"note\">Track {html.escape(str(track))} &middot; {n} scenarios &middot; "
             f"{len(runs)} method{'s' if len(runs) != 1 else ''} &middot; "
             f"S-fine &le; {p.get('s_fine_trans_m', '?')} m and &le; {p.get('s_fine_rot_deg', '?')} deg &middot; "
             f"S-coarse &le; {p.get('s_coarse_trans_m', '?')} m and &le; {p.get('s_coarse_rot_deg', '?')} deg &middot; "
             f"generated {html.escape(generated[:19])}</p>")

    sections = [f'<h2>Baselines</h2>{table_html(*summary_table(runs))}'
                f'<p class="note">Score is 100 x (1 - mean loss). Compute is reported beside the '
                f'score and is never folded into it. oracle@fine equal to SR@fine means the method '
                f'submitted a single hypothesis, so no re-ranking headroom exists.</p>'
                f'{method_glossary(runs)}']

    th, tb = tier_table(runs)
    if th:
        sections.append(f'<h2>By difficulty tier</h2>{table_html(th, tb)}'
                        f'<p class="note">Tiers come from the measured alias count (T1 = 0 aliases, '
                        f'T2 &le; 3, T3 above). A non-monotonic ordering here means the tier labels '
                        f'are not tracking real difficulty.</p>')

    sections.append(f'<h2>Casewise</h2>{table_html(*casewise_table(runs, tiers))}'
                    f'<p class="note">Per scenario: translation error in metres / rotation error in '
                    f'degrees. Green passes S-fine, red does not. '
                    f'{"Ordered hardest-first by measured rack-level aliases." if tiers else ""}</p>')

    return f"""<title>{html.escape(title)}</title>
<style>{CSS}</style>
<h1>{html.escape(title)}</h1>
{facts}
{"".join(sections)}
<footer>Generated by eval/report.py from stats.csv and summary.json.</footer>"""


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
