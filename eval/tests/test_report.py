"""The casewise table links a scenario to its rendered figure only when that
figure exists, and links it by a path relative to the report itself so the
whole directory browses offline after being moved or copied.
"""

from report import casewise_table, main as report_main


def test_casewise_links_only_supplied_ids():
    runs = {"bl_bbs": {"summary": {}, "rows": {
        "000000": {"e_t": "0.1", "e_r": "1.0", "sr_fine": "1"},
        "000001": {"e_t": "9.9", "e_r": "80.0", "sr_fine": "0"},
    }}}
    _, body = casewise_table(runs, {}, figures={"000000": "figures/000000.png"})
    linked = [row for row in body if "<a href=" in row]

    assert len(linked) == 1
    assert 'href="figures/000000.png"' in linked[0]
    assert "000001" in "".join(body)   # still listed, just not linked


def test_report_figure_paths_are_relative(tmp_path):
    results = tmp_path / "results"
    run_dir = results / "bl_bbs"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        '{"track":"A","overall":{"score":50.0,"sr_fine":0.5,"sr_coarse":0.5,'
        '"oracle_sr_fine":0.5,"mean_loss":0.5,"n_missing":0,"n_scenarios":1},'
        '"run":{"method":"bl_bbs"}}')
    (run_dir / "stats.csv").write_text("scenario_id,e_t,e_r,sr_fine\n000000,0.10,1.0,1\n")

    figures = results / "figures"
    figures.mkdir()
    (figures / "000000.png").write_bytes(b"")

    out = results / "report.html"
    assert report_main(["--results", str(results), "--out", str(out),
                        "--figures", str(figures)]) == 0

    html = out.read_text()
    assert 'href="figures/000000.png"' in html
    assert str(tmp_path) not in html   # no absolute path leaks into the report


def _poses(rows):
    return {sid: {"x": x, "y": y, "yaw": yaw} for sid, x, y, yaw in rows}


def test_identical_output_is_detected():
    """A re-ranker handed one hypothesis per scenario returns its input
    unchanged, which must not read as an independent result."""
    from report import identical_pairs

    same = [("000000", "1.0", "2.0", "0.5"), ("000001", "3.0", "4.0", "1.5")]
    runs = {
        "bl_bbs": {"poses": _poses(same)},
        "bl_vpr_rerank": {"poses": _poses(same)},
        "bl_ga": {"poses": _poses([("000000", "9.0", "9.0", "0.1"),
                                    ("000001", "8.0", "8.0", "0.2")])},
    }
    pairs = identical_pairs(runs)

    assert pairs == {"bl_vpr_rerank": "bl_bbs"}   # the later name points at the original
    assert "bl_ga" not in pairs


def test_method_names_carry_no_abbreviations():
    from report import METHODS

    for method, (name, desc) in METHODS.items():
        text = f"{name} {desc}"
        for abbrev in ("BEV", "ICP", "GICP", "VPR", "FFT", "SE(2)"):
            assert abbrev not in text, f"{method} still says {abbrev}"
        assert "—" not in text and "–" not in text
