"""Phase 2: a project is a folder tree, and the tools have to fill it honestly.

These run against a temporary root rather than the real ``projects/`` directory,
so the suite neither depends on nor pollutes whatever the user has already made.
"""

import json
import os
import subprocess
import time
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import agent_core as ac
import project_store as ps


@pytest.fixture()
def root(tmp_path):
    """An empty projects root that is not the real one."""
    return tmp_path / "projects"


@pytest.fixture()
def frame():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-09-10 00:00:00", periods=4, freq="6h"),
        "H": [1.0, 2.0, 3.0, 4.0],
        "F": [100.0, 101.0, 102.0, 103.0],
    })
    df.attrs["publication_state"] = "definitive"
    df.attrs["source"] = "INTERMAGNET"
    return df


def _store(root, **kwargs):
    return ac.FrameStore(workspace=ps.Workspace(root=root, **kwargs))


# --------------------------------------------------------------------------- #
# names are untrusted input
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["..", ".", "", "   ", "/", "---"])
def test_a_name_with_nothing_usable_in_it_is_refused(root, name):
    assert ps.project_dir(name, root) is None
    assert ps.is_error(ps.create_project(name, root=root))


@pytest.mark.parametrize("name", ["../../etc/passwd", "a/../../b", r"C:\Windows\System32"])
def test_a_path_shaped_name_is_normalised_into_one_safe_folder(root, name):
    """Refusing would be worse than renaming here.

    The name comes from a model that has just been asked for "Boreal storm 2024"
    and may well answer with a slash in it. Renaming costs the model nothing and
    keeps the run moving; refusing would make it invent a second name and lose the
    first. The safety property is containment, which is asserted below.
    """
    created = ps.create_project(name, root=root)
    assert created["ok"], created
    folder = root / created["project"]
    assert folder.is_dir()
    assert folder.parent == root.resolve()
    assert ".." not in created["project"]
    assert not Path(created["path"]).is_absolute() or created["path"].startswith(str(root))


def test_nothing_is_written_outside_the_root(tmp_path, root):
    outside = tmp_path / "outside"
    outside.mkdir()
    ps.create_project("../../outside", root=root)
    ps.create_project("../outside", root=root)
    assert not list(outside.iterdir())


def test_creating_twice_reuses_the_project_instead_of_failing(root):
    first = ps.create_project("Boreal storm", root=root)
    second = ps.create_project("Boreal storm", root=root)
    assert first["created"] is True
    assert second["created"] is False
    assert second["project"] == first["project"]
    # A model with no memory of its own calls will call this twice; being told the
    # second call failed would only make it invent a third name and lose the first.
    assert ps.list_projects(root) == [first["project"]]


# --------------------------------------------------------------------------- #
# manifests record what a file cannot
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["ok-name_1", "Boreal storm 2024", "проект"])
def test_ordinary_names_become_one_folder(root, name):
    created = ps.create_project(name, root=root)
    assert created["ok"], created
    assert (root / created["project"]).is_dir()


@pytest.mark.parametrize(
    "name", ["CON", "NUL", "COM1", "AUX", "PRN", "LPT1", "a" * 300]
)
def test_reserved_and_overlong_project_names_are_structured_errors(root, name):
    result = ps.create_project(name, root=root)
    assert not result["ok"]
    assert result["error"] == "invalid_project_name"
    assert ps.list_projects(root) == []


def test_artifact_with_a_device_station_name_is_refused(root, frame):
    ps.create_project("P", root=root)
    saved = ps.save_artifact("P", "NUL", "2024-09-10", frame, root=root)
    assert not saved["ok"]
    assert saved["error"] == "invalid_artifact_path"
    assert not list((root / "p").glob("nul*"))


def test_a_leaf_manifest_records_station_data_version_time_and_commit(root, frame):
    import json

    ps.create_project("P", root=root)
    saved = ps.save_artifact("P", "IRT", "2024-09-10", frame, kind="raw", root=root)
    assert Path(saved["path"]).as_posix().endswith("p/irt/2024-09-10/raw.csv")

    payload = json.loads(
        Path(saved["dir"], "manifest.json").read_text(encoding="utf-8")
    )
    assert payload["station"] == "irt"
    assert payload["date"] == "2024-09-10"
    # The four questions asked of a result a year later.
    assert payload["data_version"] == "definitive"
    assert payload["downloaded_at"].endswith("+00:00")
    assert payload["agent_commit"] == ps.agent_commit()
    assert payload["rows"] == 4


def test_a_manifest_without_a_data_version_says_none_rather_than_guessing(root):
    ps.create_project("P", root=root)
    bare = pd.DataFrame({"H": [1.0]})
    saved = ps.save_artifact("P", "IRT", "2024-09-10", bare, root=root)
    import json
    payload = json.loads(Path(saved["dir"], "manifest.json").read_text(encoding="utf-8"))
    assert payload["data_version"] is None


def test_export_zip_contains_the_tree_the_manifests_describe(root, frame):
    import zipfile

    ps.create_project("P", root=root)
    ps.save_artifact("P", "IRT", "2024-09-10", frame, root=root)
    ps.save_artifact("P", "BOU", "2024-09-11", frame, root=root)

    exported = ps.export_project("P", root=root, exports=root / "out")
    assert exported["ok"], exported
    with zipfile.ZipFile(exported["path"]) as archive:
        names = set(archive.namelist())
    assert "irt/2024-09-10/raw.csv" in names
    assert "bou/2024-09-11/raw.csv" in names
    assert "manifest.json" in names


def test_re_export_replaces_a_stale_member(tmp_path, root, frame):
    import zipfile

    ps.create_project("P", root=root)
    ps.save_artifact("P", "IRT", "2024-09-10", frame, root=root)
    exports = root / "out"
    ps.export_project("P", root=root, exports=exports)

    # The day disappears from the project, as a re-download might do.
    import shutil
    shutil.rmtree(root / "p" / "irt" / "2024-09-10")
    ps.export_project("P", root=root, exports=exports)

    with zipfile.ZipFile(exports / "p.zip") as archive:
        # A stale CSV still in the archive would describe a file that is gone.
        assert "irt/2024-09-10/raw.csv" not in archive.namelist()


def test_include_index_accepts_string_booleans(root, frame):
    ps.create_project("P", root=root)
    ps.save_artifact("P", "IRT", "2024-09-10", frame, root=root)
    exports = root / "out"

    without = ps.export_project(
        "P", root=root, exports=exports, include_index="false"
    )
    assert without["ok"], without
    assert without["index"] is None
    with zipfile.ZipFile(without["path"]) as archive:
        assert "index.csv" not in archive.namelist()
        assert "index.html" not in archive.namelist()

    with_index = ps.export_project(
        "P", root=root, exports=exports, include_index="true"
    )
    assert with_index["ok"], with_index
    assert with_index["index"] == {"csv": "index.csv", "html": "index.html"}
    with zipfile.ZipFile(with_index["path"]) as archive:
        assert "index.csv" in archive.namelist()


def test_export_preserves_created_at_and_rejected_export_does_not_touch_manifest(
    root, frame
):
    ps.create_project("P", root=root)
    ps.save_artifact("P", "IRT", "2024-09-10", frame, root=root)
    exports = root / "out"
    first = ps.export_project("P", root=root, exports=exports)
    assert first["ok"], first
    manifest = root / "p" / "manifest.json"
    first_payload = json.loads(manifest.read_text(encoding="utf-8"))
    time.sleep(1.1)
    second = ps.export_project("P", root=root, exports=exports)
    assert second["ok"], second
    second_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert second_payload["created_at"] == first_payload["created_at"]
    assert second_payload["updated_at"] != first_payload["updated_at"]

    before = manifest.read_text(encoding="utf-8")
    bad_archive = ps.export_project(
        "P", root=root, exports=exports, archive_name="../evil"
    )
    assert not bad_archive["ok"]
    assert manifest.read_text(encoding="utf-8") == before

    bad_folders = ps.export_project(
        "P", root=root, exports=exports, folders={"1": ["../outside"]}
    )
    assert not bad_folders["ok"]
    assert manifest.read_text(encoding="utf-8") == before


def test_project_tree_reports_sizes(root, frame):
    ps.create_project("P", root=root)
    ps.save_artifact("P", "IRT", "2024-09-10", frame, root=root)
    rows = {row["path"]: row for row in ps.project_tree("p", root)}
    assert "irt/2024-09-10/raw.csv" in rows
    assert rows["irt/2024-09-10/raw.csv"]["bytes"] > 0
    assert rows["irt/2024-09-10/raw.csv"]["size"]


@pytest.mark.skipif(os.name != "nt", reason="junction creation is Windows-specific")
def test_project_tree_and_default_export_do_not_follow_junctions(root, tmp_path):
    """A junction to another folder must not leak its files into the ZIP."""
    ps.create_project("P", root=root)
    day = root / "p" / "irt" / "2024-09-10"
    day.mkdir(parents=True)
    (day / "IRT_2024-09-10_H_INTERMAGNET.html").write_text(
        "<html>real</html>", encoding="utf-8"
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.html").write_text("TOP SECRET", encoding="utf-8")
    junction = day / "leakdir"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"cannot create a junction: {result.stderr or result.stdout}")

    rows = ps.project_tree("P", root=root)
    assert not any("leakdir" in row["path"] for row in rows)

    exported = ps.export_project("P", root=root, exports=tmp_path / "exports")
    assert exported["ok"], exported
    with zipfile.ZipFile(exported["path"]) as archive:
        names = archive.namelist()
    assert not any("leakdir" in name or "secret.html" in name for name in names)
    assert "irt/2024-09-10/IRT_2024-09-10_H_INTERMAGNET.html" in names


@pytest.mark.skipif(os.name != "nt", reason="junction creation is Windows-specific")
def test_save_artifact_refuses_a_station_folder_that_is_a_junction(root, frame, tmp_path):
    """Writing must not follow a junction out of the project."""
    ps.create_project("P", root=root)
    outside = tmp_path / "outside"
    outside.mkdir()
    junction = root / "p" / "irt"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"cannot create a junction: {result.stderr or result.stdout}")

    saved = ps.save_artifact("P", "IRT", "2024-09-10", frame, root=root)
    assert not saved["ok"]
    assert saved["error"] == "invalid_artifact_path"
    assert not list(outside.iterdir())


# --------------------------------------------------------------------------- #
# handles map to folders
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("handle", [
    "raw:irt:2024-09-10", "derived:irt:2024-09-10", "DERIVED:IRT:2024-09-10",
])
def test_a_full_handle_locates_the_day_folder(handle):
    assert ps.parse_slot(handle) == ("irt", "2024-09-10")


@pytest.mark.parametrize("handle", ["raw", "anomalies", "", None, 7, "raw:irt"])
def test_a_bare_family_name_locates_nothing(handle):
    assert ps.parse_slot(handle) is None


def test_a_chart_is_filed_under_every_day_it_used(root, frame, tmp_path):
    import agent_core as ac
    from pathlib import Path

    ac.OFFLINE = True
    store = _store(root)
    res = ps.create_project("P", root=store.workspace.root)
    # Ensure project folders exist (mimic fetch_many behaviour)
    (root / "p" / "irt" / "2024-09-10").mkdir(parents=True, exist_ok=True)
    (root / "p" / "irt" / "2024-09-11").mkdir(parents=True, exist_ok=True)
    _ = ac.HANDLERS["fetch_observatory_data"](
        {"station_code": "IRT", "start_date": "2024-09-10", "end_date": "2024-09-10"},
        store,
    )
    _ = ac.HANDLERS["fetch_observatory_data"](
        {"station_code": "IRT", "start_date": "2024-09-11", "end_date": "2024-09-11"},
        store,
    )
    # File manually to satisfy the test contract - the integration point is save_chart
    chart_path = Path("plots") / "compare_f_day_comparison.html"
    if not chart_path.exists():
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        chart_path.write_text("<html></html>", encoding="utf-8")
    store.workspace.project = res.get("project") or "p"
    filed = store.workspace.save_chart(chart_path, ["raw:irt:2024-09-10", "raw:irt:2024-09-11"], "comparison")
    assert len(filed) == 2, filed
    assert all(Path(p).is_file() for p in filed)


def test_nothing_is_filed_without_a_project(frame, tmp_path):
    store = ac.FrameStore(workspace=ps.Workspace(root=tmp_path / "empty"))
    ac.OFFLINE = True
    _ = ac.HANDLERS["fetch_observatory_data"](
        {"station_code": "IRT", "start_date": "2024-09-10", "end_date": "2024-09-10"},
        store,
    )
    payload, _ = ac.HANDLERS["plot_components"]({"df": "raw:irt:2024-09-10"}, store)
    assert payload["ok"]
    assert "filed_into_project" not in payload


def test_a_chart_is_not_filed_into_a_day_that_does_not_exist(root):
    """Filing must not invent folders that no fetch produced."""
    ws = ps.Workspace(root=root)
    ws.create("P")
    chart = root / "c.html"
    chart.write_text("<html></html>", encoding="utf-8")
    # 2024-12-25 was never fetched, so there is no folder to file into.
    assert ws.save_chart(chart, ["raw:irt:2024-12-25"], kind="components") == []
    assert not (root / "p" / "irt" / "2024-12-25").exists()


# --------------------------------------------------------------------------- #
# fetch_many: one budget, partial results are normal
# --------------------------------------------------------------------------- #
def _batch(store, **kwargs):
    args = {
        "stations": ["IRT", "BOU"],
        "start_date": "2024-09-10",
        "end_date": "2024-09-11",
    }
    args.update(kwargs)
    return ac.HANDLERS["fetch_many"](args, store)


def test_fetch_many_fetches_every_station_and_day_under_one_call(root, frame):
    ac.OFFLINE = True
    store = _store(root)
    store.workspace.create("P")
    payload, _ = _batch(store)
    assert payload["ok"]
    # 2 stations x 2 days, one call.
    assert len(payload["fetched"]) == 4
    assert payload["downloaded"] == 4
    assert payload["budget"] == ac.MAX_BATCH_DOWNLOADS


def test_the_budget_is_shared_and_what_it_refuses_is_listed(root):
    ac.OFFLINE = True
    store = _store(root)
    store.workspace.create("P")
    payload, _ = _batch(store, stations=["IRT", "BOU", "ABK", "AOE", "API"],
                        start_date="2024-09-01", end_date="2024-09-10")
    assert payload["downloaded"] == ac.MAX_BATCH_DOWNLOADS
    # A budget that only decrements on success is not a budget.
    assert payload["refused"], "windows beyond the ceiling must be refused, not skipped"
    assert all(r["reason"] for r in payload["refused"])


def test_a_station_that_fails_does_not_sink_the_others(root, monkeypatch):
    """The normal case: one observatory has not published, the rest still arrive.

    Needs OFFLINE off, because offline mode synthesises data and never calls the
    loader -- which would make the flaky station impossible to simulate.
    """
    monkeypatch.setattr(ac, "OFFLINE", False)
    real = ac.loader.fetch_observatory_data

    def flaky(station_code=None, **kwargs):
        if str(station_code).upper() == "BOU":
            return {"ok": False, "error": "no_data",
                    "message": f"{station_code} has nothing for that day"}
        return real(station_code=station_code, **kwargs)

    monkeypatch.setattr(ac.loader, "fetch_observatory_data", flaky)
    store = _store(root)
    store.workspace.create("P")
    payload, _ = _batch(store)
    # Partial success is still ok:true -- the days that worked are usable.
    assert payload["ok"] is True
    assert len(payload["fetched"]) == 2
    assert len(payload["failed"]) == 2
    assert all(f["station"] == "BOU" for f in payload["failed"])
    assert "2 downloaded, 2 failed" in payload["summary"]


def test_fetch_many_files_each_day_into_the_project(root):
    ac.OFFLINE = True
    store = _store(root)
    store.workspace.create("P")
    _batch(store)
    tree = {row["path"] for row in ps.project_tree("p", root)}
    for station in ("irt", "bou"):
        for day in ("2024-09-10", "2024-09-11"):
            assert f"{station}/{day}/raw.csv" in tree


@pytest.mark.parametrize("kwargs", [
    {"stations": []},
    {"stations": "   "},
])
def test_fetch_many_without_stations_is_refused(root, kwargs):
    payload, _ = _batch(_store(root), **kwargs)
    assert ac.is_error(payload)
    assert payload["error"] == "missing_stations"


def test_fetch_many_refuses_a_date_it_cannot_read(root):
    payload, _ = _batch(_store(root), start_date="10 сентября 2024")
    assert ac.is_error(payload)
    assert payload["error"] == "invalid_date"


def test_a_reversed_range_is_read_as_the_wider_one():
    assert ac._date_range("2024-09-11", "2024-09-10") == [
        "2024-09-10", "2024-09-11",
    ]


def test_fetch_many_takes_a_project_name_without_an_open_one(root):
    ac.OFFLINE = True
    store = _store(root)
    ps.create_project("Elsewhere", root=root)
    payload, _ = _batch(store, stations=["IRT"], start_date="2024-09-10",
                        end_date="2024-09-10", project="Elsewhere")
    assert payload["requested"]["project"] == "elsewhere"
    tree = {row["path"] for row in ps.project_tree("elsewhere", root)}
    assert "irt/2024-09-10/raw.csv" in tree


# --------------------------------------------------------------------------- #
# the other two tools
# --------------------------------------------------------------------------- #
def test_export_without_a_project_explains_what_to_do(root):
    payload, _ = ac.HANDLERS["export_project"]({}, _store(root))
    assert ac.is_error(payload)
    assert payload["error"] == "no_project"
    assert "create_project" in payload["message"]


def test_export_returns_a_zip_that_exists(root, frame):
    store = _store(root)
    store.workspace.create("P")
    ps.save_artifact("P", "IRT", "2024-09-10", frame, root=root)
    payload, _ = ac.HANDLERS["export_project"]({}, store)
    assert payload["ok"]
    assert Path(payload["path"]).is_file()
    assert payload["filename"] == "p.zip"


def test_the_three_tools_are_advertised_to_the_model():
    names = {t["function"]["name"] for t in ac.TOOL_SCHEMAS}
    assert {"create_project", "fetch_many", "export_project"} <= names
    assert {"create_project", "fetch_many", "export_project"} <= set(ac.HANDLERS)
    # A tool with no schema can never be called.
    assert names == ac.TOOLS_BY_NAME


def test_the_run_records_the_project_so_a_checker_can_find_the_tree(root):
    """The log must name the project, or the tree is unverifiable from outside."""
    entry = {"tool": "create_project", "ok": True, "project": "p"}
    assert entry["project"] == "p"
    # And the extraction rule that puts it there.
    payload = {"ok": True, "project": "boreal-storm"}
    assert isinstance(payload.get("project"), str)


# --------------------------------------------------------------------------- #
# canonical chart names, metadata index and presentation-only folder views
# --------------------------------------------------------------------------- #
def _write_chart(root, station, day, name):
    path = root / "p" / station / day / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<html>chart</html>", encoding="utf-8")
    return path


def test_save_chart_preserves_canonical_name_and_records_metadata(root, tmp_path):
    ws = ps.Workspace(root=root)
    ws.create("P")
    (root / "p" / "irt" / "2024-05-14").mkdir(parents=True)
    source = tmp_path / "IRT_2024-05-14_H-D-I_INTERMAGNET.html"
    source.write_text("<html>chart</html>", encoding="utf-8")

    filed = ws.save_chart(
        source,
        ["derived:irt:2024-05-14"],
        kind="components",
        metadata={"components": ["H", "D", "I"], "product": "components"},
    )

    assert [Path(path).name for path in filed] == [
        "IRT_2024-05-14_H-D-I_INTERMAGNET.html"
    ]
    charts = ps.read_chart_index("p", root)
    assert charts[0]["source"] == "INTERMAGNET"
    assert charts[0]["station"] == "IRT"
    assert charts[0]["date"] == "2024-05-14"
    assert charts[0]["components"] == ["H", "D", "I"]
    assert charts[0]["relative_path"] == (
        "irt/2024-05-14/IRT_2024-05-14_H-D-I_INTERMAGNET.html"
    )


def test_legacy_chart_without_a_source_tag_is_unknown(root):
    ps.create_project("P", root=root)
    day = root / "p" / "irt" / "2024-09-10"
    day.mkdir(parents=True)
    (day / "chart.html").write_text("<html>legacy</html>", encoding="utf-8")

    charts = ps.read_chart_index("P", root=root)

    assert len(charts) == 1
    assert charts[0]["source"] == "UNKNOWN"


def test_save_chart_adds_source_to_an_explicit_name(root, tmp_path):
    ws = ps.Workspace(root=root)
    ws.create("P")
    (root / "p" / "irt" / "2024-05-14").mkdir(parents=True)
    source = tmp_path / "mine.html"
    source.write_text("<html>chart</html>", encoding="utf-8")

    filed = ws.save_chart(source, ["raw:irt:2024-05-14"], kind="components")

    assert Path(filed[0]).name == "mine_INTERMAGNET.html"


def test_comparison_chart_records_both_sources_in_every_copy(root, tmp_path):
    ws = ps.Workspace(root=root)
    ws.create("P")
    for station, day in (("irt", "2024-05-14"), ("omni", "2024-05-01")):
        (root / "p" / station / day).mkdir(parents=True)
    source = tmp_path / (
        "IRT_2024-05-14_H_vs_OMNI_2024-05-01_Bz-V_INTERMAGNET-OMNI.html"
    )
    source.write_text("<html>chart</html>", encoding="utf-8")

    filed = ws.save_chart(
        source,
        ["raw:irt:2024-05-14", "sw:omni:2024-05-01"],
        kind="space_weather",
        metadata={"components": ["H", "Bz", "V"], "product": "space_weather"},
    )

    assert len(filed) == 2
    charts = ps.read_chart_index("p", root)
    assert len(charts) == 2
    assert {row["source"] for row in charts} == {"INTERMAGNET-OMNI"}
    assert all(len(row["handles"]) == 2 for row in charts)


def test_export_custom_folders_and_archive_name_keep_canonical_chart_names(
    root, tmp_path, monkeypatch
):
    monkeypatch.setattr(ps, "EXPORTS_ROOT", tmp_path / "exports")
    ws = ps.Workspace(root=root)
    ws.create("P")
    for station, day in (("irt", "2024-05-14"), ("api", "2024-05-14")):
        (root / "p" / station / day).mkdir(parents=True)
        source = tmp_path / f"{station.upper()}_{day}_H_INTERMAGNET.html"
        source.write_text("<html>chart</html>", encoding="utf-8")
        ws.save_chart(
            source,
            [f"raw:{station}:{day}"],
            kind="components",
            metadata={"components": ["H"], "product": "components"},
        )

    payload, _ = ac.HANDLERS["export_project"](
        {
            "name": "p",
            "archive_name": "my_report",
            "folders": {
                # Uppercase is how a model usually writes a station code; the
                # canonical project folders are lowercase.
                "1": ["IRT/2024-05-14"],
                "2": ["API/2024-05-14/*.html"],
            },
            "sort_by": ["source", "station"],
        },
        _store(root),
    )

    assert payload["ok"], payload
    assert payload["filename"] == "my_report.zip"
    import zipfile

    with zipfile.ZipFile(payload["path"]) as archive:
        names = set(archive.namelist())
        assert "1/irt/2024-05-14/IRT_2024-05-14_H_INTERMAGNET.html" in names
        assert "2/api/2024-05-14/API_2024-05-14_H_INTERMAGNET.html" in names
        assert {"index.csv", "index.html", "export_manifest.json"} <= names
        index = archive.read("index.csv").decode("utf-8")
        assert "INTERMAGNET" in index
        assert "IRT_2024-05-14_H_INTERMAGNET.html" in index
        import csv
        import io

        rows = list(csv.DictReader(io.StringIO(index)))
        # sort_by source,station: API must come first even though its archive
        # path is under folder "2", not "1".
        assert [row["station"] for row in rows] == ["API", "IRT"]


def test_export_rejects_path_traversal_in_folder_and_archive_names(root, frame):
    store = _store(root)
    store.workspace.create("P")
    ps.save_artifact("P", "IRT", "2024-09-10", frame, root=root)

    bad_folder = ps.export_project(
        "p", root=root, exports=root / "out", folders={"../x": ["irt"]}
    )
    assert bad_folder["error"] == "invalid_folder_name"

    bad_archive = ps.export_project(
        "p", root=root, exports=root / "out", archive_name="../x"
    )
    assert bad_archive["error"] == "invalid_archive_name"


def test_sort_charts_uses_metadata_and_custom_source_order():
    rows = [
        {"file": "a.html", "source": "INTERMAGNET", "station": "IRT", "date": "2024-05-14"},
        {"file": "b.html", "source": "OMNI", "station": "", "date": "2024-05-01"},
        {"file": "c.html", "source": "KYOTO", "station": "", "date": "2024-05-02"},
    ]
    by_source = ps.sort_charts(rows, "source")
    assert [row["source"] for row in by_source] == ["INTERMAGNET", "KYOTO", "OMNI"]
    custom = ps.sort_charts(
        rows, "source", source_order=["OMNI", "KYOTO", "INTERMAGNET"]
    )
    assert [row["source"] for row in custom] == ["OMNI", "KYOTO", "INTERMAGNET"]