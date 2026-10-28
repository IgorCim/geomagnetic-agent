"""Canonical chart names: deterministic, source-last, and collision-safe."""

from pathlib import Path

import chart_naming as cn


def test_component_names_put_source_last():
    assert (
        cn.components_filename("raw:irt:2024-12-10", ["F"])
        == "IRT_2024-12-10_F_INTERMAGNET.html"
    )
    assert (
        cn.components_filename("derived:irt:2024-12-10", ["H", "D", "I"])
        == "IRT_2024-12-10_H-D-I_INTERMAGNET.html"
    )
    assert (
        cn.components_filename("sw:omni:2024-05-01", ["Bz", "V"])
        == "2024-05-01_Bz-V_OMNI.html"
    )
    assert (
        cn.components_filename("idx:kyoto:2024-05-01", ["Dst", "Kp"])
        == "2024-05-01_Dst-Kp_KYOTO.html"
    )
    assert (
        cn.components_filename("idx:supermag:2024-05-01", ["SME", "SML", "SMU"])
        == "2024-05-01_SME-SML-SMU_SUPERMAG.html"
    )
    assert (
        cn.components_filename("smag:irt:2024-05-01", ["dX", "dY", "dZ"])
        == "IRT_2024-05-01_dX-dY-dZ_SUPERMAG.html"
    )
    # A derived frame carries raw columns too; tag only the overlapping plot so
    # raw F and derived F cannot overwrite each other.
    assert (
        cn.components_filename("derived:irt:2024-12-10", ["F"])
        == "IRT_2024-12-10_F_derived_INTERMAGNET.html"
    )
    assert (
        cn.components_filename("raw:irt:2024-12-10", ["H"])
        == "IRT_2024-12-10_H_INTERMAGNET.html"
    )


def test_comparison_and_space_weather_names_keep_both_sources():
    assert (
        cn.comparison_filename(
            "raw:irt:2024-09-10", "raw:irt:2024-09-11", "H"
        )
        == "IRT_2024-09-10_vs_2024-09-11_H_INTERMAGNET.html"
    )
    assert (
        cn.space_weather_filename(
            "raw:irt:2024-05-14",
            "sw:omni:2024-05-01",
            "H",
            ["Bz", "V"],
        )
        == "IRT_2024-05-14_H_vs_OMNI_2024-05-01_Bz-V_INTERMAGNET-OMNI.html"
    )


def test_space_weather_title_does_not_print_descriptor_as_data_date():
    title = cn.space_weather_title(
        "raw:irt:2024-05-18", "sw:omni:2024-05-01", "H", ["Bz", "V"]
    )
    assert "OMNI (sw:omni:2024-05-01)" in title
    assert "OMNI 01.05.2024" not in title
    same_day = cn.space_weather_title(
        "raw:irt:2024-05-18", "sw:omni:2024-05-18", "H", ["Bz", "V"]
    )
    assert "OMNI 18.05.2024" in same_day


def test_overlay_name_and_custom_suffix():
    assert (
        cn.overlay_filename(["BRW", "SHU", "FRD"], "2024-05-14", "H")
        == "OVERLAY_BRW-SHU-FRD_2024-05-14_H_INTERMAGNET.html"
    )
    assert cn.ensure_source_suffix("mine.html", ["INTERMAGNET"]) == (
        "mine_INTERMAGNET.html"
    )
    # Replacing a partial source suffix must not double it.
    assert cn.ensure_source_suffix(
        "IRT_2024-12-10_F_INTERMAGNET.html", ["INTERMAGNET", "OMNI"]
    ) == "IRT_2024-12-10_F_INTERMAGNET-OMNI.html"
    assert cn.safe_filename("mine.html", ["OMNI"]) == "mine_OMNI.html"
    assert cn.safe_filename("mine.html") == "mine_UNKNOWN.html"
    # The old spelling used two source tokens; replacing it must not leave the
    # first one in the stem and create a double suffix.
    assert cn.ensure_source_suffix(
        "IRT_2024-12-10_F_INTERMAGNET_OMNI.html", ["INTERMAGNET", "OMNI"]
    ) == "IRT_2024-12-10_F_INTERMAGNET-OMNI.html"


def test_long_names_are_bounded_and_source_remains_last():
    name = cn.components_filename("raw:irt:2024-12-10", ["X" * 300])
    assert len(name) <= cn.MAX_STEM + 5
    assert Path(name).stem.endswith("_INTERMAGNET")


def test_unknown_handle_is_honest():
    info = cn.parse_handle("mystery:foo:2024-01-01")
    assert info.source == "UNKNOWN"
    assert cn.source_from_filename("IRT_2024-01-01_F_INTERMAGNET.html") == (
        "INTERMAGNET"
    )
    assert cn.source_from_filename("components.html") == "UNKNOWN"


def test_duplicate_components_and_stations_are_not_repeated():
    assert (
        cn.components_filename("raw:irt:2024-12-10", ["H", "H"])
        == "IRT_2024-12-10_H_INTERMAGNET.html"
    )
    assert (
        cn.overlay_filename(["IRT", "IRT"], "2024-05-14", "H")
        == "OVERLAY_IRT_2024-05-14_H_INTERMAGNET.html"
    )


def test_an_invalid_calendar_date_is_not_used_as_a_token():
    info = cn.parse_handle("raw:irt:2024-13-01")
    assert info.start is None
    assert "2024-13-01" not in cn.components_filename(
        "raw:irt:2024-13-01", ["H"]
    )


def test_an_arbitrary_long_source_becomes_unknown_and_stays_bounded():
    name = cn.ensure_source_suffix("chart.html", ["X" * 300])
    assert len(name) <= cn.MAX_STEM + 5
    assert name.endswith("_UNKNOWN.html")
    assert cn.source_from_filename("IRT_2024-01-01_F_" + "X" * 300 + ".html") == (
        "UNKNOWN"
    )
