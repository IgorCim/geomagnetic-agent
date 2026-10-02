"""Validation contour for :mod:`geomag_coords`.

The dipole transform is a **rotation**, so its correctness is pinned by
invariants that must hold for any rotation rather than by a table of expected
numbers: the pole maps to +90, angular separation is preserved, and the inverse
is exact. Those are stronger than spot-checking published geomagnetic latitudes,
and they would catch a sign error that a single reference value might not.

The haversine function is checked against closed forms that are exact for a
sphere (1 degree of latitude, a quarter turn, antipodes) rather than against
memorised real-world distances.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import geomag_coords as gc


@pytest.fixture
def stations():
    """Three European stations plus entries that must be rejected."""
    return [
        {"code": "IRT", "name": "Tromso", "lat": 69.58, "lon": 18.95},
        {"code": "NUR", "name": "Nurmijarvi", "lat": 60.50, "lon": 26.60},
        {"code": "JFZ", "name": "Jungfraujoch", "lat": 46.55, "lon": 7.98},
        {"code": "BAD", "name": "broken", "lat": "not a number", "lon": 12.0},
        {"code": "NOCO", "name": "no coordinates"},
    ]


# --------------------------------------------------------------------------- #
# 1. haversine
# --------------------------------------------------------------------------- #
def test_one_degree_of_latitude_is_111_km():
    """The exact closed form on a sphere: R * 1 degree in radians."""
    expected = gc.EARTH_RADIUS_KM * math.radians(1.0)
    assert gc.haversine_distance(0.0, 0.0, 1.0, 0.0) == pytest.approx(expected, rel=1e-12)


def test_quarter_turn_around_the_equator():
    expected = gc.EARTH_RADIUS_KM * math.pi / 2
    assert gc.haversine_distance(0.0, 0.0, 0.0, 90.0) == pytest.approx(expected, rel=1e-12)


def test_pole_to_pole_is_half_the_circumference():
    expected = gc.EARTH_RADIUS_KM * math.pi
    assert gc.haversine_distance(-90.0, 0.0, 90.0, 0.0) == pytest.approx(expected, rel=1e-12)


def test_identical_points_are_exactly_zero():
    assert gc.haversine_distance(52.0, 13.0, 52.0, 13.0) == 0.0


def test_distance_is_symmetric():
    forward = gc.haversine_distance(69.58, 18.95, 60.50, 26.60)
    backward = gc.haversine_distance(60.50, 26.60, 69.58, 18.95)
    assert forward == backward


def test_short_separations_stay_accurate():
    """Haversine beats the acos form here; a few metres of error is the bar.

    A precision regression would make two co-located observatories look like they
    were kilometres apart, so the small-separation case is asserted explicitly
    rather than left to the large-distance checks.
    """
    distance = gc.haversine_distance(52.0, 13.0, 52.001, 13.0)
    assert distance == pytest.approx(0.11119, abs=1e-4)


def test_antipodal_points_stay_inside_the_acos_domain():
    """Rounding must not push the haversine term past 1 for antipodes."""
    distance = gc.haversine_distance(0.0, 0.0, 0.0, 180.0)
    assert math.isfinite(distance)
    assert distance == pytest.approx(gc.EARTH_RADIUS_KM * math.pi, rel=1e-9)


def test_longitude_wraps_at_the_antimeridian():
    across = gc.haversine_distance(0.0, -179.0, 0.0, 179.0)
    direct = gc.haversine_distance(0.0, 0.0, 0.0, 2.0)
    assert across == pytest.approx(direct, rel=1e-12)


def test_longitude_outside_the_normal_range_is_normalised():
    """370 and 10 degrees are the same meridian, not an error.

    Compared with a tolerance: folding 370 back into range differs from 10 by
    ~1e-12 km through the rounding of the normalisation, so an exact equality
    would be asserting something stronger than the function promises.
    """
    wrapped = gc.haversine_distance(0.0, 370.0, 0.0, 10.0)
    reference = gc.haversine_distance(0.0, 10.0, 0.0, 10.0)
    assert wrapped == pytest.approx(reference, abs=1e-6)
    # and 370 really is the same meridian as 10
    assert wrapped == pytest.approx(0.0, abs=1e-6)


def test_distance_is_bounded_by_half_the_circumference():
    for lat1, lon1, lat2, lon2 in [
        (0, 0, 0, 180),
        (90, 0, -90, 0),
        (45, -179, 45, 179),
    ]:
        assert gc.haversine_distance(lat1, lon1, lat2, lon2) <= math.pi * gc.EARTH_RADIUS_KM


@pytest.mark.parametrize(
    "args",
    [(91.0, 0.0, 0.0, 0.0), (0.0, 0.0, -90.5, 0.0), (0.0, 0.0, 0.0, float("nan"))],
)
def test_invalid_coordinates_are_reported(args):
    result = gc.haversine_distance(*args)
    assert gc.is_error(result) and result["error"] == "invalid_input"


def test_non_numeric_coordinates_are_reported():
    assert gc.is_error(gc.haversine_distance("a", 0.0, 0.0, 0.0))
    assert gc.is_error(gc.haversine_distance(None, 0.0, 0.0, 0.0))


# --------------------------------------------------------------------------- #
# 2. the dipole transform -- rotation invariants
# --------------------------------------------------------------------------- #
def test_geomagnetic_pole_maps_to_90_degrees():
    """The defining property: the pole is the point the rotation centres on."""
    result = gc.geographic_to_geomagnetic(*gc.GEOMAGNETIC_POLE)
    assert result["geomag_lat"] == pytest.approx(90.0, abs=1e-6)


def test_antipole_maps_to_minus_90():
    pole_lat, pole_lon = gc.GEOMAGNETIC_POLE
    result = gc.geographic_to_geomagnetic(-pole_lat, pole_lon + 180.0)
    assert result["geomag_lat"] == pytest.approx(-90.0, abs=1e-6)


def test_longitude_at_the_pole_is_reported_as_undefined():
    """atan2 of two denormal floats is arbitrary, so it must not be returned."""
    result = gc.geographic_to_geomagnetic(*gc.GEOMAGNETIC_POLE)
    assert result["geomag_lon"] is None
    assert result["degenerate"] is True


def test_inverse_of_the_pole_is_the_pole():
    """The pole must round-trip, with its longitude reported as undefined.

    The inverse's degeneracy test keys on the *geomagnetic* latitude, which is
    90 here while the recovered geographic latitude is 80.7 -- testing the output
    latitude instead would silently never fire.
    """
    forward = gc.geographic_to_geomagnetic(*gc.GEOMAGNETIC_POLE)
    back = gc.geomagnetic_to_geographic(forward["geomag_lat"], 0.0)
    assert back["degenerate"] is True
    assert back["lon"] is None
    assert back["lat"] == pytest.approx(gc.GEOMAGNETIC_POLE[0], abs=1e-3)


@pytest.mark.parametrize("mlon", [0.0, 45.0, 179.0, -120.0])
def test_every_longitude_at_the_pole_maps_to_the_same_point(mlon):
    """At a pole all meridians meet, so the longitude must not matter."""
    back = gc.geomagnetic_to_geographic(90.0, mlon)
    assert back["lat"] == pytest.approx(gc.GEOMAGNETIC_POLE[0], abs=1e-3)
    assert back["lon"] is None


def _angular_separation(lat1, lon1, lat2, lon2) -> float:
    phi1, lam1 = math.radians(lat1), math.radians(lon1)
    phi2, lam2 = math.radians(lat2), math.radians(lon2)
    cosine = math.sin(phi1) * math.sin(phi2) + math.cos(phi1) * math.cos(phi2) * math.cos(
        lam1 - lam2
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


@pytest.mark.parametrize(
    "point_a, point_b",
    [
        ((69.58, 18.95), (60.50, 26.60)),
        ((0.0, 0.0), (0.0, 90.0)),
        ((45.0, -120.0), (-45.0, 60.0)),
        ((80.0, 10.0), (70.0, -170.0)),
        ((-30.0, 0.0), (30.0, 180.0)),
    ],
)
def test_transform_preserves_angular_separation(point_a, point_b):
    """A rotation preserves distances, so separation is an invariant.

    Tolerance is set by the 4-decimal rounding of the returned angles.
    """
    a = gc.geographic_to_geomagnetic(*point_a)
    b = gc.geographic_to_geomagnetic(*point_b)
    before = _angular_separation(point_a[0], point_a[1], point_b[0], point_b[1])
    after = _angular_separation(a["geomag_lat"], a["geomag_lon"], b["geomag_lat"], b["geomag_lon"])
    assert after == pytest.approx(before, abs=1e-3)


def test_round_trip_is_exact_over_a_dense_grid():
    """Exercised across both hemispheres and all meridians.

    Points whose forward longitude is undefined are skipped: at a pole the
    longitude was never real information, so there is nothing to recover and the
    inverse correctly reports None.
    """
    worst_lat = 0.0
    worst_lon = 0.0
    checked = 0
    for lat in range(-90, 91, 10):
        for lon in range(-180, 180, 20):
            forward = gc.geographic_to_geomagnetic(float(lat), float(lon))
            if forward["geomag_lon"] is None:
                continue  # a pole: longitude is undefined in both directions
            back = gc.geomagnetic_to_geographic(forward["geomag_lat"], forward["geomag_lon"])
            checked += 1
            worst_lat = max(worst_lat, abs(back["lat"] - lat))
            delta = abs((back["lon"] - lon + 180.0) % 360.0 - 180.0)
            worst_lon = max(worst_lon, delta)
    assert checked > 300, "the grid should exercise most of the sphere"
    assert worst_lat < 1e-3
    assert worst_lon < 1e-3


def test_output_longitude_stays_within_range():
    for lat in range(-90, 91, 15):
        for lon in range(-180, 180, 30):
            result = gc.geographic_to_geomagnetic(float(lat), float(lon))
            if result["geomag_lon"] is not None:
                assert -180.0 <= result["geomag_lon"] <= 180.0


@pytest.mark.parametrize("pole_lat", [90.0, -90.0])
def test_geographic_poles_report_an_undefined_longitude(pole_lat):
    """All meridians meet at a geographic pole, so longitude cannot be recovered.

    The projected components are perfectly healthy here, so a degeneracy test
    that only looked at their magnitude would report a confident longitude for
    every value of ``lon`` -- and it could never round trip, since the input
    longitude was never real information.
    """
    for lon in (-180.0, -90.0, 0.0, 45.0, 180.0):
        result = gc.geographic_to_geomagnetic(pole_lat, lon)
        assert result["degenerate"] is True, (pole_lat, lon)
        assert result["geomag_lon"] is None, (pole_lat, lon)
        assert abs(result["geomag_lat"]) <= 90.0


def test_a_point_near_but_not_at_the_pole_still_has_a_longitude():
    """The degeneracy test must not be so wide that it swallows real points."""
    result = gc.geographic_to_geomagnetic(89.999, 0.0)
    assert result["degenerate"] is False
    assert result["geomag_lon"] is not None


def test_output_latitude_never_leaves_the_poles():
    for lat in range(-90, 91, 5):
        result = gc.geographic_to_geomagnetic(float(lat), 0.0)
        assert -90.0 <= result["geomag_lat"] <= 90.0


def test_a_custom_pole_is_honoured():
    """A different pole must move the result, so the pole argument is real."""
    default = gc.geographic_to_geomagnetic(60.0, 20.0)
    shifted = gc.geographic_to_geomagnetic(60.0, 20.0, pole_lat=70.0, pole_lon=0.0)
    assert default["geomag_lat"] != shifted["geomag_lat"]
    assert shifted["pole_lat"] == 70.0


def test_the_model_is_recorded_so_approximations_are_not_mistaken_for_igrf():
    """A first-order dipole must never look like a full field model."""
    result = gc.geographic_to_geomagnetic(60.0, 20.0)
    assert result["model"] == "tilted_dipole"
    assert "first-order" in result["note"].lower()


@pytest.mark.parametrize(
    "geo_lat, lon, expected_range",
    [
        # published geomagnetic latitudes are well above the geographic ones for
        # Europe, because the geomagnetic pole sits over Canada, not over Europe
        (69.58, 18.95, (60.0, 75.0)),   # Tromso
        (60.50, 26.60, (50.0, 65.0)),   # Nurmijarvi
        (46.55, 7.98, (40.0, 55.0)),    # Jungfraujoch
    ],
)
def test_european_stations_match_published_geomagnetic_latitudes(geo_lat, lon, expected_range):
    result = gc.geographic_to_geomagnetic(geo_lat, lon)
    low, high = expected_range
    assert low <= result["geomag_lat"] <= high


def test_southern_hemisphere_station_gets_a_negative_geomagnetic_latitude():
    result = gc.geographic_to_geomagnetic(-33.9, 151.2)
    assert result["geomag_lat"] < 0


def test_invalid_latitude_is_rejected_not_clamped():
    """Clamping would hide a data bug behind a plausible-looking answer."""
    result = gc.geographic_to_geomagnetic(95.0, 0.0)
    assert gc.is_error(result) and result["error"] == "invalid_input"


def test_invalid_pole_is_reported():
    assert gc.is_error(gc.geographic_to_geomagnetic(60.0, 20.0, pole_lat=200.0))


# --------------------------------------------------------------------------- #
# 3. station sorting
# --------------------------------------------------------------------------- #
def test_sort_orders_by_geomagnetic_latitude(stations):
    result = gc.sort_stations_by_lat_m(stations)
    assert result["ok"]
    mlats = [s["geomag_lat"] for s in result["stations"]]
    assert mlats == sorted(mlats)


def test_sort_can_be_descending(stations):
    result = gc.sort_stations_by_lat_m(stations, descending=True)
    mlats = [s["geomag_lat"] for s in result["stations"]]
    assert mlats == sorted(mlats, reverse=True)
    assert result["order"] == "descending"


def test_unusable_stations_are_reported_by_name(stations):
    """Dropping a station silently would hide a missing-coordinates bug."""
    result = gc.sort_stations_by_lat_m(stations)
    skipped = {entry["station"] for entry in result["skipped"]}
    assert skipped == {"BAD", "NOCO"}
    assert len(result["stations"]) == 3


def test_sort_tags_the_source_of_each_geomagnetic_latitude():
    """A supplied geomagnetic latitude must be distinguishable from a computed one."""
    supplied = [{"code": "X", "lat": 60.0, "lon": 20.0, "geomag_lat": 70.5}]
    result = gc.sort_stations_by_lat_m(supplied)
    assert result["stations"][0]["geomag_lat"] == 70.5
    assert result["stations"][0]["source"] == "provided"


def test_sort_derives_geomagnetic_latitude_when_absent():
    result = gc.sort_stations_by_lat_m([{"code": "X", "lat": 60.0, "lon": 20.0}])
    assert result["stations"][0]["source"] == "derived"
    assert "geomag_lon" in result["stations"][0]


def test_sorting_accepts_objects_with_attributes():
    class Station:
        def __init__(self):
            self.code = "OBJ"
            self.lat = 60.0
            self.lon = 20.0

    result = gc.sort_stations_by_lat_m([Station()])
    assert result["ok"] and len(result["stations"]) == 1


def test_sorting_an_empty_list_is_not_an_error():
    result = gc.sort_stations_by_lat_m([])
    assert result["ok"] and result["stations"] == []


# --------------------------------------------------------------------------- #
# 4. overlay offsets
# --------------------------------------------------------------------------- #
def test_offsets_step_by_the_base_offset(stations):
    result = gc.calculate_offsets_for_overlay(stations[:3], base_offset=100)
    assert result["ok"]
    assert sorted(result["offsets"].values()) == [0.0, 100.0, 200.0]


def test_the_first_station_starts_at_zero(stations):
    result = gc.calculate_offsets_for_overlay(stations[:3])
    assert min(result["offsets"].values()) == 0.0


def test_offsets_scale_with_the_base_offset(stations):
    small = gc.calculate_offsets_for_overlay(stations[:3], base_offset=50)
    large = gc.calculate_offsets_for_overlay(stations[:3], base_offset=200)
    assert sorted(large["offsets"].values()) == [0.0, 200.0, 400.0]
    assert sorted(small["offsets"].values()) == [0.0, 50.0, 100.0]


def test_zero_base_offset_is_refused():
    """Stacking everything at zero would make the overlay unreadable."""
    result = gc.calculate_offsets_for_overlay([{"code": "A", "lat": 1.0, "lon": 1.0}], base_offset=0)
    assert gc.is_error(result) and result["error"] == "invalid_input"


def test_offsets_for_no_usable_station_is_an_error():
    result = gc.calculate_offsets_for_overlay([])
    assert gc.is_error(result) and result["error"] == "no_stations"


def test_offsets_are_labelled_display_only(stations):
    """A caller must not mistake an offset for part of the measurement."""
    result = gc.calculate_offsets_for_overlay(stations[:2])
    assert "display" in result["note"].lower()


# --------------------------------------------------------------------------- #
# 5. apply_offsets
# --------------------------------------------------------------------------- #
@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC"),
            "H": [25000.0, 25010.0, 25020.0, 25030.0],
        }
    )


def test_apply_offsets_writes_a_new_column(frame):
    result = gc.apply_offsets(frame, 100)
    assert not gc.is_error(result)
    assert list(result["H_offset"]) == [25100.0, 25110.0, 25120.0, 25130.0]


def test_apply_offsets_never_mutates_the_input(frame):
    before = frame.copy(deep=True)
    gc.apply_offsets(frame, 100)
    pd.testing.assert_frame_equal(frame, before)


def test_apply_offsets_keeps_the_original_component(frame):
    """The unshifted series must survive for any later numeric analysis."""
    result = gc.apply_offsets(frame, 100)
    assert list(result["H"]) == [25000.0, 25010.0, 25020.0, 25030.0]


def test_apply_offsets_accepts_a_custom_target_column(frame):
    result = gc.apply_offsets(frame, 50, new_column="SHIFTED")
    assert list(result["SHIFTED"]) == [25050.0, 25060.0, 25070.0, 25080.0]


def test_apply_offsets_rejects_a_missing_column(frame):
    result = gc.apply_offsets(frame, 100, column="Z")
    assert gc.is_error(result) and result["error"] == "missing_columns"


def test_apply_offsets_rejects_a_non_frame():
    assert gc.is_error(gc.apply_offsets([1, 2, 3], 100))


def test_offsets_round_trip_through_apply(frame):
    """A stored offset must be removable, which is what keeps the data honest."""
    offset = gc.calculate_offsets_for_overlay([{"code": "A", "lat": 60.0, "lon": 20.0}], base_offset=100)
    value = offset["offsets"]["A"]
    shifted = gc.apply_offsets(frame, value)
    restored = gc.apply_offsets(shifted, -value)
    np.testing.assert_allclose(
        restored["H_offset"].to_numpy(), frame["H"].to_numpy(), rtol=1e-12
    )