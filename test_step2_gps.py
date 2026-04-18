"""
Unit tests for step2_gps.py — DBSCAN clustering, density computation,
GPS trajectory generation, and speed filtering logic.
"""
import math
import numpy as np
import pandas as pd
import pytest

from step2_gps import (
    run_dbscan,
    compute_density,
    generate_gps_points,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    DENSITY_CLUSTER_WEIGHT,
    DENSITY_BASE_WEIGHT,
    MAX_DBSCAN_POINTS,
    CROWD_HIGH_THRESHOLD,
    CROWD_MEDIUM_THRESHOLD,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

def _make_poi(base_density=0.80, avg_dwell=45, simulated_tracks=10):
    return {
        "name": "Test POI",
        "name_en": "Test POI",
        "lat": 18.7871,
        "lon": 98.9849,
        "category": "Temple",
        "description": "A test POI.",
        "base_density": base_density,
        "avg_dwell_minutes": avg_dwell,
        "simulated_tracks": simulated_tracks,
        "overtourism_risk": "high",
        "peak_hours": "09:00–12:00",
    }


def _make_points_df(n=20, center=(18.787, 98.985), spread=0.0005, poi_name="Test POI"):
    """Create a tight cluster of GPS points around `center`."""
    rng = np.random.default_rng(42)
    lats = center[0] + rng.normal(0, spread, n)
    lons = center[1] + rng.normal(0, spread, n)
    return pd.DataFrame({
        "lat": lats,
        "lon": lons,
        "dwell_minutes": [45.0] * n,
        "poi_name": [poi_name] * n,
    })


def _make_clustered_df(n_inliers=80, n_noise=20, cluster_id=0):
    """Create a DataFrame that looks like DBSCAN output."""
    rows = []
    for i in range(n_inliers):
        rows.append({"lat": 18.787, "lon": 98.985, "dwell_minutes": 45.0,
                     "poi_name": "POI", "cluster_id": cluster_id})
    for i in range(n_noise):
        rows.append({"lat": 18.700, "lon": 98.900, "dwell_minutes": 45.0,
                     "poi_name": "POI", "cluster_id": -1})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# run_dbscan
# ─────────────────────────────────────────────────────────────

class TestRunDbscan:
    def test_returns_dataframe_with_cluster_id_column(self):
        df = _make_points_df(n=20)
        result = run_dbscan(df)
        assert "cluster_id" in result.columns

    def test_dense_cluster_has_inliers(self):
        # 30 tightly packed points should form at least one DBSCAN cluster
        df = _make_points_df(n=30, spread=0.0003)
        result = run_dbscan(df)
        inliers = result[result["cluster_id"] >= 0]
        assert len(inliers) > 0

    def test_too_few_points_all_noise(self):
        # Fewer than DBSCAN_MIN_SAMPLES → early-return, all labelled -1
        df = _make_points_df(n=DBSCAN_MIN_SAMPLES - 1)
        result = run_dbscan(df)
        assert (result["cluster_id"] == -1).all()

    def test_original_columns_preserved(self):
        df = _make_points_df(n=20)
        result = run_dbscan(df)
        for col in ["lat", "lon", "dwell_minutes", "poi_name"]:
            assert col in result.columns

    def test_input_dataframe_not_mutated(self):
        df = _make_points_df(n=20)
        original_cols = list(df.columns)
        run_dbscan(df)
        assert list(df.columns) == original_cols
        assert "cluster_id" not in df.columns

    def test_output_length_equals_input_length(self):
        df = _make_points_df(n=25)
        result = run_dbscan(df)
        assert len(result) == len(df)

    def test_widely_spread_points_all_noise(self):
        # 4 points < DBSCAN_MIN_SAMPLES (5) → early-return path, all labelled -1
        lats = [18.70, 18.75, 18.80, 18.85]
        lons = [98.90, 98.95, 99.00, 99.05]
        df = pd.DataFrame({
            "lat": lats, "lon": lons,
            "dwell_minutes": [45.0] * 4,
            "poi_name": ["POI"] * 4,
        })
        result = run_dbscan(df)
        assert (result["cluster_id"] == -1).all()

    def test_exceeds_max_points_raises(self):
        # run_dbscan has an assertion that input <= MAX_DBSCAN_POINTS
        n = MAX_DBSCAN_POINTS + 1
        df = pd.DataFrame({
            "lat": [18.787] * n,
            "lon": [98.985] * n,
            "dwell_minutes": [45.0] * n,
            "poi_name": ["POI"] * n,
        })
        with pytest.raises(AssertionError):
            run_dbscan(df)


# ─────────────────────────────────────────────────────────────
# compute_density
# ─────────────────────────────────────────────────────────────

class TestComputeDensity:
    def test_all_inliers_gives_max_cluster_ratio(self):
        poi = _make_poi(base_density=0.80, avg_dwell=45)
        df = _make_clustered_df(n_inliers=100, n_noise=0)
        density, _ = compute_density(poi, df)
        # cluster_ratio = 1.0 → density = 0.6*1.0 + 0.4*0.8 = 0.92
        expected = round(DENSITY_CLUSTER_WEIGHT * 1.0 + DENSITY_BASE_WEIGHT * 0.80, 3)
        assert density == pytest.approx(expected, abs=0.001)

    def test_no_inliers_uses_base_density_only(self):
        poi = _make_poi(base_density=0.65, avg_dwell=30)
        df = _make_clustered_df(n_inliers=0, n_noise=50)
        density, _ = compute_density(poi, df)
        # cluster_ratio = 0.0 → density = 0.6*0.0 + 0.4*0.65 = 0.26
        expected = round(DENSITY_CLUSTER_WEIGHT * 0.0 + DENSITY_BASE_WEIGHT * 0.65, 3)
        assert density == pytest.approx(expected, abs=0.001)

    def test_empty_dataframe_falls_back_to_base_density(self):
        poi = _make_poi(base_density=0.70, avg_dwell=60)
        df = pd.DataFrame(columns=["lat", "lon", "dwell_minutes", "poi_name", "cluster_id"])
        density, avg_dwell = compute_density(poi, df)
        assert density == poi["base_density"]
        assert avg_dwell == poi["avg_dwell_minutes"]

    def test_mixed_inliers_noise_blends_correctly(self):
        poi = _make_poi(base_density=0.80, avg_dwell=45)
        df = _make_clustered_df(n_inliers=60, n_noise=40)
        density, _ = compute_density(poi, df)
        cluster_ratio = 60 / 100
        expected = round(
            DENSITY_CLUSTER_WEIGHT * cluster_ratio + DENSITY_BASE_WEIGHT * 0.80, 3
        )
        assert density == pytest.approx(expected, abs=0.001)

    def test_density_clamped_to_unit_interval(self):
        poi = _make_poi(base_density=1.0)
        df = _make_clustered_df(n_inliers=100, n_noise=0)
        density, _ = compute_density(poi, df)
        assert 0.0 <= density <= 1.0

    def test_avg_dwell_computed_from_data(self):
        poi = _make_poi(avg_dwell=45)
        df = _make_clustered_df(n_inliers=10, n_noise=10)
        # All rows have dwell_minutes=45 in _make_clustered_df
        _, avg_dwell = compute_density(poi, df)
        assert avg_dwell == pytest.approx(45.0, abs=0.5)

    def test_avg_dwell_falls_back_when_no_data(self):
        poi = _make_poi(avg_dwell=90)
        df = pd.DataFrame(columns=["lat", "lon", "dwell_minutes", "poi_name", "cluster_id"])
        _, avg_dwell = compute_density(poi, df)
        assert avg_dwell == 90


# ─────────────────────────────────────────────────────────────
# generate_gps_points
# ─────────────────────────────────────────────────────────────

class TestGenerateGpsPoints:
    def test_returns_list_of_dicts(self):
        poi = _make_poi(simulated_tracks=2)
        points = generate_gps_points(poi, n_tracks=2)
        assert isinstance(points, list)
        assert all(isinstance(p, dict) for p in points)

    def test_each_point_has_required_keys(self):
        poi = _make_poi(simulated_tracks=2)
        points = generate_gps_points(poi, n_tracks=2)
        required = {"lat", "lon", "dwell_minutes", "poi_name"}
        for p in points:
            assert required.issubset(p.keys())

    def test_points_not_empty_for_valid_poi(self):
        poi = _make_poi(simulated_tracks=3)
        points = generate_gps_points(poi, n_tracks=3)
        assert len(points) > 0

    def test_poi_name_matches(self):
        poi = _make_poi(simulated_tracks=2)
        poi["name"] = "My Test POI"
        points = generate_gps_points(poi, n_tracks=2)
        assert all(p["poi_name"] == "My Test POI" for p in points)

    def test_respects_max_dbscan_points_cap(self):
        poi = _make_poi(simulated_tracks=10_000)
        points = generate_gps_points(poi, n_tracks=10_000)
        assert len(points) <= MAX_DBSCAN_POINTS

    def test_coordinates_are_finite(self):
        poi = _make_poi(simulated_tracks=3)
        points = generate_gps_points(poi, n_tracks=3)
        for p in points:
            assert math.isfinite(p["lat"])
            assert math.isfinite(p["lon"])

    def test_dwell_minutes_positive(self):
        poi = _make_poi(avg_dwell=45, simulated_tracks=3)
        points = generate_gps_points(poi, n_tracks=3)
        for p in points:
            assert p["dwell_minutes"] > 0

    def test_deterministic_with_same_poi(self):
        # generate_gps_points seeds the RNG from hash(poi["name"]) % 100_000
        # so two calls with the identical POI dict must produce identical output.
        poi = _make_poi(simulated_tracks=3)
        points1 = generate_gps_points(poi, n_tracks=3)
        points2 = generate_gps_points(poi, n_tracks=3)
        assert len(points1) == len(points2)
        for p1, p2 in zip(points1, points2):
            assert p1["lat"] == pytest.approx(p2["lat"], abs=1e-9)
            assert p1["lon"] == pytest.approx(p2["lon"], abs=1e-9)


# ─────────────────────────────────────────────────────────────
# Constants sanity checks
# ─────────────────────────────────────────────────────────────

class TestConstants:
    def test_density_weights_sum_to_one(self):
        assert DENSITY_CLUSTER_WEIGHT + DENSITY_BASE_WEIGHT == pytest.approx(1.0)

    def test_dbscan_eps_positive(self):
        assert DBSCAN_EPS > 0

    def test_dbscan_min_samples_at_least_2(self):
        assert DBSCAN_MIN_SAMPLES >= 2

    def test_crowd_thresholds_ordered(self):
        assert CROWD_MEDIUM_THRESHOLD < CROWD_HIGH_THRESHOLD

    def test_max_dbscan_points_reasonable(self):
        assert 100 <= MAX_DBSCAN_POINTS <= 100_000