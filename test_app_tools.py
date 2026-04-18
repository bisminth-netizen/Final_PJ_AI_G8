"""
Unit tests for app.py — RAG retrieval, agent tools, and tool routing.

Streamlit and UI libraries are mocked in conftest.py before this module
is imported, so importing from app is safe.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from app import (
    rag_retrieve,
    tool_get_hotspot,
    tool_get_sentiment,
    tool_search_poi,
    exec_tool,
)


# ─────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────

SAMPLE_HOTSPOTS = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "name": "Wat Phra Singh",
                "name_en": "Wat Phra Singh",
                "category": "Temple / Heritage",
                "description": "Most revered temple in Chiang Mai.",
                "gps_density": 0.85,
                "crowd_level": "High",
                "avg_dwell_minutes": 45.0,
                "total_tracks": 280,
                "peak_hours": "09:00–12:00",
                "overtourism_risk": "high",
                "visit_advice": "Visit early morning to avoid crowds.",
            },
            "geometry": {"type": "Point", "coordinates": [98.9849, 18.7871]},
        },
        {
            "type": "Feature",
            "properties": {
                "name": "Sunday Walking Street",
                "name_en": "Sunday Walking Street (Wualai Rd)",
                "category": "Market / Walking Street",
                "description": "Lively Sunday night market.",
                "gps_density": 0.92,
                "crowd_level": "Very High",
                "avg_dwell_minutes": 90.0,
                "total_tracks": 420,
                "peak_hours": "17:00–22:00",
                "overtourism_risk": "very_high",
                "visit_advice": "Arrive before 17:00 for easier access.",
            },
            "geometry": {"type": "Point", "coordinates": [98.9951, 18.7869]},
        },
        {
            "type": "Feature",
            "properties": {
                "name": "Mae Kampong Village",
                "name_en": "Ban Mae Kampong Eco-Village",
                "category": "Eco-Tourism Village",
                "description": "Charming mountain eco-village.",
                "gps_density": 0.48,
                "crowd_level": "Low",
                "avg_dwell_minutes": 180.0,
                "total_tracks": 90,
                "peak_hours": "09:00–17:00",
                "overtourism_risk": "low",
                "visit_advice": "Any time is good.",
            },
            "geometry": {"type": "Point", "coordinates": [99.1653, 18.8219]},
        },
    ],
}


def _make_sentiment_df():
    return pd.DataFrame([
        {"poi": "Wat Phra Singh", "text": "Beautiful temple, highly recommend!", "sentiment": "positive", "sentiment_score": 0.95},
        {"poi": "Wat Phra Singh", "text": "Very crowded, hard to enjoy.", "sentiment": "negative", "sentiment_score": 0.82},
        {"poi": "Wat Phra Singh", "text": "Typical temple visit.", "sentiment": "neutral", "sentiment_score": 0.71},
        {"poi": "Sunday Walking Street", "text": "Amazing night market!", "sentiment": "positive", "sentiment_score": 0.93},
    ])


def _make_docs():
    return [
        {"text": "Wat Phra Singh is Chiang Mai's most revered temple.", "poi": "Wat Phra Singh", "type": "guide"},
        {"text": "GPS density at Doi Suthep is 0.88, crowd level High.", "poi": "Wat Doi Suthep", "type": "hotspot"},
        {"text": "Night Bazaar has many shopping options and food stalls.", "poi": "Night Bazaar", "type": "guide"},
    ]


# ─────────────────────────────────────────────────────────────
# rag_retrieve
# ─────────────────────────────────────────────────────────────

class TestRagRetrieve:
    def _mock_model(self, dim=384):
        m = MagicMock()
        m.transform.return_value = np.ones((1, dim), dtype=np.float32)
        return m

    def _mock_index(self, scores, indices):
        idx = MagicMock()
        idx.search.return_value = (
            np.array([scores], dtype=np.float32),
            np.array([indices], dtype=np.int64),
        )
        return idx

    def test_returns_list_of_dicts(self):
        docs = _make_docs()
        model = self._mock_model()
        index = self._mock_index([0.9, 0.8], [0, 1])
        results = rag_retrieve(model, "temple", index, docs, top_k=2)
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_result_has_required_keys(self):
        docs = _make_docs()
        model = self._mock_model()
        index = self._mock_index([0.9], [0])
        results = rag_retrieve(model, "temple", index, docs, top_k=1)
        assert {"text", "score", "poi", "type"}.issubset(results[0].keys())

    def test_correct_text_returned(self):
        docs = _make_docs()
        model = self._mock_model()
        index = self._mock_index([0.9, 0.8, 0.7], [0, 1, 2])
        results = rag_retrieve(model, "temple", index, docs, top_k=3)
        assert results[0]["text"] == docs[0]["text"]
        assert results[1]["text"] == docs[1]["text"]

    def test_score_matches_index_output(self):
        docs = _make_docs()
        model = self._mock_model()
        index = self._mock_index([0.95], [0])
        results = rag_retrieve(model, "query", index, docs, top_k=1)
        assert results[0]["score"] == pytest.approx(0.95, abs=1e-4)

    def test_poi_field_populated(self):
        docs = _make_docs()
        model = self._mock_model()
        index = self._mock_index([0.9], [0])
        results = rag_retrieve(model, "query", index, docs, top_k=1)
        assert results[0]["poi"] == "Wat Phra Singh"

    def test_out_of_bounds_index_skipped(self):
        docs = _make_docs()
        model = self._mock_model()
        # Index 99 is out of bounds for a 3-doc list
        index = self._mock_index([0.9, 0.8], [0, 99])
        results = rag_retrieve(model, "query", index, docs, top_k=2)
        assert len(results) == 1  # only the valid index
        assert results[0]["poi"] == "Wat Phra Singh"

    def test_model_error_returns_error_dict(self):
        model = MagicMock()
        model.transform.side_effect = RuntimeError("embedding failed")
        index = MagicMock()
        results = rag_retrieve(model, "query", index, _make_docs(), top_k=3)
        assert len(results) == 1
        assert results[0]["type"] == "error"
        assert "error" in results[0]["text"].lower()

    def test_empty_docs_returns_empty(self):
        model = self._mock_model()
        index = self._mock_index([], [])
        results = rag_retrieve(model, "query", index, [], top_k=5)
        assert results == []


# ─────────────────────────────────────────────────────────────
# tool_get_hotspot
# ─────────────────────────────────────────────────────────────

class TestToolGetHotspot:
    def test_exact_name_match_returns_data(self):
        result = tool_get_hotspot("Wat Phra Singh", SAMPLE_HOTSPOTS)
        assert "GPS Hotspot" in result
        assert "0.85" in result

    def test_case_insensitive_match(self):
        result = tool_get_hotspot("wat phra singh", SAMPLE_HOTSPOTS)
        assert "GPS Hotspot" in result

    def test_partial_name_match(self):
        result = tool_get_hotspot("Singh", SAMPLE_HOTSPOTS)
        assert "GPS Hotspot" in result

    def test_crowd_level_included(self):
        result = tool_get_hotspot("Wat Phra Singh", SAMPLE_HOTSPOTS)
        assert "High" in result

    def test_avg_dwell_included(self):
        result = tool_get_hotspot("Wat Phra Singh", SAMPLE_HOTSPOTS)
        assert "45" in result

    def test_overtourism_risk_included(self):
        result = tool_get_hotspot("Wat Phra Singh", SAMPLE_HOTSPOTS)
        assert "high" in result.lower()

    def test_not_found_returns_helpful_message(self):
        # app.py: "No GPS data found for '{name}'."
        result = tool_get_hotspot("Unknown Place XYZ", SAMPLE_HOTSPOTS)
        assert "No GPS data found" in result

    def test_no_hotspot_data_returns_unavailable(self):
        # app.py: "No GPS hotspot data available."
        result = tool_get_hotspot("Wat Phra Singh", None)
        assert "No GPS hotspot data" in result

    def test_empty_features_returns_not_found(self):
        result = tool_get_hotspot("Wat Phra Singh", {"features": []})
        assert "No GPS data found" in result

    def test_name_en_also_searched(self):
        result = tool_get_hotspot("Wualai", SAMPLE_HOTSPOTS)
        assert "GPS Hotspot" in result


# ─────────────────────────────────────────────────────────────
# tool_get_sentiment
# ─────────────────────────────────────────────────────────────

class TestToolGetSentiment:
    def test_matching_poi_returns_report(self):
        df = _make_sentiment_df()
        result = tool_get_sentiment("Wat Phra Singh", df)
        assert "Sentiment Report" in result

    def test_review_count_included(self):
        df = _make_sentiment_df()
        result = tool_get_sentiment("Wat Phra Singh", df)
        assert "3" in result  # 3 reviews for Wat Phra Singh

    def test_positive_percentage_included(self):
        df = _make_sentiment_df()
        result = tool_get_sentiment("Wat Phra Singh", df)
        # 1/3 positive → ~33%
        assert "33" in result or "34" in result

    def test_negative_review_included(self):
        df = _make_sentiment_df()
        result = tool_get_sentiment("Wat Phra Singh", df)
        # tool_get_sentiment uses "Top concern" label for the negative example
        assert "concern" in result.lower() or "crowded" in result.lower()

    def test_case_insensitive_match(self):
        df = _make_sentiment_df()
        result = tool_get_sentiment("wat phra singh", df)
        assert "Sentiment Report" in result

    def test_partial_name_match(self):
        df = _make_sentiment_df()
        result = tool_get_sentiment("Phra", df)
        assert "Sentiment Report" in result

    def test_no_match_returns_stop_message(self):
        # app.py returns "STOP — No sentiment data for '{name}'." (not "No sentiment data found")
        df = _make_sentiment_df()
        result = tool_get_sentiment("Doi Inthanon", df)
        assert "No sentiment data" in result

    def test_none_dataframe_returns_unavailable(self):
        # app.py: "No sentiment data available."
        result = tool_get_sentiment("Wat Phra Singh", None)
        assert "No sentiment data available" in result


# ─────────────────────────────────────────────────────────────
# tool_search_poi
# ─────────────────────────────────────────────────────────────

class TestToolSearchPoi:
    def test_keyword_match_by_name(self):
        result = tool_search_poi("temple", SAMPLE_HOTSPOTS)
        assert "Wat Phra Singh" in result

    def test_keyword_match_by_category(self):
        result = tool_search_poi("Market", SAMPLE_HOTSPOTS)
        assert "Walking Street" in result

    def test_keyword_match_by_description(self):
        result = tool_search_poi("eco", SAMPLE_HOTSPOTS)
        assert "Mae Kampong" in result

    def test_no_match_returns_not_found(self):
        # app.py: "No POI found for '{kw}'."
        result = tool_search_poi("underwater cave diving", SAMPLE_HOTSPOTS)
        assert "No POI found" in result

    def test_none_hotspot_returns_unavailable(self):
        # app.py: "No POI data available."
        result = tool_search_poi("temple", None)
        assert "No POI data available" in result

    def test_case_insensitive_search(self):
        result = tool_search_poi("TEMPLE", SAMPLE_HOTSPOTS)
        assert "Wat Phra Singh" in result

    def test_density_shown_in_results(self):
        result = tool_search_poi("temple", SAMPLE_HOTSPOTS)
        assert "density=" in result

    def test_multiple_matches_returned(self):
        # "village" appears in Mae Kampong description and name_en
        result = tool_search_poi("village", SAMPLE_HOTSPOTS)
        assert "Mae Kampong" in result


# ─────────────────────────────────────────────────────────────
# exec_tool (routing)
# ─────────────────────────────────────────────────────────────

class TestExecTool:
    def _exec(self, tool_name, args, hs=None, df=None, em=None, docs=None, idx=None):
        return exec_tool(tool_name, args, hs, df, em, docs, idx)

    def test_routes_to_get_hotspot(self):
        result = self._exec("get_hotspot", "Wat Phra Singh", hs=SAMPLE_HOTSPOTS)
        assert "GPS Hotspot" in result

    def test_routes_to_get_sentiment(self):
        df = _make_sentiment_df()
        result = self._exec("get_sentiment", "Wat Phra Singh", df=df)
        assert "Sentiment Report" in result

    def test_routes_to_search_poi(self):
        result = self._exec("search_poi", "temple", hs=SAMPLE_HOTSPOTS)
        assert "Wat Phra Singh" in result

    def test_routes_to_rag_retrieve(self):
        docs = _make_docs()
        mock_em = MagicMock()
        mock_em.transform.return_value = np.ones((1, 384), dtype=np.float32)
        mock_idx = MagicMock()
        mock_idx.search.return_value = (
            np.array([[0.9]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )
        result = self._exec("rag_retrieve", "temple visit", em=mock_em, docs=docs, idx=mock_idx)
        # app.py exec_tool wraps results as "Knowledge Base Results:\n..."
        assert "Wat Phra Singh" in result or "Knowledge Base" in result

    def test_rag_retrieve_no_results_returns_message(self):
        mock_em = MagicMock()
        mock_em.transform.return_value = np.ones((1, 384), dtype=np.float32)
        mock_idx = MagicMock()
        mock_idx.search.return_value = (
            np.array([[]], dtype=np.float32),
            np.array([[]], dtype=np.int64),
        )
        result = self._exec("rag_retrieve", "query", em=mock_em, docs=[], idx=mock_idx)
        # app.py: "No relevant knowledge base results found." when rag_retrieve returns []
        assert "No relevant" in result or "Knowledge Base" in result or "error" in result.lower()

    def test_unknown_tool_returns_error_message(self):
        # app.py: "Unknown tool: {name}"
        result = self._exec("fly_to_moon", "arg")
        assert "Unknown tool" in result