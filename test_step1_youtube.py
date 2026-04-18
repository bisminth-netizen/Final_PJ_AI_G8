"""
Unit tests for step1_youtube.py — language detection, place-relevance filter,
sentiment analysis pipeline, and API key validation.
"""
import sys
from unittest.mock import MagicMock, patch
import pytest

# Mock heavy optional dependencies before importing step1
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("googleapiclient", MagicMock())
sys.modules.setdefault("googleapiclient.discovery", MagicMock())
sys.modules.setdefault("googleapiclient.errors", MagicMock())

from step1_youtube import (
    detect_language,
    is_english,
    is_place_relevant,
    accept_comment,
    analyse_sentiment,
    _validate_api_key,
    CONFIDENCE_FLOOR,
    MIN_WORD_COUNT,
)


# ─────────────────────────────────────────────────────────────
# detect_language / is_english
# ─────────────────────────────────────────────────────────────

class TestDetectLanguage:
    def test_pure_english_returns_en(self):
        assert detect_language("This temple is absolutely beautiful!") == "en"

    def test_thai_script_returns_other(self):
        assert detect_language("วัดพระสิงห์สวยมากเลย") == "other"

    def test_empty_string_returns_other(self):
        assert detect_language("") == "other"

    def test_whitespace_only_returns_other(self):
        assert detect_language("   \t\n") == "other"

    def test_numbers_only_returns_other(self):
        # No alphabetic chars → ratio undefined → other
        assert detect_language("12345 6789") == "other"

    def test_mixed_mostly_english_returns_en(self):
        # High ASCII ratio → en
        assert detect_language("Great view from Doi Suthep วัด") == "en"

    def test_chinese_characters_returns_other(self):
        assert detect_language("这个寺庙非常漂亮") == "other"

    def test_english_with_emoji_returns_en(self):
        assert detect_language("Amazing place! 🌟🏯 Must visit!") == "en"

    def test_japanese_returns_other(self):
        assert detect_language("素晴らしい場所です") == "other"


class TestIsEnglish:
    def test_english_text_is_true(self):
        assert is_english("Crowded but worth the visit") is True

    def test_thai_text_is_false(self):
        assert is_english("สวยมากค่ะ") is False

    def test_empty_string_is_false(self):
        assert is_english("") is False


# ─────────────────────────────────────────────────────────────
# is_place_relevant
# ─────────────────────────────────────────────────────────────

class TestIsPlaceRelevant:
    def test_place_focused_comment_accepted(self):
        text = "The temple is beautiful and very crowded with tourists."
        assert is_place_relevant(text) is True

    def test_temple_keyword_with_poi_name(self):
        text = "Wat Phra Singh was stunning, the architecture is incredible."
        assert is_place_relevant(text, poi="Wat Phra Singh") is True

    def test_too_short_rejected(self):
        # Fewer than MIN_WORD_COUNT words
        text = "Nice place"
        assert is_place_relevant(text) is False

    def test_exactly_min_word_count_boundary(self):
        # A borderline word count with no clear place keywords — result may vary
        # depending on keyword scoring, but the function must not crash.
        text = "I saw the thing there it"  # 6 words, no place keywords
        result = is_place_relevant(text)
        assert isinstance(result, bool)

    def test_spam_subscribe_rejected(self):
        text = "Subscribe to my channel for more travel content videos"
        assert is_place_relevant(text) is False

    def test_spam_follow_me_rejected(self):
        text = "Please follow me on instagram for amazing travel photos"
        assert is_place_relevant(text) is False

    def test_spam_check_out_my_rejected(self):
        text = "Check out my channel for more beautiful temple videos here"
        assert is_place_relevant(text) is False

    def test_spam_nice_video_rejected(self):
        # "nice video" matches _SPAM_PATTERNS: (nice|good|...) (video|content|...)
        text = "This is a nice video with amazing content about travel"
        assert is_place_relevant(text) is False

    def test_person_focused_rejected(self):
        # "this vlogger is" matches _SPAM_PATTERNS
        text = "This vlogger is so funny and talented in presenting places"
        assert is_place_relevant(text) is False

    def test_crowd_density_comment_accepted(self):
        text = "It was extremely crowded and busy during peak morning hours."
        assert is_place_relevant(text) is True

    def test_food_and_market_comment_accepted(self):
        text = "The street market food was amazing, so many vendors and stalls."
        assert is_place_relevant(text) is True

    def test_access_and_timing_comment_accepted(self):
        text = "Morning visit is best, parking is easy and entrance fee is cheap."
        assert is_place_relevant(text) is True

    def test_empty_string_rejected(self):
        assert is_place_relevant("") is False


# ─────────────────────────────────────────────────────────────
# accept_comment (combined gate)
# ─────────────────────────────────────────────────────────────

class TestAcceptComment:
    def test_english_place_relevant_accepted(self):
        text = "The temple view is stunning and the crowd is manageable."
        assert accept_comment(text) is True

    def test_thai_place_relevant_rejected(self):
        # Thai language → fails language filter even if place-relevant
        text = "วัดสวยมากค่ะ นักท่องเที่ยวเยอะมาก"
        assert accept_comment(text) is False

    def test_english_spam_rejected(self):
        text = "Subscribe to my channel for more amazing travel content videos"
        assert accept_comment(text) is False

    def test_english_too_short_rejected(self):
        assert accept_comment("Nice") is False

    def test_empty_string_rejected(self):
        assert accept_comment("") is False


# ─────────────────────────────────────────────────────────────
# analyse_sentiment
# ─────────────────────────────────────────────────────────────

class TestAnalyseSentiment:
    def _make_pipeline(self, label: str, score: float):
        mock_fn = MagicMock()
        mock_fn.return_value = [{"label": label, "score": score}]
        return mock_fn

    def test_label2_mapped_to_positive(self):
        pipe = self._make_pipeline("LABEL_2", 0.95)
        results = analyse_sentiment(pipe, ["Great view!"])
        assert results[0]["sentiment"] == "positive"

    def test_label0_mapped_to_negative(self):
        pipe = self._make_pipeline("LABEL_0", 0.88)
        results = analyse_sentiment(pipe, ["Terrible place."])
        assert results[0]["sentiment"] == "negative"

    def test_label1_mapped_to_neutral(self):
        pipe = self._make_pipeline("LABEL_1", 0.75)
        results = analyse_sentiment(pipe, ["It was okay."])
        assert results[0]["sentiment"] == "neutral"

    def test_string_label_positive_accepted(self):
        pipe = self._make_pipeline("positive", 0.91)
        results = analyse_sentiment(pipe, ["Loved it!"])
        assert results[0]["sentiment"] == "positive"

    def test_score_below_confidence_floor_remapped_to_neutral(self):
        # Score 0.65 < CONFIDENCE_FLOOR (0.70) → re-label to neutral;
        # score itself is still stored as-is (rounded to 4 dp).
        pipe = self._make_pipeline("LABEL_2", 0.65)
        results = analyse_sentiment(pipe, ["Alright, I guess."])
        assert results[0]["sentiment"] == "neutral"
        assert results[0]["sentiment_score"] == pytest.approx(0.65, abs=0.001)

    def test_score_exactly_at_floor_is_kept(self):
        # Score == CONFIDENCE_FLOOR → label is NOT remapped (strict < check)
        pipe = self._make_pipeline("LABEL_2", CONFIDENCE_FLOOR)
        results = analyse_sentiment(pipe, ["Decent spot."])
        assert results[0]["sentiment"] == "positive"

    def test_score_stored_rounded_to_4_decimals(self):
        pipe = self._make_pipeline("LABEL_2", 0.912345)
        results = analyse_sentiment(pipe, ["Nice!"])
        assert results[0]["sentiment_score"] == pytest.approx(0.9123, abs=0.0001)

    def test_batch_error_falls_back_to_neutral(self):
        # When pipeline_fn raises, the batch falls back to neutral / 0.5
        mock_fn = MagicMock(side_effect=RuntimeError("model error"))
        results = analyse_sentiment(mock_fn, ["some text"])
        assert results[0]["sentiment"] == "neutral"
        assert results[0]["sentiment_score"] == 0.5

    def test_multiple_texts_returns_same_count(self):
        mock_fn = MagicMock()
        mock_fn.return_value = [
            {"label": "LABEL_2", "score": 0.9},
            {"label": "LABEL_0", "score": 0.8},
        ]
        results = analyse_sentiment(mock_fn, ["text1", "text2"])
        assert len(results) == 2

    def test_empty_text_list_returns_empty(self):
        mock_fn = MagicMock()
        results = analyse_sentiment(mock_fn, [])
        assert results == []


# ─────────────────────────────────────────────────────────────
# _validate_api_key
# ─────────────────────────────────────────────────────────────

class TestValidateApiKey:
    def test_valid_key_returned(self):
        key = "AIzaSy1234567890abcdefghijklmnopqrstuvwxyz"
        assert _validate_api_key(key) == key

    def test_empty_string_returns_none(self):
        assert _validate_api_key("") is None

    def test_whitespace_only_returns_none(self):
        assert _validate_api_key("   ") is None

    def test_placeholder_value_returns_none(self):
        assert _validate_api_key("your_key_here") is None

    def test_none_sentinel_returns_none(self):
        # "none" is in _PLACEHOLDER_VALUES (checked after .lower())
        assert _validate_api_key("none") is None

    def test_null_sentinel_returns_none(self):
        # "null" is in _PLACEHOLDER_VALUES
        assert _validate_api_key("null") is None

    def test_short_key_returns_none(self):
        # Real keys are 39 chars; anything < 20 is rejected
        assert _validate_api_key("tooshort") is None

    def test_key_with_leading_trailing_spaces_accepted(self):
        # _validate_api_key strips the key before returning
        key = "  AIzaSy1234567890abcdefghijklmnopqrstuvwxyz  "
        result = _validate_api_key(key)
        assert result == key.strip()

    def test_case_insensitive_placeholder_check(self):
        # _PLACEHOLDER_VALUES checked against stripped.lower()
        assert _validate_api_key("None") is None
        assert _validate_api_key("NULL") is None