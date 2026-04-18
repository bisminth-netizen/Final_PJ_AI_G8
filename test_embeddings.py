"""
Unit tests for embeddings.py — SentenceTransformerWrapper.
"""
import pickle
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from embeddings import SentenceTransformerWrapper


class TestSentenceTransformerWrapperInit:
    def test_stores_model_name(self):
        w = SentenceTransformerWrapper("paraphrase-multilingual-MiniLM-L12-v2")
        assert w.model_name == "paraphrase-multilingual-MiniLM-L12-v2"

    def test_default_batch_size(self):
        w = SentenceTransformerWrapper("some-model")
        assert w._batch_size == 64

    def test_custom_batch_size(self):
        w = SentenceTransformerWrapper("some-model", batch_size=16)
        assert w._batch_size == 16

    def test_model_initially_none(self):
        w = SentenceTransformerWrapper("some-model")
        assert w._model is None


class TestSentenceTransformerWrapperPickle:
    def test_getstate_excludes_model_weights(self):
        w = SentenceTransformerWrapper("some-model", batch_size=32)
        w._model = MagicMock()  # simulate loaded model
        state = w.__getstate__()
        assert "model_name" in state
        assert "_batch_size" in state
        assert "_model" not in state

    def test_setstate_restores_name_and_batch(self):
        w = SentenceTransformerWrapper.__new__(SentenceTransformerWrapper)
        w.__setstate__({"model_name": "test-model", "_batch_size": 32})
        assert w.model_name == "test-model"
        assert w._batch_size == 32

    def test_setstate_resets_model_to_none(self):
        w = SentenceTransformerWrapper.__new__(SentenceTransformerWrapper)
        w.__setstate__({"model_name": "test-model", "_batch_size": 32})
        assert w._model is None

    def test_setstate_default_batch_when_missing(self):
        w = SentenceTransformerWrapper.__new__(SentenceTransformerWrapper)
        w.__setstate__({"model_name": "test-model"})  # no _batch_size key
        assert w._batch_size == 64  # falls back to _DEFAULT_BATCH_SIZE

    def test_pickle_round_trip_preserves_config(self):
        w = SentenceTransformerWrapper("round-trip-model", batch_size=8)
        data = pickle.dumps(w)
        w2 = pickle.loads(data)
        assert w2.model_name == "round-trip-model"
        assert w2._batch_size == 8
        assert w2._model is None  # model weights not pickled

    def test_pickle_survives_without_model_loaded(self):
        w = SentenceTransformerWrapper("no-load-model")
        # _model is None — pickle must not raise
        data = pickle.dumps(w)
        w2 = pickle.loads(data)
        assert w2.model_name == "no-load-model"


class TestSentenceTransformerWrapperTransform:
    def _make_mock_st(self, output_shape=(2, 384)):
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones(output_shape, dtype=np.float32)
        mock_st.SentenceTransformer.return_value = mock_model
        return mock_st, mock_model

    def test_transform_returns_float32_array(self):
        mock_st, _ = self._make_mock_st((2, 384))
        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            w = SentenceTransformerWrapper("test-model")
            result = w.transform(["hello", "world"])
        assert result.dtype == np.float32

    def test_transform_output_shape_matches_inputs(self):
        mock_st, _ = self._make_mock_st((3, 384))
        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            w = SentenceTransformerWrapper("test-model")
            result = w.transform(["a", "b", "c"])
        assert result.shape == (3, 384)

    def test_transform_calls_encode_with_normalize(self):
        mock_st, mock_model = self._make_mock_st((1, 384))
        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            w = SentenceTransformerWrapper("test-model")
            w.transform(["hello"])
        _, kwargs = mock_model.encode.call_args
        assert kwargs.get("normalize_embeddings") is True

    def test_transform_lazy_loads_model_only_once(self):
        mock_st, _ = self._make_mock_st((1, 384))
        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            w = SentenceTransformerWrapper("test-model")
            w.transform(["first call"])
            w.transform(["second call"])
        assert mock_st.SentenceTransformer.call_count == 1

    def test_transform_reuses_cached_model(self):
        mock_st, mock_model = self._make_mock_st((1, 384))
        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            w = SentenceTransformerWrapper("test-model")
            w.transform(["first"])
            # Reset the constructor call count; model should not reload
            mock_st.SentenceTransformer.reset_mock()
            w.transform(["second"])
        mock_st.SentenceTransformer.assert_not_called()

    def test_transform_raises_import_error_when_unavailable(self):
        w = SentenceTransformerWrapper("test-model")
        with patch.object(w, "_load", side_effect=ImportError("sentence-transformers is required")):
            with pytest.raises(ImportError, match="sentence-transformers"):
                w.transform(["hello"])

    def test_transform_passes_batch_size(self):
        mock_st, mock_model = self._make_mock_st((1, 384))
        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            w = SentenceTransformerWrapper("test-model", batch_size=16)
            w.transform(["hello"])
        _, kwargs = mock_model.encode.call_args
        assert kwargs.get("batch_size") == 16