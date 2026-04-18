"""
=============================================================================
embeddings.py — Sentence Transformer Wrapper
=============================================================================
Project : Agentic RAG for Smart Tourism — Chiang Mai, Thailand
Course  : AI for Remote Sensing & Geoinformatics (Graduate)
Team    : Boonyoros Pheechaphuth (LS2525207) · Teh Bismin (LS2525222)

Purpose
-------
Defines SentenceTransformerWrapper, a thin pickle-safe wrapper around
sentence-transformers' SentenceTransformer.

This class is kept in its own file so that both step3_rag.py (which builds
and pickles the model) and app.py (which unpickles it at runtime) can import
the same class definition without triggering each other's module-level
side-effects (makedirs, heavy constants, etc.).

Usage
-----
  # In step3_rag.py
  from embeddings import SentenceTransformerWrapper
  wrapper = SentenceTransformerWrapper("paraphrase-multilingual-MiniLM-L12-v2")
  embeddings = wrapper.transform(texts)          # numpy float32 array
  pickle.dump(wrapper, open("vectorizer.pkl", "wb"))

  # In app.py
  from embeddings import SentenceTransformerWrapper  # registers class for unpickling
  wrapper = pickle.load(open("knowledge_base/vectorizer.pkl", "rb"))
  query_vec = wrapper.transform(["Is Wat Doi Suthep crowded?"])

Model
-----
  paraphrase-multilingual-MiniLM-L12-v2
  · Output dim  : 384
  · Languages   : 50+ (Thai + English both supported)
  · Download    : ~275 MB on first run, cached to ~/.cache/huggingface/
  · Vectors are L2-normalised → FAISS IndexFlatIP == cosine similarity
=============================================================================
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Default model — must match ST_MODEL_NAME in step3_rag.py
_DEFAULT_MODEL      = "paraphrase-multilingual-MiniLM-L12-v2"
_DEFAULT_BATCH_SIZE = 64


class SentenceTransformerWrapper:
    """
    Pickle-safe wrapper around sentence_transformers.SentenceTransformer.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier (also used as metadata label).
    _batch_size : int
        Sentences per forward-pass batch during encoding.

    Notes
    -----
    * __getstate__ / __setstate__ are implemented so that only lightweight
      metadata is pickled; the underlying SentenceTransformer model is
      re-loaded from the HuggingFace cache on the first call to transform()
      after unpickling.  This avoids storing ~275 MB model weights inside the
      pickle file.
    * The internal ``_model`` attribute can be injected directly (e.g. to
      reuse an already-loaded model in step3_rag.py) to avoid a redundant
      reload.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_name  = model_name
        self._batch_size = batch_size
        self._model      = None  # lazy-loaded on first transform() call

    # ------------------------------------------------------------------
    # Pickle protocol — store only lightweight metadata
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Exclude the heavy SentenceTransformer object from pickle.

        Keys stored: "model_name" and "_batch_size" (note the underscore
        prefix on _batch_size so __setstate__ can restore it to the private
        attribute without name-mangling issues).
        """
        return {
            "model_name":  self.model_name,
            "_batch_size": self._batch_size,
        }

    def __setstate__(self, state: dict) -> None:
        """Restore lightweight state; _model will be lazy-loaded later."""
        self.model_name  = state["model_name"]
        self._batch_size = state.get("_batch_size", _DEFAULT_BATCH_SIZE)
        self._model      = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """
        Load the SentenceTransformer model on demand.

        Raises ImportError (with install hint) if sentence-transformers is
        not installed — this surfaces a clear message rather than a cryptic
        AttributeError.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required but not installed.\n"
                "Install it with:  pip install sentence-transformers"
            ) from exc

        logger.info(
            "[SentenceTransformerWrapper] Loading model '%s' "
            "(cached after first download).",
            self.model_name,
        )
        self._model = SentenceTransformer(self.model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into L2-normalised float32 embeddings.

        Parameters
        ----------
        texts : list[str]
            Input sentences / passages to encode.

        Returns
        -------
        numpy.ndarray, shape (len(texts), 384), dtype float32
            L2-normalised dense vectors.  Inner-product with another
            normalised vector equals cosine similarity.
        """
        if self._model is None:
            self._load()

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self._batch_size,
        )
        return embeddings.astype(np.float32)

    def __repr__(self) -> str:
        loaded = "loaded" if self._model is not None else "not loaded"
        return (
            f"SentenceTransformerWrapper("
            f"model_name={self.model_name!r}, "
            f"_batch_size={self._batch_size}, "
            f"_model={loaded})"
        )