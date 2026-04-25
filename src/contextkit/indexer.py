"""Vector indexing for semantic context search.

Provides embedding-based indexing using OpenAI embeddings with local
file-based persistence. Supports cosine similarity search for finding
contextually relevant messages.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


class VectorIndex:
    """In-memory vector index with local file persistence.

    Stores message embeddings and provides cosine similarity search.
    Uses OpenAI's text-embedding models for generating embeddings.
    """

    def __init__(
        self,
        storage_dir: str = "./.contextkit",
        embedding_model: str = "text-embedding-3-small",
        base_url: str | None = None,
    ) -> None:
        """Initialize the vector index.

        Args:
            storage_dir: Directory for persisting index data.
            embedding_model: OpenAI embedding model name.
            base_url: Optional custom API base URL for OpenAI-compatible services.
        """
        self.storage_dir = Path(storage_dir)
        self.embedding_model = embedding_model
        self.base_url = base_url
        self._vectors: np.ndarray | None = None
        self._ids: list[str] = []
        self._client: Any = None

    def _get_client(self) -> Any | None:
        """Lazy-init the OpenAI client. Returns None if openai not installed."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                return None
            kwargs: dict[str, str] = {}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            2D numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])

        client = self._get_client()
        if client is None:
            # No OpenAI client available — return empty, will use keyword fallback
            return np.array([])
        # OpenAI supports batching — max 2048 texts per call
        all_embeddings: list[list[float]] = []
        batch_size = 512

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(
                model=self.embedding_model,
                input=batch,
            )
            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([d.embedding for d in sorted_data])

        return np.array(all_embeddings, dtype=np.float32)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query vector and matrix rows.

        Args:
            a: Query vector of shape (dim,).
            b: Matrix of shape (n, dim).

        Returns:
            Array of similarity scores of shape (n,).
        """
        if b.size == 0:
            return np.array([])
        norms = np.linalg.norm(b, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        normalized = b / norms[:, np.newaxis]
        query_norm = np.linalg.norm(a)
        if query_norm == 0:
            return np.zeros(b.shape[0])
        return normalized @ (a / query_norm)

    def add(self, msg_id: str, text: str) -> None:
        """Add a single message to the index.

        Args:
            msg_id: Unique message identifier.
            text: Text content to embed and index.
        """
        self._ids.append(msg_id)
        embedding = self._embed([text])
        if embedding.size == 0:
            return

        if self._vectors is None:
            self._vectors = embedding
        else:
            self._vectors = np.vstack([self._vectors, embedding])

    def add_batch(self, items: list[tuple[str, str]]) -> list[int]:
        """Add multiple messages in a single embedding call.

        Args:
            items: List of (msg_id, text) tuples.

        Returns:
            List of original indices for each added item.
        """
        if not items:
            return []

        texts = [text for _, text in items]
        embeddings = self._embed(texts)

        start_indices: list[int] = []
        for i, (msg_id, _) in enumerate(items):
            start_indices.append(len(self._ids))
            self._ids.append(msg_id)

        if self._vectors is None:
            self._vectors = embeddings
        else:
            self._vectors = np.vstack([self._vectors, embeddings])

        return start_indices

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search for most similar messages.

        Args:
            query: Query text to search for.
            top_k: Number of results to return.

        Returns:
            List of (msg_id, similarity_score) tuples, sorted by relevance.
        """
        if self._vectors is None or self._vectors.size == 0 or not self._ids:
            return []

        query_embedding = self._embed([query])[0]
        similarities = self._cosine_similarity(query_embedding, self._vectors)

        # Get top-k indices
        k = min(top_k, len(similarities))
        if k == 0:
            return []

        top_indices = np.argsort(similarities)[::-1][:k]
        return [(self._ids[i], float(similarities[i])) for i in top_indices]

    def remove(self, msg_id: str) -> bool:
        """Remove a message from the index.

        Args:
            msg_id: The message ID to remove.

        Returns:
            True if the message was found and removed.
        """
        if msg_id not in self._ids:
            return False

        idx = self._ids.index(msg_id)
        self._ids.pop(idx)

        if self._vectors is not None:
            self._vectors = np.delete(self._vectors, idx, axis=0)
            if self._vectors.shape[0] == 0:
                self._vectors = None

        return True

    def remove_batch(self, msg_ids: list[str]) -> int:
        """Remove multiple messages from the index.

        Args:
            msg_ids: List of message IDs to remove.

        Returns:
            Number of messages successfully removed.
        """
        removed = 0
        for msg_id in msg_ids:
            if self.remove(msg_id):
                removed += 1
        return removed

    @property
    def size(self) -> int:
        """Number of indexed messages."""
        return len(self._ids)

    def save(self) -> None:
        """Persist the index to disk."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        index_file = self.storage_dir / "index.npz"
        meta_file = self.storage_dir / "index_meta.json"

        # Save vectors
        if self._vectors is not None and self._vectors.size > 0:
            np.savez_compressed(str(index_file), vectors=self._vectors)
        else:
            # Create empty file to indicate empty index
            if index_file.exists():
                index_file.unlink()

        # Save metadata (ids + config)
        meta = {
            "ids": self._ids,
            "embedding_model": self.embedding_model,
            "count": len(self._ids),
            "saved_at": time.time(),
        }
        meta_file.write_text(json.dumps(meta, indent=2))

    def load(self) -> bool:
        """Load the index from disk.

        Returns:
            True if index was loaded successfully, False if no saved index exists.
        """
        index_file = self.storage_dir / "index.npz"
        meta_file = self.storage_dir / "index_meta.json"

        if not meta_file.exists():
            return False

        try:
            meta = json.loads(meta_file.read_text())
            self._ids = meta.get("ids", [])

            if index_file.exists() and self._ids:
                data = np.load(str(index_file))
                self._vectors = data["vectors"]
            else:
                self._vectors = None

            return True
        except Exception:
            # If loading fails, start fresh
            self._ids = []
            self._vectors = None
            return False

    def clear(self) -> None:
        """Remove all entries from the index."""
        self._ids = []
        self._vectors = None
        # Clean up files
        for fname in ["index.npz", "index_meta.json"]:
            fpath = self.storage_dir / fname
            if fpath.exists():
                fpath.unlink()
