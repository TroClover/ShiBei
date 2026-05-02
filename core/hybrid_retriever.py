from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from core.vector_store import VectorStore


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridRetriever:
    """BM25 (keyword) + FAISS (semantic) hybrid retrieval with RRF fusion."""

    def __init__(self, vector_store: VectorStore, k: int = 60):
        self._vector_store = vector_store
        self._bm25: BM25Okapi | None = None
        self._docs: list[Document] = []
        self._k = k

    @property
    def documents(self) -> list[Document]:
        return self._docs

    def index(self, docs: Sequence[Document]) -> None:
        self._docs = list(docs)
        corpus = [_tokenize(d.page_content) for d in self._docs]
        self._bm25 = BM25Okapi(corpus)

    def add_documents(self, docs: Sequence[Document]) -> None:
        self._docs.extend(docs)
        corpus = [_tokenize(d.page_content) for d in self._docs]
        self._bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        if self._bm25 is None or not self._docs:
            return []

        # BM25 search
        tokenized = _tokenize(query)
        bm25_scores = self._bm25.get_scores(tokenized)
        bm25_ranked = sorted(
            enumerate(bm25_scores), key=lambda x: x[1], reverse=True
        )

        # Vector search — get more candidates for fusion
        fetch_k = max(top_k * 2, 10)
        vector_docs = self._vector_store.query(query, k=fetch_k)

        # Build lookup: doc index → rank
        vector_rank: dict[int, int] = {}
        for rank, doc in enumerate(vector_docs):
            for idx, stored in enumerate(self._docs):
                if stored.page_content == doc.page_content:
                    vector_rank[idx] = rank + 1
                    break

        # RRF fusion
        rrf: dict[int, float] = {}
        for rank, (idx, _score) in enumerate(bm25_ranked):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self._k + rank + 1)

        for idx, rank in vector_rank.items():
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self._k + rank)

        sorted_ids = sorted(rrf, key=rrf.get, reverse=True)[:top_k]
        return [self._docs[i] for i in sorted_ids]
