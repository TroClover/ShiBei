from __future__ import annotations

from langchain_core.documents import Document

from config import RAGConfig
from core.hybrid_retriever import HybridRetriever


class RetrievalAgent:
    def __init__(self, retriever: HybridRetriever, rag_config: RAGConfig):
        self._retriever = retriever
        self._top_k = rag_config.top_k

    def run(self, query: str) -> list[Document]:
        return self._retriever.retrieve(query, top_k=self._top_k)
