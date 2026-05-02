from __future__ import annotations

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from core.model_factory import OpenAICompatEmbedding


class VectorStore:
    def __init__(self, embedding: OpenAICompatEmbedding):
        self.embedding = embedding
        self.db: FAISS | None = None

    def build(self, docs: list[Document]) -> None:
        texts = [d.page_content for d in docs]
        metadatas = [d.metadata for d in docs]
        self.db = FAISS.from_texts(texts, self.embedding, metadatas=metadatas)

    def add_documents(self, docs: list[Document]) -> None:
        if self.db is None:
            self.build(docs)
            return
        texts = [d.page_content for d in docs]
        metadatas = [d.metadata for d in docs]
        self.db.add_texts(texts, metadatas=metadatas)

    def query(self, query: str, k: int = 5) -> list[Document]:
        if self.db is None:
            return []
        return self.db.similarity_search(query, k=k)
