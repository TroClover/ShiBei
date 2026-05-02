from __future__ import annotations

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import RAGConfig
from tools.document_loader import load_documents


class ParserAgent:
    def __init__(self, rag_config: RAGConfig):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=rag_config.chunk_size,
            chunk_overlap=rag_config.chunk_overlap,
        )

    def run(self, file_paths: List[str]) -> list[Document]:
        docs = load_documents(file_paths)
        return self._splitter.split_documents(docs)
