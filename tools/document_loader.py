from __future__ import annotations

import os
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def _load_single(path: str) -> list[Document]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext == ".txt":
        loader = TextLoader(path, encoding="utf-8")
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        loader = Docx2txtLoader(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()
    source = os.path.basename(path)
    for doc in docs:
        doc.metadata["source"] = source
    return docs


def load_documents(paths: List[str]) -> list[Document]:
    all_docs: list[Document] = []
    for path in paths:
        if not os.path.exists(path):
            print(f"⚠ 文件不存在，跳过: {path}")
            continue
        all_docs.extend(_load_single(path))
    return all_docs
