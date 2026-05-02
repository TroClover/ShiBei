from __future__ import annotations

from typing import List

from agents.parser_agent import ParserAgent
from agents.retrieval_agent import RetrievalAgent
from agents.summary_agent import SummaryAgent
from agents.workflow import RAGWorkflow
from config import Config, config
from core.evaluator import Evaluator
from core.hybrid_retriever import HybridRetriever
from core.model_factory import ModelFactory
from core.vector_store import VectorStore


class OrchestratorAgent:
    def __init__(self, cfg: Config | None = None):
        self._cfg = cfg or config

        chat = ModelFactory.create_chat(self._cfg.model)
        embedding = ModelFactory.create_embedding(self._cfg.model)

        self._vector_store = VectorStore(embedding)
        self._hybrid = HybridRetriever(self._vector_store)
        self._parser = ParserAgent(self._cfg.rag)
        self._retriever = RetrievalAgent(self._hybrid, self._cfg.rag)
        self._summarizer = SummaryAgent(chat)
        self._evaluator = Evaluator(chat)

        self._workflow = RAGWorkflow(
            retriever=self._retriever,
            summarizer=self._summarizer,
            evaluator=self._evaluator,
            rag_config=self._cfg.rag,
        )

    def ingest(self, file_paths: List[str]) -> int:
        docs = self._parser.run(file_paths)
        self._vector_store.build(docs)
        self._hybrid.index(docs)
        return len(docs)

    def ask(self, query: str) -> dict:
        return self._workflow.run(query)
