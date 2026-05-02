from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings
from openai import OpenAI

from config import ModelConfig


class BaseChat(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


class OpenAICompatChat(BaseChat):
    """OpenAI-compatible chat client — works with DeepSeek, Ollama, etc."""

    def __init__(self, config: ModelConfig):
        self._client = OpenAI(
            api_key=config.api_key or "not-needed",
            base_url=config.chat_base_url,
        )
        self._model = config.chat_model

    def generate(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "你是一个专业的文档分析助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""


class OpenAICompatEmbedding(Embeddings):
    """OpenAI-compatible embeddings — satisfies LangChain Embeddings interface."""

    def __init__(self, config: ModelConfig):
        self._client = OpenAI(
            api_key=config.embedding_api_key or config.api_key or "not-needed",
            base_url=config.embedding_base_url,
        )
        self._model = config.embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class ModelFactory:
    @staticmethod
    def create_chat(config: ModelConfig) -> BaseChat:
        return OpenAICompatChat(config)

    @staticmethod
    def create_embedding(config: ModelConfig) -> OpenAICompatEmbedding:
        return OpenAICompatEmbedding(config)
