import os
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的环境变量

Provider = Literal["deepseek", "ollama"]
EmbeddingProvider = Literal["ollama", "openai"]  # 嵌入模型提供方

# ============================================================
# API Key 默认值（优先用环境变量/ .env 文件，此处仅作占位）
# ============================================================
_DEFAULT_DEEPSEEK_KEY = "sk-your-deepseek-key-here"
_DEFAULT_OPENAI_KEY = ""       # 如嵌入用 OpenAI 则填 .env 中 EMBEDDING_API_KEY
_DEFAULT_OLLAMA_KEY = ""       # Ollama 本地不需要 key

# DeepSeek 没有 Embeddings API，嵌入默认走 Ollama 本地
_DEFAULT_EMBEDDING_PROVIDER: EmbeddingProvider = "ollama"

# 各 provider 支持的已知模型列表
_CHAT_MODELS: dict[str, list[str]] = {
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro"],
    "ollama": [],  # 运行时通过 /api/tags 获取
}
_EMBEDDING_MODELS: dict[str, list[str]] = {
    "ollama": [],       # 运行时通过 /api/tags 获取
    "openai": ["text-embedding-3-small", "text-embedding-3-large",
               "text-embedding-ada-002"],
}


def get_default_api_key(provider: Provider) -> str:
    if provider == "deepseek":
        return _DEFAULT_DEEPSEEK_KEY
    return _DEFAULT_OLLAMA_KEY


def get_preset_models(provider: Provider) -> dict[str, list[str]]:
    """Return {chat: [...], embedding: [...]} for the given provider."""
    chat = list(_CHAT_MODELS.get(provider, []))
    emb: list[str] = []

    if provider == "ollama":
        live = _fetch_ollama_models()
        if live["chat"]:
            chat = live["chat"]
        else:
            chat = chat or ["qwen2.5", "qwen3", "llama3", "mistral"]
        emb = live["embedding"] or ["nomic-embed-text", "bge-m3"]
    else:
        # DeepSeek: 对话用 DeepSeek 模型，嵌入走 Ollama
        emb = _EMBEDDING_MODELS.get(_DEFAULT_EMBEDDING_PROVIDER, [])
        # 也尝试从 Ollama 获取嵌入模型
        ollama_emb = _fetch_ollama_models().get("embedding", [])
        if ollama_emb:
            emb = ollama_emb

    return {"chat": chat, "embedding": emb}


def _fetch_ollama_models() -> dict[str, list[str]]:
    """Query Ollama /api/tags to discover installed models."""
    try:
        import json
        import urllib.request

        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        # 简单区分 chat 和 embedding 模型
        chat = [m for m in models if "embed" not in m.lower() and "nomic" not in m.lower()]
        emb = [m for m in models if "embed" in m.lower() or "nomic" in m.lower() or "bge" in m.lower()]
        return {"chat": chat or ["qwen2.5"], "embedding": emb or ["nomic-embed-text"]}
    except Exception:
        return {"chat": [], "embedding": []}


@dataclass
class ModelConfig:
    provider: Provider = "deepseek"
    chat_model: str = "deepseek-v4-flash"
    embedding_model: str = "nomic-embed-text"
    embedding_provider: EmbeddingProvider = "ollama"
    api_key: str = ""
    embedding_api_key: str = ""
    base_url: str = ""

    @property
    def chat_base_url(self) -> str:
        if self.base_url and self.provider == "deepseek":
            return self.base_url
        if self.provider == "deepseek":
            return "https://api.deepseek.com"
        return "http://localhost:11434/v1"

    @property
    def embedding_base_url(self) -> str:
        if self.base_url and self.embedding_provider == "openai":
            return self.base_url
        if self.embedding_provider == "openai":
            return "https://api.openai.com/v1"
        return "http://localhost:11434/v1"  # Ollama 默认


@dataclass
class RAGConfig:
    chunk_size: int = 1500
    chunk_overlap: int = 300
    top_k: int = 10
    max_reflections: int = 2
    quality_threshold: float = 0.75


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)

    @classmethod
    def from_env(cls) -> "Config":
        provider = os.getenv("PROVIDER", "deepseek")
        if provider not in ("deepseek", "ollama"):
            raise ValueError(f"PROVIDER must be 'deepseek' or 'ollama', got '{provider}'")

        emb_provider = os.getenv("EMBEDDING_PROVIDER", _DEFAULT_EMBEDDING_PROVIDER)
        if emb_provider not in ("ollama", "openai"):
            raise ValueError(f"EMBEDDING_PROVIDER must be 'ollama' or 'openai', got '{emb_provider}'")

        model = ModelConfig(
            provider=provider,
            chat_model=os.getenv(
                "CHAT_MODEL",
                "deepseek-v4-flash" if provider == "deepseek" else "qwen2.5",
            ),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "nomic-embed-text" if emb_provider == "ollama" else "text-embedding-3-small",
            ),
            embedding_provider=emb_provider,
            api_key=os.getenv("API_KEY")
                    or os.getenv("OPENAI_API_KEY")
                    or get_default_api_key(provider),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY")
                             or os.getenv("OPENAI_API_KEY")
                             or _DEFAULT_OPENAI_KEY,
            base_url=os.getenv("BASE_URL", ""),
        )

        rag = RAGConfig(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "300")),
            top_k=int(os.getenv("TOP_K", "10")),
            max_reflections=int(os.getenv("MAX_REFLECTIONS", "2")),
            quality_threshold=float(os.getenv("QUALITY_THRESHOLD", "0.75")),
        )

        return cls(model=model, rag=rag)


config = Config.from_env()
