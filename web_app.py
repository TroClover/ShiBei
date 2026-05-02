from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from agents.orchestrator import OrchestratorAgent
from config import (
    Config,
    ModelConfig,
    RAGConfig,
    _DEFAULT_EMBEDDING_PROVIDER,
    config as default_config,
    get_default_api_key,
    get_preset_models,
    _DEFAULT_OPENAI_KEY,
)

st.set_page_config(page_title="ShiBei 文档知识库", layout="wide")
st.title("📚 拾贝 · 多 Agent 文档理解与知识库系统")

# ---- Sidebar: model config ----
st.sidebar.header("⚙️ 模型配置")

provider = st.sidebar.selectbox(
    "对话 Provider",
    ["deepseek", "ollama"],
    index=0 if default_config.model.provider == "deepseek" else 1,
)

# 嵌入 provider：DeepSeek 没有 embedding API，默认走 Ollama
emb_provider = st.sidebar.selectbox(
    "嵌入 Provider",
    ["ollama", "openai"],
    index=0 if _DEFAULT_EMBEDDING_PROVIDER == "ollama" else 1,
    help="DeepSeek 不提供 Embeddings API，嵌入默认走 Ollama 本地。也可选 OpenAI。",
)

# 获取可用模型列表
chat_presets = get_preset_models(provider)
chat_options = chat_presets["chat"]
emb_options = chat_presets["embedding"]  # 已经按 emb_provider 返回了

# 默认值
default_chat = default_config.model.chat_model
if default_chat not in chat_options and chat_options:
    default_chat = chat_options[0]
default_emb = default_config.model.embedding_model
if default_emb not in emb_options and emb_options:
    default_emb = emb_options[0]

if chat_options:
    chat_model = st.sidebar.selectbox(
        "对话模型",
        chat_options,
        index=chat_options.index(default_chat) if default_chat in chat_options else 0,
    )
else:
    chat_model = st.sidebar.text_input("对话模型（手动输入）", value=default_chat)

if emb_options:
    embedding_model = st.sidebar.selectbox(
        "嵌入模型",
        emb_options,
        index=emb_options.index(default_emb) if default_emb in emb_options else 0,
    )
else:
    embedding_model = st.sidebar.text_input("嵌入模型（手动输入）", value=default_emb)

# API Key: 取 config 中的硬编码值，不暴露在前端
api_key = get_default_api_key(provider)
embedding_api_key = get_default_api_key("ollama") if emb_provider == "ollama" else _DEFAULT_OPENAI_KEY

# Base URL: 高级选项
with st.sidebar.expander("🔗 高级设置"):
    base_url = st.text_input(
        "Base URL",
        value="",
        placeholder="留空使用默认地址",
    )

# RAG params
st.sidebar.header("🔧 检索参数")
top_k = st.sidebar.slider("Top-K", 1, 10, default_config.rag.top_k)
chunk_size = st.sidebar.slider("Chunk Size", 200, 1500, default_config.rag.chunk_size)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 300, default_config.rag.chunk_overlap)
max_reflections = st.sidebar.slider("最大反思次数", 0, 5, default_config.rag.max_reflections)
quality_threshold = st.sidebar.slider(
    "质量阈值", 0.0, 1.0, default_config.rag.quality_threshold, 0.05
)

# Build config
cfg = Config(
    model=ModelConfig(
        provider=provider,
        chat_model=chat_model,
        embedding_model=embedding_model,
        embedding_provider=emb_provider,
        api_key=api_key,
        embedding_api_key=embedding_api_key,
        base_url=base_url,
    ),
    rag=RAGConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        max_reflections=max_reflections,
        quality_threshold=quality_threshold,
    ),
)

# ---- Main area ----
st.header("📤 上传文档")
uploaded_files = st.file_uploader(
    "支持 PDF / TXT / DOCX，可多选",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
)

if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
    if st.button("🚀 解析文档", type="primary"):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_paths = []
            for uf in uploaded_files:
                fpath = Path(tmpdir) / uf.name
                fpath.write_bytes(uf.read())
                file_paths.append(str(fpath))

            with st.spinner(f"正在解析 {len(file_paths)} 个文件..."):
                try:
                    st.session_state.agent = OrchestratorAgent(cfg)
                    chunk_count = st.session_state.agent.ingest(file_paths)
                    st.session_state.chat_history = []
                    st.success(f"✅ 解析完成！共 {chunk_count} 个文本块")
                except Exception as e:
                    st.error(f"解析失败: {e}")

# ---- Q&A ----
st.header("💬 问答")

if st.session_state.agent is not None:
    query = st.chat_input("请输入你的问题")

    if query:
        with st.spinner("检索与生成中..."):
            result = st.session_state.agent.ask(query)

        answer = result.get("answer", "无法生成回答")
        eval_r = result.get("eval_result", {})
        reflections = result.get("reflection_count", 0)
        docs = result.get("retrieved_docs", [])

        st.session_state.chat_history.append(
            {"query": query, "result": result}
        )

        # Display answer
        st.markdown(f"### 💡 回答\n{answer}")

        # Evaluation metrics
        if eval_r:
            col1, col2, col3 = st.columns(3)
            col1.metric("综合评分", f"{eval_r.get('overall', 0):.2f}")
            col2.metric("忠实度", f"{eval_r.get('faithfulness', 0):.2f}")
            col3.metric("相关性", f"{eval_r.get('relevance', 0):.2f}")
            if reflections > 0:
                st.caption(f"🔄 自我反思次数: {reflections}")
            if eval_r.get("feedback"):
                with st.expander("📝 评估反馈"):
                    st.write(eval_r["feedback"])

        # Retrieved chunks
        if docs:
            with st.expander(f"📚 检索到的文档片段 ({len(docs)} 条)"):
                for i, doc in enumerate(docs, 1):
                    src = doc.metadata.get("source", "未知")
                    st.caption(f"[{i}] 来源: {src}")
                    st.text(doc.page_content[:300])

    # Chat history
    if st.session_state.chat_history:
        with st.expander(f"📜 对话历史 ({len(st.session_state.chat_history)} 条)"):
            for i, entry in enumerate(reversed(st.session_state.chat_history), 1):
                st.markdown(f"**Q{i}:** {entry['query']}")
                st.markdown(
                    f"**A{i}:** {entry['result'].get('answer', '')[:200]}..."
                )
                st.divider()
else:
    st.info("👆 请先上传文档并点击「解析文档」")
