# ShiBei (拾贝) — 多 Agent 文档理解与知识库系统

基于 **LangGraph** 编排的 Multi-Agent RAG 系统，支持语义检索、混合检索、自动评估与自我反思机制。

## 架构

```
┌──────────────────────────────────────────────────────────┐
│  LangGraph Workflow                                      │
│                                                          │
│  Parse → Index → Retrieve(hybrid) → Generate             │
│                        ↑               ↓                 │
│                        └── Reflect ← Evaluate             │
│                                                          │
│  CLI: app.py  │  Web: web_app.py (Streamlit)             │
└──────────────────────────────────────────────────────────┘

Agent 层:
  ParserAgent      — 文档解析与分块 (PDF/TXT/DOCX)
  RetrievalAgent   — 混合检索 (BM25 + FAISS + RRF)
  SummaryAgent     — RAG 生成回答
  Evaluator        — LLM-as-Judge 质量评估
  OrchestratorAgent — 工作流封装

模型层:
  DeepSeek (API)   — api.deepseek.com
  Ollama (本地)    — localhost:11434，支持 Qwen 等
```

## 功能特性

- **双模型支持**: DeepSeek API + Ollama 本地模型 (Qwen)，统一 OpenAI-compatible 接口
- **多文件知识库**: 支持 PDF / TXT / DOCX，批量上传与解析
- **混合检索**: BM25 (关键词) + FAISS (语义向量) + RRF 融合排序
- **自动评估**: LLM-as-Judge 对回答的 Faithfulness 和 Relevance 打分
- **Self-Reflection Loop**: 评估不达标时自动反思优化，检索-生成-评估循环
- **LangGraph 工作流**: 状态图驱动的 Agent 编排，支持条件路由

## 环境要求

- Python 3.10+
- [Ollama](https://ollama.com/) (如使用本地模型)

## 安装

```bash
# 1. 克隆项目
cd ShiBei

# 2. 安装依赖
pip install -r requirements.txt
```

## 配置

通过环境变量配置：

```bash
# Provider: deepseek 或 ollama
export PROVIDER=deepseek

# DeepSeek 配置 (PROVIDER=deepseek)
export API_KEY=sk-your-deepseek-key
export CHAT_MODEL=deepseek-chat
export EMBEDDING_MODEL=text-embedding-3-small

# Ollama 配置 (PROVIDER=ollama)
# 先拉取模型: ollama pull qwen2.5 && ollama pull nomic-embed-text
export PROVIDER=ollama
export CHAT_MODEL=qwen2.5
export EMBEDDING_MODEL=nomic-embed-text
# API_KEY 可留空

# 可选 RAG 参数
export CHUNK_SIZE=1500
export TOP_K=10
export MAX_REFLECTIONS=2
export QUALITY_THRESHOLD=0.75
```

或创建 `.env` 文件在项目根目录。

## 使用方式

### CLI 命令行

```bash
# 指定文件启动
python app.py docs/report.pdf docs/notes.txt

# 或交互式输入路径
python app.py
```

### Web 界面

```bash
streamlit run web_app.py
```

在浏览器中打开 http://localhost:8501 ，上传文档并提问。

## 项目结构

```
ShiBei/
├── agents/
│   ├── orchestrator.py    # 工作流封装
│   ├── parser_agent.py    # 文档解析 Agent
│   ├── retrieval_agent.py # 检索 Agent
│   ├── summary_agent.py   # 生成 Agent
│   └── workflow.py        # LangGraph 状态图
├── core/
│   ├── vector_store.py    # FAISS 向量库
│   ├── model_factory.py   # 模型工厂 (DeepSeek/Ollama)
│   ├── hybrid_retriever.py # BM25 + FAISS 混合检索
│   └── evaluator.py       # 回答质量评估
├── tools/
│   └── document_loader.py # 多格式文档加载
├── config.py              # 全局配置
├── app.py                 # CLI 入口
├── web_app.py             # Streamlit Web 入口
├── requirements.txt       # 依赖清单
└── README.md
```

## 工作流说明

1. **Parse**: 加载文档 → 文本分块 (RecursiveCharacterTextSplitter)
2. **Index**: 构建 FAISS 向量索引 + BM25 关键词索引
3. **Retrieve**: 用户提问 → BM25 + Vector + RRF 融合 → Top-K 文档片段
4. **Generate**: 检索上下文 + Query → LLM 生成回答
5. **Evaluate**: LLM-as-Judge 评估忠实度与相关性
6. **Reflect** (条件触发): 评分不足时 → 改写回答并重新检索 → 循环至达标或达上限

## Ollama 本地部署指引

```bash
# 安装 Ollama
# macOS/Windows: https://ollama.com/download
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型
ollama pull qwen2.5          # 对话模型 (~4.7GB)
ollama pull nomic-embed-text  # 嵌入模型 (~0.3GB)

# 启动服务 (默认 11434 端口)
ollama serve
```
