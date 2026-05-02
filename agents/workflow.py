from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from agents.retrieval_agent import RetrievalAgent
from agents.summary_agent import SummaryAgent
from config import RAGConfig
from core.evaluator import Evaluator

QUERY_REWRITE_PROMPT = """你是一个搜索查询优化专家。当前回答质量不达标，需要改写查询以获得更好的检索结果。

## 原始问题
{query}

## 评估反馈
{feedback}

## 当前回答
{answer}

## 要求
请将原始问题改写为 2~3 个更具体、更精准的搜索查询，每个查询聚焦一个方面。
用中文关键词形式，每行一个查询，以 "QUERY:" 开头。

示例：
原始问题：论文中使用了哪些数据集？
输出：
QUERY: 实验数据集 训练集 测试集 数据来源
QUERY: 数据分布 划分方式 客户端数据量
QUERY: CIFAR MNIST FEMNIST 基准数据集名"""

REFLECT_PROMPT = """你是一个严谨的问答系统优化助手。以下是根据新的检索结果重新生成的回答。

## 原始问题
{query}

## 补充检索关键词
{rewritten_query}

## 新的检索文档
{context}

## 上一轮回答（质量不达标）
{prev_answer}

## 评估反馈
{feedback}

请基于以上所有信息给出一个完整、详尽、准确的最终回答。确保覆盖问题的所有方面。"""


class WorkflowState(TypedDict):
    query: str
    rewritten_query: str
    retrieved_docs: list
    answer: str
    eval_result: dict
    reflection_count: int


class RAGWorkflow:
    def __init__(
        self,
        retriever: RetrievalAgent,
        summarizer: SummaryAgent,
        evaluator: Evaluator,
        rag_config: RAGConfig,
    ):
        self.retriever = retriever
        self.summarizer = summarizer
        self.evaluator = evaluator
        self.rag_config = rag_config
        self._graph = self._build_graph()

    def _retrieve(self, state: WorkflowState) -> dict:
        # 如果有改写后的 query，用改写后的检索；否则用原始 query
        search_query = state.get("rewritten_query") or state["query"]
        docs = self.retriever.run(search_query)
        return {"retrieved_docs": docs}

    def _generate(self, state: WorkflowState) -> dict:
        answer = self.summarizer.run(state["retrieved_docs"], state["query"])
        return {"answer": answer}

    def _evaluate(self, state: WorkflowState) -> dict:
        result = self.evaluator.evaluate(
            state["answer"], state["retrieved_docs"], state["query"]
        )
        return {"eval_result": result}

    def _reflect(self, state: WorkflowState) -> dict:
        eval_r = state["eval_result"]
        count = state.get("reflection_count", 0) + 1

        # Step 1: 让 LLM 改写查询词
        rewrite_prompt = QUERY_REWRITE_PROMPT.format(
            query=state["query"],
            answer=state["answer"],
            feedback=eval_r.get("feedback", ""),
        )
        rewritten_raw = self.summarizer._llm.generate(rewrite_prompt)

        # 提取 "QUERY:" 行
        rewritten_lines = [
            line.replace("QUERY:", "").strip()
            for line in rewritten_raw.split("\n")
            if "QUERY:" in line.upper()
        ]
        rewritten_query = " | ".join(rewritten_lines) if rewritten_lines else state["query"]

        # Step 2: 用改写后的 query 重新检索
        docs = self.retriever.run(rewritten_query)

        # Step 3: 用新检索结果 + 改写 query 重新生成回答
        context = "\n\n---\n\n".join(
            f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
            for d in docs
        )
        reflect_prompt = REFLECT_PROMPT.format(
            query=state["query"],
            rewritten_query=rewritten_query,
            context=context,
            prev_answer=state["answer"],
            feedback=eval_r.get("feedback", ""),
        )
        improved_answer = self.summarizer._llm.generate(reflect_prompt)

        return {
            "answer": improved_answer,
            "retrieved_docs": docs,
            "rewritten_query": rewritten_query,
            "reflection_count": count,
        }

    def _should_continue(self, state: WorkflowState) -> str:
        overall = state["eval_result"].get("overall", 0.5)
        count = state.get("reflection_count", 0)

        if overall >= self.rag_config.quality_threshold:
            return END
        if count >= self.rag_config.max_reflections:
            return END
        return "reflect"

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(WorkflowState)

        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.add_node("evaluate", self._evaluate)
        graph.add_node("reflect", self._reflect)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {END: END, "reflect": "reflect"},
        )
        graph.add_edge("reflect", "retrieve")

        return graph.compile()

    def run(self, query: str) -> dict:
        initial: WorkflowState = {
            "query": query,
            "rewritten_query": "",
            "retrieved_docs": [],
            "answer": "",
            "eval_result": {},
            "reflection_count": 0,
        }
        return self._graph.invoke(initial)
