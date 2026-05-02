from __future__ import annotations

import json
import re

from langchain_core.documents import Document

from core.model_factory import BaseChat

EVAL_PROMPT = """你是一个严格的答案质量评估专家。请基于以下信息对回答进行评分。

## 检索到的文档内容
{context}

## 用户问题
{query}

## 生成的回答
{answer}

## 评分标准
1. **Faithfulness（忠实度，0-1）**：回答中的每一句话是否能从文档内容中找到支撑？如果有任何信息是文档中没有的（幻觉），扣分。
2. **Relevance（相关性，0-1）**：回答是否完整覆盖了用户的问题？是否切题？

请严格以JSON格式输出（不要包含其他文字）：
{{"faithfulness": <0-1之间的浮点数>, "relevance": <0-1之间的浮点数>, "feedback": "<用中文写的一两句改进建议>"}}"""


class Evaluator:
    def __init__(self, llm: BaseChat):
        self._llm = llm

    def evaluate(
        self, answer: str, docs: list[Document], query: str
    ) -> dict:
        context = "\n\n".join(
            f"[来源 {i+1}] {d.page_content}" for i, d in enumerate(docs)
        )
        prompt = EVAL_PROMPT.format(context=context, query=query, answer=answer)
        raw = self._llm.generate(prompt)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = _fallback_parse(raw)

        result.setdefault("faithfulness", 0.5)
        result.setdefault("relevance", 0.5)
        result["overall"] = (result["faithfulness"] + result["relevance"]) / 2
        return result


def _fallback_parse(text: str) -> dict:
    faithfulness = 0.5
    relevance = 0.5
    feedback = ""

    m = re.search(r"faithfulness[:\s]*([0-9.]+)", text, re.IGNORECASE)
    if m:
        faithfulness = float(m.group(1))

    m = re.search(r"relevance[:\s]*([0-9.]+)", text, re.IGNORECASE)
    if m:
        relevance = float(m.group(1))

    m = re.search(r"feedback[:\s]*[\"']?(.+?)[\"']?\s*[}\]]", text, re.IGNORECASE)
    if m:
        feedback = m.group(1)

    return {"faithfulness": faithfulness, "relevance": relevance, "feedback": feedback}
