from __future__ import annotations

from langchain_core.documents import Document

from core.model_factory import BaseChat

SUMMARY_PROMPT = """你是一个严谨的学术论文分析专家。请仔细阅读以下所有文档片段，逐段扫描，不要遗漏任何信息。

## 文档内容
{context}

## 用户问题
{query}

## 回答要求
1. **穷尽检索**：逐一检查每个文档片段，列出所有相关信息，不要只举一两个例子就停止。
2. **原文引用**：尽可能引用文档中的原文表述，保留数字、名称等精确信息。
3. **区分确定与推测**：明确标注哪些是文档中明确出现的，哪些是推断的。
4. **如信息不全**：不要直接说"没有"，而是列出在文档中找到了什么，哪些方面文档未覆盖。
5. **格式化输出**：如问题是列出多项内容，请用编号列表清晰呈现。

请给出详尽、全面的回答："""


class SummaryAgent:
    def __init__(self, llm: BaseChat):
        self._llm = llm

    def run(self, docs: list[Document], query: str) -> str:
        context = "\n\n---\n\n".join(
            f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
            for d in docs
        )
        prompt = SUMMARY_PROMPT.format(context=context, query=query)
        return self._llm.generate(prompt)
