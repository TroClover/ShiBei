from __future__ import annotations

import os
import sys

from agents.orchestrator import OrchestratorAgent
from config import config


def main():
    print(f"🤖 ShiBei 文档知识库 (Provider: {config.model.provider})")
    print(f"   对话模型: {config.model.chat_model}")
    print(f"   嵌入模型: {config.model.embedding_model}")
    print()

    file_paths = sys.argv[1:] if len(sys.argv) > 1 else None

    if file_paths is None:
        print("用法: python app.py <file1.pdf> <file2.txt> [...]")
        print("或拖放文件到此处:")
        raw = input("> ").strip()
        file_paths = [p.strip().strip('"') for p in raw.split() if p.strip()]

    if not file_paths:
        print("未提供文件，退出。")
        return

    for fp in file_paths:
        if not os.path.exists(fp):
            print(f"⚠ 文件不存在: {fp}")
            file_paths.remove(fp)

    if not file_paths:
        print("没有有效文件，退出。")
        return

    agent = OrchestratorAgent()

    print(f"📄 加载 {len(file_paths)} 个文件...")
    chunk_count = agent.ingest(file_paths)
    print(f"✅ 解析完成，共 {chunk_count} 个文本块\n")

    while True:
        try:
            q = input("🙋 请输入问题（exit 退出）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见!")
            break

        if q.lower() == "exit":
            print("👋 再见!")
            break
        if not q:
            continue

        result = agent.ask(q)
        answer = result.get("answer", "无法生成回答")
        eval_r = result.get("eval_result", {})
        reflections = result.get("reflection_count", 0)

        print(f"\n💡 回答: {answer}")

        if eval_r:
            score = eval_r.get("overall", 0)
            faith = eval_r.get("faithfulness", 0)
            rel = eval_r.get("relevance", 0)
            print(f"📊 评估: 综合 {score:.2f} | 忠实度 {faith:.2f} | 相关性 {rel:.2f}")
            if reflections > 0:
                print(f"🔄 反思次数: {reflections}")

        # Show source chunks
        docs = result.get("retrieved_docs", [])
        if docs:
            print(f"\n📚 检索到的文档片段 ({len(docs)} 条):")
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get("source", "未知")
                snippet = doc.page_content[:120].replace("\n", " ")
                print(f"  [{i}] {src}: {snippet}...")
        print()


if __name__ == "__main__":
    main()
