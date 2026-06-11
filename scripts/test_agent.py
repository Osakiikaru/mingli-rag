"""
端到端 Agent 测试
-----------------
测试两类问题：
  1. 纯命理知识问答（不需要排盘）
  2. 提供生辰的个人命理分析（需要排盘）

用法：python scripts/test_agent.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import mingli_graph

TEST_CASES = [
    {
        "label": "纯知识问答",
        "query": "七杀格如何制化？食神制杀和印绶化杀有什么区别？",
    },
    {
        "label": "个人命理分析",
        "query": "我1990年3月15日午时出生，男，想问事业运势如何，适合什么方向发展？",
    },
]


def run(query: str) -> dict:
    initial_state = {
        "user_query":   query,
        "chat_history": [],
        "query_type":   "",
        "needs_bazi":   False,
        "birth_info":   {},
        "search_query": "",
        "bazi_str":     "",
        "chunks":       [],
        "draft_answer": "",
        "final_answer": "",
    }
    return mingli_graph.invoke(initial_state)


def main():
    for case in TEST_CASES:
        print("\n" + "=" * 70)
        print(f"【{case['label']}】")
        print(f"问题：{case['query']}")
        print("=" * 70)

        result = run(case["query"])

        if result.get("bazi_str"):
            print(f"\n📅 排盘结果：\n{result['bazi_str']}")

        print(f"\n🔍 检索词：{result['search_query']}")
        print(f"\n📚 召回片段（Top-5）：")
        for i, c in enumerate(result["chunks"], 1):
            loc = c["source"]
            if c.get("chapter"):
                loc += f"·{c['chapter']}"
            score = c.get("rerank_score", c.get("rrf_score", ""))
            print(f"  [{i}] {loc}  {score}")

        print(f"\n💬 最终回答：")
        print(result["final_answer"])


if __name__ == "__main__":
    main()
