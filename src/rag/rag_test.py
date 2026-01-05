"""
Iterate through all RAG queries and save query-answer pairs to a text file.
"""

from src.rag.utils.agent_response import ChatSession
from src.rag.utils.queries1 import QUERIES
from src.rag.rag_agent import answer_query

OUTPUT_FILE = "rag_evaluation_output.txt"


def run_all_queries(output_path=OUTPUT_FILE):
    session = ChatSession()

    with open(output_path, "w", encoding="utf-8") as f:
        for topic_idx, topic in enumerate(QUERIES, start=1):
            for q_idx, query in topic.items():

                result = answer_query(query, session)

                f.write(f"Topic {topic_idx} — Query {q_idx}\n")
                f.write("\n")
                f.write(f"Query:\n{result['query']}\n\n")
                f.write(f"Answer:\n{result['answer']}\n")
                f.write("\n\n")

    print(f"Saved all query-answer pairs to {output_path}")


if __name__ == "__main__":
    run_all_queries()
