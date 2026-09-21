from graph import graph, format_sources


def run_rag(question: str):
    initial_state = {
        "original_question": question,
        "search_query": question,
        "documents": [],
        "answer": "",
        "documents_relevant": False,
        "answer_supported": False,
        "verification_feedback": "",
        "retrieval_retry_count": 0,
        "generation_retry_count": 0,
    }

    result = graph.invoke(initial_state)

    return result


if __name__ == "__main__":
    question = input("Question: ")

    result = run_rag(question)

    print("\n===== Final State =====")
    print(result)

    print("\n===== Answer =====")
    print(result["answer"])

    if result["documents_relevant"]:
        sources = format_sources(result["documents"])

        print("\n===== Sources =====")
        for i, source in enumerate(sources, start=1):
            print(f"[{i}] {source}")
