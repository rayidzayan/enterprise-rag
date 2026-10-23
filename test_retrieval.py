from src.retrieval import ask


if __name__ == "__main__":
    question = input(
        "Ask a question about the company policy: "
    ).strip()

    if not question:
        raise SystemExit("Please enter a question.")

    result = ask(question)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "=" * 60)
    print("RETRIEVAL SUMMARY")
    print("=" * 60)
    print("Vector results:", len(result["vector_results"]))
    print("Graph results:", len(result["graph_results"]))
    print("Combined evidence:", len(result["evidence"]))

    for item in result["evidence"]:
        methods = ", ".join(item["retrieval_methods"])
        print(
            f"\n[{methods}] {item['filename']} "
            f"(chunk {item['chunk_index']})"
        )
        if item["graph_entities"]:
            print("Entities:", ", ".join(item["graph_entities"]))
        print(item["content"][:350])
