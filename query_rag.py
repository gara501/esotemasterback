import argparse
import json

from src.rag import answer_question


def parse_filter(values: list[str] | None) -> dict:
    filters = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Invalid filter '{value}'. Use key=value.")
        key, raw = value.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if key == "page":
            filters[key] = int(raw)
        else:
            filters[key] = raw
    return filters


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Qdrant-backed RAG.")
    parser.add_argument("question", help="Question to ask.")
    parser.add_argument("--mode", default="general", help="RAG mode.")
    parser.add_argument("--initial-k", type=int, default=None)
    parser.add_argument("--final-k", type=int, default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--debug-chunks", action="store_true")
    parser.add_argument(
        "--filter",
        action="append",
        help="Optional metadata filter. Repeat as needed: --filter title=... --filter page=7",
    )
    args = parser.parse_args()

    rag_config = {
        "debug_chunks": args.debug_chunks,
    }
    if args.initial_k is not None:
        rag_config["initial_k"] = args.initial_k
    if args.final_k is not None:
        rag_config["final_k"] = args.final_k
    if args.no_rerank:
        rag_config["use_rerank"] = False

    result = answer_question(
        question=args.question,
        mode=args.mode,
        rag_config=rag_config,
        filters=parse_filter(args.filter),
    )
    print(json.dumps(json.loads(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
