"""
CLI entry point for querying the RAG pipeline.

Usage:
    python query.py "What is Self-RAG?"
    python query.py "Explain retrieval augmented generation" --stream
"""
import argparse
from dotenv import load_dotenv

load_dotenv()

from src.pipeline.rag_pipeline import RAGPipeline   # noqa: E402
from rich.console import Console                      # noqa: E402
from rich.rule import Rule                            # noqa: E402

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Query the Self-RAG pipeline")
    parser.add_argument("query", type=str, help="Your question")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the answer token by token",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline()
    console.print(Rule("[bold blue]Self-RAG Query[/bold blue]"))
    console.print(f"[dim]Q:[/dim] {args.query}\n")

    if args.stream:
        console.print("[dim]A:[/dim] ", end="")
        for token in pipeline.stream(args.query):
            print(token, end="", flush=True)
        print()
    else:
        result = pipeline.run(args.query)
        console.print(f"[dim]A:[/dim] {result.answer}\n")
        sources = sorted({
            doc.metadata.get("source", "unknown")
            for doc in result.source_documents
        })
        console.print(Rule("[dim]Sources[/dim]"))
        for src in sources:
            console.print(f"  [green]•[/green] {src}")


if __name__ == "__main__":
    main()
