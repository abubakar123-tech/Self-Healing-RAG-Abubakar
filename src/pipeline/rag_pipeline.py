"""
Basic RAG pipeline — orchestrates: retrieve → generate.

This is Phase 2. Phase 3 will extend this with Self-RAG reflection nodes
(ISREL, ISSUP, ISUSE) using LangGraph.
"""
from dataclasses import dataclass
from langchain_core.documents import Document
from src.retriever.retriever import Retriever
from src.generator.generator import Generator


@dataclass
class RAGResult:
    query: str
    answer: str
    source_documents: list[Document]

    def __str__(self) -> str:
        sources = {
            doc.metadata.get("source", "unknown")
            for doc in self.source_documents
        }
        sources_str = "\n  ".join(sorted(sources))
        return (
            f"Answer:\n{self.answer}\n\n"
            f"Sources ({len(self.source_documents)} chunks):\n  {sources_str}"
        )


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
    ):
        self._retriever = retriever or Retriever()
        self._generator = generator or Generator()

    def run(self, query: str) -> RAGResult:
        """
        Full RAG pass: retrieve relevant chunks, generate an answer.

        Args:
            query: The user's question.

        Returns:
            RAGResult with the answer and source documents.
        """
        documents = self._retriever.retrieve(query)
        answer = self._generator.generate(query, documents)
        return RAGResult(query=query, answer=answer, source_documents=documents)

    def stream(self, query: str):
        """
        Stream the answer token by token.

        Yields:
            str tokens as they are generated.
        Also returns source documents after streaming via RAGResult.source_documents.
        """
        documents = self._retriever.retrieve(query)
        for token in self._generator.generate_stream(query, documents):
            yield token
