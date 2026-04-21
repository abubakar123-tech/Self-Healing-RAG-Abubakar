"""
Tests for the basic RAG pipeline (Phase 2).

Uses mocks so no API key or vector store is needed to run.
"""
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.generator.generator import Generator, format_context
from src.pipeline.rag_pipeline import RAGPipeline, RAGResult


# ── format_context ────────────────────────────────────────────────────────────

def test_format_context_includes_content():
    docs = [
        Document(page_content="RAG stands for Retrieval-Augmented Generation.", metadata={"source": "doc1.pdf"}),
        Document(page_content="Self-RAG adds reflection tokens.", metadata={"source": "doc2.pdf"}),
    ]
    ctx = format_context(docs)
    assert "RAG stands for" in ctx
    assert "Self-RAG adds" in ctx
    assert "doc1.pdf" in ctx
    assert "[1]" in ctx
    assert "[2]" in ctx


def test_format_context_empty():
    assert format_context([]) == ""


# ── Generator ─────────────────────────────────────────────────────────────────

def test_generator_calls_llm():
    mock_llm = MagicMock()
    mock_llm.__or__ = lambda self, other: MagicMock(
        __or__=lambda s, o: MagicMock(invoke=lambda x: "mocked answer")
    )

    with patch("src.generator.generator.get_llm", return_value=mock_llm):
        gen = Generator(llm=mock_llm)
        # Patch the internal chain directly
        gen._chain = MagicMock()
        gen._chain.invoke.return_value = "Self-RAG improves faithfulness."

        docs = [Document(page_content="context text", metadata={})]
        answer = gen.generate("What is Self-RAG?", docs)

    assert answer == "Self-RAG improves faithfulness."
    gen._chain.invoke.assert_called_once()


# ── RAGPipeline ───────────────────────────────────────────────────────────────

def test_rag_pipeline_run():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Document(page_content="RAG retrieves relevant docs.", metadata={"source": "test.pdf"})
    ]

    mock_generator = MagicMock()
    mock_generator.generate.return_value = "RAG uses retrieved context to answer."

    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
    result = pipeline.run("What is RAG?")

    assert isinstance(result, RAGResult)
    assert result.answer == "RAG uses retrieved context to answer."
    assert result.query == "What is RAG?"
    assert len(result.source_documents) == 1
    mock_retriever.retrieve.assert_called_once_with("What is RAG?")
    mock_generator.generate.assert_called_once()


def test_rag_result_str():
    docs = [Document(page_content="text", metadata={"source": "file.pdf"})]
    result = RAGResult(query="q", answer="the answer", source_documents=docs)
    output = str(result)
    assert "the answer" in output
    assert "file.pdf" in output
