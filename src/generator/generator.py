"""
Generator — takes a query + retrieved documents and produces an answer via LLM.

Uses a structured prompt that clearly separates context from the question,
making it easy to later extend for Self-RAG reflection steps.
"""
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.generator.llm import get_llm


RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that answers questions strictly based on "
        "the provided context. If the context does not contain enough information "
        "to answer the question, say so clearly instead of making things up.",
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}",
    ),
])


def format_context(documents: list[Document]) -> str:
    """Combine retrieved documents into a single context string."""
    parts = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


class Generator:
    def __init__(self, llm: BaseChatModel | None = None):
        self._llm = llm or get_llm()
        self._chain = RAG_PROMPT | self._llm | StrOutputParser()

    def generate(self, query: str, documents: list[Document]) -> str:
        """Generate an answer for `query` grounded in `documents`."""
        context = format_context(documents)
        return self._chain.invoke({"question": query, "context": context})

    def generate_stream(self, query: str, documents: list[Document]):
        """Stream the answer token by token (for API/UI streaming)."""
        context = format_context(documents)
        return self._chain.stream({"question": query, "context": context})
