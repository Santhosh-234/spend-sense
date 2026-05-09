"""RAG chain using LangChain 0.3 LCEL — no legacy langchain.chains dependency."""
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from loguru import logger


_EXPENSE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are Spend-Sense AI, a precise personal finance assistant.
Analyse the receipt data below and answer the user's query.
Rules:
- Always cite specific receipts: merchant, date, amount.
- Sum amounts when the query asks for totals.
- If the receipts don't contain enough information, say so explicitly.
- Never hallucinate amounts or dates.

Receipt Data:
{context}

User Question: {question}

Answer:""",
)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


class ExpenseRAGChain:
    """
    LCEL-based RAG chain: retriever → prompt → ChatOllama → StrOutputParser.

    Requires Ollama running at config['llm']['base_url'] with the target
    model already pulled: `ollama pull llama3.1:8b`
    """

    def __init__(self, retriever: Any, config: dict) -> None:
        llm_cfg = config["llm"]

        llm = ChatOllama(
            model=llm_cfg["model"],
            base_url=llm_cfg["base_url"],
            temperature=llm_cfg["temperature"],
            num_predict=llm_cfg["max_tokens"],
            num_ctx=llm_cfg["num_ctx"],
        )

        # LCEL chain: str → str (answer only)
        self._answer_chain = _EXPENSE_PROMPT | llm | StrOutputParser()
        self._retriever = retriever
        logger.info(f"RAG chain ready — Ollama model: {llm_cfg['model']} @ {llm_cfg['base_url']}")

    def query(self, question: str) -> dict:
        """
        Retrieve relevant receipts then generate an answer.

        Returns:
            {"answer": str, "sources": [{"merchant", "date", "amount", "category", "file"}, ...]}
        """
        logger.info(f"RAG query: {question!r}")

        docs: list[Document] = self._retriever.invoke(question)
        context = _format_docs(docs)

        answer = self._answer_chain.invoke({"context": context, "question": question})

        sources = [
            {
                "merchant": d.metadata.get("merchant"),
                "date":     d.metadata.get("date"),
                "amount":   d.metadata.get("total_amount"),
                "file":     d.metadata.get("source_file"),
                "category": d.metadata.get("category"),
            }
            for d in docs
        ]
        return {"answer": answer, "sources": sources}
