"""FAISS-backed receipt vector store via LangChain + HuggingFace embeddings."""
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from loguru import logger


class ReceiptVectorStore:
    """
    Wraps FAISS with receipt-aware metadata schema.

    Stored metadata per document:
        source_file, merchant, date, total_amount, category, processed_at
    """

    def __init__(self, config: dict) -> None:
        self._store_path = Path(config["paths"]["vector_store"]) / "faiss_index"
        self._embeddings = HuggingFaceEmbeddings(
            model_name=config["embedding"]["model"],
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._store: Optional[FAISS] = None
        self._load_or_init()

    # ------------------------------------------------------------------
    def _load_or_init(self) -> None:
        if self._store_path.exists():
            self._store = FAISS.load_local(
                str(self._store_path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"FAISS index loaded ← {self._store_path}")
        else:
            logger.info("No FAISS index found — will create on first receipt ingestion")

    # ------------------------------------------------------------------
    def add_receipt(self, receipt: dict) -> None:
        """Embed a parsed receipt dict and persist the updated index."""
        doc = Document(
            page_content=self._format_doc(receipt),
            metadata={
                "source_file":   receipt.get("source_file", ""),
                "merchant":      receipt.get("merchant", ""),
                "date":          receipt.get("date", ""),
                "total_amount":  float(receipt.get("total_amount") or 0.0),
                "category":      receipt.get("category", "Other"),
                "processed_at":  receipt.get("processed_at", ""),
            },
        )
        if self._store is None:
            self._store = FAISS.from_documents([doc], self._embeddings)
        else:
            self._store.add_documents([doc])

        self._persist()
        logger.info(f"Indexed receipt: {receipt.get('source_file')} → {receipt.get('category')}")

    # ------------------------------------------------------------------
    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        if self._store is None:
            logger.warning("Vector store is empty — no receipts ingested yet")
            return []
        return self._store.similarity_search(query, k=k)

    def as_retriever(self, k: int = 5):
        if self._store is None:
            raise RuntimeError("Vector store is empty — ingest at least one receipt first")
        return self._store.as_retriever(search_kwargs={"k": k})

    # ------------------------------------------------------------------
    @staticmethod
    def _format_doc(r: dict) -> str:
        amount = r.get("total_amount")
        amount_str = f"${float(amount):.2f}" if amount is not None else "unknown"
        return (
            f"Merchant: {r.get('merchant', 'Unknown')}\n"
            f"Date: {r.get('date', 'Unknown')}\n"
            f"Amount: {amount_str}\n"
            f"Category: {r.get('category', 'Other')}\n"
            f"---\n"
            f"{r.get('raw_text', '')}"
        )

    def _persist(self) -> None:
        self._store_path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self._store_path))
