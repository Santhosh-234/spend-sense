"""Orchestration layer: Image → OCR → Classify → Embed → Archive."""
import json
from pathlib import Path
from typing import Optional

from loguru import logger

from src.ingestion.ocr_engine import OCREngine
from src.models.classifier import ExpenseClassifier
from src.vector_store.embeddings import ReceiptVectorStore
from src.vector_store.rag_chain import ExpenseRAGChain


class SpendSensePipeline:
    """
    Stateful pipeline object.  Owns one instance of each component and
    wires them together.  Thread-safe for single-threaded watcher use.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._processed_dir = Path(config["paths"]["processed_data"])
        self._processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialising pipeline components…")
        self._ocr        = OCREngine(config)
        self._classifier = ExpenseClassifier(config)
        self._classifier.load()
        self._store      = ReceiptVectorStore(config)
        self._rag: Optional[ExpenseRAGChain] = None  # lazy — needs ANTHROPIC_API_KEY

    # ------------------------------------------------------------------
    def process_receipt(self, image_path: Path) -> dict:
        """
        End-to-end ingestion for one receipt image.

        Steps:
            1. OCR extraction           → raw_text + heuristic metadata
            2. RF classification        → category + confidence
            3. FAISS embedding + store  → semantic index updated
            4. JSON archive             → data/processed/<stem>.json
        """
        # 1. Extract
        receipt = self._ocr.process(image_path)

        # 2. Classify
        category, confidence = self._classifier.predict(receipt["raw_text"])
        receipt["category"]            = category
        receipt["category_confidence"] = round(confidence, 4)
        logger.info(
            f"{image_path.name} classified as '{category}' "
            f"(confidence: {confidence:.1%})"
        )

        # 3. Index
        self._store.add_receipt(receipt)

        # 4. Archive
        out_path = self._processed_dir / f"{image_path.stem}.json"
        out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Archived metadata → {out_path}")

        return receipt

    # ------------------------------------------------------------------
    def ask(self, question: str) -> dict:
        """Natural-language expense query via RAG."""
        if self._rag is None:
            retriever  = self._store.as_retriever(k=6)
            self._rag  = ExpenseRAGChain(retriever, self._config)
        return self._rag.query(question)
