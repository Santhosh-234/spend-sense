"""OCR extraction layer — strategy pattern over EasyOCR / Tesseract backends."""
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from PIL import Image


# ---------------------------------------------------------------------------
# Backend abstractions
# ---------------------------------------------------------------------------

class _OCRBackend(ABC):
    @abstractmethod
    def extract_text(self, image_path: Path) -> str: ...


class _EasyOCRBackend(_OCRBackend):
    def __init__(self, languages: list[str]) -> None:
        import easyocr  # lazy import — heavy init only when used
        self._reader = easyocr.Reader(languages, gpu=True)
        logger.debug("EasyOCR reader initialised (CPU mode)")

    def extract_text(self, image_path: Path) -> str:
        results: list[str] = self._reader.readtext(str(image_path), detail=0)
        return "\n".join(results)


class _TesseractBackend(_OCRBackend):
    def __init__(self) -> None:
        import pytesseract
        self._tess = pytesseract

    def extract_text(self, image_path: Path) -> str:
        img = Image.open(image_path)
        return self._tess.image_to_string(img)


# ---------------------------------------------------------------------------
# Regex-based receipt parser (heuristic, no ML)
# ---------------------------------------------------------------------------

class ReceiptParser:
    _DATE_PATTERNS = [
        r"\b(\d{4}-\d{2}-\d{2})\b",                      # ISO: 2024-07-15
        r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b",       # 07/15/2024
        r"\b(\w{3,9} \d{1,2},?\s*\d{4})\b",              # July 15, 2024
    ]
    _TOTAL_PATTERNS = [
        r"(?:total|amount due|grand total|balance)[:\s]*\$?\s*([\d,]+\.\d{2})",
        r"\bTOTAL\b.*?\$([\d,]+\.\d{2})",
        r"\$([\d,]+\.\d{2})\s*$",                         # last dollar amount as fallback
    ]

    def parse(self, raw_text: str, source_path: Path) -> dict:
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        return {
            "raw_text": raw_text,
            "source_file": source_path.name,
            "merchant": self._extract_merchant(lines),
            "date": self._extract_date(raw_text),
            "total_amount": self._extract_total(raw_text),
            "processed_at": datetime.utcnow().isoformat(),
        }

    def _extract_merchant(self, lines: list[str]) -> Optional[str]:
        # First non-trivial line is usually the merchant name
        for line in lines[:4]:
            if len(line) >= 4 and not re.fullmatch(r"[\d\s\-/:.,$]+", line):
                return line
        return lines[0] if lines else None

    def _extract_date(self, text: str) -> Optional[str]:
        for pattern in self._DATE_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    def _extract_total(self, text: str) -> Optional[float]:
        for pattern in self._TOTAL_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except ValueError:
                    continue
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class OCREngine:
    """Unified OCR engine; selects backend from config['ocr']['engine']."""

    def __init__(self, config: dict) -> None:
        engine = config["ocr"]["engine"]
        langs = config["ocr"]["languages"]
        if engine == "easyocr":
            self._backend: _OCRBackend = _EasyOCRBackend(langs)
        elif engine == "tesseract":
            self._backend = _TesseractBackend()
        else:
            raise ValueError(f"Unknown OCR engine: {engine!r}. Choose 'easyocr' or 'tesseract'.")
        self._parser = ReceiptParser()
        logger.info(f"OCREngine ready — backend: {engine}")

    def process(self, image_path: Path) -> dict:
        """Extract text from image and return parsed receipt dict."""
        logger.info(f"OCR processing: {image_path.name}")
        raw_text = self._backend.extract_text(image_path)
        # [n_lines] char count guard
        logger.debug(f"Extracted {len(raw_text)} chars from {image_path.name}")
        return self._parser.parse(raw_text, image_path)
