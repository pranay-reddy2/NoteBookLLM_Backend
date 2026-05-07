"""
rag/loader.py
─────────────
Text extraction from PDF and TXT files.

Supports:
  - PDF via pdfplumber (with page-level metadata)
  - Plain text files (UTF-8 / latin-1 fallback)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def extract_text_from_file(file_path: str, ext: str) -> str:
    """
    Extract raw text from a file.

    Args:
        file_path: Absolute or relative path to the file on disk.
        ext: File extension ('.pdf' or '.txt').

    Returns:
        Extracted text as a single string.

    Raises:
        ValueError: If the extension is unsupported.
        RuntimeError: If extraction fails.
    """
    ext = ext.lower()
    if ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext == ".txt":
        return _extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def _extract_pdf(file_path: str) -> str:
    """
    Extract text from a PDF using pdfplumber.

    pdfplumber preserves layout better than PyPDF2 for most documents.
    Falls back to PyPDF2 if pdfplumber fails.

    Returns text with page markers so downstream chunker can track page numbers.
    """
    try:
        import pdfplumber

        pages_text: list[str] = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    # Embed page marker so chunker can extract it
                    pages_text.append(f"[PAGE {i}]\n{text}")

        if pages_text:
            return "\n\n".join(pages_text)

        # pdfplumber returned empty – try PyPDF2
        logger.warning("pdfplumber returned empty text; trying PyPDF2 fallback.")
        return _extract_pdf_pypdf2(file_path)

    except ImportError:
        logger.warning("pdfplumber not installed; falling back to PyPDF2.")
        return _extract_pdf_pypdf2(file_path)
    except Exception as e:
        logger.error(f"pdfplumber failed: {e}; trying PyPDF2 fallback.")
        return _extract_pdf_pypdf2(file_path)


def _extract_pdf_pypdf2(file_path: str) -> str:
    """Fallback PDF extraction using PyPDF2."""
    import PyPDF2

    pages_text: list[str] = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(f"[PAGE {i}]\n{text}")

    return "\n\n".join(pages_text)


def _extract_txt(file_path: str) -> str:
    """Read a plain text file with UTF-8 / latin-1 fallback."""
    path = Path(file_path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")