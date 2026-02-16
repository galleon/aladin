"""
Document extraction: text, JSON, external Docling API, in-process docling, fallback PDF.
"""
from __future__ import annotations

import os
import base64
import json
import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Lazy imports for docling (heavy)
def _get_settings():
    from shared.config import settings
    return settings


def get_file_format_map():
    """Return file extension to InputFormat mapping (lazy to avoid docling import at load)."""
    from docling.datamodel.base_models import InputFormat
    return {
        ".pdf": InputFormat.PDF,
        ".docx": InputFormat.DOCX,
        ".doc": InputFormat.DOCX,
        ".pptx": InputFormat.PPTX,
        ".ppt": InputFormat.PPTX,
        ".html": InputFormat.HTML,
        ".htm": InputFormat.HTML,
        ".md": InputFormat.MD,
        ".txt": None,
        ".json": None,
    }


def _get_docling_prompt() -> str:
    """Get docling conversion prompt from config or default."""
    from worker.prompts_loader import get_prompt as get_config_prompt
    return get_config_prompt("file.docling_convert") or "Convert this page to docling."


def load_text_file(file_path: str) -> str:
    """Load a plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_json_file(file_path: str) -> str:
    """Load a JSON file and convert to readable text."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json_to_text(data)


def json_to_text(data: Any, indent: int = 0) -> str:
    """Convert JSON data to readable text."""
    lines = []
    prefix = "  " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(json_to_text(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {value}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}[{i}]:")
                lines.append(json_to_text(item, indent + 1))
            else:
                lines.append(f"{prefix}- {item}")
    else:
        lines.append(f"{prefix}{data}")
    return "\n".join(lines)


async def call_external_docling(file_path: str, original_filename: str) -> Optional[str]:
    """
    Call external Docling at DOCLING_API_BASE / LLM_API_BASE.
    Returns markdown string or None on failure.
    """
    settings = _get_settings()
    base = (settings.DOCLING_API_BASE or "").strip().rstrip("/")
    if not base:
        return None

    # 1) Try docling-api style: POST /documents/convert
    try:
        url = f"{base}/documents/convert"
        with open(file_path, "rb") as f:
            files = {"document": (original_filename, f)}
            async with httpx.AsyncClient(timeout=600.0) as client:
                r = await client.post(url, files=files)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "application/json" in ct:
            data = r.json()
            out = (
                data.get("markdown")
                or data.get("text")
                or data.get("content")
                or (
                    data.get("data", {}).get("markdown")
                    if isinstance(data.get("data"), dict)
                    else None
                )
            )
            if out:
                return out
        return r.text or None
    except Exception as e:
        logger.debug("External Docling /documents/convert failed: %s", e)

    # 2) Try /convert
    try:
        url2 = f"{base}/convert"
        with open(file_path, "rb") as f:
            files = {"document": (original_filename, f)}
            async with httpx.AsyncClient(timeout=600.0) as client:
                r2 = await client.post(url2, files=files)
        r2.raise_for_status()
        ct = r2.headers.get("content-type", "")
        if "application/json" in ct:
            data = r2.json()
            out = (
                data.get("markdown")
                or data.get("text")
                or data.get("content")
                or None
            )
            if out:
                return out
        return r2.text or None
    except Exception as e2:
        logger.debug("External Docling /convert failed: %s", e2)

    # 3) OpenAI-compatible API: /chat/completions
    model = settings.DOCLING_MODEL
    try:
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        file_ext = os.path.splitext(original_filename)[1].lower()
        mime = "application/pdf" if file_ext == ".pdf" else "application/octet-stream"
        data_url = f"data:{mime};base64,{b64}"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Convert this document to markdown."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": 128 * 1024,
        }
        async with httpx.AsyncClient(timeout=600.0) as client:
            r3 = await client.post(
                f"{base}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        r3.raise_for_status()
        data = r3.json()
        choice = (data.get("choices") or [None])[0]
        if choice and isinstance(choice.get("message"), dict):
            content = choice["message"].get("content")
            if isinstance(content, str) and content.strip():
                return content
        return None
    except Exception as e3:
        logger.warning("External Docling chat/completions (%s) failed: %s", model, e3)
    return None


def get_pdf_metadata(file_path: str) -> dict:
    """Extract document-level metadata from a PDF."""
    out: dict = {}
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        raw = reader.metadata
        if not raw:
            return out
        if raw.get("/Author"):
            out["author"] = str(raw["/Author"]).strip()
        if raw.get("/Title"):
            out["title"] = str(raw["/Title"]).strip()
        if raw.get("/CreationDate"):
            out["creation_date"] = str(raw["/CreationDate"]).strip()
        if raw.get("/ModDate"):
            out["modification_date"] = str(raw["/ModDate"]).strip()
    except Exception as e:
        logger.debug("Could not read PDF metadata from %s: %s", file_path, e)
    return out


def fallback_pdf_extraction(file_path: str) -> str:
    """Fallback PDF extraction using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        logger.error("Fallback PDF extraction failed: %s", e)
        return ""
