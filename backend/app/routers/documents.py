"""Document-level API endpoints (not tied to a specific data domain)."""

import io
import os

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Document, User
from ..config import settings
from ..services.auth import get_current_active_user
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get(
    "/{document_id}/pages/{page_no}",
    summary="Render a PDF page as a JPEG thumbnail",
)
async def get_document_page_image(
    document_id: int,
    page_no: int,
    scale: float = Query(default=1.5, ge=0.5, le=3.0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Return a JPEG rendering of a single PDF page for use as a citation thumbnail.

    ``page_no`` is 1-based (matches the ``page`` field in source references).
    ``scale`` controls render DPI (default 1.5 ≈ 108 dpi for a standard 72-dpi PDF).
    The caller must own a data domain that contains this document.
    """
    document = (
        db.query(Document)
        .filter(Document.id == document_id)
        .first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Ownership check: the document's data domain must belong to the requesting user
    if document.data_domain.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    if document.file_type != "pdf":
        raise HTTPException(status_code=400, detail="Page images are only available for PDF documents")

    file_path = document.filename  # stored as absolute path by the upload handler
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document file not found on disk")

    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.warning("pypdfium2 not installed; page thumbnail unavailable", document_id=document_id)
        raise HTTPException(status_code=501, detail="PDF page rendering is not available on this server (pypdfium2 not installed)")

    try:
        pdf = pdfium.PdfDocument(file_path)
        if page_no < 1 or page_no > len(pdf):
            pdf.close()
            raise HTTPException(status_code=404, detail=f"Page {page_no} does not exist (document has {len(pdf)} pages)")

        page = pdf[page_no - 1]  # pypdfium2 is 0-indexed
        bitmap = page.render(scale=scale, rotation=0)
        pil_image = bitmap.to_pil()

        buf = io.BytesIO()
        pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
        buf.seek(0)
        pdf.close()

        return Response(
            content=buf.read(),
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Page render failed", document_id=document_id, page_no=page_no, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to render page image")
