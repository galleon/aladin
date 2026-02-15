"""Document converter service for PDF <-> Markdown conversion using marker-pdf."""

from __future__ import annotations

import os
import asyncio
from pathlib import Path
from datetime import datetime
import structlog

from ..config import settings

logger = structlog.get_logger()


class DocumentConverterService:
    """Service for converting documents between formats using marker-pdf."""

    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._artifact_dict = None

    def _get_artifact_dict(self):
        """Lazy-load the marker model artifacts (heavy operation)."""
        if self._artifact_dict is None:
            from marker.models import create_model_dict

            logger.info("Loading marker model artifacts...")
            self._artifact_dict = create_model_dict()
            logger.info("Marker model artifacts loaded successfully")
        return self._artifact_dict

    def _create_converter(
        self,
        use_llm: bool | None = None,
        llm_model: str | None = None,
    ):
        """Create a marker PDF converter with the given configuration.

        Args:
            use_llm: Override the default LLM setting. If None, uses MARKER_USE_LLM from env.
            llm_model: Specific LLM model to use (optional).
        """
        from marker.converters.pdf import PdfConverter

        artifact_dict = self._get_artifact_dict()

        # Determine if LLM should be used (env default or explicit override)
        should_use_llm = use_llm if use_llm is not None else settings.MARKER_USE_LLM

        llm_service = None
        if should_use_llm and settings.LLM_API_BASE and settings.LLM_API_KEY:
            llm_service = "marker.services.openai.OpenAIService"

            # Set environment variables for OpenAI service
            os.environ["OPENAI_BASE_URL"] = settings.LLM_API_BASE.rstrip("/")
            os.environ["OPENAI_API_KEY"] = settings.LLM_API_KEY
            if llm_model:
                os.environ["OPENAI_MODEL"] = llm_model

            logger.info(
                "Creating marker converter with LLM",
                base_url=settings.LLM_API_BASE,
                model=llm_model,
            )
        else:
            logger.info("Creating marker converter without LLM (using built-in models)")

        return PdfConverter(
            artifact_dict=artifact_dict,
            llm_service=llm_service,
        )

    async def extract_markdown_from_pdf(
        self,
        pdf_path: str,
        use_llm: bool | None = None,
        llm_model: str | None = None,
    ) -> dict[str, str]:
        """Extract markdown from a PDF file using marker-pdf.

        Args:
            pdf_path: Path to the PDF file
            use_llm: Override default LLM setting. If None, uses MARKER_USE_LLM from env.
            llm_model: Specific LLM model to use (optional)

        Returns:
            Dictionary with 'markdown' content and metadata
        """
        # Create converter with configuration
        converter = self._create_converter(use_llm=use_llm, llm_model=llm_model)

        # Run conversion in thread pool to not block async
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: converter(pdf_path))

        # Extract markdown from result
        markdown_content = (
            result.markdown if hasattr(result, "markdown") else str(result)
        )

        # Get metadata if available
        metadata = {}
        if hasattr(result, "metadata"):
            metadata = result.metadata

        logger.info(
            "PDF extraction with marker completed",
            pdf_path=pdf_path,
            markdown_length=len(markdown_content),
            use_llm=use_llm if use_llm is not None else settings.MARKER_USE_LLM,
        )

        return {
            "markdown": markdown_content,
            "source_file": pdf_path,
            "metadata": metadata,
        }

    async def markdown_to_html(
        self,
        markdown_content: str,
        output_path: str,
        title: str = "Translated Document",
        target_language: str = "en",
        simplified: bool = False,
    ) -> str:
        """Convert markdown to a styled HTML file.

        Args:
            markdown_content: The markdown text to convert
            output_path: Where to save the HTML file
            title: Document title
            target_language: Language code for the document
            simplified: Whether this is a simplified translation

        Returns:
            Path to the generated HTML file
        """
        import markdown

        # Convert markdown to HTML
        md = markdown.Markdown(
            extensions=[
                "tables",
                "fenced_code",
                "toc",
                "nl2br",
            ]
        )
        body_html = md.convert(markdown_content)

        mode_text = "Simplified" if simplified else "Standard"
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Create full HTML document with modern styling
        html_content = f"""<!DOCTYPE html>
<html lang="{target_language}">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        * {{ box-sizing: border-box; }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 850px;
            margin: 0 auto;
            padding: 40px 20px;
            line-height: 1.7;
            color: #1f2937;
            background: #fafafa;
        }}

        .document {{
            background: white;
            padding: 60px;
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1);
        }}

        .header {{
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            padding: 30px 40px;
            margin: -60px -60px 40px;
            border-radius: 16px 16px 0 0;
        }}

        .header h1 {{
            margin: 0 0 10px;
            font-size: 1.75rem;
            font-weight: 700;
        }}

        .header .meta {{
            font-size: 0.875rem;
            opacity: 0.9;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}

        .header .meta span {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        h1 {{ font-size: 1.875rem; font-weight: 700; color: #111827; margin-top: 2rem; }}
        h2 {{ font-size: 1.5rem; font-weight: 600; color: #1f2937; margin-top: 2rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem; }}
        h3 {{ font-size: 1.25rem; font-weight: 600; color: #374151; margin-top: 1.5rem; }}

        p {{ margin: 1rem 0; }}

        code {{
            background: #f3f4f6;
            padding: 2px 8px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.875em;
        }}

        pre {{
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            border-radius: 12px;
            overflow-x: auto;
            font-size: 0.875rem;
        }}

        pre code {{
            background: none;
            padding: 0;
        }}

        blockquote {{
            border-left: 4px solid #6366f1;
            margin: 1.5rem 0;
            padding: 1rem 1.5rem;
            background: #f5f3ff;
            border-radius: 0 8px 8px 0;
            color: #4c1d95;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1.5rem 0;
            font-size: 0.9375rem;
        }}

        th, td {{
            border: 1px solid #e5e7eb;
            padding: 12px 16px;
            text-align: left;
        }}

        th {{
            background: #f9fafb;
            font-weight: 600;
            color: #374151;
        }}

        tr:hover td {{
            background: #f9fafb;
        }}

        ul, ol {{
            padding-left: 1.5rem;
        }}

        li {{
            margin: 0.5rem 0;
        }}

        a {{
            color: #6366f1;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        hr {{
            border: none;
            border-top: 2px solid #e5e7eb;
            margin: 2rem 0;
        }}

        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }}

        @media print {{
            body {{ background: white; padding: 0; }}
            .document {{ box-shadow: none; padding: 20px; }}
            .header {{ margin: -20px -20px 20px; }}
        }}
    </style>
</head>
<body>
    <div class="document">
        <div class="header">
            <h1>{title}</h1>
            <div class="meta">
                <span>📄 {mode_text} Translation</span>
                <span>🌐 {target_language.upper()}</span>
                <span>📅 {date_str}</span>
            </div>
        </div>
        {body_html}
    </div>
</body>
</html>"""

        # Save as HTML
        html_path = output_path.replace(".pdf", ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(
            "HTML document generated",
            output_path=html_path,
            markdown_length=len(markdown_content),
        )

        return html_path

    async def markdown_to_pdf(
        self,
        markdown_content: str,
        output_path: str,
        title: str = "Translated Document",
        target_language: str = "en",
        simplified: bool = False,
    ) -> str:
        """Convert markdown to PDF via HTML.

        Args:
            markdown_content: The markdown text to convert
            output_path: Where to save the PDF file (will be created as PDF)
            title: Document title
            target_language: Language code for the document
            simplified: Whether this is a simplified translation

        Returns:
            Path to the generated PDF file
        """
        # First generate HTML
        html_path = output_path.replace(".pdf", ".html")
        await self.markdown_to_html(
            markdown_content=markdown_content,
            output_path=html_path,
            title=title,
            target_language=target_language,
            simplified=simplified,
        )

        # Try to convert HTML to PDF
        try:
            from weasyprint import HTML

            # If output_path is PDF, generate PDF
            if output_path.endswith(".pdf"):
                HTML(filename=html_path).write_pdf(output_path)
                # Clean up HTML file
                if os.path.exists(html_path):
                    os.remove(html_path)
                logger.info("PDF generated successfully", output_path=output_path)
                return output_path
            else:
                # Keep HTML if not requesting PDF
                return html_path

        except ImportError:
            logger.warning("weasyprint not available, falling back to HTML output")
            # Rename HTML to requested output path if different
            if html_path != output_path:
                import shutil

                new_path = output_path.replace(".pdf", ".html")
                shutil.move(html_path, new_path)
                return new_path
            return html_path
        except Exception as e:
            logger.error(f"PDF generation failed: {e}, falling back to HTML")
            # Fall back to HTML
            return html_path

    async def extract_text_from_file(
        self,
        file_path: str,
        file_type: str,
        use_llm: bool | None = None,
        llm_model: str | None = None,
    ) -> str:
        """Extract text content from various file types.

        Args:
            file_path: Path to the file
            file_type: Type of file (pdf, txt, md, docx)
            use_llm: Override default LLM setting for PDF extraction
            llm_model: Specific LLM model to use for PDF extraction
        """
        file_path = Path(file_path)

        if file_type == "pdf":
            result = await self.extract_markdown_from_pdf(
                str(file_path),
                use_llm=use_llm,
                llm_model=llm_model,
            )
            return result["markdown"]

        elif file_type in ("txt", "md", "markdown"):
            return file_path.read_text(encoding="utf-8")

        elif file_type == "docx":
            from docx import Document

            doc = Document(str(file_path))
            paragraphs = [para.text for para in doc.paragraphs]
            return "\n\n".join(paragraphs)

        else:
            raise ValueError(f"Unsupported file type: {file_type}")


# Singleton instance
document_converter_service = DocumentConverterService()
