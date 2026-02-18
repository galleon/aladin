"""Tests for file validation service."""

import pytest
from app.services.file_validation import (
    FileValidationService,
    ValidationErrorCode,
    MIN_FILE_SIZE,
)


@pytest.fixture
def validation_service():
    """Create a validation service instance."""
    service = FileValidationService(
        max_file_size=1024 * 1024,  # 1MB for testing
        max_video_size=5 * 1024 * 1024,  # 5MB for testing
        enable_duplicate_check=True,
    )
    # Clear cache before each test to ensure test isolation
    service.clear_checksum_cache()
    yield service
    # Clear cache after each test
    service.clear_checksum_cache()


class TestFileValidation:
    """Test file validation functionality."""

    def test_valid_pdf_file(self, validation_service):
        """Test validation of a valid PDF file."""
        # PDF header
        pdf_content = b"%PDF-1.4\n" + b"x" * 100
        result = validation_service.validate_file(pdf_content, "test.pdf")
        
        assert result.is_valid
        assert "checksum" in result.metadata
        assert result.metadata["extension"] == ".pdf"

    def test_valid_text_file(self, validation_service):
        """Test validation of a valid text file."""
        text_content = b"Hello, world! This is a test file.\n" * 10
        result = validation_service.validate_file(text_content, "test.txt")
        
        assert result.is_valid
        assert result.metadata["extension"] == ".txt"

    def test_valid_json_file(self, validation_service):
        """Test validation of a valid JSON file."""
        json_content = b'{"key": "value", "number": 123}'
        result = validation_service.validate_file(json_content, "test.json")
        
        assert result.is_valid
        assert result.metadata["extension"] == ".json"

    def test_empty_file(self, validation_service):
        """Test that empty files are rejected."""
        result = validation_service.validate_file(b"", "empty.txt")
        
        assert not result.is_valid
        assert result.error_code == ValidationErrorCode.EMPTY_FILE

    def test_file_too_small(self, validation_service):
        """Test that files below minimum size are rejected."""
        small_content = b"tiny"  # 4 bytes, less than MIN_FILE_SIZE (10 bytes)
        result = validation_service.validate_file(small_content, "tiny.txt")
        
        assert not result.is_valid
        assert result.error_code == ValidationErrorCode.FILE_TOO_SMALL

    def test_file_too_large(self, validation_service):
        """Test that files above maximum size are rejected."""
        # Create content larger than 1MB
        large_content = b"x" * (2 * 1024 * 1024)
        result = validation_service.validate_file(large_content, "large.txt")
        
        assert not result.is_valid
        assert result.error_code == ValidationErrorCode.FILE_TOO_LARGE

    def test_video_file_size_limits(self, validation_service):
        """Test that video files have different size limits."""
        # Create content that's 2MB (too large for regular, ok for video)
        video_content = b"\x00\x00\x00\x18ftypmp42" + b"x" * (2 * 1024 * 1024)
        result = validation_service.validate_file(video_content, "video.mp4")
        
        assert result.is_valid  # Should pass for video

    def test_invalid_encoding(self, validation_service):
        """Test that invalid UTF-8 encoding is detected in text files."""
        # Invalid UTF-8 sequence
        invalid_content = b"Hello " + b"\xff\xfe" + b" World" + b"x" * 100
        result = validation_service.validate_file(invalid_content, "invalid.txt")
        
        assert not result.is_valid
        assert result.error_code == ValidationErrorCode.INVALID_ENCODING

    def test_markdown_encoding(self, validation_service):
        """Test that markdown files are validated for UTF-8 encoding."""
        valid_md = "# Header\n\nThis is valid UTF-8 content.".encode("utf-8")
        result = validation_service.validate_file(valid_md, "test.md")
        
        assert result.is_valid

    def test_csv_encoding(self, validation_service):
        """Test that CSV files are validated for UTF-8 encoding."""
        valid_csv = "col1,col2,col3\nval1,val2,val3\n".encode("utf-8")
        result = validation_service.validate_file(valid_csv, "test.csv")
        
        assert result.is_valid

    def test_duplicate_file_detection(self, validation_service):
        """Test that duplicate files are detected."""
        content = b"This is unique content for testing" + b"x" * 100
        
        # First upload should succeed
        result1 = validation_service.validate_file(content, "original.txt")
        assert result1.is_valid
        
        # Second upload with same content should be rejected
        result2 = validation_service.validate_file(content, "duplicate.txt")
        assert not result2.is_valid
        assert result2.error_code == ValidationErrorCode.DUPLICATE_FILE
        assert "original.txt" in result2.error_message

    def test_duplicate_check_different_content(self, validation_service):
        """Test that different content is not flagged as duplicate."""
        content1 = b"First file content" + b"x" * 100
        content2 = b"Second file content" + b"y" * 100
        
        result1 = validation_service.validate_file(content1, "file1.txt")
        result2 = validation_service.validate_file(content2, "file2.txt")
        
        assert result1.is_valid
        assert result2.is_valid

    def test_pdf_magic_bytes(self, validation_service):
        """Test PDF magic bytes detection."""
        # Valid PDF header
        pdf_content = b"%PDF-1.4\n" + b"x" * 100
        result = validation_service.validate_file(pdf_content, "test.pdf")
        assert result.is_valid

    def test_docx_file_validation(self, validation_service):
        """Test DOCX file validation (ZIP-based format)."""
        # DOCX files are ZIP archives with specific structure
        # PK zip header
        docx_content = b"PK\x03\x04" + b"x" * 100
        result = validation_service.validate_file(docx_content, "test.docx")
        assert result.is_valid

    def test_mp4_magic_bytes(self, validation_service):
        """Test MP4 magic bytes detection."""
        # MP4 with ftyp box at offset 4
        mp4_content = b"\x00\x00\x00\x18ftypmp42" + b"x" * 100
        result = validation_service.validate_file(mp4_content, "test.mp4")
        assert result.is_valid

    def test_html_file_validation(self, validation_service):
        """Test HTML file validation."""
        html_content = b"<!DOCTYPE html><html><body>Test</body></html>"
        result = validation_service.validate_file(html_content, "test.html")
        assert result.is_valid

    def test_no_extension_file(self, validation_service):
        """Test that files without extension are rejected."""
        content = b"some content" + b"x" * 100
        result = validation_service.validate_file(content, "noextension")
        
        assert not result.is_valid
        assert result.error_code == ValidationErrorCode.INVALID_FILE_TYPE

    def test_checksum_metadata(self, validation_service):
        """Test that checksum is included in metadata."""
        content = b"test content for checksum" + b"x" * 100
        result = validation_service.validate_file(content, "test.txt")
        
        assert result.is_valid
        assert "checksum" in result.metadata
        assert len(result.metadata["checksum"]) == 64  # SHA-256 hex

    def test_clear_checksum_cache(self, validation_service):
        """Test that checksum cache can be cleared."""
        content = b"test content" + b"x" * 100
        
        # First upload
        result1 = validation_service.validate_file(content, "file1.txt")
        assert result1.is_valid
        
        # Second upload should fail (duplicate)
        result2 = validation_service.validate_file(content, "file2.txt")
        assert not result2.is_valid
        
        # Clear cache
        validation_service.clear_checksum_cache()
        
        # Third upload should succeed after cache clear
        result3 = validation_service.validate_file(content, "file3.txt")
        assert result3.is_valid

    def test_validation_without_duplicate_check(self):
        """Test validation with duplicate checking disabled."""
        service = FileValidationService(
            max_file_size=1024 * 1024,
            enable_duplicate_check=False,
        )
        
        content = b"test content" + b"x" * 100
        
        # Both uploads should succeed
        result1 = service.validate_file(content, "file1.txt")
        result2 = service.validate_file(content, "file2.txt")
        
        assert result1.is_valid
        assert result2.is_valid

    def test_expected_extension_mismatch(self, validation_service):
        """Test that extension mismatch is detected when expected extension is provided."""
        content = b"test content" + b"x" * 100
        result = validation_service.validate_file(
            content, "test.txt", expected_extension=".pdf"
        )
        
        assert not result.is_valid
        assert result.error_code == ValidationErrorCode.INVALID_FILE_TYPE

    def test_binary_file_no_encoding_check(self, validation_service):
        """Test that binary files (PDF, DOCX) are not checked for encoding."""
        # PDF with potentially invalid UTF-8 bytes in content
        pdf_content = b"%PDF-1.4\n" + b"\xff\xfe\xfd" + b"x" * 100
        result = validation_service.validate_file(pdf_content, "test.pdf")
        
        # Should pass because PDFs are binary files
        assert result.is_valid

    def test_pptx_validation(self, validation_service):
        """Test PPTX file validation (ZIP-based format)."""
        pptx_content = b"PK\x03\x04" + b"x" * 100
        result = validation_service.validate_file(pptx_content, "test.pptx")
        assert result.is_valid

    def test_validation_result_bool(self, validation_service):
        """Test that ValidationResult can be used as boolean."""
        valid_content = b"valid content" + b"x" * 100
        result = validation_service.validate_file(valid_content, "test.txt")
        
        assert bool(result) is True
        assert result  # Can use directly in if statements
        
        invalid_content = b""
        result2 = validation_service.validate_file(invalid_content, "empty.txt")
        assert bool(result2) is False
        assert not result2


class TestValidationService:
    """Test validation service helper functions."""

    def test_get_validation_service(self):
        """Test that get_validation_service returns a service instance."""
        from app.services.file_validation import get_validation_service
        
        service = get_validation_service()
        assert service is not None
        assert isinstance(service, FileValidationService)

    def test_get_validation_service_singleton(self):
        """Test that get_validation_service returns the same instance."""
        from app.services.file_validation import get_validation_service, reset_validation_service
        
        # Reset global instance for this test
        reset_validation_service()
        
        service1 = get_validation_service()
        service2 = get_validation_service()
        
        assert service1 is service2
        
        # Clean up
        reset_validation_service()
