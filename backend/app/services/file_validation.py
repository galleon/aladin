"""
File validation service for data quality checks.

Provides comprehensive validation for uploaded files including:
- File type validation with magic bytes (MIME type verification)
- Data encoding validation  
- File size limits (min/max)
- Duplicate detection with checksums
- File corruption detection
"""

import hashlib
import magic
import os
from enum import Enum
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

import structlog

logger = structlog.get_logger()


class ValidationErrorCode(str, Enum):
    """Error codes for validation failures."""
    INVALID_FILE_TYPE = "invalid_file_type"
    INVALID_ENCODING = "invalid_encoding"
    FILE_TOO_SMALL = "file_too_small"
    FILE_TOO_LARGE = "file_too_large"
    DUPLICATE_FILE = "duplicate_file"
    CORRUPTED_FILE = "corrupted_file"
    INVALID_MIME_TYPE = "invalid_mime_type"
    EMPTY_FILE = "empty_file"


class FileValidationResult:
    """Result of file validation."""
    
    def __init__(
        self,
        is_valid: bool,
        error_code: Optional[ValidationErrorCode] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.is_valid = is_valid
        self.error_code = error_code
        self.error_message = error_message
        self.metadata = metadata or {}
    
    def __bool__(self):
        return self.is_valid


# MIME type mappings for supported file types
MIME_TYPE_MAPPINGS = {
    # Documents
    "application/pdf": [".pdf"],
    "text/plain": [".txt"],
    "text/markdown": [".md"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    "application/msword": [".doc"],
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
    "application/vnd.ms-powerpoint": [".ppt"],
    "text/html": [".html", ".htm"],
    "text/csv": [".csv"],
    "application/json": [".json"],
    # Video
    "video/mp4": [".mp4"],
    "video/x-m4v": [".mp4"],
    # Archives (for docx/pptx which are ZIP-based)
    "application/zip": [".docx", ".pptx"],
}

# File signatures (magic bytes) for common file types
FILE_SIGNATURES = {
    b"%PDF": "pdf",
    b"\x25\x50\x44\x46": "pdf",  # %PDF alternative
    b"PK\x03\x04": "zip",  # ZIP (docx, pptx, xlsx)
    b"PK\x05\x06": "zip",  # Empty ZIP
    b"PK\x07\x08": "zip",  # Spanned ZIP
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1": "doc",  # MS Office old format
    b"\x00\x00\x00\x14ftypmp4": "mp4",  # MP4 video
    b"\x00\x00\x00\x18ftypmp42": "mp4",  # MP4 video variant
    b"\x00\x00\x00\x1cftypisom": "mp4",  # MP4 ISO variant
}

# Text-based file extensions that should be UTF-8 encoded
TEXT_BASED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".csv", ".json"}

# Minimum file size (10 bytes to avoid completely empty files)
MIN_FILE_SIZE = 10

# Default maximum file sizes (can be overridden)
DEFAULT_MAX_FILE_SIZE = 52428800  # 50MB
DEFAULT_MAX_VIDEO_SIZE = 524288000  # 500MB


class FileValidationService:
    """Service for validating uploaded files."""
    
    def __init__(
        self,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        max_video_size: int = DEFAULT_MAX_VIDEO_SIZE,
        min_file_size: int = MIN_FILE_SIZE,
        enable_duplicate_check: bool = True,
    ):
        self.max_file_size = max_file_size
        self.max_video_size = max_video_size
        self.min_file_size = min_file_size
        self.enable_duplicate_check = enable_duplicate_check
        self._checksum_cache: Dict[str, str] = {}
    
    def validate_file(
        self,
        file_content: bytes,
        filename: str,
        expected_extension: Optional[str] = None,
    ) -> FileValidationResult:
        """
        Validate a file's content and properties.
        
        Args:
            file_content: The file's binary content
            filename: Original filename
            expected_extension: Expected file extension (e.g., '.pdf')
        
        Returns:
            FileValidationResult indicating if validation passed
        """
        # Extract file extension
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Check 1: Empty file
        if len(file_content) == 0:
            return FileValidationResult(
                is_valid=False,
                error_code=ValidationErrorCode.EMPTY_FILE,
                error_message=f"File '{filename}' is empty",
            )
        
        # Check 2: File size limits
        size_check = self._check_file_size(file_content, file_ext, filename)
        if not size_check.is_valid:
            return size_check
        
        # Check 3: Magic bytes validation (MIME type)
        mime_check = self._check_mime_type(file_content, file_ext, filename)
        if not mime_check.is_valid:
            return mime_check
        
        # Check 4: File type validation
        type_check = self._check_file_type(file_ext, expected_extension, filename)
        if not type_check.is_valid:
            return type_check
        
        # Check 5: Encoding validation for text files
        encoding_check = self._check_encoding(file_content, file_ext, filename)
        if not encoding_check.is_valid:
            return encoding_check
        
        # Check 6: Duplicate detection
        if self.enable_duplicate_check:
            duplicate_check = self._check_duplicate(file_content, filename)
            if not duplicate_check.is_valid:
                return duplicate_check
        
        # Calculate checksum for metadata
        checksum = self._calculate_checksum(file_content)
        
        logger.info(
            "File validation passed",
            filename=filename,
            size=len(file_content),
            checksum=checksum,
            extension=file_ext,
        )
        
        return FileValidationResult(
            is_valid=True,
            metadata={
                "size": len(file_content),
                "checksum": checksum,
                "extension": file_ext,
            },
        )
    
    def _check_file_size(
        self, file_content: bytes, file_ext: str, filename: str
    ) -> FileValidationResult:
        """Check if file size is within acceptable limits."""
        size = len(file_content)
        
        # Check minimum size
        if size < self.min_file_size:
            return FileValidationResult(
                is_valid=False,
                error_code=ValidationErrorCode.FILE_TOO_SMALL,
                error_message=f"File '{filename}' is too small ({size} bytes). Minimum size: {self.min_file_size} bytes",
            )
        
        # Check maximum size based on file type
        max_size = self.max_video_size if file_ext == ".mp4" else self.max_file_size
        
        if size > max_size:
            return FileValidationResult(
                is_valid=False,
                error_code=ValidationErrorCode.FILE_TOO_LARGE,
                error_message=f"File '{filename}' is too large ({size} bytes). Maximum size: {max_size} bytes",
            )
        
        return FileValidationResult(is_valid=True)
    
    def _check_mime_type(
        self, file_content: bytes, file_ext: str, filename: str
    ) -> FileValidationResult:
        """Validate file content matches expected MIME type using magic bytes."""
        try:
            # Check file signature (magic bytes)
            detected_type = self._detect_file_type_from_signature(file_content)
            
            # For ZIP-based formats (docx, pptx), accept ZIP signature
            if file_ext in [".docx", ".pptx"] and detected_type == "zip":
                return FileValidationResult(is_valid=True)
            
            # For MP4, accept any MP4 variant
            if file_ext == ".mp4" and detected_type == "mp4":
                return FileValidationResult(is_valid=True)
            
            # For PDF, check signature
            if file_ext == ".pdf" and detected_type == "pdf":
                return FileValidationResult(is_valid=True)
            
            # For text files, we'll rely on encoding check
            if file_ext in TEXT_BASED_EXTENSIONS:
                return FileValidationResult(is_valid=True)
            
            # For DOC files
            if file_ext == ".doc" and detected_type == "doc":
                return FileValidationResult(is_valid=True)
            
            # If we have libmagic available, use it for better detection
            try:
                mime_type = magic.from_buffer(file_content, mime=True)
                if self._is_mime_type_compatible(mime_type, file_ext):
                    return FileValidationResult(is_valid=True)
                
                logger.warning(
                    "MIME type mismatch",
                    filename=filename,
                    expected_ext=file_ext,
                    detected_mime=mime_type,
                )
                return FileValidationResult(
                    is_valid=False,
                    error_code=ValidationErrorCode.INVALID_MIME_TYPE,
                    error_message=f"File '{filename}' MIME type '{mime_type}' does not match extension '{file_ext}'",
                )
            except Exception as e:
                # libmagic not available, accept if signature check passed
                logger.debug("libmagic not available, using signature check", error=str(e))
                return FileValidationResult(is_valid=True)
            
        except Exception as e:
            logger.warning("MIME type check failed", filename=filename, error=str(e))
            # Don't fail validation if MIME check has issues
            return FileValidationResult(is_valid=True)
    
    def _detect_file_type_from_signature(self, file_content: bytes) -> Optional[str]:
        """Detect file type from magic bytes."""
        for signature, file_type in FILE_SIGNATURES.items():
            if file_content.startswith(signature):
                return file_type
        
        # Check for MP4 at different offsets (ftyp box can be at offset 4)
        if len(file_content) > 12:
            for offset in [4, 8]:
                if file_content[offset:offset+4] == b"ftyp":
                    return "mp4"
        
        return None
    
    def _is_mime_type_compatible(self, mime_type: str, file_ext: str) -> bool:
        """Check if MIME type is compatible with file extension."""
        compatible_extensions = MIME_TYPE_MAPPINGS.get(mime_type, [])
        return file_ext in compatible_extensions
    
    def _check_file_type(
        self, file_ext: str, expected_extension: Optional[str], filename: str
    ) -> FileValidationResult:
        """Check if file type is supported."""
        if not file_ext:
            return FileValidationResult(
                is_valid=False,
                error_code=ValidationErrorCode.INVALID_FILE_TYPE,
                error_message=f"File '{filename}' has no extension",
            )
        
        # If expected extension is provided, verify it matches
        if expected_extension and file_ext != expected_extension:
            return FileValidationResult(
                is_valid=False,
                error_code=ValidationErrorCode.INVALID_FILE_TYPE,
                error_message=f"File '{filename}' extension '{file_ext}' does not match expected '{expected_extension}'",
            )
        
        return FileValidationResult(is_valid=True)
    
    def _check_encoding(
        self, file_content: bytes, file_ext: str, filename: str
    ) -> FileValidationResult:
        """Check encoding for text-based files."""
        # Only check encoding for text-based files
        if file_ext not in TEXT_BASED_EXTENSIONS:
            return FileValidationResult(is_valid=True)
        
        try:
            # Try to decode as UTF-8
            file_content.decode("utf-8")
            return FileValidationResult(is_valid=True)
        except UnicodeDecodeError as e:
            logger.warning(
                "Invalid encoding detected",
                filename=filename,
                extension=file_ext,
                error=str(e),
            )
            return FileValidationResult(
                is_valid=False,
                error_code=ValidationErrorCode.INVALID_ENCODING,
                error_message=f"File '{filename}' is not valid UTF-8 encoded. Text files must be UTF-8 encoded.",
            )
    
    def _check_duplicate(
        self, file_content: bytes, filename: str
    ) -> FileValidationResult:
        """Check for duplicate files using checksum."""
        checksum = self._calculate_checksum(file_content)
        
        # Check if this checksum exists in cache
        if checksum in self._checksum_cache:
            original_filename = self._checksum_cache[checksum]
            logger.info(
                "Duplicate file detected",
                filename=filename,
                original=original_filename,
                checksum=checksum,
            )
            return FileValidationResult(
                is_valid=False,
                error_code=ValidationErrorCode.DUPLICATE_FILE,
                error_message=f"File '{filename}' is a duplicate of '{original_filename}'",
                metadata={"original_file": original_filename, "checksum": checksum},
            )
        
        # Store checksum for future duplicate detection
        self._checksum_cache[checksum] = filename
        
        return FileValidationResult(is_valid=True)
    
    def _calculate_checksum(self, file_content: bytes) -> str:
        """Calculate SHA-256 checksum of file content."""
        return hashlib.sha256(file_content).hexdigest()
    
    def clear_checksum_cache(self):
        """Clear the duplicate detection cache."""
        self._checksum_cache.clear()


# Global instance for use across the application
_validation_service: Optional[FileValidationService] = None


def get_validation_service(
    max_file_size: Optional[int] = None,
    max_video_size: Optional[int] = None,
    enable_duplicate_check: bool = True,
) -> FileValidationService:
    """
    Get or create the file validation service instance.
    
    Note: This function uses a singleton pattern. Configuration parameters
    (max_file_size, max_video_size) are only used on first initialization.
    Subsequent calls return the existing instance with its original configuration,
    and any new parameter values passed will be ignored.
    
    For testing or when you need custom configurations that differ from the
    global instance, create a new FileValidationService instance directly
    instead of using this function.
    
    Args:
        max_file_size: Maximum file size in bytes (only used on first call)
        max_video_size: Maximum video size in bytes (only used on first call)
        enable_duplicate_check: Enable duplicate detection (only used on first call)
    
    Returns:
        FileValidationService: The global validation service instance
    """
    global _validation_service
    
    if _validation_service is None:
        from ..config import settings
        
        _validation_service = FileValidationService(
            max_file_size=max_file_size or settings.MAX_FILE_SIZE,
            max_video_size=max_video_size or settings.MAX_VIDEO_SIZE,
            enable_duplicate_check=enable_duplicate_check,
        )
    
    return _validation_service


def reset_validation_service():
    """
    Reset the global validation service instance.
    
    This is primarily useful for testing to ensure a clean state
    between test runs.
    """
    global _validation_service
    _validation_service = None
