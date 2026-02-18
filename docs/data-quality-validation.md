# Data Quality Validation

This document describes the data quality validation system implemented in Aladin to ensure reliable and secure file uploads.

## Overview

The file validation system provides multiple layers of protection:

1. **File Type Validation** - Ensures only supported formats are uploaded
2. **Content Verification** - Validates file content matches the claimed type
3. **Encoding Validation** - Ensures text files are properly encoded
4. **Size Constraints** - Prevents excessively large or small files
5. **Duplicate Detection** - Identifies duplicate uploads
6. **Integrity Checks** - Detects corrupted or malformed files

## Implementation

### File Validation Service

The validation logic is implemented in `backend/app/services/file_validation.py` and provides:

- **FileValidationService**: Main validation class with configurable limits
- **FileValidationResult**: Structured result with error details
- **ValidationErrorCode**: Enumeration of possible validation failures

### Integration Points

Validation is integrated at three upload endpoints:

1. **Data Domain Upload**: `POST /data-domains/{domain_id}/documents`
2. **Ingestion File Upload**: `POST /ingestion/file`
3. **Markdown Extraction**: `POST /ingestion/extract`

## Validation Rules

### Supported File Types

| Extension | Format | MIME Type | Encoding Check |
|-----------|--------|-----------|----------------|
| `.pdf` | PDF Document | `application/pdf` | No |
| `.docx` | Word Document | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | No |
| `.doc` | Word 97-2003 | `application/msword` | No |
| `.pptx` | PowerPoint | `application/vnd.openxmlformats-officedocument.presentationml.presentation` | No |
| `.ppt` | PowerPoint 97-2003 | `application/vnd.ms-powerpoint` | No |
| `.txt` | Plain Text | `text/plain` | Yes |
| `.md` | Markdown | `text/markdown` | Yes |
| `.html` | HTML | `text/html` | Yes |
| `.csv` | CSV | `text/csv` | Yes |
| `.json` | JSON | `application/json` | Yes |
| `.mp4` | MP4 Video | `video/mp4` | No |

### Magic Bytes (File Signatures)

The system validates files using magic bytes to prevent file extension spoofing:

| Format | Magic Bytes | Description |
|--------|-------------|-------------|
| PDF | `%PDF` | PDF header |
| ZIP/DOCX/PPTX | `PK\x03\x04` | ZIP archive header (Office files are ZIP-based) |
| DOC | `\xd0\xcf\x11\xe0` | Microsoft Compound File Binary |
| MP4 | `ftyp` at offset 4-8 | MP4 file type box |

### File Size Limits

- **Minimum Size**: 10 bytes
  - Prevents completely empty files
  - Catches truncated or corrupted files
  
- **Maximum Size (Documents)**: 50 MB (default)
  - Configurable via `MAX_FILE_SIZE` environment variable
  - Applied to: PDF, DOCX, TXT, MD, HTML, CSV, JSON
  
- **Maximum Size (Videos)**: 500 MB (default)
  - Configurable via `MAX_VIDEO_SIZE` environment variable
  - Applied to: MP4 files

### Encoding Validation

Text-based files must be valid UTF-8:

- **Validated formats**: TXT, MD, HTML, CSV, JSON
- **Error**: Returns `INVALID_ENCODING` if UTF-8 decoding fails
- **Binary formats**: PDF, DOCX, DOC, PPTX, MP4 are not checked for encoding

### Duplicate Detection

Files are checked for duplicates using SHA-256 checksums:

- **Scope**: Per-session duplicate detection
- **Algorithm**: SHA-256 hash of file content
- **Cache**: In-memory cache cleared on service restart
- **Configurable**: Can be disabled via `enable_duplicate_check=False`

## Error Handling

### Validation Error Codes

All validation failures return HTTP 400 with one of these error codes:

```python
EMPTY_FILE          # File contains no data
FILE_TOO_SMALL      # File smaller than 10 bytes
FILE_TOO_LARGE      # File exceeds size limit
INVALID_FILE_TYPE   # Extension not supported or missing
INVALID_MIME_TYPE   # Content doesn't match extension
INVALID_ENCODING    # Text file is not UTF-8
DUPLICATE_FILE      # Identical file already uploaded
```

### Example Error Response

```json
{
  "detail": "File 'document.txt' is not valid UTF-8 encoded. Text files must be UTF-8 encoded."
}
```

## Security Considerations

The validation system provides several security benefits:

1. **File Type Spoofing Prevention**
   - Magic byte verification prevents malicious files with fake extensions
   - Example: A PDF renamed to .txt will be rejected

2. **Resource Protection**
   - File size limits prevent DoS attacks via large file uploads
   - Minimum size prevents empty file spam

3. **Data Integrity**
   - UTF-8 validation prevents encoding corruption
   - Checksum verification ensures file integrity

4. **Duplicate Prevention**
   - Reduces storage waste from duplicate uploads
   - Prevents accidental re-processing of same content

## Usage Examples

### Python API

```python
from app.services.file_validation import get_validation_service

# Get validation service instance
validation_service = get_validation_service(
    max_file_size=50 * 1024 * 1024,  # 50MB
    max_video_size=500 * 1024 * 1024,  # 500MB
    enable_duplicate_check=True,
)

# Validate a file
file_content = await file.read()
result = validation_service.validate_file(
    file_content=file_content,
    filename="document.pdf",
    expected_extension=".pdf",
)

if result.is_valid:
    # File is valid
    checksum = result.metadata["checksum"]
    print(f"Valid file with checksum: {checksum}")
else:
    # Validation failed
    print(f"Error: {result.error_message}")
    print(f"Code: {result.error_code}")
```

### HTTP API

```bash
# Upload a valid file
curl -X POST "http://localhost:3000/data-domains/1/documents" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@document.pdf"

# Response on success (HTTP 200)
{
  "id": 123,
  "filename": "abc123.pdf",
  "original_filename": "document.pdf",
  "file_type": "pdf",
  "file_size": 1024000,
  "status": "pending"
}

# Response on validation failure (HTTP 400)
{
  "detail": "File 'document.pdf' is too large (60000000 bytes). Maximum size: 52428800 bytes"
}
```

## Configuration

### Environment Variables

```bash
# File size limits
MAX_FILE_SIZE=52428800      # 50MB for documents
MAX_VIDEO_SIZE=524288000    # 500MB for videos

# Allowed file extensions
ALLOWED_EXTENSIONS=pdf,txt,md,docx,doc,csv,json,mp4
```

### Service Configuration

```python
from app.services.file_validation import FileValidationService

# Custom configuration
service = FileValidationService(
    max_file_size=100 * 1024 * 1024,  # 100MB
    max_video_size=1024 * 1024 * 1024,  # 1GB
    min_file_size=1,  # 1 byte minimum
    enable_duplicate_check=False,  # Disable duplicate detection
)
```

## Testing

The validation system includes comprehensive tests in `backend/tests/test_file_validation.py`:

```bash
# Run validation tests
cd backend
pytest tests/test_file_validation.py -v

# Run specific test
pytest tests/test_file_validation.py::TestFileValidation::test_valid_pdf_file -v
```

### Test Coverage

- Empty file detection
- File size validation (min/max)
- MIME type verification
- UTF-8 encoding validation
- Duplicate detection
- Magic bytes validation
- Checksum calculation
- Error message formatting

## Future Enhancements

Potential improvements to the validation system:

1. **Malware Scanning**
   - Integration with ClamAV or similar
   - Real-time threat detection

2. **Advanced Content Analysis**
   - Deep content inspection
   - Metadata extraction and validation

3. **Rate Limiting**
   - Per-user upload limits
   - Throttling for large files

4. **Persistent Duplicate Detection**
   - Database-backed checksum storage
   - Global duplicate detection across all users

5. **File Repair**
   - Automatic encoding conversion
   - Corrupted file recovery

## References

- [Magic Numbers (File Signatures)](https://en.wikipedia.org/wiki/List_of_file_signatures)
- [MIME Types](https://www.iana.org/assignments/media-types/media-types.xhtml)
- [UTF-8 Encoding](https://en.wikipedia.org/wiki/UTF-8)
- [SHA-256 Hashing](https://en.wikipedia.org/wiki/SHA-2)
