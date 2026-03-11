"""Tests for the clips endpoint."""

from unittest.mock import MagicMock, patch


def test_get_clip_no_minio(client):
    """Returns 503 when MINIO_ENDPOINT is not configured."""
    # Default settings have MINIO_ENDPOINT=None, so _get_minio_client raises 503
    response = client.get("/api/clips/my_collection/some-point-id")
    assert response.status_code == 503


def test_get_clip_point_not_found(client):
    """Returns 404 when Qdrant has no matching point."""
    mock_qdrant = MagicMock()
    mock_qdrant.retrieve.return_value = []

    with (
        patch("app.routers.clips._get_qdrant_client", return_value=mock_qdrant),
        patch("app.routers.clips._get_minio_client", return_value=MagicMock()),
    ):
        response = client.get("/api/clips/my_collection/missing-id")

    assert response.status_code == 404


def test_get_clip_no_clip_key(client):
    """Returns 404 when Qdrant point exists but has no clip_key in payload."""
    mock_point = MagicMock()
    mock_point.payload = {"source_file": "video.mp4", "t_start": 0.0}

    mock_qdrant = MagicMock()
    mock_qdrant.retrieve.return_value = [mock_point]

    with (
        patch("app.routers.clips._get_qdrant_client", return_value=mock_qdrant),
        patch("app.routers.clips._get_minio_client", return_value=MagicMock()),
    ):
        response = client.get("/api/clips/my_collection/point-no-clip")

    assert response.status_code == 404


def test_get_clip_redirect(client):
    """Returns 307 redirect to presigned URL when clip_key is present."""
    presigned_url = "http://minio:9000/clips/vid/0.mp4?X-Amz-Signature=abc"

    mock_point = MagicMock()
    mock_point.payload = {"clip_key": "vid/0.mp4"}

    mock_qdrant = MagicMock()
    mock_qdrant.retrieve.return_value = [mock_point]

    mock_minio = MagicMock()
    mock_minio.presigned_get_object.return_value = presigned_url

    with (
        patch("app.routers.clips._get_qdrant_client", return_value=mock_qdrant),
        patch("app.routers.clips._get_minio_client", return_value=mock_minio),
    ):
        response = client.get(
            "/api/clips/my_collection/existing-point-id",
            follow_redirects=False,
        )

    assert response.status_code == 307
    assert response.headers["location"] == presigned_url
