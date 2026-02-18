"""Tests for voice API endpoints."""

import io
from unittest.mock import patch, MagicMock
import pytest


def test_transcribe_audio_success(client, test_user):
    """Test successful audio transcription."""
    # Create a mock audio file
    audio_data = b"fake audio data"
    
    # Mock the httpx client response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "Hello, this is a test transcription.",
        "language": "en"
    }
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        response = client.post(
            "/api/voice/transcribe",
            files={"file": ("test.mp3", io.BytesIO(audio_data), "audio/mpeg")},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hello, this is a test transcription."
        assert data["language"] == "en"


def test_transcribe_audio_with_language(client, test_user):
    """Test audio transcription with specified language."""
    audio_data = b"fake audio data"
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "Hola, esta es una prueba.",
        "language": "es"
    }
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        response = client.post(
            "/api/voice/transcribe",
            files={"file": ("test.mp3", io.BytesIO(audio_data), "audio/mpeg")},
            data={"language": "es"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hola, esta es una prueba."
        assert data["language"] == "es"


def test_transcribe_audio_empty_file(client, test_user):
    """Test transcription with empty audio file."""
    response = client.post(
        "/api/voice/transcribe",
        files={"file": ("test.mp3", io.BytesIO(b""), "audio/mpeg")},
    )
    
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_transcribe_audio_invalid_format(client, test_user):
    """Test transcription with invalid audio format."""
    response = client.post(
        "/api/voice/transcribe",
        files={"file": ("test.txt", io.BytesIO(b"text data"), "text/plain")},
    )
    
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_transcribe_audio_api_error(client, test_user):
    """Test transcription when API returns error."""
    audio_data = b"fake audio data"
    
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        response = client.post(
            "/api/voice/transcribe",
            files={"file": ("test.mp3", io.BytesIO(audio_data), "audio/mpeg")},
        )
        
        assert response.status_code == 502
        assert "api error" in response.json()["detail"].lower()


def test_text_to_speech_success(client, test_user):
    """Test successful text-to-speech conversion."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake audio content"
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        response = client.post(
            "/api/voice/synthesize",
            json={
                "text": "Hello, this is a test.",
                "voice": "alloy",
                "speed": 1.0,
            },
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert len(response.content) > 0


def test_text_to_speech_different_voice(client, test_user):
    """Test TTS with different voice."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake audio content"
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        response = client.post(
            "/api/voice/synthesize",
            json={
                "text": "Testing different voice.",
                "voice": "nova",
                "speed": 1.5,
            },
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"


def test_text_to_speech_empty_text(client, test_user):
    """Test TTS with empty text."""
    response = client.post(
        "/api/voice/synthesize",
        json={
            "text": "",
            "voice": "alloy",
            "speed": 1.0,
        },
    )
    
    assert response.status_code == 422  # Validation error


def test_text_to_speech_api_error(client, test_user):
    """Test TTS when API returns error."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        response = client.post(
            "/api/voice/synthesize",
            json={
                "text": "Test speech synthesis.",
                "voice": "alloy",
                "speed": 1.0,
            },
        )
        
        assert response.status_code == 502
        assert "api error" in response.json()["detail"].lower()


def test_transcribe_audio_unauthenticated(unauthenticated_client):
    """Test transcription without authentication."""
    audio_data = b"fake audio data"
    
    response = unauthenticated_client.post(
        "/api/voice/transcribe",
        files={"file": ("test.mp3", io.BytesIO(audio_data), "audio/mpeg")},
    )
    
    assert response.status_code == 401


def test_text_to_speech_unauthenticated(unauthenticated_client):
    """Test TTS without authentication."""
    response = unauthenticated_client.post(
        "/api/voice/synthesize",
        json={
            "text": "Test speech.",
            "voice": "alloy",
            "speed": 1.0,
        },
    )
    
    assert response.status_code == 401
