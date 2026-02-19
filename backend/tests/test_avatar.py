"""Tests for avatar agent endpoints."""

from unittest.mock import AsyncMock, patch, MagicMock

from app.models import Agent, AgentType


class FakeModel:
    def __init__(self, id):
        self.id = id


def _mock_model_service():
    mock = MagicMock()
    mock.get_llm_models = AsyncMock(return_value=[FakeModel("gpt-4")])
    return mock


@patch("app.services.model_service.model_service", new_callable=_mock_model_service)
def test_create_avatar_agent(mock_svc, client):
    """Avatar agents can be created via POST /agents/avatar."""
    response = client.post(
        "/api/agents/avatar",
        json={
            "name": "My Avatar",
            "description": "A live avatar agent",
            "llm_model": "gpt-4",
            "avatar_config": {
                "video_source_url": "rtmp://example.com/live/stream",
                "image_url": "https://example.com/avatar.jpg",
            },
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["agent_type"] == "avatar"
    assert data["avatar_config"]["video_source_url"] == "rtmp://example.com/live/stream"


@patch("app.services.model_service.model_service", new_callable=_mock_model_service)
def test_create_avatar_agent_without_config(mock_svc, client):
    """Avatar agent can be created without avatar_config (nullable)."""
    response = client.post(
        "/api/agents/avatar",
        json={
            "name": "Minimal Avatar",
            "llm_model": "gpt-4",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["agent_type"] == "avatar"
    assert data["avatar_config"] is None


def test_create_avatar_session_wrong_type(client, test_agent):
    """POST /agents/{id}/session returns 400 for non-avatar agents."""
    response = client.post(f"/api/agents/{test_agent.id}/session")
    assert response.status_code == 400
    assert "avatar" in response.json()["detail"].lower()


def test_create_avatar_session_not_found(client):
    """POST /agents/9999/session returns 404 for unknown agents."""
    response = client.post("/api/agents/9999/session")
    assert response.status_code == 404


@patch("app.services.model_service.model_service", new_callable=_mock_model_service)
def test_create_avatar_session_returns_token(mock_svc, client, db, test_user):
    """POST /agents/{id}/session returns a LiveKit token for avatar agents."""
    # Create an avatar agent directly
    agent = Agent(
        name="Avatar Test Agent",
        agent_type=AgentType.AVATAR.value,
        llm_model="gpt-4",
        system_prompt="You are an avatar.",
        temperature=0.7,
        max_tokens=512,
        owner_id=test_user.id,
        avatar_config={"video_source_url": "rtmp://example.com/live"},
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)

    # Patch livekit and arq dependencies
    mock_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test"
    mock_instance = MagicMock()
    mock_instance.with_identity.return_value = mock_instance
    mock_instance.with_name.return_value = mock_instance
    mock_instance.with_grants.return_value = mock_instance
    mock_instance.to_jwt.return_value = mock_token

    mock_access_token_cls = MagicMock(return_value=mock_instance)

    mock_pool = MagicMock()
    mock_pool.enqueue_job = AsyncMock()

    with (
        patch("app.routers.avatar_session.AccessToken", mock_access_token_cls),
        patch("app.routers.avatar_session.VideoGrants", MagicMock()),
        patch("app.routers.avatar_session._LIVEKIT_AVAILABLE", True),
        patch("app.routers.avatar_session.get_arq_pool", new=AsyncMock(return_value=mock_pool)),
    ):
        response = client.post(f"/api/agents/{agent.id}/session")

    assert response.status_code == 201
    data = response.json()
    assert "token" in data
    assert "room_name" in data
    assert "livekit_url" in data
    assert data["token"] == mock_token
    assert data["room_name"].startswith(f"avatar-{agent.id}-")
