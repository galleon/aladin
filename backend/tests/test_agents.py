"""Tests for agent CRUD endpoints."""

from unittest.mock import AsyncMock, patch, MagicMock


class FakeModel:
    def __init__(self, id):
        self.id = id


def _mock_model_service():
    """Create a mock model_service with get_llm_models returning gpt-4."""
    mock = MagicMock()
    mock.get_llm_models = AsyncMock(return_value=[FakeModel("gpt-4")])
    return mock


@patch("app.services.model_service.model_service", new_callable=_mock_model_service)
def test_create_rag_agent(mock_svc, client, test_data_domain):
    response = client.post(
        "/api/agents/",
        json={
            "name": "New RAG Agent",
            "description": "Test agent",
            "llm_model": "gpt-4",
            "system_prompt": "You are helpful.",
            "data_domain_ids": [test_data_domain.id],
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "New RAG Agent"
    assert data["agent_type"] == "rag"


@patch("app.services.model_service.model_service", new_callable=_mock_model_service)
def test_list_agents(mock_svc, client, test_agent):
    response = client.get("/api/agents/")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert any(a["name"] == "Test Agent" for a in data)


@patch("app.services.model_service.model_service", new_callable=_mock_model_service)
def test_get_agent(mock_svc, client, test_agent):
    response = client.get(f"/api/agents/{test_agent.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Agent"
    assert data["id"] == test_agent.id


@patch("app.services.model_service.model_service", new_callable=_mock_model_service)
def test_get_agent_not_found(mock_svc, client):
    response = client.get("/api/agents/9999")
    assert response.status_code == 404


@patch("app.services.model_service.model_service", new_callable=_mock_model_service)
def test_update_agent(mock_svc, client, test_agent):
    response = client.put(
        f"/api/agents/{test_agent.id}",
        json={"name": "Updated Agent", "description": "Updated description"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Agent"
    assert data["description"] == "Updated description"


def test_delete_agent(client, test_agent):
    response = client.delete(f"/api/agents/{test_agent.id}")
    assert response.status_code == 204
