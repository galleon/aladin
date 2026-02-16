"""Tests for conversation CRUD endpoints."""


def test_create_conversation(client, test_agent):
    response = client.post(
        "/api/conversations/",
        json={"agent_id": test_agent.id},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["agent_id"] == test_agent.id
    assert "id" in data


def test_list_conversations(client, test_conversation):
    response = client.get("/api/conversations/")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1


def test_get_conversation(client, test_conversation):
    response = client.get(f"/api/conversations/{test_conversation.id}")
    assert response.status_code == 200
    data = response.json()
    assert int(data["id"]) == test_conversation.id


def test_get_conversation_not_found(client):
    response = client.get("/api/conversations/9999")
    assert response.status_code == 404


def test_delete_conversation(client, test_conversation):
    response = client.delete(f"/api/conversations/{test_conversation.id}")
    assert response.status_code == 204

    # Verify deleted
    response = client.get(f"/api/conversations/{test_conversation.id}")
    assert response.status_code == 404
