"""Tests for data domain CRUD endpoints."""

from unittest.mock import patch, MagicMock


QDRANT_SERVICE_PATH = "app.routers.data_domains.qdrant_service"


@patch(f"{QDRANT_SERVICE_PATH}.create_collection")
def test_create_data_domain(mock_create, client):
    response = client.post(
        "/api/data-domains/",
        json={
            "name": "Test Domain",
            "description": "A test domain",
            "embedding_model": "text-embedding-3-small",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Domain"
    assert data["embedding_model"] == "text-embedding-3-small"
    assert "qdrant_collection" in data
    mock_create.assert_called_once()


def test_list_data_domains(client, test_data_domain):
    response = client.get("/api/data-domains/")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert any(d["name"] == "Test Domain" for d in data)


def test_get_data_domain(client, test_data_domain):
    response = client.get(f"/api/data-domains/{test_data_domain.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Domain"
    assert data["id"] == test_data_domain.id


def test_get_data_domain_not_found(client):
    response = client.get("/api/data-domains/9999")
    assert response.status_code == 404


@patch(f"{QDRANT_SERVICE_PATH}.delete_collection", return_value=True)
def test_delete_data_domain(mock_delete, client, test_data_domain):
    response = client.delete(f"/api/data-domains/{test_data_domain.id}")
    assert response.status_code == 204
    mock_delete.assert_called_once()
