"""Tests for authentication endpoints."""


def test_register_success(unauthenticated_client):
    response = unauthenticated_client.post(
        "/api/auth/register",
        json={
            "email": "newuser@example.com",
            "password": "securepassword123",
            "full_name": "New User",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "newuser@example.com"
    assert data["full_name"] == "New User"
    assert data["is_active"] is True
    assert "id" in data


def test_register_duplicate_email(unauthenticated_client):
    # Register first user
    unauthenticated_client.post(
        "/api/auth/register",
        json={
            "email": "duplicate@example.com",
            "password": "securepassword123",
        },
    )
    # Try registering with same email
    response = unauthenticated_client.post(
        "/api/auth/register",
        json={
            "email": "duplicate@example.com",
            "password": "anotherpassword123",
        },
    )
    assert response.status_code == 400
    assert "already registered" in response.json()["detail"].lower()


def test_login_success(unauthenticated_client):
    # Register user first
    unauthenticated_client.post(
        "/api/auth/register",
        json={
            "email": "login@example.com",
            "password": "securepassword123",
        },
    )
    # Login
    response = unauthenticated_client.post(
        "/api/auth/login",
        data={
            "username": "login@example.com",
            "password": "securepassword123",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password(unauthenticated_client):
    # Register user first
    unauthenticated_client.post(
        "/api/auth/register",
        json={
            "email": "wrongpw@example.com",
            "password": "securepassword123",
        },
    )
    # Login with wrong password
    response = unauthenticated_client.post(
        "/api/auth/login",
        data={
            "username": "wrongpw@example.com",
            "password": "wrongpassword",
        },
    )
    assert response.status_code == 401


def test_login_nonexistent_user(unauthenticated_client):
    response = unauthenticated_client.post(
        "/api/auth/login",
        data={
            "username": "nonexistent@example.com",
            "password": "somepassword",
        },
    )
    assert response.status_code == 401


def test_get_me_authenticated(client):
    response = client.get("/api/auth/me")
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["full_name"] == "Test User"


def test_get_me_unauthenticated(unauthenticated_client):
    response = unauthenticated_client.get("/api/auth/me")
    assert response.status_code == 401
