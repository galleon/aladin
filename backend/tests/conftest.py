"""Test configuration and fixtures for backend API tests."""

import os
import tempfile

# Set test environment variables before importing app modules
os.environ.setdefault("UPLOAD_DIR", tempfile.mkdtemp())
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-purposes-min-32-chars")

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.database import Base, get_db
from app.main import app
from app.models import User, Agent, DataDomain, Conversation, AgentType
from app.services.auth import get_password_hash, get_current_user, get_current_active_user

# SQLite in-memory database for tests
SQLALCHEMY_DATABASE_URL = "sqlite:///file::memory:?cache=shared"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

# Enable foreign key support for SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(autouse=True)
def setup_database():
    """Create tables before each test and drop after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db():
    """Get a test database session."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_user(db):
    """Create a test user."""
    user = User(
        email="test@example.com",
        hashed_password=get_password_hash("testpassword123"),
        full_name="Test User",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def client(db, test_user):
    """Create a test client with auth overrides."""

    def override_get_db():
        try:
            yield db
        finally:
            pass

    async def override_get_current_user():
        return test_user

    async def override_get_current_active_user():
        return test_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_current_active_user] = override_get_current_active_user

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture
def unauthenticated_client(db):
    """Create a test client without auth overrides."""

    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture
def test_agent(db, test_user):
    """Create a test RAG agent."""
    agent = Agent(
        name="Test Agent",
        description="A test agent",
        agent_type=AgentType.RAG.value,
        llm_model="gpt-4",
        system_prompt="You are a helpful assistant.",
        temperature=0.7,
        owner_id=test_user.id,
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


@pytest.fixture
def test_data_domain(db, test_user):
    """Create a test data domain."""
    domain = DataDomain(
        name="Test Domain",
        description="A test domain",
        embedding_model="text-embedding-3-small",
        qdrant_collection="test_collection_abc",
        owner_id=test_user.id,
    )
    db.add(domain)
    db.commit()
    db.refresh(domain)
    return domain


@pytest.fixture
def test_conversation(db, test_user, test_agent):
    """Create a test conversation."""
    conversation = Conversation(
        conversation_metadata={"topic": "Test Conversation"},
        user_id=test_user.id,
        agent_id=test_agent.id,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation
