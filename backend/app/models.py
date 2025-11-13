"""SQLAlchemy database models."""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Tenant(Base):
    """Tenant model."""
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    namespace = Column(String(255), unique=True, nullable=False, index=True)
    status = Column(String(50), nullable=False, default="pending")
    helm_release_name = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    token_usage = relationship("TokenUsage", back_populates="tenant", cascade="all, delete-orphan")
    deployed_applications = relationship(
        "DeployedApplication", back_populates="tenant", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Tenant(id={self.id}, name='{self.name}', namespace='{self.namespace}', status='{self.status}')>"


class TokenUsage(Base):
    """Token usage tracking model."""
    __tablename__ = "token_usage"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    model = Column(String(255), nullable=True)
    endpoint = Column(String(255), nullable=True)

    # Relationships
    tenant = relationship("Tenant", back_populates="token_usage")

    __table_args__ = (
        Index("idx_token_usage_tenant", "tenant_id"),
        Index("idx_token_usage_timestamp", "timestamp"),
    )

    def __repr__(self):
        return f"<TokenUsage(id={self.id}, tenant_id={self.tenant_id}, total_tokens={self.total_tokens})>"


class DeployedApplication(Base):
    """Deployed application model."""
    __tablename__ = "deployed_applications"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    application_id = Column(String(255), nullable=False)
    release_name = Column(String(255), nullable=False)
    namespace = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="deploying")
    helm_values = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    tenant = relationship("Tenant", back_populates="deployed_applications")

    __table_args__ = (
        Index("idx_deployed_apps_tenant", "tenant_id"),
    )

    def __repr__(self):
        return (
            f"<DeployedApplication(id={self.id}, tenant_id={self.tenant_id}, "
            f"application_id='{self.application_id}', status='{self.status}')>"
        )


class Task(Base):
    """Task model for tracking background jobs."""
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(255), unique=True, nullable=False, index=True)  # Celery task ID
    task_type = Column(String(50), nullable=False, index=True)  # e.g., "create_tenant", "delete_tenant", "deploy_app"
    status = Column(String(50), nullable=False, default="pending", index=True)  # pending, running, success, failed
    result = Column(JSON, nullable=True)  # Task result or error details
    error_message = Column(String(1000), nullable=True)
    progress = Column(Integer, default=0)  # 0-100
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Foreign keys for related resources
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=True, index=True)
    deployment_id = Column(Integer, ForeignKey("deployed_applications.id", ondelete="CASCADE"), nullable=True, index=True)

    __table_args__ = (
        Index("idx_tasks_status", "status"),
        Index("idx_tasks_type", "task_type"),
        Index("idx_tasks_created", "created_at"),
    )

    def __repr__(self):
        return f"<Task(id={self.id}, task_id='{self.task_id}', type='{self.task_type}', status='{self.status}')>"

