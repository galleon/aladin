"""Pydantic schemas for request/response validation."""
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, Any


# Tenant schemas
class TenantBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    namespace: str = Field(..., min_length=1, max_length=255)
    helm_values: Optional[dict[str, Any]] = None

    @field_validator("namespace")
    @classmethod
    def validate_namespace(cls, v: str) -> str:
        import re
        if not re.match(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$", v):
            raise ValueError("Namespace must match Kubernetes naming requirements")
        return v


class TenantCreate(TenantBase):
    pass


class TenantResponse(TenantBase):
    id: int
    status: str
    helm_release_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Application schemas
class ApplicationResponse(BaseModel):
    id: str
    name: str
    description: str
    version: str
    chart_path: str
    icon: Optional[str] = None
    category: Optional[str] = None

    class Config:
        from_attributes = True


class DeployApplicationRequest(BaseModel):
    application_id: str = Field(..., min_length=1)
    release_name: Optional[str] = None
    values: Optional[dict[str, Any]] = None


class DeployedApplicationResponse(BaseModel):
    id: int
    tenant_id: int
    application_id: str
    release_name: str
    namespace: str
    status: str
    helm_values: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Token usage schemas
class TokenUsageCreate(BaseModel):
    tenant_id: int
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    model: Optional[str] = None
    endpoint: Optional[str] = None


class TokenUsageResponse(BaseModel):
    id: int
    tenant_id: Optional[int] = None
    timestamp: datetime
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: Optional[str] = None
    endpoint: Optional[str] = None

    class Config:
        from_attributes = True


# Status schemas
class TenantStatusResponse(BaseModel):
    tenant: TenantResponse
    cluster: Optional[dict[str, Any]] = None
    namespace: Optional[dict[str, Any]] = None
    deployment: Optional[dict[str, Any]] = None


# Task schemas
class TaskResponse(BaseModel):
    id: int
    task_id: str
    task_type: str
    status: str
    result: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    progress: int
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    tenant_id: Optional[int] = None
    deployment_id: Optional[int] = None

    class Config:
        from_attributes = True


class TaskStatusUpdate(BaseModel):
    task_id: str
    status: str
    progress: int
    result: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
