"""Celery tasks module."""
from app.tasks.tenant_tasks import (
    create_tenant_task,
    delete_tenant_task,
)
from app.tasks.application_tasks import (
    deploy_application_task,
    uninstall_application_task,
)

__all__ = [
    "create_tenant_task",
    "delete_tenant_task",
    "deploy_application_task",
    "uninstall_application_task",
]

