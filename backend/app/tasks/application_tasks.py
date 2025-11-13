"""Celery tasks for application deployment operations."""
import asyncio
from celery import Task
from sqlalchemy.orm import Session
from app.celery_app import celery_app
from app.database import SessionLocal
from app.models import DeployedApplication, Tenant, Task as TaskModel
from app.services.applications import applications_service
from app.services.helm import helm_service
from app.services.cluster_api import cluster_api_service
from app.logger import logger
from datetime import datetime
import tempfile
import os
from kubernetes import client, config
from typing import Any
import redis
import json
from app.config import settings


def broadcast_task_update(task_id: str, status: str, progress: int, result: dict[str, Any] | None = None, error_message: str | None = None):
    """Broadcast task update via Redis pub/sub (for WebSocket)."""
    try:
        # Use synchronous Redis client for Celery tasks (Celery runs in sync context)
        r = redis.Redis.from_url(settings.redis_url, decode_responses=False)
        message = {
            "type": "task_update",
            "task_id": task_id,
            "status": status,
            "progress": progress,
        }
        if result:
            message["result"] = result
        if error_message:
            message["error_message"] = error_message
        r.publish("task_updates", json.dumps(message))
        r.close()
    except Exception as e:
        logger.warn("Failed to broadcast task update", task_id=task_id, error=str(e))


def update_task_status(
    db: Session,
    task_id: str,
    status: str,
    progress: int = 0,
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
):
    """Update task status in database and broadcast update."""
    try:
        task = db.query(TaskModel).filter(TaskModel.task_id == task_id).first()
        if task:
            task.status = status
            task.progress = progress
            if result:
                task.result = result
            if error_message:
                task.error_message = error_message
            if status in ("success", "failed"):
                task.completed_at = datetime.utcnow()
            db.commit()

            # Broadcast update
            broadcast_task_update(task_id, status, progress, result, error_message)
    except Exception as e:
        logger.error("Failed to update task status", task_id=task_id, error=str(e))
        db.rollback()


class DatabaseTask(Task):
    """Base task class that provides database session."""
    _db: Session | None = None

    @property
    def db(self) -> Session:
        """Get database session."""
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    def after_return(self, *args, **kwargs):
        """Close database session after task completion."""
        if self._db:
            self._db.close()
            self._db = None


@celery_app.task(bind=True, base=DatabaseTask, name="tasks.deploy_application")
def deploy_application_task(
    self, tenant_id: int, application_id: str, release_name: str | None = None, values: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Deploy an application to a tenant cluster asynchronously."""
    task_id = self.request.id
    db = self.db

    try:
        update_task_status(db, task_id, "running", progress=10)

        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            error_msg = "Tenant not found"
            update_task_status(db, task_id, "failed", error_message=error_msg)
            return {"error": error_msg}

        if tenant.status != "active":
            error_msg = f"Tenant is not active (status: {tenant.status})"
            update_task_status(db, task_id, "failed", error_message=error_msg)
            return {"error": error_msg}

        update_task_status(db, task_id, "running", progress=20)

        # Get application details
        application = asyncio.run(applications_service.get_application(application_id))
        if not application:
            error_msg = f"Application {application_id} not found"
            update_task_status(db, task_id, "failed", error_message=error_msg)
            return {"error": error_msg}

        update_task_status(db, task_id, "running", progress=30)

        # Get tenant cluster kubeconfig
        kubeconfig = asyncio.run(cluster_api_service.get_tenant_cluster_kubeconfig(tenant.namespace))
        if not kubeconfig:
            error_msg = "Could not retrieve tenant cluster kubeconfig"
            update_task_status(db, task_id, "failed", error_message=error_msg)
            return {"error": error_msg}

        kubeconfig_path = None
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
            f.write(kubeconfig)
            kubeconfig_path = f.name

        update_task_status(db, task_id, "running", progress=50)

        # Prepare Helm values
        helm_values = values or {}
        helm_values.setdefault("tenant", {}).update({
            "name": tenant.name,
            "namespace": tenant.namespace,
        })

        # Determine release name
        final_release_name = release_name or f"{tenant.name}-{application_id}"

        update_task_status(db, task_id, "running", progress=60)

        # Deploy using Helm
        try:
            asyncio.run(helm_service.install_release(
                final_release_name,
                application["chart_path"],
                tenant.namespace,
                helm_values,
                kubeconfig_path,
            ))
        except Exception as e:
            error_msg = f"Helm deployment failed: {str(e)}"
            logger.error("Helm deployment failed", error=str(e))
            update_task_status(db, task_id, "failed", error_message=error_msg)
            if kubeconfig_path and os.path.exists(kubeconfig_path):
                os.unlink(kubeconfig_path)
            return {"error": error_msg}

        update_task_status(db, task_id, "running", progress=80)

        # Create deployment record
        deployment = DeployedApplication(
            tenant_id=tenant_id,
            application_id=application_id,
            release_name=final_release_name,
            namespace=tenant.namespace,
            status="deployed",
            helm_values=helm_values,
        )
        db.add(deployment)
        db.commit()
        db.refresh(deployment)

        # Update task with deployment_id
        task = db.query(TaskModel).filter(TaskModel.task_id == task_id).first()
        if task:
            task.deployment_id = deployment.id
            task.tenant_id = tenant_id
            db.commit()

        if kubeconfig_path and os.path.exists(kubeconfig_path):
            os.unlink(kubeconfig_path)

        update_task_status(
            db,
            task_id,
            "success",
            progress=100,
            result={"deployment_id": deployment.id, "status": "deployed"},
        )

        return {"deployment_id": deployment.id, "status": "deployed"}

    except Exception as e:
        error_msg = f"Failed to deploy application: {str(e)}"
        logger.error("Failed to deploy application", error=str(e))
        update_task_status(db, task_id, "failed", error_message=error_msg)
        return {"error": error_msg}


@celery_app.task(bind=True, base=DatabaseTask, name="tasks.uninstall_application")
def uninstall_application_task(self, deployment_id: int) -> dict[str, Any]:
    """Uninstall an application from a tenant cluster asynchronously."""
    task_id = self.request.id
    db = self.db

    try:
        update_task_status(db, task_id, "running", progress=10)

        deployment = db.query(DeployedApplication).filter(DeployedApplication.id == deployment_id).first()
        if not deployment:
            error_msg = "Deployment not found"
            update_task_status(db, task_id, "failed", error_message=error_msg)
            return {"error": error_msg}

        tenant = db.query(Tenant).filter(Tenant.id == deployment.tenant_id).first()
        if not tenant:
            error_msg = "Tenant not found"
            update_task_status(db, task_id, "failed", error_message=error_msg)
            return {"error": error_msg}

        # Update task with deployment_id and tenant_id
        task = db.query(TaskModel).filter(TaskModel.task_id == task_id).first()
        if task:
            task.deployment_id = deployment_id
            task.tenant_id = deployment.tenant_id
            db.commit()

        update_task_status(db, task_id, "running", progress=30)

        # Get tenant cluster kubeconfig
        kubeconfig = asyncio.run(cluster_api_service.get_tenant_cluster_kubeconfig(tenant.namespace))
        kubeconfig_path = None
        if kubeconfig:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
                f.write(kubeconfig)
                kubeconfig_path = f.name

        update_task_status(db, task_id, "running", progress=50)

        # Uninstall Helm release
        try:
            asyncio.run(helm_service.uninstall_release(
                deployment.release_name,
                deployment.namespace,
                kubeconfig_path,
            ))
        except Exception as e:
            logger.warn("Failed to uninstall Helm release, continuing", error=str(e))

        update_task_status(db, task_id, "running", progress=80)

        # Delete deployment record
        db.delete(deployment)
        db.commit()

        if kubeconfig_path and os.path.exists(kubeconfig_path):
            os.unlink(kubeconfig_path)

        update_task_status(
            db,
            task_id,
            "success",
            progress=100,
            result={"deployment_id": deployment_id, "status": "uninstalled"},
        )

        return {"deployment_id": deployment_id, "status": "uninstalled"}

    except Exception as e:
        error_msg = f"Failed to uninstall application: {str(e)}"
        logger.error("Failed to uninstall application", error=str(e))
        update_task_status(db, task_id, "failed", error_message=error_msg)
        return {"error": error_msg}

