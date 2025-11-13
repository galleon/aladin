"""Celery tasks for tenant operations."""
import asyncio
from celery import Task
from sqlalchemy.orm import Session
from app.celery_app import celery_app
from app.database import SessionLocal
from app.models import Tenant, Task as TaskModel
from app.services.cluster_api import cluster_api_service
from app.services.helm import helm_service
from app.services.kubernetes import kubernetes_service
from app.logger import logger
from datetime import datetime
import tempfile
import os
from kubernetes import client, config
from typing import Any
import redis
import json
from app.config import settings


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


@celery_app.task(bind=True, base=DatabaseTask, name="tasks.create_tenant")
def create_tenant_task(self, tenant_id: int) -> dict[str, Any]:
    """Create a tenant cluster asynchronously."""
    task_id = self.request.id
    db = self.db

    try:
        update_task_status(db, task_id, "running", progress=10)

        # Get tenant record
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            error_msg = "Tenant not found"
            update_task_status(db, task_id, "failed", error_message=error_msg)
            return {"error": error_msg}

        update_task_status(db, task_id, "running", progress=20)

        # Create Cluster API resources
        try:
            asyncio.run(cluster_api_service.create_tenant_cluster(tenant.namespace))
            logger.info("Cluster API resources created", tenant=tenant.name)
        except Exception as e:
            error_msg = f"Failed to create tenant cluster: {str(e)}"
            logger.error("Failed to create tenant cluster", tenant=tenant.name, error=str(e))
            tenant.status = "failed"
            db.commit()
            update_task_status(db, task_id, "failed", error_message=error_msg)
            return {"error": error_msg}

        update_task_status(db, task_id, "running", progress=60)

        # Wait for cluster to be ready
        is_ready = asyncio.run(cluster_api_service.wait_for_cluster_ready(
            tenant.namespace, 300
        ))

        if not is_ready:
            tenant.status = "failed"
            db.commit()
            error_msg = "Cluster did not become ready within timeout"
            update_task_status(db, task_id, "failed", error_message=error_msg)
            return {"error": error_msg}

        update_task_status(db, task_id, "running", progress=90)

        tenant.status = "active"
        db.commit()
        logger.info("Tenant cluster is ready", tenant=tenant.name)

        update_task_status(
            db,
            task_id,
            "success",
            progress=100,
            result={"tenant_id": tenant.id, "status": "active"},
        )

        return {"tenant_id": tenant.id, "status": "active"}

    except Exception as e:
        error_msg = f"Failed to create tenant: {str(e)}"
        logger.error("Failed to create tenant", error=str(e))
        update_task_status(db, task_id, "failed", error_message=error_msg)
        return {"error": error_msg}


@celery_app.task(bind=True, base=DatabaseTask, name="tasks.delete_tenant")
def delete_tenant_task(self, tenant_id: int) -> dict[str, Any]:
    """Delete a tenant cluster asynchronously."""
    task_id = self.request.id
    db = self.db

    try:
        update_task_status(db, task_id, "running", progress=10)

        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            error_msg = "Tenant not found"
            update_task_status(db, task_id, "failed", error_message=error_msg)
            return {"error": error_msg}

        # Update task with tenant_id
        task = db.query(TaskModel).filter(TaskModel.task_id == task_id).first()
        if task:
            task.tenant_id = tenant.id
            db.commit()

        update_task_status(db, task_id, "running", progress=30)

        # Try to get tenant cluster kubeconfig
        kubeconfig_path = None
        try:
            kubeconfig = asyncio.run(cluster_api_service.get_tenant_cluster_kubeconfig(
                tenant.namespace
            ))
            if kubeconfig:
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".yaml"
                ) as f:
                    f.write(kubeconfig)
                    kubeconfig_path = f.name
        except Exception as e:
            logger.warn("Could not retrieve tenant cluster kubeconfig", error=str(e))

        update_task_status(db, task_id, "running", progress=50)

        # Uninstall Helm release
        if tenant.helm_release_name:
            try:
                asyncio.run(helm_service.uninstall_release(
                    tenant.helm_release_name,
                    tenant.namespace,
                    kubeconfig_path,
                ))
            except Exception as e:
                logger.warn("Failed to uninstall Helm release, continuing", error=str(e))

        update_task_status(db, task_id, "running", progress=70)

        # Delete namespace from tenant cluster
        if kubeconfig_path:
            try:
                kc = config.new_client_from_config(kubeconfig_path)
                tenant_k8s_api = client.CoreV1Api(kc)
                try:
                    tenant_k8s_api.delete_namespace(tenant.namespace, _request_timeout=(5, 15))
                    logger.info("Deleted namespace from tenant cluster", namespace=tenant.namespace)
                except Exception as e:
                    error_str = str(e)
                    if "404" not in error_str and "timeout" not in error_str.lower():
                        logger.warn("Failed to delete namespace from tenant cluster", error=error_str)
            except Exception as e:
                logger.warn("Failed to delete namespace", error=str(e))
            finally:
                if kubeconfig_path and os.path.exists(kubeconfig_path):
                    try:
                        os.unlink(kubeconfig_path)
                    except Exception:
                        pass

        update_task_status(db, task_id, "running", progress=85)

        # Delete Cluster API cluster
        try:
            asyncio.run(cluster_api_service.delete_tenant_cluster(tenant.namespace))
        except Exception as e:
            logger.warn("Failed to delete Cluster API cluster", error=str(e))

        update_task_status(db, task_id, "running", progress=95)

        # Delete tenant record
        try:
            db.delete(tenant)
            db.commit()
            logger.info("Tenant record deleted", tenant_id=tenant_id)
        except Exception as e:
            logger.error("Failed to delete tenant record", tenant_id=tenant_id, error=str(e))
            db.rollback()
            # Continue anyway - cluster is already deleted

        update_task_status(
            db,
            task_id,
            "success",
            progress=100,
            result={"tenant_id": tenant_id, "status": "deleted"},
        )

        return {"tenant_id": tenant_id, "status": "deleted"}

    except Exception as e:
        error_msg = f"Failed to delete tenant: {str(e)}"
        logger.error("Failed to delete tenant", error=str(e))
        update_task_status(db, task_id, "failed", error_message=error_msg)
        return {"error": error_msg}

