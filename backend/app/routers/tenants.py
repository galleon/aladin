"""Tenants router."""
import tempfile
import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.database import get_db
from app.models import Tenant, Task as TaskModel
from app.schemas import TenantCreate, TenantResponse, TenantStatusResponse
from app.services.cluster_api import cluster_api_service
from app.services.helm import helm_service
from app.services.kubernetes import kubernetes_service
from app.tasks.tenant_tasks import create_tenant_task, delete_tenant_task
from app.routers.tasks import manager
from app.logger import logger
from kubernetes import client, config
import os
from datetime import datetime
import urllib3

router = APIRouter()


@router.get("/", response_model=list[TenantResponse])
async def get_tenants(db: Session = Depends(get_db)):
    """Get all tenants."""
    try:
        tenants = db.query(Tenant).order_by(Tenant.created_at.desc()).all()
        return tenants
    except Exception as e:
        logger.error("Failed to get tenants", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch tenants",
        )


@router.get("/{tenant_id}", response_model=TenantResponse)
async def get_tenant(tenant_id: int, db: Session = Depends(get_db)):
    """Get tenant by ID."""
    try:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found",
            )
        return tenant
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get tenant", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch tenant",
        )


@router.post("/", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
async def create_tenant(tenant_data: TenantCreate, db: Session = Depends(get_db)):
    """Create a new tenant (queues background task)."""
    try:
        # Check if tenant already exists
        existing = (
            db.query(Tenant)
            .filter(or_(Tenant.name == tenant_data.name, Tenant.namespace == tenant_data.namespace))
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Tenant already exists",
            )

        # Create tenant record with "creating" status
        tenant = Tenant(
            name=tenant_data.name,
            namespace=tenant_data.namespace,
            status="creating",
            helm_release_name=f"{tenant_data.name}-llm",
        )
        db.add(tenant)
        db.commit()
        db.refresh(tenant)

        # Queue Celery task with tenant ID
        task = create_tenant_task.delay(tenant.id)

        # Create task record
        task_record = TaskModel(
            task_id=task.id,
            task_type="create_tenant",
            status="pending",
            tenant_id=tenant.id,
        )
        db.add(task_record)
        db.commit()

        # Broadcast task creation via WebSocket
        await manager.broadcast({
            "type": "task_created",
            "task_id": task.id,
            "task_type": "create_tenant",
            "tenant_id": tenant.id,
            "status": "pending",
        })

        return tenant
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Failed to create tenant", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create tenant",
        )


@router.delete("/{tenant_id}")
async def delete_tenant(tenant_id: int, db: Session = Depends(get_db)):
    """Delete a tenant (queues background task)."""
    try:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found",
            )

        # Update tenant status
        tenant.status = "deleting"
        db.commit()

        # Queue Celery task
        task = delete_tenant_task.delay(tenant_id)

        # Create task record
        task_record = TaskModel(
            task_id=task.id,
            task_type="delete_tenant",
            status="pending",
            tenant_id=tenant_id,
        )
        db.add(task_record)
        db.commit()

        # Broadcast task creation via WebSocket
        await manager.broadcast({
            "type": "task_created",
            "task_id": task.id,
            "task_type": "delete_tenant",
            "tenant_id": tenant_id,
            "status": "pending",
        })

        return {"message": "Tenant deletion queued", "task_id": task.id}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Failed to delete tenant", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete tenant",
        )


@router.get("/{tenant_id}/status", response_model=TenantStatusResponse)
async def get_tenant_status(tenant_id: int, db: Session = Depends(get_db)):
    """Get tenant status."""
    try:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found",
            )

        namespace_status = "Unknown"
        helm_status = None
        deployment_status = None
        kubeconfig_path = None

        try:
            kubeconfig = await cluster_api_service.get_tenant_cluster_kubeconfig(tenant.namespace)
            if kubeconfig:
                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
                    f.write(kubeconfig)
                    kubeconfig_path = f.name

                kc = config.new_client_from_config(kubeconfig_path)
                timeout = urllib3.Timeout(connect=5.0, read=10.0)
                kc.rest_client.pool_manager.connection_pool_kw["timeout"] = timeout
                tenant_k8s_api = client.CoreV1Api(kc)
                tenant_apps_api = client.AppsV1Api(kc)

                try:
                    ns = await asyncio.wait_for(
                        asyncio.to_thread(
                            tenant_k8s_api.read_namespace,
                            tenant.namespace,
                            _request_timeout=(3, 8),
                        ),
                        timeout=10.0,
                    )
                    namespace_status = ns.status.phase if ns.status else "Unknown"
                except asyncio.TimeoutError:
                    logger.warn("Namespace status check timed out", tenant_id=tenant_id)
                    namespace_status = "Timeout"
                except Exception as e:
                    if "404" in str(e):
                        namespace_status = "NotFound"
                    else:
                        namespace_status = "Unknown"

                if tenant.helm_release_name:
                    # Wrap Helm status check in asyncio timeout
                    try:
                        helm_status = await asyncio.wait_for(
                            helm_service.get_release_status(
                                tenant.helm_release_name,
                                tenant.namespace,
                                kubeconfig_path,
                            ),
                            timeout=12.0,  # Slightly longer than Helm timeout
                        )
                    except asyncio.TimeoutError:
                        logger.warn("Helm status check timed out", tenant_id=tenant_id)
                        helm_status = None

                    # Wrap deployment check in asyncio timeout
                    try:
                        deployment = await asyncio.wait_for(
                            asyncio.to_thread(
                                tenant_apps_api.read_namespaced_deployment,
                                f"{tenant.helm_release_name}-deployment",
                                tenant.namespace,
                                _request_timeout=(3, 8),
                            ),
                            timeout=10.0,
                        )
                        deployment_status = {
                            "replicas": deployment.spec.replicas if deployment.spec else 0,
                            "ready_replicas": deployment.status.ready_replicas if deployment.status else 0,
                            "available_replicas": deployment.status.available_replicas if deployment.status else 0,
                            "conditions": [
                                {
                                    "type": c.type,
                                    "status": c.status,
                                    "message": getattr(c, "message", None),
                                }
                                for c in (deployment.status.conditions if deployment.status else [])
                            ],
                        }
                    except (asyncio.TimeoutError, Exception) as e:
                        if isinstance(e, asyncio.TimeoutError):
                            logger.warn("Deployment status check timed out", tenant_id=tenant_id)
                        pass

        except Exception as e:
            logger.warn("Could not retrieve tenant cluster kubeconfig", error=str(e))
        finally:
            if kubeconfig_path and os.path.exists(kubeconfig_path):
                try:
                    os.unlink(kubeconfig_path)
                except Exception:
                    pass

        cluster_status = await cluster_api_service.get_cluster_status(tenant.namespace)

        return {
            "tenant": tenant,
            "cluster": cluster_status,
            "namespace": {"status": namespace_status},
            "deployment": deployment_status,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get tenant status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get tenant status",
        )

