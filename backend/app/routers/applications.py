"""Applications router."""
import tempfile
import asyncio
import os
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Tenant, DeployedApplication, Task as TaskModel
from app.schemas import (
    ApplicationResponse,
    DeployApplicationRequest,
    DeployedApplicationResponse,
)
from app.services.applications import applications_service
from app.services.helm import helm_service
from app.services.cluster_api import cluster_api_service
from app.services.kubernetes import kubernetes_service
from app.tasks.application_tasks import deploy_application_task, uninstall_application_task
from app.routers.tasks import manager
from app.logger import logger
from kubernetes import client, config

router = APIRouter()


@router.get("/", response_model=list[ApplicationResponse])
async def get_applications():
    """Get all available applications."""
    try:
        applications = await applications_service.list_applications()
        return applications
    except Exception as e:
        logger.error("Failed to get applications", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch applications",
        )


@router.get("/{application_id}", response_model=ApplicationResponse)
async def get_application(application_id: str):
    """Get a specific application."""
    try:
        application = await applications_service.get_application(application_id)
        if not application:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Application not found",
            )
        return application
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get application", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch application",
        )


@router.post(
    "/tenants/{tenant_id}/deploy",
    response_model=DeployedApplicationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def deploy_application(
    tenant_id: int,
    deploy_data: DeployApplicationRequest,
    db: Session = Depends(get_db),
):
    """Deploy an application to a tenant."""
    try:
        # Get tenant
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found",
            )

        # Get application
        application = await applications_service.get_application(deploy_data.application_id)
        if not application:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Application not found",
            )

        # Check if already deployed
        existing = (
            db.query(DeployedApplication)
            .filter(
                DeployedApplication.tenant_id == tenant_id,
                DeployedApplication.application_id == deploy_data.application_id,
            )
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Application already deployed to this tenant",
            )

        # Generate release name
        release_name = deploy_data.release_name or f"{tenant.name}-{deploy_data.application_id}"

        # Create deployment record with "deploying" status
        deployment = DeployedApplication(
            tenant_id=tenant_id,
            application_id=deploy_data.application_id,
            release_name=release_name,
            namespace=tenant.namespace,
            status="deploying",
            helm_values=deploy_data.values,
        )
        db.add(deployment)
        db.commit()
        db.refresh(deployment)

        # Queue Celery task
        task = deploy_application_task.delay(
            tenant_id=tenant_id,
            application_id=deploy_data.application_id,
            release_name=release_name,
            values=deploy_data.values,
        )

        # Create task record
        task_record = TaskModel(
            task_id=task.id,
            task_type="deploy_application",
            status="pending",
            tenant_id=tenant_id,
            deployment_id=deployment.id,
        )
        db.add(task_record)
        db.commit()

        # Broadcast task creation via WebSocket
        await manager.broadcast({
            "type": "task_created",
            "task_id": task.id,
            "task_type": "deploy_application",
            "tenant_id": tenant_id,
            "deployment_id": deployment.id,
            "status": "pending",
        })

        return deployment
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Failed to deploy application", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deploy application",
        )


@router.get(
    "/tenants/{tenant_id}/deployments",
    response_model=list[DeployedApplicationResponse],
)
async def get_deployed_applications(tenant_id: int, db: Session = Depends(get_db)):
    """Get deployed applications for a tenant."""
    try:
        deployments = (
            db.query(DeployedApplication)
            .filter(DeployedApplication.tenant_id == tenant_id)
            .order_by(DeployedApplication.created_at.desc())
            .all()
        )
        return deployments
    except Exception as e:
        logger.error("Failed to get deployed applications", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch deployed applications",
        )


@router.delete("/tenants/{tenant_id}/deployments/{deployment_id}")
async def uninstall_application(
    tenant_id: int, deployment_id: int, db: Session = Depends(get_db)
):
    """Uninstall an application from a tenant (queues background task)."""
    try:
        deployment = (
            db.query(DeployedApplication)
            .filter(
                DeployedApplication.id == deployment_id,
                DeployedApplication.tenant_id == tenant_id,
            )
            .first()
        )
        if not deployment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deployment not found",
            )

        # Update deployment status
        deployment.status = "uninstalling"
        db.commit()

        # Queue Celery task
        task = uninstall_application_task.delay(deployment_id)

        # Create task record
        task_record = TaskModel(
            task_id=task.id,
            task_type="uninstall_application",
            status="pending",
            tenant_id=tenant_id,
            deployment_id=deployment_id,
        )
        db.add(task_record)
        db.commit()

        # Broadcast task creation via WebSocket
        await manager.broadcast({
            "type": "task_created",
            "task_id": task.id,
            "task_type": "uninstall_application",
            "tenant_id": tenant_id,
            "deployment_id": deployment_id,
            "status": "pending",
        })

        return {"message": "Application uninstallation queued", "task_id": task.id}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Failed to uninstall application", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to uninstall application",
        )

