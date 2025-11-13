"""Monitoring router."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Tenant
from app.services import kubernetes_service
from app.logger import logger
from datetime import datetime

router = APIRouter()


@router.get("/tenant/{tenant_id}")
async def get_tenant_monitoring(tenant_id: int, db: Session = Depends(get_db)):
    """Get monitoring data for a tenant."""
    try:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found",
            )

        # Get pod metrics
        pod_metrics = await kubernetes_service.get_pod_metrics(tenant.namespace)

        # Get deployment status
        deployment_status = None
        if tenant.helm_release_name:
            deployment_status = await kubernetes_service.get_deployment_status(
                tenant.namespace,
                f"{tenant.helm_release_name}-deployment",
            )

        return {
            "tenant": tenant,
            "pods": pod_metrics,
            "deployment": deployment_status,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get monitoring data", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get monitoring data",
        )


@router.get("/tenants")
async def get_all_tenants_monitoring(db: Session = Depends(get_db)):
    """Get monitoring data for all tenants."""
    try:
        tenants = db.query(Tenant).all()
        monitoring_data = []

        for tenant in tenants:
            try:
                pod_metrics = await kubernetes_service.get_pod_metrics(tenant.namespace)
                deployment_status = None
                if tenant.helm_release_name:
                    deployment_status = await kubernetes_service.get_deployment_status(
                        tenant.namespace,
                        f"{tenant.helm_release_name}-deployment",
                    )

                monitoring_data.append({
                    "tenant": tenant,
                    "pods": pod_metrics,
                    "deployment": deployment_status,
                })
            except Exception as e:
                logger.error("Failed to get monitoring for tenant", tenant_id=tenant.id, error=str(e))
                monitoring_data.append({
                    "tenant": tenant,
                    "pods": [],
                    "deployment": None,
                    "error": "Failed to fetch monitoring data",
                })

        return {
            "tenants": monitoring_data,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("Failed to get tenants monitoring data", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get monitoring data",
        )

