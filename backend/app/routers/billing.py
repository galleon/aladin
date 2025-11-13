"""Billing router."""
from typing import Optional, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Tenant
from app.services.billing import billing_service
from app.schemas import TokenUsageCreate, TokenUsageResponse
from app.logger import logger

router = APIRouter()


@router.post("/usage", status_code=status.HTTP_201_CREATED)
async def record_token_usage(
    usage_data: TokenUsageCreate,
    db: Session = Depends(get_db),
):
    """Record token usage."""
    try:
        # Verify tenant exists
        tenant = db.query(Tenant).filter(Tenant.id == usage_data.tenant_id).first()
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found",
            )

        await billing_service.record_token_usage(
            db=db,
            tenant_id=usage_data.tenant_id,
            input_tokens=usage_data.input_tokens,
            output_tokens=usage_data.output_tokens,
            model=usage_data.model,
            endpoint=usage_data.endpoint,
        )

        return {"message": "Token usage recorded successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to record token usage", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record token usage",
        )


@router.get("/usage/tenant/{tenant_id}", response_model=list[dict[str, Any]])
async def get_token_usage(
    tenant_id: int,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
):
    """Get token usage for a tenant."""
    try:
        usage = await billing_service.get_token_usage(
            db=db,
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date,
        )
        return usage
    except Exception as e:
        logger.error("Failed to get token usage", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get token usage",
        )


@router.get("/usage/tenant/{tenant_id}/summary")
async def get_token_usage_summary(tenant_id: int, db: Session = Depends(get_db)):
    """Get token usage summary for a tenant."""
    try:
        summary = await billing_service.get_token_usage_summary(db=db, tenant_id=tenant_id)
        return summary
    except Exception as e:
        logger.error("Failed to get token usage summary", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get token usage summary",
        )


@router.get("/usage/tenant/{tenant_id}/by-model")
async def get_token_usage_by_model(tenant_id: int, db: Session = Depends(get_db)):
    """Get token usage by model for a tenant."""
    try:
        usage_by_model = await billing_service.get_token_usage_by_model(
            db=db, tenant_id=tenant_id
        )
        return usage_by_model
    except Exception as e:
        logger.error("Failed to get token usage by model", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get token usage by model",
        )

