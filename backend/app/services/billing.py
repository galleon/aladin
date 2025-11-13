"""Billing service for tracking token usage."""
from typing import Optional, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import TokenUsage
import structlog

logger = structlog.get_logger()


class BillingService:
    """Service for managing billing and token usage."""

    async def record_token_usage(
        self,
        db: Session,
        tenant_id: int,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        """Record token usage for a tenant."""
        try:
            total_tokens = input_tokens + output_tokens

            token_usage = TokenUsage(
                tenant_id=tenant_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                model=model,
                endpoint=endpoint,
            )

            db.add(token_usage)
            db.commit()
            logger.info("Token usage recorded", tenant_id=tenant_id)
        except Exception as e:
            db.rollback()
            logger.error("Failed to record token usage", error=str(e))
            raise

    async def get_token_usage(
        self,
        db: Session,
        tenant_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """Get token usage for a tenant."""
        try:
            query = db.query(TokenUsage).filter(TokenUsage.tenant_id == tenant_id)

            if start_date:
                query = query.filter(TokenUsage.timestamp >= start_date)
            if end_date:
                query = query.filter(TokenUsage.timestamp <= end_date)

            usage_records = query.order_by(TokenUsage.timestamp.desc()).all()

            return [
                {
                    "timestamp": record.timestamp.isoformat(),
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "total_tokens": record.total_tokens,
                    "model": record.model,
                    "endpoint": record.endpoint,
                }
                for record in usage_records
            ]
        except Exception as e:
            logger.error("Failed to get token usage", error=str(e))
            raise

    async def get_token_usage_summary(self, db: Session, tenant_id: int) -> dict[str, Any]:
        """Get token usage summary for a tenant."""
        try:
            result = (
                db.query(
                    func.sum(TokenUsage.input_tokens).label("total_input_tokens"),
                    func.sum(TokenUsage.output_tokens).label("total_output_tokens"),
                    func.sum(TokenUsage.total_tokens).label("total_tokens"),
                    func.count(TokenUsage.id).label("request_count"),
                )
                .filter(TokenUsage.tenant_id == tenant_id)
                .first()
            )

            return {
                "total_input_tokens": result.total_input_tokens or 0,
                "total_output_tokens": result.total_output_tokens or 0,
                "total_tokens": result.total_tokens or 0,
                "request_count": result.request_count or 0,
            }
        except Exception as e:
            logger.error("Failed to get token usage summary", error=str(e))
            raise

    async def get_token_usage_by_model(self, db: Session, tenant_id: int) -> list[dict[str, Any]]:
        """Get token usage grouped by model."""
        try:
            results = (
                db.query(
                    TokenUsage.model,
                    func.sum(TokenUsage.input_tokens).label("total_input_tokens"),
                    func.sum(TokenUsage.output_tokens).label("total_output_tokens"),
                    func.sum(TokenUsage.total_tokens).label("total_tokens"),
                    func.count(TokenUsage.id).label("request_count"),
                )
                .filter(TokenUsage.tenant_id == tenant_id)
                .filter(TokenUsage.model.isnot(None))
                .group_by(TokenUsage.model)
                .order_by(func.sum(TokenUsage.total_tokens).desc())
                .all()
            )

            return [
                {
                    "model": result.model,
                    "total_input_tokens": result.total_input_tokens or 0,
                    "total_output_tokens": result.total_output_tokens or 0,
                    "total_tokens": result.total_tokens or 0,
                    "request_count": result.request_count or 0,
                }
                for result in results
            ]
        except Exception as e:
            logger.error("Failed to get token usage by model", error=str(e))
            raise


billing_service = BillingService()

