"""Applications service for discovering and managing Helm charts."""
import os
import yaml
from pathlib import Path
from typing import Optional, Any
import structlog

logger = structlog.get_logger()


class ApplicationsService:
    """Service for discovering and managing applications."""

    def __init__(self):
        """Initialize applications service."""
        self.charts_base_path = Path(os.getcwd()) / "helm-charts"

    async def list_applications(self) -> list[dict[str, Any]]:
        """List all available applications."""
        applications = []

        if not self.charts_base_path.exists():
            logger.warn("Helm charts directory not found", path=str(self.charts_base_path))
            return applications

        try:
            chart_dirs = [
                d.name
                for d in self.charts_base_path.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]

            for chart_dir in chart_dirs:
                chart_path = self.charts_base_path / chart_dir
                chart_yaml_path = chart_path / "Chart.yaml"

                if not chart_yaml_path.exists():
                    logger.warn("Chart.yaml not found, skipping", chart_path=str(chart_path))
                    continue

                try:
                    with open(chart_yaml_path, "r") as f:
                        chart_data = yaml.safe_load(f)

                    application = {
                        "id": chart_dir,
                        "name": chart_data.get("name", chart_dir),
                        "description": chart_data.get("description", ""),
                        "version": chart_data.get("version", "0.0.0"),
                        "chart_path": str(chart_path),
                        "icon": chart_data.get("icon"),
                        "category": chart_data.get("keywords", [None])[0] if chart_data.get("keywords") else None,
                    }
                    applications.append(application)
                except Exception as e:
                    logger.warn("Failed to parse Chart.yaml", chart_path=str(chart_path), error=str(e))
                    continue

        except Exception as e:
            logger.error("Failed to list applications", error=str(e))

        return applications

    async def get_application(self, application_id: str) -> Optional[dict[str, Any]]:
        """Get a specific application by ID."""
        applications = await self.list_applications()
        return next((app for app in applications if app["id"] == application_id), None)


applications_service = ApplicationsService()

