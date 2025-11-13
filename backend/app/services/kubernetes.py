"""Kubernetes service for managing Kubernetes resources."""
from typing import Optional, Any
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import structlog

logger = structlog.get_logger()


class KubernetesService:
    """Service for managing Kubernetes resources."""

    def __init__(self):
        """Initialize Kubernetes clients."""
        self._initialized = False
        self.core_v1: Optional[client.CoreV1Api] = None
        self.apps_v1: Optional[client.AppsV1Api] = None

    def _initialize(self):
        """Initialize Kubernetes API clients."""
        if self._initialized:
            return

        try:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()

            self.core_v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self._initialized = True
        except Exception as e:
            logger.warn("Kubernetes not configured, some features will be unavailable", error=str(e))
            self._initialized = True  # Mark as initialized to avoid repeated attempts

    async def create_namespace(self, name: str) -> None:
        """Create a Kubernetes namespace."""
        self._initialize()
        if not self.core_v1:
            raise Exception("Kubernetes API not available. Please configure kubectl.")

        try:
            namespace = client.V1Namespace(
                metadata=client.V1ObjectMeta(
                    name=name,
                    labels={
                        "app.kubernetes.io/managed-by": "aladin",
                        "app.kubernetes.io/name": name,
                    },
                )
            )
            self.core_v1.create_namespace(body=namespace)
            logger.info("Namespace created successfully", name=name)
        except ApiException as e:
            if e.status == 409:
                logger.warn("Namespace already exists", name=name)
            else:
                logger.error("Failed to create namespace", name=name, error=str(e))
                raise

    async def delete_namespace(self, name: str) -> None:
        """Delete a Kubernetes namespace."""
        self._initialize()
        if not self.core_v1:
            raise Exception("Kubernetes API not available. Please configure kubectl.")

        try:
            self.core_v1.delete_namespace(name)
            logger.info("Namespace deleted successfully", name=name)
        except ApiException as e:
            if e.status == 404:
                logger.warn("Namespace not found", name=name)
            else:
                logger.error("Failed to delete namespace", name=name, error=str(e))
                raise

    async def get_namespace_status(self, name: str) -> str:
        """Get namespace status."""
        self._initialize()
        if not self.core_v1:
            return "Unavailable"

        try:
            response = self.core_v1.read_namespace(name)
            return response.status.phase or "Unknown"
        except ApiException as e:
            if e.status == 404:
                return "NotFound"
            logger.error("Failed to get namespace status", name=name, error=str(e))
            raise

    async def get_deployment_status(
        self, namespace: str, deployment_name: str
    ) -> Optional[dict[str, Any]]:
        """Get deployment status."""
        self._initialize()
        if not self.apps_v1:
            return None

        try:
            response = self.apps_v1.read_namespaced_deployment(deployment_name, namespace)
            return {
                "replicas": response.spec.replicas or 0,
                "ready_replicas": response.status.ready_replicas or 0,
                "available_replicas": response.status.available_replicas or 0,
                "conditions": [
                    {
                        "type": c.type,
                        "status": c.status,
                        "message": c.message,
                    }
                    for c in (response.status.conditions or [])
                ],
            }
        except ApiException as e:
            if e.status == 404:
                return None
            logger.error("Failed to get deployment status", error=str(e))
            raise

    async def get_pod_metrics(self, namespace: str) -> list:
        """Get pod metrics for namespace."""
        self._initialize()
        if not self.core_v1:
            return []

        try:
            pods = self.core_v1.list_namespaced_pod(namespace)
            return [
                {
                    "name": pod.metadata.name,
                    "status": pod.status.phase,
                    "containers": [
                        {"name": c.name, "image": c.image}
                        for c in (pod.spec.containers or [])
                    ],
                }
                for pod in pods.items
            ]
        except Exception as e:
            logger.error("Failed to get pod metrics", namespace=namespace, error=str(e))
            raise


kubernetes_service = KubernetesService()

