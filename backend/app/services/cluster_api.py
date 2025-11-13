"""Cluster API service for managing tenant clusters."""
import re
import os
import yaml
import base64
import asyncio
from pathlib import Path
from typing import Optional, Any
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import structlog

logger = structlog.get_logger()

TEMPLATE_PATH = Path(__file__).parent.parent.parent / "templates" / "capi" / "cluster-template.yaml"


def sanitize_tenant_slug(value: str) -> str:
    """Sanitize tenant name to valid Kubernetes resource name."""
    slug = re.sub(r"[^a-z0-9-]", "-", value.lower())
    slug = re.sub(r"^-+", "", slug)
    slug = re.sub(r"-+$", "", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or "tenant"


class ClusterApiService:
    """Service for managing Cluster API resources."""

    def __init__(self):
        """Initialize Kubernetes clients."""
        self.core_v1: Optional[client.CoreV1Api] = None
        self.custom_objects_api: Optional[client.CustomObjectsApi] = None
        self._template_cache: Optional[str] = None
        self._initialized = False

        try:
            try:
                config.load_incluster_config()
                logger.info("Loaded Kubernetes in-cluster config")
            except config.ConfigException:
                try:
                    config.load_kube_config()
                    logger.info("Loaded Kubernetes kubeconfig")
                except config.ConfigException as e:
                    logger.warn("Could not load Kubernetes config", error=str(e))
                    return

            self.core_v1 = client.CoreV1Api()
            self.custom_objects_api = client.CustomObjectsApi()
            self._initialized = True
        except Exception as e:
            logger.error("Failed to initialize Kubernetes clients", error=str(e))
            self._initialized = False

    def _load_template(self) -> str:
        """Load and cache cluster template."""
        if self._template_cache is None:
            with open(TEMPLATE_PATH, "r") as f:
                self._template_cache = f.read()
        return self._template_cache

    def _build_manifests(self, tenant: str) -> list[dict[str, Any]]:
        """Build Kubernetes manifests from template."""
        template = self._load_template()
        slug = sanitize_tenant_slug(tenant)
        rendered = template.replace("{{TENANT}}", slug)

        manifests = []
        for doc in yaml.safe_load_all(rendered):
            if not doc or not isinstance(doc, dict):
                continue
            if not doc.get("kind") or not doc.get("apiVersion"):
                logger.warn("Skipping manifest without kind/apiVersion", manifest=doc)
                continue
            if not doc.get("metadata", {}).get("name"):
                logger.warn("Skipping manifest without metadata.name", manifest=doc)
                continue
            if "namespace" not in doc.get("metadata", {}):
                doc["metadata"]["namespace"] = "default"
            manifests.append(doc)

        return manifests

    async def create_tenant_cluster(self, tenant: str) -> dict[str, str]:
        """Create a tenant cluster using Cluster API."""
        if not self._initialized or not self.custom_objects_api:
            raise Exception("Kubernetes API not available. Please configure kubectl.")

        manifests = self._build_manifests(tenant)
        slug = sanitize_tenant_slug(tenant)

        for manifest in manifests:
            group = manifest["apiVersion"].split("/")[0]
            version = manifest["apiVersion"].split("/")[1] if "/" in manifest["apiVersion"] else "v1"
            kind = manifest["kind"]
            namespace = manifest["metadata"].get("namespace", "default")
            name = manifest["metadata"]["name"]

            try:
                # Use appropriate API based on resource type
                if group == "v1" or group == "":
                    # Core API resources
                    if kind == "Namespace":
                        ns_body = client.V1Namespace(
                            metadata=client.V1ObjectMeta(
                                name=manifest["metadata"]["name"],
                                labels=manifest["metadata"].get("labels", {}),
                            )
                        )
                        self.core_v1.create_namespace(body=ns_body)
                    else:
                        logger.warn("Unsupported core resource", kind=kind)
                        continue
                else:
                    # Custom resources - need to determine plural form
                    plural_map = {
                        "Cluster": "clusters",
                        "KubeadmControlPlane": "kubeadmcontrolplanes",
                        "DockerCluster": "dockerclusters",
                        "DockerMachineTemplate": "dockermachinetemplates",
                        "MachineDeployment": "machinedeployments",
                        "KubeadmConfigTemplate": "kubeadmconfigtemplates",
                    }
                    plural = plural_map.get(kind, kind.lower() + "s")

                    self.custom_objects_api.create_namespaced_custom_object(
                        group=group,
                        version=version,
                        namespace=namespace,
                        plural=plural,
                        body=manifest,
                    )

                logger.info(
                    "Applied CAPI manifest",
                    kind=kind,
                    name=name,
                    namespace=namespace,
                )
            except ApiException as e:
                if e.status == 409:  # Already exists
                    logger.warn(
                        "CAPI manifest already exists, skipping",
                        kind=kind,
                        name=name,
                        namespace=namespace,
                    )
                    continue
                error_msg = e.body if hasattr(e, "body") else str(e)
                logger.error("Failed to apply CAPI manifest", manifest=manifest, error=error_msg)
                raise Exception(f"Failed to create cluster: {error_msg}")
            except Exception as e:
                if "409" in str(e) or "AlreadyExists" in str(e):
                    logger.warn("CAPI manifest already exists, skipping", manifest=manifest)
                    continue
                logger.error("Failed to apply CAPI manifest", manifest=manifest, error=str(e))
                raise

        return {"slug": slug}

    async def get_cluster_status(self, tenant: str) -> Optional[dict[str, Any]]:
        """Get cluster status."""
        if not self._initialized or not self.custom_objects_api:
            return None

        slug = sanitize_tenant_slug(tenant)
        name = f"tenant-{slug}"

        try:
            response = self.custom_objects_api.get_namespaced_custom_object(
                group="cluster.x-k8s.io",
                version="v1beta1",
                namespace="default",
                plural="clusters",
                name=name,
            )
            return response
        except ApiException as e:
            if e.status == 404:
                return None
            logger.error("Failed to fetch cluster status", name=name, error=str(e))
            return None

    async def wait_for_cluster_ready(self, tenant: str, max_wait_seconds: int = 300) -> bool:
        """Wait for cluster to be ready."""
        slug = sanitize_tenant_slug(tenant)
        cluster_name = f"tenant-{slug}"
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < max_wait_seconds:
            cluster = await self.get_cluster_status(tenant)
            if not cluster:
                await asyncio.sleep(5)
                continue

            phase = cluster.get("status", {}).get("phase")
            conditions = cluster.get("status", {}).get("conditions", [])
            infrastructure_ready = any(
                c.get("type") == "InfrastructureReady" and c.get("status") == "True"
                for c in conditions
            )
            control_plane_ready = any(
                c.get("type") == "ControlPlaneReady" and c.get("status") == "True"
                for c in conditions
            )
            ready_condition = any(
                c.get("type") == "Ready" and c.get("status") == "True"
                for c in conditions
            )

            is_ready = phase == "Provisioned" or (
                infrastructure_ready and control_plane_ready and ready_condition
            )

            if is_ready:
                kubeconfig = await self.get_tenant_cluster_kubeconfig(tenant)
                if kubeconfig:
                    logger.info(
                        "Cluster is ready and kubeconfig is available",
                        cluster_name=cluster_name,
                    )
                    return True
                else:
                    logger.info(
                        "Cluster appears ready but kubeconfig secret not yet available, continuing to wait",
                        cluster_name=cluster_name,
                    )

            logger.info(
                "Waiting for cluster to be ready",
                cluster_name=cluster_name,
                phase=phase,
                infrastructure_ready=infrastructure_ready,
                control_plane_ready=control_plane_ready,
            )
            await asyncio.sleep(5)

        logger.warn(
            "Cluster did not become ready within timeout period",
            cluster_name=cluster_name,
            max_wait_seconds=max_wait_seconds,
        )
        return False

    async def get_tenant_cluster_kubeconfig(self, tenant: str) -> Optional[str]:
        """Get tenant cluster kubeconfig from secret."""
        if not self._initialized or not self.core_v1:
            return None

        slug = sanitize_tenant_slug(tenant)
        cluster_name = f"tenant-{slug}"
        secret_name = f"{cluster_name}-kubeconfig"

        try:
            secret = self.core_v1.read_namespaced_secret(secret_name, "default")
            kubeconfig_data = secret.data.get("value")
            if not kubeconfig_data:
                logger.error(
                    "Kubeconfig secret does not contain 'value' key",
                    secret_name=secret_name,
                )
                return None
            kubeconfig = base64.b64decode(kubeconfig_data).decode("utf-8")
            logger.info("Retrieved kubeconfig for cluster", cluster_name=cluster_name)
            return kubeconfig
        except ApiException as e:
            if e.status == 404:
                logger.warn("Kubeconfig secret not found yet", secret_name=secret_name)
                return None
            logger.error("Failed to retrieve kubeconfig", cluster_name=cluster_name, error=str(e))
            return None

    async def delete_tenant_cluster(self, tenant: str) -> None:
        """Delete tenant cluster."""
        if not self._initialized or not self.custom_objects_api:
            logger.warn("Kubernetes API not available, skipping cluster deletion")
            return

        slug = sanitize_tenant_slug(tenant)
        cluster_name = f"tenant-{slug}"

        try:
            self.custom_objects_api.delete_namespaced_custom_object(
                group="cluster.x-k8s.io",
                version="v1beta1",
                namespace="default",
                plural="clusters",
                name=cluster_name,
            )
            logger.info("Deleted Cluster API cluster", cluster_name=cluster_name)
        except ApiException as e:
            if e.status == 404:
                logger.warn("Cluster not found, skipping deletion", cluster_name=cluster_name)
                return
            logger.error("Failed to delete cluster", cluster_name=cluster_name, error=str(e))
            raise


cluster_api_service = ClusterApiService()

