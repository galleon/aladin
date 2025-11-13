"""Helm service for managing Helm releases."""
import subprocess
import tempfile
import os
import yaml
import structlog
from typing import Optional, Any

logger = structlog.get_logger()


class HelmService:
    """Service for managing Helm releases."""

    def _values_to_yaml(self, values: dict[str, Any]) -> str:
        """Convert values dict to YAML string."""
        return yaml.dump(values, default_flow_style=False)

    async def install_release(
        self,
        release_name: str,
        chart_path: str,
        namespace: str,
        values: Optional[dict[str, Any]] = None,
        kubeconfig_path: Optional[str] = None,
    ) -> None:
        """Install a Helm release."""
        if values is None:
            values = {}

        values_file = None
        try:
            # Create temporary values file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write(self._values_to_yaml(values))
                values_file = f.name

            kubeconfig_flag = f"--kubeconfig {kubeconfig_path}" if kubeconfig_path else ""
            command = (
                f"helm install {release_name} {chart_path} "
                f"--namespace {namespace} --create-namespace "
                f"-f {values_file} {kubeconfig_flag}"
            ).strip()

            logger.info("Executing helm install", command=command)

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.stderr and "WARNING" not in result.stderr:
                logger.warn("Helm install stderr", stderr=result.stderr)

            if result.returncode != 0:
                raise Exception(f"Helm install failed: {result.stderr}")

            logger.info("Helm release installed", release_name=release_name, stdout=result.stdout)

        except Exception as e:
            logger.error("Failed to install Helm release", release_name=release_name, error=str(e))
            raise
        finally:
            if values_file and os.path.exists(values_file):
                try:
                    os.unlink(values_file)
                except Exception as e:
                    logger.warn("Failed to delete temp file", file=values_file, error=str(e))

    async def upgrade_release(
        self,
        release_name: str,
        chart_path: str,
        namespace: str,
        values: Optional[dict[str, Any]] = None,
        kubeconfig_path: Optional[str] = None,
    ) -> None:
        """Upgrade a Helm release."""
        if values is None:
            values = {}

        values_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write(self._values_to_yaml(values))
                values_file = f.name

            kubeconfig_flag = f"--kubeconfig {kubeconfig_path}" if kubeconfig_path else ""
            command = (
                f"helm upgrade {release_name} {chart_path} "
                f"--namespace {namespace} -f {values_file} {kubeconfig_flag}"
            ).strip()

            logger.info("Executing helm upgrade", command=command)

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.stderr and "WARNING" not in result.stderr:
                logger.warn("Helm upgrade stderr", stderr=result.stderr)

            if result.returncode != 0:
                raise Exception(f"Helm upgrade failed: {result.stderr}")

            logger.info("Helm release upgraded", release_name=release_name, stdout=result.stdout)

        except Exception as e:
            logger.error("Failed to upgrade Helm release", release_name=release_name, error=str(e))
            raise
        finally:
            if values_file and os.path.exists(values_file):
                try:
                    os.unlink(values_file)
                except Exception as e:
                    logger.warn("Failed to delete temp file", file=values_file, error=str(e))

    async def uninstall_release(
        self,
        release_name: str,
        namespace: str,
        kubeconfig_path: Optional[str] = None,
    ) -> None:
        """Uninstall a Helm release."""
        try:
            kubeconfig_flag = f"--kubeconfig {kubeconfig_path}" if kubeconfig_path else ""
            command = (
                f"helm uninstall {release_name} --namespace {namespace} {kubeconfig_flag}"
            ).strip()

            logger.info("Executing helm uninstall", command=command)

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,  # 10 second timeout
            )

            if result.stderr and "WARNING" not in result.stderr:
                logger.warn("Helm uninstall stderr", stderr=result.stderr)

            if result.returncode != 0:
                message = result.stderr or result.stdout or str(result.returncode)
                if "release: not found" in message:
                    logger.warn(
                        "Helm release not found during uninstall",
                        release_name=release_name,
                    )
                    return
                if "unreachable" in message or "timeout" in message or "i/o timeout" in message:
                    logger.warn(
                        "Helm uninstall timed out or cluster unreachable",
                        release_name=release_name,
                    )
                    return
                raise Exception(f"Helm uninstall failed: {message}")

            logger.info("Helm release uninstalled", release_name=release_name, stdout=result.stdout)

        except subprocess.TimeoutExpired:
            logger.warn(
                "Helm uninstall timed out, continuing with cleanup",
                release_name=release_name,
            )
            return
        except Exception as e:
            message = str(e)
            if "release: not found" in message or "unreachable" in message or "timeout" in message:
                logger.warn(
                    "Helm uninstall failed but continuing",
                    release_name=release_name,
                    error=message,
                )
                return
            logger.error("Failed to uninstall Helm release", release_name=release_name, error=str(e))
            raise

    async def get_release_status(
        self,
        release_name: str,
        namespace: str,
        kubeconfig_path: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Get Helm release status."""
        try:
            kubeconfig_flag = f"--kubeconfig {kubeconfig_path}" if kubeconfig_path else ""
            command = (
                f"helm status {release_name} --namespace {namespace} -o json {kubeconfig_flag}"
            ).strip()

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,  # Reduced from 30 to 10 seconds
            )

            if result.returncode != 0:
                message = result.stderr or result.stdout or ""
                if "release: not found" in message:
                    logger.warn("Helm release not found when fetching status", release_name=release_name)
                    return None
                logger.error("Failed to get Helm release status", release_name=release_name, error=message)
                return None

            import json
            return json.loads(result.stdout)

        except subprocess.TimeoutExpired:
            logger.warn("Helm status check timed out", release_name=release_name, namespace=namespace)
            return None
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "unreachable" in error_msg.lower():
                logger.warn("Helm status check failed due to timeout or unreachable cluster", release_name=release_name, error=error_msg)
            else:
                logger.error("Failed to get Helm release status", release_name=release_name, error=error_msg)
            return None

    async def list_releases(self, namespace: Optional[str] = None) -> list:
        """List Helm releases."""
        try:
            command = (
                f"helm list --namespace {namespace} -o json"
                if namespace
                else "helm list -A -o json"
            )

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error("Failed to list Helm releases", error=result.stderr)
                return []

            import json
            return json.loads(result.stdout)

        except Exception as e:
            logger.error("Failed to list Helm releases", error=str(e))
            return []


helm_service = HelmService()

