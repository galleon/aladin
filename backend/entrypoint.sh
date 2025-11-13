#!/bin/bash
set -e

# If running as celery worker, don't wait for postgres (it's already handled by depends_on)
if [ "${CELERY_WORKER}" != "true" ]; then
  # Wait for database to be ready
  until pg_isready -h "${DB_HOST:-postgres}" -p "${DB_PORT:-5432}" -U "${DB_USER:-postgres}"; do
    echo "Waiting for database..."
    sleep 2
  done
fi

# Kubeconfig setup
KUBECONFIG_SOURCE="/root/.kube/config"
KUBECONFIG_DEST="/tmp/kubeconfig"

echo "Preparing kubeconfig..."
if [ -f "$KUBECONFIG_SOURCE" ]; then
  cp "$KUBECONFIG_SOURCE" "$KUBECONFIG_DEST"

  # Rewrite 127.0.0.1 to host.docker.internal for API server
  sed -i "s#https://127\.0\.0\.1:#https://host.docker.internal:#g" "$KUBECONFIG_DEST"

  # Add tls-server-name: localhost to cluster entry for Kind certs
  # Using python to parse and modify YAML as yq might not be available
  python -c '
import yaml
import sys

kubeconfig_path = sys.argv[1]
with open(kubeconfig_path, "r") as f:
    cfg = yaml.safe_load(f)

if cfg and "clusters" in cfg:
    for cluster in cfg["clusters"]:
        if "cluster" in cluster and "server" in cluster["cluster"]:
            # Assuming Kind clusters use "kind-" prefix in their name
            # or you can make this more specific if needed
            cluster_name = cluster.get("name", "")
            if "kind-mgmt" in cluster_name: # Target the management cluster
                cluster["cluster"]["tls-server-name"] = "localhost"
                print(f"Added tls-server-name: localhost to cluster {cluster_name}")

with open(kubeconfig_path, "w") as f:
    yaml.safe_dump(cfg, f)
' "$KUBECONFIG_DEST"

  export KUBECONFIG="$KUBECONFIG_DEST"
  echo "Kubeconfig prepared and KUBECONFIG env var set to $KUBECONFIG_DEST"
else
  echo "Warning: Kubeconfig source $KUBECONFIG_SOURCE not found. Kubernetes API might be unavailable."
fi

# If running as celery worker, execute the provided command
if [ "${CELERY_WORKER}" = "true" ]; then
  exec "$@"
else
  # Run the application
  exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-3000}"
fi
