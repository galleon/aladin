#!/bin/bash
set -e

echo "=========================================="
echo "Recreating Kind Cluster with Docker Access"
echo "=========================================="

CLUSTER_NAME="mgmt"
KIND_CONFIG="kind-config.yaml"

# Check if kind is installed
if ! command -v kind &> /dev/null; then
    echo "Error: kind is not installed. Please install it first:"
    echo "  brew install kind  # macOS"
    echo "  or visit: https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
    exit 1
fi

# Check if clusterctl is installed
if ! command -v clusterctl &> /dev/null; then
    echo "Error: clusterctl is not installed. Please install it first:"
    echo "  brew install clusterctl  # macOS"
    echo "  or visit: https://cluster-api.sigs.k8s.io/user/quick-start.html#install-clusterctl"
    exit 1
fi

# Check if config file exists
if [ ! -f "$KIND_CONFIG" ]; then
    echo "Error: $KIND_CONFIG not found in current directory"
    exit 1
fi

echo ""
echo "Step 1: Deleting existing kind cluster (if it exists)..."
if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
    kind delete cluster --name "$CLUSTER_NAME"
    echo "✓ Cluster deleted"
else
    echo "✓ No existing cluster found"
fi

echo ""
echo "Step 2: Creating new kind cluster with Docker socket access..."
kind create cluster --name "$CLUSTER_NAME" --config "$KIND_CONFIG"
echo "✓ Cluster created"

echo ""
echo "Step 3: Waiting for cluster to be ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=120s || true
echo "✓ Cluster is ready"

echo ""
echo "Step 4: Initializing Cluster API providers..."
echo "  - Initializing core CAPI..."
clusterctl init --infrastructure docker

echo ""
echo "Step 5: Waiting for CAPI controllers to be ready..."
echo "  Waiting for CAPI core controllers..."
kubectl wait --for=condition=Available deployment/capi-controller-manager -n capi-system --timeout=300s || true

echo "  Waiting for CAPD controllers..."
kubectl wait --for=condition=Available deployment/capd-controller-manager -n capd-system --timeout=300s || true

echo "  Waiting for Kubeadm Bootstrap controllers..."
kubectl wait --for=condition=Available deployment/capi-kubeadm-bootstrap-controller-manager -n capi-kubeadm-bootstrap-system --timeout=300s || true

echo "  Waiting for Kubeadm Control Plane controllers..."
kubectl wait --for=condition=Available deployment/capi-kubeadm-control-plane-controller-manager -n capi-kubeadm-control-plane-system --timeout=300s || true

echo ""
echo "=========================================="
echo "✓ Kind cluster recreated successfully!"
echo "=========================================="
echo ""
echo "Cluster name: $CLUSTER_NAME"
echo "Kubeconfig: ~/.kube/config"
echo ""
echo "You can now:"
echo "  1. Restart the Docker Compose application"
echo "  2. Create new tenants (existing ones may need to be recreated)"
echo ""

