# Getting Started with Aladin

This guide will help you set up and run the Aladin LLM Platform Management system.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python** 3.11+ and uv (Python package manager)
- **PostgreSQL** 12+
- **Kubernetes** cluster (minikube, kind, or cloud-based)
- **kubectl** configured to access your cluster
- **Helm** 3.0+
- **Docker** and Docker Compose (recommended for containerized deployment)

## Quick Start

### 1. Clone and Setup

```bash
# Navigate to the project directory
cd aladin
```

### 2. Database Setup

Start PostgreSQL (if not already running):

```bash
# Using Docker
docker run -d \
  --name postgres \
  -e POSTGRES_DB=aladin \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  postgres:15

# Or use your existing PostgreSQL instance
createdb aladin
```

### 3. Backend Setup

The backend is now Python-based using FastAPI. The easiest way to run it is with Docker Compose (see below). For local development:

```bash
cd backend

# Install uv if not already installed
pip install uv

# Install dependencies
uv pip install -e .

# Set environment variables (or use .env file)
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=aladin
export DB_USER=postgres
export DB_PASSWORD=postgres

# Start the backend server
uvicorn app.main:app --host 0.0.0.0 --port 3000 --reload
```

The backend will be available at `http://localhost:3000`

### 4. Frontend Setup

In a new terminal:

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:5173` (dev) or `http://localhost:5174` (Docker)

### 5. Kubernetes Configuration

#### Setting Up Kind Cluster with Docker Access

If you're using kind as your management cluster, it **must** be created with Docker socket access for Cluster API Provider Docker (CAPD) to work. Use the provided script:

```bash
# Run the cluster recreation script
./scripts/recreate-kind-cluster.sh
```

This script will:
1. Delete any existing `mgmt` kind cluster
2. Create a new cluster with Docker socket mounted
3. Initialize Cluster API providers (CAPI + CAPD)
4. Wait for all controllers to be ready

**Manual Setup** (if you prefer to do it manually):

```bash
# Create kind cluster with Docker socket access
kind create cluster --name mgmt --config kind-config.yaml

# Initialize Cluster API
clusterctl init --infrastructure docker

# Wait for controllers to be ready
kubectl wait --for=condition=Available deployment/capi-controller-manager -n capi-system --timeout=300s
kubectl wait --for=condition=Available deployment/capd-controller-manager -n capd-system --timeout=300s
```

**Important**: Without Docker socket access, tenant cluster provisioning will fail with errors like:
```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock
```

#### Verify Kubernetes Setup

Ensure your Kubernetes cluster is accessible:

```bash
# Verify cluster access
kubectl get nodes

# Verify Helm is installed
helm version

# Verify Cluster API management cluster (kind + CAPD example)
kubectl get clusters.cluster.x-k8s.io -A

# Verify CAPI controllers are running
kubectl get pods -n capi-system
kubectl get pods -n capd-system

# Optional: check clusterctl is installed
clusterctl version
```

### Cluster API Tenant Provisioning

The backend uses Cluster API (CAPI) to provision tenant clusters from the template stored in `backend/templates/capi/cluster-template.yaml`. By default the template targets a Docker-based management cluster (kind + CAPD) and replaces the `{{TENANT}}` placeholder with the tenant namespace.

**Before creating tenants from the UI/API:**

1. **Set up your management cluster** (if using kind):
   ```bash
   ./scripts/recreate-kind-cluster.sh
   ```
   This script will:
   - Create a kind cluster with Docker socket access
   - Initialize Cluster API providers (CAPI + CAPD)
   - Wait for all controllers to be ready

2. **Ensure kubeconfig access**: The backend container needs access to your Kubernetes cluster. By default, Docker Compose mounts `~/.kube/config` from your host.

3. **Verify Cluster API is ready**:
   ```bash
   kubectl get pods -n capi-system
   kubectl get pods -n capd-system
   ```

When a tenant is created, the backend:

1. Renders the template with the tenant slug.
2. Applies the manifests through the Kubernetes API using the python-kubernetes client.
3. Waits for the tenant cluster to be provisioned and ready.
4. Retrieves the tenant cluster's kubeconfig.
5. Deploys applications to the tenant cluster using Helm.

## Using Docker Compose

**Recommended**: Use Docker Compose to run the entire stack:

```bash
# First, set up your Kubernetes cluster (if using kind)
./scripts/recreate-kind-cluster.sh

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop all services
docker-compose down
```

**Important**:
- If you're using kind, **you must run `./scripts/recreate-kind-cluster.sh` first** to ensure Docker socket access is properly configured for Cluster API Provider Docker (CAPD).
- The backend container needs access to your Kubernetes cluster. By default, it mounts `~/.kube/config` from your host.
- Ensure your kubeconfig is configured to access the management cluster before starting the backend.

## Creating Your First Tenant

1. Open the frontend at `http://localhost:5173` (dev) or `http://localhost:5174` (Docker)
2. Click "New Tenant" or navigate to `/tenants/new`
3. Fill in:
   - **Tenant Name**: A unique name (e.g., `my-llm-platform`)
   - **Kubernetes Namespace**: A valid namespace name (e.g., `my-llm-platform`)
   - **Helm Values** (optional): JSON configuration for customizing the deployment
4. Click "Create Tenant"

The system will:
- Create a Kubernetes namespace
- Deploy the Helm chart
- Start monitoring the platform

## Monitoring

- View tenant status on the dashboard
- Check individual tenant details for deployment status
- Monitor pod health and resource usage

## Token Tracking

To record token usage for billing:

```bash
curl -X POST http://localhost:3000/api/billing/usage \
  -H "Content-Type: application/json" \
  -d '{
    "tenantId": 1,
    "inputTokens": 100,
    "outputTokens": 50,
    "model": "gpt-3.5-turbo",
    "endpoint": "/v1/chat/completions"
  }'
```

View billing information in the tenant detail page.

## Customizing the Helm Chart

The Helm chart is located in `helm-charts/llm-platform/`. You can customize:

- Container images
- Resource limits
- Environment variables
- Service configuration

Edit `values.yaml` or provide custom values when creating a tenant.

## Troubleshooting

### Backend can't connect to Kubernetes

- Ensure `kubectl` is configured correctly on your host
- Check that `~/.kube/config` exists and is valid
- Verify cluster access: `kubectl get nodes`
- If using Docker Compose, ensure the kubeconfig volume mount is correct in `docker-compose.yml`
- Check backend logs for Kubernetes initialization errors

### Helm deployment fails

- Check Helm is installed: `helm version`
- Verify chart path is correct
- Check Kubernetes namespace permissions

### Database connection errors

- Verify PostgreSQL is running
- Check connection settings in `.env`
- Ensure database `aladin` exists

### Frontend can't reach backend

- Check backend is running on port 3000
- Verify CORS settings in backend (should allow all origins by default)
- Check that the frontend is configured to use the correct API URL
- Verify Docker Compose networking if using containers

## Next Steps

- Customize the LLM platform container image
- Configure monitoring and alerting
- Set up production deployment
- Integrate with your LLM provider APIs
- Configure token tracking endpoints

## Architecture Overview

```
┌─────────────┐
│   Frontend  │ (React + Vite)
│  Port 5174  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Backend   │ (Python + FastAPI)
│  Port 3000  │
└──────┬──────┘
       │
       ├──► PostgreSQL (Database)
       │
       ├──► Kubernetes API (python-kubernetes)
       │
       ├──► Cluster API (Tenant Clusters)
       │
       └──► Helm (Chart Deployment)
```

Each tenant gets:
- A Kubernetes namespace
- A Helm release
- A deployment with pods
- Monitoring and billing tracking

