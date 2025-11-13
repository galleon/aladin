# Aladin Backend

Backend API server for managing AI/LLM platforms across Kubernetes tenants.

## Features

- Tenant management (create, delete, list)
- Kubernetes namespace management
- Helm chart deployment
- Platform monitoring
- Token usage tracking and billing

## Prerequisites

- Node.js 18+
- PostgreSQL database
- Kubernetes cluster access (kubeconfig)
- Helm CLI installed

## Setup

1. Install dependencies:
```bash
npm install
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Ensure PostgreSQL is running and create the database:
```bash
createdb aladin
```

4. Ensure Kubernetes access is configured:
```bash
kubectl get nodes  # Should work
```

5. Run the server:
```bash
npm run dev  # Development mode
# or
npm run build && npm start  # Production mode
```

## API Endpoints

### Tenants
- `GET /api/tenants` - List all tenants
- `GET /api/tenants/:id` - Get tenant details
- `POST /api/tenants` - Create new tenant
- `DELETE /api/tenants/:id` - Delete tenant
- `GET /api/tenants/:id/status` - Get tenant status

### Monitoring
- `GET /api/monitoring/tenant/:id` - Get monitoring data for a tenant
- `GET /api/monitoring/tenants` - Get monitoring data for all tenants

### Billing
- `POST /api/billing/usage` - Record token usage
- `GET /api/billing/usage/tenant/:id` - Get token usage for a tenant
- `GET /api/billing/usage/tenant/:id/summary` - Get token usage summary
- `GET /api/billing/usage/tenant/:id/by-model` - Get token usage by model

