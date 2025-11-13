# Aladin - AI/LLM Platform Management System

A comprehensive platform for managing multiple AI/LLM platforms across Kubernetes tenants. Each platform runs in its own Kubernetes tenant, deployed via Helm charts, with monitoring and token-based billing.

## Architecture

- **Backend**: Node.js/Express API for tenant management, Kubernetes operations, and billing
- **Frontend**: React application for tenant creation and management
- **Helm Charts**: Kubernetes deployment templates for LLM platforms
- **Monitoring**: Platform health and performance metrics
- **Billing**: Token usage tracking (input/output) per tenant

## Project Structure

```
aladin/
├── backend/          # Backend API server
├── frontend/         # React frontend application
├── helm-charts/      # Helm chart templates for LLM deployment
├── monitoring/       # Monitoring configuration
└── docs/            # Documentation
```

## Features

- Create and manage Kubernetes tenants
- Deploy LLM platforms via Helm charts
- Monitor platform health and performance
- Track input/output tokens for billing
- Multi-tenant isolation

## Getting Started

See individual README files in each directory for setup instructions.

