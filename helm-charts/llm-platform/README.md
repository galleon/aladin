# LLM Platform Helm Chart

This Helm chart deploys an LLM platform on Kubernetes.

## Installation

```bash
helm install my-llm-platform ./helm-charts/llm-platform \
  --namespace my-namespace \
  --create-namespace \
  --set tenant.name=my-tenant \
  --set tenant.namespace=my-namespace
```

## Configuration

The following table lists the configurable parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicas` | Number of replicas | `1` |
| `image.repository` | Container image repository | `nginx` |
| `image.tag` | Container image tag | `latest` |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `service.type` | Kubernetes service type | `ClusterIP` |
| `service.port` | Service port | `80` |
| `resources.requests.memory` | Memory request | `1Gi` |
| `resources.requests.cpu` | CPU request | `500m` |
| `resources.limits.memory` | Memory limit | `2Gi` |
| `resources.limits.cpu` | CPU limit | `1000m` |
| `tenant.name` | Tenant name | `""` |
| `tenant.namespace` | Tenant namespace | `""` |
| `llm.model` | LLM model name | `gpt-3.5-turbo` |
| `llm.apiKey` | LLM API key (optional) | `""` |
| `monitoring.enabled` | Enable monitoring | `true` |
| `tokenTracking.enabled` | Enable token tracking | `true` |

## Notes

- This chart creates a basic deployment structure. You should customize the container image to your actual LLM platform.
- Token tracking requires the backend URL to be configured.
- The chart includes health checks and resource limits.

