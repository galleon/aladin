# OpenTelemetry and observability

## How OpenTelemetry is installed

- **In the app**: OpenTelemetry is installed as **Python packages** inside the worker image (`worker.Dockerfile`). The worker uses:
  - `opentelemetry-api`, `opentelemetry-sdk`
  - `opentelemetry-exporter-otlp-proto-grpc` (sends traces/metrics to an OTLP gRPC endpoint)
  - `opentelemetry-instrumentation-httpx`, `opentelemetry-instrumentation-redis`
- If these packages are missing or `OTEL_ENABLED` is false, the worker uses no-op spans and runs without sending telemetry.

## Docker: Jaeger (already in this repo)

You already have **Jaeger** in Docker Compose. It accepts OTLP gRPC and exposes a UI.

1. Start the stack with the observability profile:
   ```bash
   docker compose --profile observability up -d
   ```
2. Set env so the worker sends traces to Jaeger:
   ```bash
   export OTEL_ENABLED=true
   export OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
   docker compose --profile observability up -d worker
   ```
   Or in `.env`:
   ```
   OTEL_ENABLED=true
   OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
   ```
3. Open the Jaeger UI: **http://localhost:16686**

The worker will send traces to `jaeger:4317` (OTLP gRPC). No extra “OpenTelemetry Docker” service is required for this.

## Docker: OpenTelemetry Collector (optional)

If you want a dedicated **OpenTelemetry Collector** (e.g. to buffer, sample, or send to multiple backends), you can run the official image:

- **Image**: [otel/opentelemetry-collector-contrib](https://hub.docker.com/r/otel/opentelemetry-collector-contrib) (includes many exporters)
- Or slimmer: [otel/opentelemetry-collector](https://hub.docker.com/r/otel/opentelemetry-collector)

Example service (add to `docker-compose.yml` or run separately):

```yaml
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./otel-collector-config.yaml:/etc/otel-collector-config.yaml:ro
    ports:
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
```

Then point the worker at the collector instead of Jaeger:

- `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317`

The collector can then export to Jaeger, Prometheus, or other backends via its config.

## Summary

| What you want            | How                                                                   |
| ------------------------ | --------------------------------------------------------------------- |
| Install OpenTelemetry    | Already in worker image (see `worker.Dockerfile`).                    |
| Run a backend for traces | Use **Jaeger** (already in compose with `--profile observability`).   |
| Optional middle layer    | Run **OpenTelemetry Collector** Docker image and point the app at it. |

No separate “OpenTelemetry server” Docker is required; your app sends OTLP to Jaeger (or a Collector) that you run in Docker.
