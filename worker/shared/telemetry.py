"""
OpenTelemetry setup for ingestion pipeline.

When opentelemetry is not installed, no-op implementations are used so the
worker can run without optional telemetry dependencies.
"""

import logging
from contextlib import nullcontext
from typing import Any, Optional

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False
    trace = None
    metrics = None

# FastAPI instrumentation is optional (not needed for workers)
if _HAS_OTEL:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        _HAS_FASTAPI = True
    except ImportError:
        _HAS_FASTAPI = False
else:
    _HAS_FASTAPI = False

from .config import settings

logger = logging.getLogger(__name__)

_tracer: Any = None
_meter: Any = None


# No-op helpers when opentelemetry is not installed
class _NoOpSpan:
    """Span that does nothing; used when opentelemetry is not installed or disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass


class _NoOpTracer:
    """Tracer that does nothing; used when opentelemetry is not installed."""

    def start_as_current_span(self, name: str, **kwargs):
        return nullcontext(_NoOpSpan())


class _NoOpMetric:
    """Metric that does nothing."""

    def add(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass


def setup_telemetry(service_name: str, service_version: str = "1.0.0"):
    """
    Set up OpenTelemetry tracing and metrics.

    Args:
        service_name: Name of the service (e.g., 'ingestion-api', 'ingestion-worker')
        service_version: Version of the service
    """
    global _tracer, _meter

    if not _HAS_OTEL:
        logger.debug("OpenTelemetry not installed; telemetry disabled")
        return

    if not settings.OTEL_ENABLED:
        logger.info("OpenTelemetry is disabled")
        return

    # Create resource
    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
            SERVICE_VERSION: service_version,
            "deployment.environment": settings.ENVIRONMENT,
        }
    )

    # Set up tracing
    try:
        tracer_provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter(
            endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
            insecure=True,
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)
        _tracer = trace.get_tracer(service_name, service_version)
        logger.info(
            f"Tracing enabled, exporting to {settings.OTEL_EXPORTER_OTLP_ENDPOINT}"
        )
    except Exception as e:
        logger.warning(f"Failed to set up tracing: {e}")

    # Set up metrics
    try:
        metric_exporter = OTLPMetricExporter(
            endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
            insecure=True,
        )
        metric_reader = PeriodicExportingMetricReader(
            metric_exporter,
            export_interval_millis=60000,  # Export every 60 seconds
        )
        meter_provider = MeterProvider(
            resource=resource, metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(meter_provider)
        _meter = metrics.get_meter(service_name, service_version)
        logger.info("Metrics enabled")
    except Exception as e:
        logger.warning(f"Failed to set up metrics: {e}")

    # Instrument libraries
    try:
        HTTPXClientInstrumentor().instrument()
        RedisInstrumentor().instrument()
        logger.info("Auto-instrumentation enabled for httpx and redis")
    except Exception as e:
        logger.warning(f"Failed to instrument libraries: {e}")


def instrument_fastapi(app):
    """Instrument a FastAPI application."""
    if not _HAS_FASTAPI:
        logger.debug("FastAPI instrumentation not available")
        return
    if _HAS_OTEL and settings.OTEL_ENABLED:
        try:
            FastAPIInstrumentor.instrument_app(app)
            logger.info("FastAPI instrumented")
        except Exception as e:
            logger.warning(f"Failed to instrument FastAPI: {e}")


def get_tracer():
    """Get the global tracer instance (or no-op if opentelemetry not installed)."""
    global _tracer
    if not _HAS_OTEL:
        return _NoOpTracer()
    if _tracer is None:
        _tracer = trace.get_tracer("ingestion-pipeline")
    return _tracer


def get_meter():
    """Get the global meter instance (or no-op if opentelemetry not installed)."""
    global _meter
    if not _HAS_OTEL:
        return None
    if _meter is None:
        _meter = metrics.get_meter("ingestion-pipeline")
    return _meter


# Pre-defined metrics (or no-op when opentelemetry not installed)
class IngestionMetrics:
    """Ingestion pipeline metrics."""

    _instance = None
    _noop = _NoOpMetric()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        if not _HAS_OTEL:
            self.documents_processed = self._noop
            self.chunks_created = self._noop
            self.extraction_failures = self._noop
            self.pages_crawled = self._noop
            self.processing_duration = self._noop
            self.crawl_duration = self._noop
            self.embedding_duration = self._noop
            self.active_jobs = self._noop
            self.queue_depth = self._noop
            self.tables_per_document = self._noop
            self._initialized = True
            return

        meter = get_meter()

        # Counters
        self.documents_processed = meter.create_counter(
            name="ingestion.documents_processed",
            description="Total number of documents processed",
            unit="1",
        )

        self.chunks_created = meter.create_counter(
            name="ingestion.chunks_created",
            description="Total number of chunks created",
            unit="1",
        )

        self.extraction_failures = meter.create_counter(
            name="ingestion.extraction_failures",
            description="Number of extraction failures",
            unit="1",
        )

        self.pages_crawled = meter.create_counter(
            name="ingestion.pages_crawled",
            description="Total number of pages crawled",
            unit="1",
        )

        # Histograms
        self.processing_duration = meter.create_histogram(
            name="ingestion.processing_duration",
            description="Time to process a document",
            unit="ms",
        )

        self.crawl_duration = meter.create_histogram(
            name="ingestion.crawl_duration",
            description="Time to crawl a page",
            unit="ms",
        )

        self.embedding_duration = meter.create_histogram(
            name="ingestion.embedding_duration",
            description="Time to embed chunks",
            unit="ms",
        )

        # Gauges (using up-down counters)
        self.active_jobs = meter.create_up_down_counter(
            name="ingestion.active_jobs",
            description="Number of currently active jobs",
            unit="1",
        )

        self.queue_depth = meter.create_up_down_counter(
            name="ingestion.queue_depth",
            description="Current queue depth",
            unit="1",
        )

        # Custom metrics
        self.tables_per_document = meter.create_histogram(
            name="ingestion.tables_per_document",
            description="Number of tables extracted per document",
            unit="1",
        )

        self._initialized = True

    def record_document_processed(self, collection: str, source_type: str):
        """Record a document was processed."""
        self.documents_processed.add(
            1, {"collection": collection, "source_type": source_type}
        )

    def record_chunks_created(self, count: int, collection: str):
        """Record chunks were created."""
        self.chunks_created.add(count, {"collection": collection})

    def record_extraction_failure(self, source_type: str, error_type: str):
        """Record an extraction failure."""
        self.extraction_failures.add(
            1, {"source_type": source_type, "error_type": error_type}
        )

    def record_processing_time(self, duration_ms: float, stage: str):
        """Record processing time for a stage."""
        self.processing_duration.record(duration_ms, {"stage": stage})
