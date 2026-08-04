"""Structured logging and OpenTelemetry helpers for the FastAPI serving app."""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import contextmanager
from typing import Iterator

from fastapi import Request
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import SpanKind


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
        }
        return json.dumps(payload)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]


def get_request_id(request: Request | None) -> str:
    if request is None:
        return str(uuid.uuid4())
    return request.headers.get("x-request-id") or str(uuid.uuid4())


@contextmanager
def tracing_scope(name: str, request_id: str) -> Iterator[None]:
    tracer = trace.get_tracer("uav-gnn")
    with tracer.start_as_current_span(name, kind=SpanKind.SERVER) as span:
        span.set_attribute("request.id", request_id)
        yield


def setup_tracing() -> None:
    resource = Resource.create({"service.name": "uav-gnn-serving"})
    provider = TracerProvider(resource=resource)
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
