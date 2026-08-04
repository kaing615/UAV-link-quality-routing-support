"""Prometheus metrics and async prediction export helpers for the serving app."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

REQUEST_COUNT = Counter(
    "uav_gnn_requests_total",
    "Total inference requests",
    ["model_id"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "uav_gnn_request_latency_seconds",
    "Inference request latency",
    ["model_id"],
    registry=REGISTRY,
)
PREDICTION_STABILITY = Histogram(
    "uav_gnn_prediction_stability_score",
    "Distribution of prediction stability scores",
    ["model_id"],
    registry=REGISTRY,
)
STABLE_RATIO = Gauge(
    "uav_gnn_stable_predictions_ratio",
    "Ratio of stable predictions",
    ["model_id"],
    registry=REGISTRY,
)
EDGE_COUNT = Histogram(
    "uav_gnn_edges_per_request",
    "Number of edges per inference request",
    ["model_id"],
    registry=REGISTRY,
)
MODEL_LOADED = Gauge(
    "uav_gnn_model_loaded",
    "Whether the model is loaded",
    ["model_id"],
    registry=REGISTRY,
)

_QUEUE: list[dict[str, Any]] = []
_QUEUE_LOCK = threading.Lock()


def _s3_client() -> Any:
    endpoint_url = os.getenv("AWS_S3_ENDPOINT_URL")
    if endpoint_url:
        return boto3.client("s3", endpoint_url=endpoint_url, region_name=os.getenv("AWS_REGION", "us-east-1"))
    return boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))


def record_request(model_id: str) -> None:
    REQUEST_COUNT.labels(model_id=model_id).inc()


def record_latency(model_id: str, duration_seconds: float) -> None:
    REQUEST_LATENCY.labels(model_id=model_id).observe(duration_seconds)


def record_prediction_metrics(model_id: str, edge_count: int, stability_scores: list[float], stable_count: int) -> None:
    EDGE_COUNT.labels(model_id=model_id).observe(edge_count)
    for score in stability_scores:
        PREDICTION_STABILITY.labels(model_id=model_id).observe(score)
    if stability_scores:
        STABLE_RATIO.labels(model_id=model_id).set(stable_count / len(stability_scores))
    else:
        STABLE_RATIO.labels(model_id=model_id).set(0.0)


def record_model_loaded(model_id: str, loaded: bool) -> None:
    MODEL_LOADED.labels(model_id=model_id).set(1 if loaded else 0)


def enqueue_prediction(payload: dict[str, Any]) -> None:
    with _QUEUE_LOCK:
        _QUEUE.append(payload)


def flush_predictions() -> None:
    if not _QUEUE:
        return

    with _QUEUE_LOCK:
        batch = list(_QUEUE)
        _QUEUE.clear()

    bucket = os.getenv("PREDICTION_BUCKET")
    if not bucket:
        return

    client = _s3_client()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    key = f"predictions/{timestamp}-{threading.get_ident()}.json"
    client.put_object(Bucket=bucket, Key=key, Body=json.dumps(batch).encode("utf-8"))


def start_background_flush(interval_seconds: float = 5.0) -> None:
    def _worker() -> None:
        while True:
            time.sleep(interval_seconds)
            flush_predictions()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
