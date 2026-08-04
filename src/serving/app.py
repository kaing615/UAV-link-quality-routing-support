"""FastAPI inference server for UAV-GNN link quality prediction."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.models.gnn.edge_gnn import EdgeAwareSAGEEdgeClassifier, GATEdgeClassifier, GraphSAGEEdgeClassifier
from src.serving.auth import handle_rate_limit_exceeded, limiter, require_api_key
from src.serving.logging_tracing import configure_logging, get_request_id, setup_tracing, tracing_scope
from src.serving.metrics import (
    REGISTRY,
    enqueue_prediction,
    record_latency,
    record_model_loaded,
    record_prediction_metrics,
    record_request,
    start_background_flush,
)
from src.serving.schemas import EdgePrediction, HealthResponse, PredictionRequest, PredictionResponse

logger = logging.getLogger("uav_gnn.serving")
configure_logging()
setup_tracing()


async def _body_size_limit(request: Request, call_next):
    max_bytes = int(os.getenv("MAX_REQUEST_BODY_BYTES", "1000000"))
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_bytes:
        return Response(status_code=413, content="Payload Too Large")
    return await call_next(request)

_MODELS = {
    "graphsage": GraphSAGEEdgeClassifier,
    "gat": GATEdgeClassifier,
    "edge-sage": EdgeAwareSAGEEdgeClassifier,
}
NODE_IN = 8
EDGE_IN = 7

# Global model state
_model = None
_model_id = None
_threshold = 0.5


def load_model() -> None:
    global _model, _model_id, _threshold

    model_dir = Path(os.environ.get("MODEL_DIR", "models"))
    _model_id = os.environ.get("MODEL_ID", "edge-sage")
    hidden = int(os.environ.get("HIDDEN_CHANNELS", "128"))
    num_layers = int(os.environ.get("NUM_LAYERS", "2"))
    _threshold = float(os.environ.get("THRESHOLD", "0.5"))

    base_id = _model_id.replace("-noedge", "")
    if base_id not in _MODELS:
        raise ValueError(f"Unknown model_id: {_model_id}")

    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        hidden = metadata.get("hidden_channels", hidden)
        num_layers = metadata.get("num_layers", num_layers)
        # metadata threshold wins unless explicitly overridden by env
        if "THRESHOLD" not in os.environ:
            _threshold = metadata.get("threshold", _threshold)

    weights_path = model_dir / "best_model.pt"
    if not weights_path.exists():
        # Fail fast: never serve a randomly-initialized model.
        raise RuntimeError(
            f"Model weights not found at {weights_path}. "
            f"Set MODEL_DIR to a directory containing best_model.pt "
            f"(and optionally metadata.json), or stage one with scripts/mlops/stage_serving_model.sh."
        )

    model_cls = _MODELS[base_id]
    model = model_cls(
        node_in_channels=NODE_IN,
        edge_in_channels=EDGE_IN,
        hidden_channels=hidden,
        num_layers=num_layers,
        dropout=0.0,  # inference mode
        use_edge_features="-noedge" not in _model_id,
    )
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location="cpu"))
    model.eval()
    _model = model
    logger.info("Loaded model_id=%s from %s (threshold=%.3f)", _model_id, model_dir, _threshold)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    record_model_loaded(_model_id or "none", _model is not None)
    start_background_flush()
    yield


app = FastAPI(
    title="UAV-GNN Link Quality Prediction API",
    description="Predict link stability in UAV networks using GNN",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(BaseHTTPMiddleware, dispatch=_body_size_limit)
app.add_exception_handler(RateLimitExceeded, handle_rate_limit_exceeded)
Instrumentator(registry=REGISTRY).instrument(app)


@app.get("/health", response_model=HealthResponse)
def health():
    logger.info("Health check requested", extra={"request_id": None})
    return HealthResponse(
        status="ok",
        model_id=_model_id or "none",
        model_loaded=_model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("20/minute")
def predict(req: PredictionRequest, request: Request, _: None = Depends(require_api_key)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = get_request_id(request)
    start_time = time.perf_counter()
    model_id = _model_id or "none"
    record_request(model_id)

    with tracing_scope("predict", request_id):
        logger.info("Prediction request received", extra={"request_id": request_id})

        node_id_map = {n.node_id: i for i, n in enumerate(req.nodes)}
        x = torch.tensor(
            [[n.x, n.y, n.z, n.vx, n.vy, n.vz, n.speed, n.degree] for n in req.nodes],
            dtype=torch.float32,
        )

        edge_index_list = []
        edge_attr_list = []
        for e in req.edges:
            if e.src in node_id_map and e.dst in node_id_map:
                si, di = node_id_map[e.src], node_id_map[e.dst]
                for s, d in [(si, di), (di, si)]:
                    edge_index_list.append([s, d])
                    edge_attr_list.append(
                        [
                            e.distance,
                            e.rssi,
                            e.snr,
                            e.delay,
                            e.packet_loss,
                            e.relative_speed,
                            e.throughput,
                        ]
                    )

        if not edge_index_list:
            logger.info("No edges to predict", extra={"request_id": request_id})
            return PredictionResponse(model_id=_model_id, threshold=_threshold, predictions=[])

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).T
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

        query_pairs = req.query_edges or [(e.src, e.dst) for e in req.edges]
        query_idx = []
        query_edge_attr = []
        for src, dst in query_pairs:
            if src in node_id_map and dst in node_id_map:
                si, di = node_id_map[src], node_id_map[dst]
                query_idx.append([si, di])
                e = next((e for e in req.edges if (e.src == src and e.dst == dst) or (e.src == dst and e.dst == src)), None)
                if e:
                    query_edge_attr.append(
                        [
                            e.distance,
                            e.rssi,
                            e.snr,
                            e.delay,
                            e.packet_loss,
                            e.relative_speed,
                            e.throughput,
                        ]
                    )
                else:
                    query_edge_attr.append([0.0] * EDGE_IN)

        edge_label_index = torch.tensor(query_idx, dtype=torch.long).T
        labeled_edge_attr = torch.tensor(query_edge_attr, dtype=torch.float32)

        with torch.no_grad():
            logits = _model(x, edge_index, edge_attr, edge_label_index, labeled_edge_attr)
            scores = torch.sigmoid(logits).numpy()

        predictions = []
        stability_scores = []
        stable_count = 0
        for i, (src, dst) in enumerate(query_pairs):
            if src in node_id_map and dst in node_id_map:
                score = float(scores[i])
                stability_scores.append(score)
                stable = score >= _threshold
                if stable:
                    stable_count += 1
                predictions.append(
                    EdgePrediction(
                        src=src,
                        dst=dst,
                        stability_score=score,
                        stable=stable,
                        routing_weight=1.0 - score,
                    )
                )

        record_prediction_metrics(model_id, len(req.edges), stability_scores, stable_count)
        record_latency(model_id, time.perf_counter() - start_time)

        payload = {
            "model_id": model_id,
            "threshold": _threshold,
            "request": req.model_dump(),
            "response": [p.model_dump() for p in predictions],
        }
        enqueue_prediction(payload)

        logger.info(
            "Prediction completed",
            extra={"request_id": request_id, "model_id": model_id, "prediction_count": len(predictions)},
        )
        return PredictionResponse(model_id=_model_id, threshold=_threshold, predictions=predictions)
