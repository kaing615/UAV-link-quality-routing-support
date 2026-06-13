"""FastAPI inference server for UAV-GNN link quality prediction."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException

from src.models.gnn.edge_gnn import EdgeAwareSAGEEdgeClassifier, GATEdgeClassifier, GraphSAGEEdgeClassifier
from src.serving.schemas import EdgePrediction, HealthResponse, PredictionRequest, PredictionResponse

_MODELS = {
    "graphsage": GraphSAGEEdgeClassifier,
    "gat": GATEdgeClassifier,
    "edge-sage": EdgeAwareSAGEEdgeClassifier,
}
NODE_IN = 8
EDGE_IN = 7

app = FastAPI(
    title="UAV-GNN Link Quality Prediction API",
    description="Predict link stability in UAV networks using GNN",
    version="1.0.0",
)

# Global model state
_model = None
_model_id = None
_threshold = 0.5


@app.on_event("startup")
def load_model():
    global _model, _model_id, _threshold

    model_dir = Path(os.environ.get("MODEL_DIR", "outputs/gnn/edge-sage/ns3big_000"))
    _model_id = os.environ.get("MODEL_ID", "edge-sage")
    hidden = int(os.environ.get("HIDDEN_CHANNELS", "128"))
    num_layers = int(os.environ.get("NUM_LAYERS", "2"))

    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            hidden = metadata.get("hidden_channels", hidden)
            num_layers = metadata.get("num_layers", num_layers)
            _threshold = metadata.get("threshold", 0.5)
        except Exception:
            pass

    base_id = _model_id.replace("-noedge", "")
    if base_id not in _MODELS:
        raise ValueError(f"Unknown model_id: {_model_id}")

    model_cls = _MODELS[base_id]
    _model = model_cls(
        node_in_channels=NODE_IN,
        edge_in_channels=EDGE_IN,
        hidden_channels=hidden,
        num_layers=num_layers,
        dropout=0.0,  # inference mode
        use_edge_features="-noedge" not in _model_id,
    )
    weights_path = model_dir / "best_model.pt"
    if weights_path.exists():
        _model.load_state_dict(torch.load(weights_path, weights_only=True, map_location="cpu"))
    _model.eval()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_id=_model_id or "none",
        model_loaded=_model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build tensors from request
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
                edge_attr_list.append([
                    e.distance, e.rssi, e.snr, e.delay,
                    e.packet_loss, e.relative_speed, e.throughput,
                ])

    if not edge_index_list:
        return PredictionResponse(model_id=_model_id, threshold=_threshold, predictions=[])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).T
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

    # Query edges
    query_pairs = req.query_edges or [(e.src, e.dst) for e in req.edges]
    query_idx = []
    query_edge_attr = []
    for src, dst in query_pairs:
        if src in node_id_map and dst in node_id_map:
            si, di = node_id_map[src], node_id_map[dst]
            query_idx.append([si, di])
            e = next((e for e in req.edges if (e.src == src and e.dst == dst) or (e.src == dst and e.dst == src)), None)
            if e:
                query_edge_attr.append([
                    e.distance, e.rssi, e.snr, e.delay,
                    e.packet_loss, e.relative_speed, e.throughput,
                ])
            else:
                query_edge_attr.append([0.0] * EDGE_IN)

    edge_label_index = torch.tensor(query_idx, dtype=torch.long).T
    labeled_edge_attr = torch.tensor(query_edge_attr, dtype=torch.float32)

    with torch.no_grad():
        logits = _model(x, edge_index, edge_attr, edge_label_index, labeled_edge_attr)
        scores = torch.sigmoid(logits).numpy()

    predictions = []
    for i, (src, dst) in enumerate(query_pairs):
        if src in node_id_map and dst in node_id_map:
            score = float(scores[i])
            predictions.append(EdgePrediction(
                src=src, dst=dst,
                stability_score=score,
                stable=score >= _threshold,
                routing_weight=1.0 - score,
            ))

    return PredictionResponse(
        model_id=_model_id, threshold=_threshold, predictions=predictions
    )
