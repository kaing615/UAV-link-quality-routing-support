from __future__ import annotations
from pydantic import BaseModel, Field

class NodeFeatures(BaseModel):
    node_id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    speed: float
    degree: int

class EdgeFeatures(BaseModel):
    src: int
    dst: int
    distance: float
    rssi: float
    snr: float
    delay: float
    packet_loss: float
    relative_speed: float
    throughput: float

class PredictionRequest(BaseModel):
    nodes: list[NodeFeatures]
    edges: list[EdgeFeatures]
    query_edges: list[tuple[int, int]] = Field(default=[], description='Edges to predict (src, dst). Empty = predict all.')

class EdgePrediction(BaseModel):
    src: int
    dst: int
    stability_score: float = Field(ge=0.0, le=1.0)
    stable: bool
    routing_weight: float = Field(description='w = 1 - stability_score, for Dijkstra shortest path')

class PredictionResponse(BaseModel):
    model_id: str
    threshold: float
    predictions: list[EdgePrediction]

class HealthResponse(BaseModel):
    status: str
    model_id: str
    model_loaded: bool