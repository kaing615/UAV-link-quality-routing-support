from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

Float32 = Annotated[float, Field(allow_inf_nan=False, ge=-3.4028234663852886e38, le=3.4028234663852886e38)]
NonnegativeFloat32 = Annotated[Float32, Field(ge=0)]


class NodeFeatures(BaseModel):
    node_id: int = Field(ge=0)
    x: Float32
    y: Float32
    z: Float32
    vx: Float32
    vy: Float32
    vz: Float32
    degree: int = Field(ge=0, le=int(3.4028234663852886e38))
    load: NonnegativeFloat32


class EdgeFeatures(BaseModel):
    src: int = Field(ge=0)
    dst: int = Field(ge=0)
    distance: NonnegativeFloat32
    rssi: Float32
    snr: Float32
    delay: NonnegativeFloat32
    packet_loss: float = Field(allow_inf_nan=False, ge=0, le=1)
    relative_speed: NonnegativeFloat32
    throughput: NonnegativeFloat32


class PredictionRequest(BaseModel):
    nodes: list[NodeFeatures]
    edges: list[EdgeFeatures]
    query_edges: list[tuple[int, int]] = Field(
        default=[], description="Edges to predict (src, dst). Empty = predict all."
    )

    @model_validator(mode="after")
    def validate_node_references(self):
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate node IDs")
        known = set(ids)
        if any(edge.src not in known or edge.dst not in known for edge in self.edges) or any(
            src not in known or dst not in known for src, dst in self.query_edges
        ):
            raise ValueError("Edge references an unknown node")
        return self


class EdgePrediction(BaseModel):
    src: int
    dst: int
    stability_score: float = Field(ge=0.0, le=1.0)
    stable: bool
    routing_weight: float = Field(description="w = 1 - stability_score, for Dijkstra shortest path")


class PredictionResponse(BaseModel):
    model_id: str
    threshold: float
    predictions: list[EdgePrediction]


class HealthResponse(BaseModel):
    status: str
    model_id: str
    model_loaded: bool
