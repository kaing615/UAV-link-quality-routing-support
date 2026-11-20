"""Evidence-only online service feasibility assessment (contract v1).

The assessment consumes observations from the current horizon/frame only.  It
does not infer packet identity, predict future connectivity, tune radio/load,
or alter routing.  ``service_ok`` is deliberately strict: a connected probe
graph alone is never treated as delivered application service.
"""

from __future__ import annotations

from typing import Any


SERVICE_FEASIBILITY_CONTRACT_VERSION = "service-feasibility-v1"


def assess_service_feasibility(
    *,
    route_found_rate: float,
    pdr: float,
    probe_connected_windows: int,
    snapshots: int,
    arp_cache_retries: int = 0,
) -> dict[str, Any]:
    """Classify the currently observed service seam conservatively.

    ``route_found_rate`` and ``pdr`` are measured values for the same
    observation horizon.  Probe connectivity is only supporting evidence.
    The result is a serialisable, versioned contract for a controller or
    evaluator; no future samples are consulted.
    Count fields accept native Python integers only (not bool, float, Fraction
    or integer subclasses), preserving exact values and JSON serialization.
    """
    values = (route_found_rate, pdr)
    if any(not isinstance(v, (int, float)) for v in values):
        raise TypeError("route_found_rate and pdr must be numeric")
    if not 0.0 <= float(route_found_rate) <= 1.0:
        raise ValueError("route_found_rate must be in [0,1]")
    if not 0.0 <= float(pdr) <= 1.0:
        raise ValueError("pdr must be in [0,1]")
    def validate_count(name: str, value: Any) -> int:
        if type(value) is not int:
            raise TypeError(f"{name} must be a native non-negative integer")
        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    probe_connected_windows = validate_count("probe_connected_windows", probe_connected_windows)
    snapshots = validate_count("snapshots", snapshots)
    arp_cache_retries = validate_count("arp_cache_retries", arp_cache_retries)
    if probe_connected_windows > snapshots:
        raise ValueError("invalid probe window counts")

    probe_complete = snapshots > 0 and probe_connected_windows == snapshots
    reasons: list[str] = []
    if not probe_complete:
        reasons.append("probe_graph_not_connected_for_all_windows")
    if route_found_rate < 1.0:
        reasons.append("route_traversal_not_available_for_all_windows")
    if pdr == 0.0:
        reasons.append("application_delivery_zero")
    elif pdr < 1.0:
        reasons.append("application_delivery_degraded")
    if arp_cache_retries:
        reasons.append("arp_cache_retry_exhaustion_observed")

    if not probe_complete:
        status = "probe_disconnected"
    elif route_found_rate < 1.0:
        status = "route_unavailable"
    elif pdr == 0.0 and arp_cache_retries:
        status = "forwarding_unavailable"
    elif pdr < 1.0:
        status = "service_degraded"
    else:
        status = "service_ok"

    return {
        "contract_version": SERVICE_FEASIBILITY_CONTRACT_VERSION,
        "status": status,
        "service_ok": status == "service_ok",
        "observed": {
            "route_found_rate": float(route_found_rate),
            "pdr": float(pdr),
            "probe_connected_windows": probe_connected_windows,
            "snapshots": snapshots,
            "arp_cache_retries": arp_cache_retries,
        },
        "reasons": reasons,
        "future_data_used": False,
    }
