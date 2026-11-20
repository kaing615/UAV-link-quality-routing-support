import json
from fractions import Fraction

import pytest

from src.routing.service_feasibility import assess_service_feasibility


def test_probe_connected_zero_service_is_not_ok():
    """Illustrative contract boundary, not a literal replay of the .004267 case.

    The route0/PDR0 inputs exercise the no-route branch; they are not copied
    from static_n8_s0.55_r256_seed71001, which received 8/1875 packets.
    """
    result = assess_service_feasibility(
        route_found_rate=0.0, pdr=0.0,
        probe_connected_windows=31, snapshots=31, arp_cache_retries=3,
    )
    assert result["contract_version"] == "service-feasibility-v1"
    assert result["status"] == "route_unavailable"
    assert not result["service_ok"]
    assert result["future_data_used"] is False


def test_route_present_zero_pdr_identifies_forwarding_seam():
    """Minimized historical static_n2_s0.55_r256_seed71001/summary.json.

    That case has 31/31 probe/route success, zero of 1875 packets received,
    and three ARP-cache retry observations. The separate n8/.55/r256/71001
    degradation example has PDR .004267 and is not this zero-PDR fixture.
    """
    result = assess_service_feasibility(
        route_found_rate=1.0,
        pdr=0.0,
        probe_connected_windows=31,
        snapshots=31,
        arp_cache_retries=3,
    )
    assert result["status"] == "forwarding_unavailable"
    assert "arp_cache_retry_exhaustion_observed" in result["reasons"]


@pytest.mark.parametrize("field", ["snapshots", "probe_connected_windows", "arp_cache_retries"])
@pytest.mark.parametrize("value", [True, False, 31.0, 1.5, float("nan"), float("inf"),
                                   Fraction(31, 1), Fraction(18014398509481985, 18014398509481984)])
def test_only_native_integer_counts_are_supported(field, value):
    base = dict(route_found_rate=1.0, pdr=1.0, probe_connected_windows=31,
                snapshots=31, arp_cache_retries=0)
    with pytest.raises(TypeError):
        assess_service_feasibility(**{**base, field: value})


@pytest.mark.parametrize("count", [0, 31, 2**100])
def test_supported_counts_roundtrip_json_exactly(count):
    result = assess_service_feasibility(
        route_found_rate=1.0, pdr=1.0, probe_connected_windows=count,
        snapshots=count, arp_cache_retries=count)
    decoded = json.loads(json.dumps(result, allow_nan=False))
    assert decoded == result
    for field in ("snapshots", "probe_connected_windows", "arp_cache_retries"):
        assert type(decoded["observed"][field]) is int
        assert decoded["observed"][field] == count


def test_invalid_observation_is_rejected():
    try:
        assess_service_feasibility(
            route_found_rate=1.1,
            pdr=0.0,
            probe_connected_windows=1,
            snapshots=1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range route rate must be rejected")


def test_count_observations_reject_fractional_bool_and_nonfinite_values():
    base = dict(route_found_rate=1.0, pdr=1.0, probe_connected_windows=1,
                snapshots=1, arp_cache_retries=0)
    for key, value in (("snapshots", 1.5), ("probe_connected_windows", True),
                       ("arp_cache_retries", float("nan"))):
        try:
            assess_service_feasibility(**{**base, key: value})
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid count {key}={value!r} was accepted")


def test_integral_observations_are_preserved_without_truncation():
    result = assess_service_feasibility(
        route_found_rate=1.0, pdr=0.0, probe_connected_windows=31,
        snapshots=31, arp_cache_retries=3,
    )
    assert result["observed"] == {
        "route_found_rate": 1.0, "pdr": 0.0,
        "probe_connected_windows": 31, "snapshots": 31,
        "arp_cache_retries": 3,
    }
    assert result["status"] == "forwarding_unavailable"
