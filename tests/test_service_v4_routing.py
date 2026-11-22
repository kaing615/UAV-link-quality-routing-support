import numpy as np
import pytest

from src.control.service_v4 import routing as r


def matrix(direct, relay=1.):
    """4 nodes, source 0, destination 3: chain 0-1-2-3 with `relay` links, direct link 0-3."""
    m = np.zeros((4, 4))
    for a, b in ((0, 1), (1, 2), (2, 3)):
        m[a, b] = m[b, a] = relay
    m[0, 3] = m[3, 0] = direct
    return m


def run(decider, steps, rx, olsr):
    return [decider.decide(t, rx(t) if callable(rx) else rx, olsr(t) if callable(olsr) else olsr) for t in steps]


def test_paths_and_etx():
    assert sorted(r.simple_paths(4, 0, 3)) == [(0, 1, 2, 3), (0, 1, 3), (0, 2, 1, 3), (0, 2, 3), (0, 3)]
    q = r.LinkQuality(); q.update(matrix(.5))
    assert q.etx((0, 3)) == pytest.approx(4.) and q.etx((0, 1, 2, 3)) == pytest.approx(3.) and q.etx(()) == np.inf


def test_weak_direct_link_engages_relay_path_after_warmup():
    d = r.RouteDecider(4, 0, 3, margin=.5)
    out = run(d, range(5), matrix(.4), [0, 3])           # direct ETX 6.25 vs relay 3.0
    assert [o[1]['reason'] for o in out] == ['warmup', 'warmup', 'engage', 'dwell', 'dwell']
    assert out[2][0] == (0, 1, 2, 3)


def test_good_direct_link_or_good_olsr_relay_keeps_olsr():
    assert run(r.RouteDecider(4, 0, 3, .5), range(6), matrix(.9), [0, 3])[-1][0] == ()      # 1.23 < 3
    assert run(r.RouteDecider(4, 0, 3, 0.), range(6), matrix(.1), [0, 1, 2, 3])[-1][0] == ()  # OLSR already on it


def test_no_olsr_route_engages_viable_relay_only():
    assert run(r.RouteDecider(4, 0, 3, 1.), range(4), matrix(0.), [])[-1][0] == (0, 1, 2, 3)
    assert run(r.RouteDecider(4, 0, 3, 1.), range(4), matrix(0., relay=.05), [])[-1][0] == ()


def test_release_after_dwell_and_immediately_when_broken():
    d = r.RouteDecider(4, 0, 3, .5)
    run(d, range(3), matrix(.4), [0, 3])                  # engaged at step 2
    out = run(d, range(3, 12), matrix(1.), [0, 3])         # direct recovers: ETX 1 + .5 < 3
    reasons = [o[1]['reason'] for o in out]
    assert reasons[:4] == ['dwell'] * 4 and reasons[4] == 'release' and out[-1][0] == ()
    d = r.RouteDecider(4, 0, 3, .5)
    run(d, range(3), matrix(.4), [0, 3])
    broken = matrix(.4, relay=0.)                          # every relay link fails
    out = run(d, range(3, 6), broken, [0, 3])
    assert out[0][1]['reason'] == 'release_broken' and out[0][0] == ()


def test_invalid_inputs():
    with pytest.raises(ValueError):
        r.RouteDecider(4, 0, 3, margin=-1)
    with pytest.raises(ValueError):
        r.RouteDecider(4, 0, 3, margin=0).decide(0, [[2.]], [])
