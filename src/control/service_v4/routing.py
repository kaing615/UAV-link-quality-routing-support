"""Link-quality-aware route override (v4), pure and causal.

Each second the controller sees the measured probe delivery matrix rx[a][b]
and the route live OLSR would use. It pins the data route to a relay path when
that path's ETX (De Couto et al., 2003) beats OLSR's current choice by a
margin, with a minimum dwell so the override does not itself cause churn.
Constants are fixed by docs/service_v4_protocol.md; only the margin is tuned.
"""
from dataclasses import dataclass, field
import itertools
import math

import numpy as np

ALPHA = .5          # EWMA weight of the newest probe window
WARMUP_STEPS = 3    # observations before any override
DWELL_S = 5         # minimum seconds between voluntary route changes
BROKEN_LINK = .1    # newest raw window below this: release/replace at once (smoothing would lag ~4 s)
ETX_CAP = 1e3
MARGINS = (0., .5, 1.)


def simple_paths(n, source, destination):
    middle = [i for i in range(n) if i not in (source, destination)]
    for k in range(len(middle) + 1):
        for chosen in itertools.permutations(middle, k):
            yield (source, *chosen, destination)


class LinkQuality:
    def __init__(self):
        self.q, self.raw, self.count = None, None, 0

    def update(self, rx):
        rx = np.asarray(rx, float)
        if rx.ndim != 2 or rx.shape[0] != rx.shape[1] or not np.isfinite(rx).all() or (rx < 0).any() or (rx > 1).any():
            raise ValueError('rx_ratio must be a finite square matrix in [0, 1]')
        self.q = rx.copy() if self.q is None else ALPHA * rx + (1 - ALPHA) * self.q
        self.raw = rx
        self.count += 1

    def link_etx(self, a, b):
        return min(ETX_CAP, 1 / max(self.q[a, b] * self.q[b, a], 1 / ETX_CAP))

    def etx(self, path):
        return math.inf if not path or len(path) < 2 else sum(self.link_etx(a, b) for a, b in zip(path, path[1:]))

    def worst(self, path, raw=False):
        m = self.raw if raw else self.q
        return min(min(m[a, b], m[b, a]) for a, b in zip(path, path[1:]))

    def usable(self, path):
        return self.worst(path) >= BROKEN_LINK and self.worst(path, raw=True) >= BROKEN_LINK


@dataclass
class RouteDecider:
    n: int
    source: int
    destination: int
    margin: float
    quality: LinkQuality = field(default_factory=LinkQuality)
    override: tuple = ()
    since: int = -10**9

    def __post_init__(self):
        if not (math.isfinite(self.margin) and self.margin >= 0): raise ValueError('margin must be finite and >= 0')
        self.relay_paths = [p for p in simple_paths(self.n, self.source, self.destination) if len(p) > 2]

    def _change(self, step, route, reason):
        self.override, self.since = tuple(route), step
        return reason

    def decide(self, step, rx, olsr_route):
        """Return (route to pin, () = let OLSR decide; diagnostic)."""
        self.quality.update(rx)
        q = self.quality
        olsr = tuple(olsr_route or ())
        olsr_etx = q.etx(olsr) if len(olsr) >= 2 else math.inf
        best = min(self.relay_paths, key=lambda p: (q.etx(p), p)) if self.relay_paths else None
        best_etx = q.etx(best) if best else math.inf
        viable = best is not None and q.usable(best)
        if q.count < WARMUP_STEPS:
            reason = 'warmup'
        elif not self.override:
            reason = (self._change(step, best, 'engage') if viable and best_etx + self.margin < olsr_etx else 'olsr')
        else:
            current_etx = q.etx(self.override)
            broken = q.worst(self.override, raw=True) < BROKEN_LINK
            if broken:
                better = viable and best != self.override and best_etx + self.margin < olsr_etx
                reason = self._change(step, best if better else (), 'replace_broken' if better else 'release_broken')
            elif step - self.since < DWELL_S:
                reason = 'dwell'
            elif olsr_etx + self.margin < current_etx:
                reason = self._change(step, (), 'release')
            elif viable and best != self.override and best_etx + self.margin < current_etx:
                reason = self._change(step, best, 'switch')
            else:
                reason = 'keep'
        finite = lambda v: None if not math.isfinite(v) else float(v)
        return self.override, dict(reason=reason, route=list(self.override), olsr_route=list(olsr),
                                   olsr_etx=finite(olsr_etx), best_relay_path=list(best) if best else None,
                                   best_relay_etx=finite(best_etx), override_etx=finite(q.etx(self.override)),
                                   margin=self.margin)
