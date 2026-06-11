"""Generate link parameters between nodes from layer relations (Spec sec.6).

Bandwidth / latency / failure-rate base values come from the per-layer-pair
table in the config; each directed link gets a fixed U(0.8, 1.2) perturbation
(seed from config) so identical layer pairs are not perfectly identical.

To avoid materialising a full O(N^2) link set for large node pools, links are
generated lazily on demand and cached. ``build_links`` returns a
:class:`LinkFactory` whose ``get(k, l)`` produces a deterministic Link.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..core.model import Continuum, Link, Node


class LinkFactory:
    """Deterministic, lazily-evaluated link parameter provider."""

    def __init__(self, nodes: dict[int, Node], cfg: Config):
        self.nodes = nodes
        self.cfg = cfg
        lk = cfg["links"]
        self.local_latency = float(lk["local_latency"])
        self.local_bw = float(lk["local_bandwidth"])
        self.table = lk["table"]
        self.low = float(lk["perturb_low"])
        self.high = float(lk["perturb_high"])
        self.seed = int(lk["perturb_seed"])
        self._cache: dict[tuple[int, int], Link] = {}

    def _key(self, la: str, lb: str) -> str:
        a, b = sorted([la, lb])
        return f"{a}-{b}"

    def _perturb(self, k: int, l: int) -> tuple[float, float]:
        # Stable per-pair RNG keyed by sorted node ids + global seed.
        a, b = sorted([k, l])
        rng = np.random.default_rng((self.seed, a, b))
        eta = float(rng.uniform(self.low, self.high))   # bandwidth factor
        xi = float(rng.uniform(self.low, self.high))    # latency factor
        return eta, xi

    def get(self, k: int, l: int) -> Link | None:
        if k == l:
            return None
        key = (min(k, l), max(k, l))
        if key in self._cache:
            return self._cache[key]
        la = self.nodes[k].layer
        lb = self.nodes[l].layer
        base = self.table[self._key(la, lb)]
        eta, xi = self._perturb(k, l)
        link = Link(
            src=key[0],
            dst=key[1],
            bandwidth=float(base["bw"]) * eta,
            latency=float(base["lat"]) * xi,
            failure_rate=float(base["fail"]),
            cost_coef=0.0,
        )
        self._cache[key] = link
        return link

    def comm_delay(self, k: int, l: int, data_mb: float) -> float:
        """comm_ij^{kl} = lat + d/bw  (0 when k == l up to local latency)."""
        if k == l:
            return self.local_latency
        link = self.get(k, l)
        assert link is not None
        return link.latency + data_mb / link.bandwidth


def get_factory(instance, cfg: Config) -> LinkFactory:
    """Rebuild a deterministic LinkFactory for an instance's continuum.

    Link parameters are a pure function of node ids/layers + config seed, so the
    factory can be reconstructed at any time (e.g. after loading from JSON).
    """
    return LinkFactory(instance.continuum.nodes, cfg)


def build_continuum(nodes: dict[int, Node], cfg: Config,
                    materialize: bool = False) -> tuple[Continuum, LinkFactory]:
    """Build a Continuum plus its LinkFactory.

    When ``materialize`` is True (small instances), all pairwise links are
    pre-computed and stored in ``Continuum.links``; otherwise links are
    produced lazily through the factory.
    """
    factory = LinkFactory(nodes, cfg)
    links: dict[tuple[int, int], Link] = {}
    if materialize:
        ids = list(nodes.keys())
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                k, l = ids[a], ids[b]
                link = factory.get(k, l)
                if link is not None:
                    links[(k, l)] = link
    return Continuum(nodes=nodes, links=links), factory
