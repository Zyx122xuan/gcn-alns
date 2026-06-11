"""GCN-guided ALNS (Spec sec.15, 5.5).

Subclasses :class:`ALNS` and injects a learned ``gcn_score(i, k)`` preference
into the insertion-cost evaluation (the ``- w6 * GCNScore`` term). Everything
else - destroy/repair operators, simulated-annealing acceptance, and the exact
recovery verification gate - is inherited unchanged, so feasibility is still
judged by the *same* verifier as every other method.

The trained model is loaded lazily (once) from ``cfg["gcn"]["model_path"]`` or
an explicit ``model_path``. The GCN inference time is recorded in
``self.gcn_infer_time`` for the metrics pipeline. If the model is missing or
torch is unavailable, it degrades gracefully to plain ALNS.
"""
from __future__ import annotations

import time
from pathlib import Path

from ..config import Config
from .alns.solver import ALNS


class GCNALNS(ALNS):
    name = "GCN-ALNS"
    use_gcn = True

    def __init__(self, cfg: Config, seed: int = 0, ablation: dict | None = None,
                 model_path: str | None = None):
        super().__init__(cfg, seed, ablation)
        self.model_path = model_path or cfg.get("gcn.model_path", "results/gcn_model.pt")
        self._runner = None
        self.gcn_infer_time: float = 0.0

    def _load_runner(self):
        if self._runner is not None:
            return self._runner
        from ..gcn.infer import GCNScorerRunner
        self._runner = GCNScorerRunner.load(self.model_path, self.cfg)
        return self._runner

    def _gcn_hook(self, instance, factory):
        # ablation switch: w/o GCN -> behave as plain ALNS
        if self.ablation.get("gcn", True) is False:
            return None
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"GCN model not found at {path}; train it first via "
                f"`python -m gcnalns.gcn.train` (or run plain ALNS)."
            )
        t0 = time.perf_counter()
        runner = self._load_runner()
        hook = runner.make_hook(instance, factory)
        self.gcn_infer_time = time.perf_counter() - t0
        return hook
