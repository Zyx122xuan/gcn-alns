"""GCN-guided ALNS (Spec sec.15, 5.5).

Subclasses :class:`ALNS` and injects the learned guidance at several points:

* placement re-ranking - the ``- w6 * GCNScore`` term inside InsertCost;
* warm start (A1)      - a GCN-guided greedy construction seeds the search;
* FT-mode / replicas (A2) - a learned mode head can request active backups and
  bias replica placement (only when the loaded model has a mode head);
* directed destroy (A3) - the ``low_gcn`` removal operator (enabled in the base
  solver when a GCN score hook is present).

Everything downstream - simulated-annealing acceptance and the exact recovery
verification gate - is inherited unchanged, so feasibility is still judged by
the *same* verifier as every other method.

The trained model is loaded lazily (once) from ``cfg["gcn"]["model_path"]`` or
an explicit ``model_path``. GCN inference time is recorded in
``self.gcn_infer_time`` for the metrics pipeline.
"""
from __future__ import annotations

import time
from pathlib import Path

from ..config import Config
from ..core.model import Solution
from ..core.recovery import effective_budget
from ..core.schedule_eval import compute_schedule
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
        # cached per-instance hooks (built once by :meth:`_ensure_hooks`)
        self._hook_key = None
        self._score_hook = None
        self._mode_hook = None

    def _load_runner(self):
        if self._runner is not None:
            return self._runner
        from ..gcn.infer import GCNScorerRunner
        self._runner = GCNScorerRunner.load(self.model_path, self.cfg)
        return self._runner

    def _gcn_disabled(self) -> bool:
        return self.ablation.get("gcn", True) is False

    def _ensure_hooks(self, instance, factory) -> None:
        """Build the score (and optional mode) hooks once per instance."""
        if self._hook_key == id(instance):
            return
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"GCN model not found at {path}; train it first via "
                f"`python -m gcnalns.gcn.train` (or run plain ALNS)."
            )
        t0 = time.perf_counter()
        runner = self._load_runner()
        score_hook, mode_hook = runner.make_hooks(instance, factory)
        self.gcn_infer_time = time.perf_counter() - t0
        self._score_hook = score_hook
        self._mode_hook = mode_hook
        self._hook_key = id(instance)

    def _gcn_hook(self, instance, factory):
        if self._gcn_disabled():
            return None
        self._ensure_hooks(instance, factory)
        return self._score_hook

    def _gcn_mode_hook(self, instance, factory):
        # requires the learned guidance on, the mode switch on, and a model that
        # actually has a trained mode head (old checkpoints degrade gracefully).
        if self._gcn_disabled() or self.ablation.get("gcn_mode", True) is False:
            return None
        if not Path(self.model_path).exists():
            return None
        self._ensure_hooks(instance, factory)
        return self._mode_hook

    # -- warm start (A1) ----------------------------------------------------
    def _candidate_inits(self, instance, factory, gcn_score, gcn_mode):
        inits = super()._candidate_inits(instance, factory, gcn_score, gcn_mode)
        if gcn_score is not None and self.ablation.get("warm_start", True):
            gsol = self._gcn_greedy(instance, factory, gcn_score, gcn_mode)
            if gsol is not None:
                inits.insert(0, gsol)
        return inits

    def _gcn_greedy(self, instance, factory, gcn_score, gcn_mode) -> Solution | None:
        """Construct a full solution by GCN-guided greedy insertion over all
        tasks (reuses the repair engine so placement / FT-mode / replicas are
        decided exactly as during search)."""
        from .alns.insertion import Repairer
        from .alns.operators import _repair_core, _score_insert
        try:
            from .heft import HEFT
            heft = HEFT(self.cfg).schedule(instance)
            sched = compute_schedule(instance, heft, factory, self.cfg)
            cbase = min(sched.cmax, instance.deadline)
        except Exception:
            cbase = instance.deadline
        budget = max(effective_budget(instance, cbase), 0.0)
        repairer = Repairer(instance, factory, self.cfg, budget, gcn_score,
                            ablation=self.ablation, gcn_mode=gcn_mode)
        removed = list(instance.workflow.tasks.keys())
        sol = _repair_core(instance, Solution(), removed, repairer,
                           _score_insert, self.m_max)
        return sol
