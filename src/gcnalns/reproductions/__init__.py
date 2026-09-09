"""Static, fault-tolerant adaptations of external (CCF-B) scheduling papers.

These are *not* bit-exact reproductions. Each paper's online / uncertainty /
energy / elastic-VM machinery is stripped away; only its **node-selection +
fault-tolerance decision logic** is reproduced, producing a
:class:`~gcnalns.core.model.Solution` that is then judged by the SAME
``core.recovery.verify`` gate as GCN-ALNS / ALNS / the built-in baselines.

Baselines:

* :class:`~gcnalns.reproductions.recr_d.RECRD`      -- RECR-D (IEEE TSC 2025, CCF-B)
* :class:`~gcnalns.reproductions.hfwpf.HFWPF`        -- Hybrid-FT static core (IEEE TSC 2026, CCF-B)
* :class:`~gcnalns.reproductions.ga_placement.GAPlacement` -- GA evolutionary reference (MTGP stand-in)
* :class:`~gcnalns.reproductions.ra_moma.RAMOMA`     -- RA-MOMA (IEEE TPDS 2023, CCF-A)

See ``README.md`` in this package for the per-paper cropping notes.
"""
from __future__ import annotations

__all__ = ["RECRD", "HFWPF", "GAPlacement", "RAMOMA"]


def __getattr__(name: str):  # lazy re-export, avoids eager imports
    if name == "RECRD":
        from .recr_d import RECRD
        return RECRD
    if name == "HFWPF":
        from .hfwpf import HFWPF
        return HFWPF
    if name == "GAPlacement":
        from .ga_placement import GAPlacement
        return GAPlacement
    if name == "RAMOMA":
        from .ra_moma import RAMOMA
        return RAMOMA
    raise AttributeError(name)
