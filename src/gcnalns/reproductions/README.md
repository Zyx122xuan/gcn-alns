# External baseline reproductions

Static, fault-tolerant adaptations of three CCF-B scheduling papers, evaluated
under the **same `core.recovery.verify` gate** as GCN-ALNS / ALNS / the built-in
baselines. Each baseline only reproduces the paper's **node-selection +
fault-tolerance decision logic**, producing a `Solution(placement, mode)` that
is then scored by our shared `compute_schedule -> verify -> compute_cost`
pipeline.

> These are **not** bit-exact reproductions. Every paper's online /
> uncertainty / energy / elastic-VM machinery is intentionally stripped so all
> methods face the identical *static, single-instance, deterministic-exec*
> problem and the identical feasibility gate. The tables below list exactly what
> was dropped or re-mapped. Report this honestly in `results/REPORT.md`.

All parameters live in `config/default.yaml` under `reproductions:`; nothing is
hard-coded. Register/run via `algorithms/registry.py` (`RECR-D`, `HFWPF`,
`GA-Placement`) and `experiments/run_main.py`.

---

## RECR-D (`recr_d.py`)

A. Taghinezhad-Niar, J. Taheri, *"Fault-Tolerant Cost-Efficient Scheduling for
Energy and Deadline-Constrained IoT Workflows in Edge-Cloud Continuum,"* IEEE
TSC 18(5):2892-2903, 2025. DOI:10.1109/TSC.2025.3599497.

| Paper element | Our adaptation |
|---|---|
| Alg.1 constraint distribution (EST/EFT/LST/LFT, dynamic balance) | Per-task sub-deadlines `dl_i = D*(db*EFT+(1-db)*LFT)/cmax` using structural timing (`_shared.sub_deadlines`). |
| Alg.2 penalty `PN = zeta*cost + alpha*energy + beta*deadline + gamma*reliability` (eq.43) | Same, but **energy dropped** (`alpha_energy=0`); tasks placed in **EDF** order onto the min-`PN` resource. Norms: deadline `max(0,(EFT-dl_i)/dl_i)`, cost `MC/(MC_max+1)`, reliability `max(0, Theta - exp(-lambda_k*p_ik))`. |
| Alg.3 TRM (CTMC success-probability backup) | **Reliability口径 replaced**: a task is promoted to active backup if it misses its sub-deadline / falls below `Theta`, **or** its reactive recovery exceeds the effective budget under our `estimate_rmax` gate. "Add backup" = `add_active_replicas(m=m_max)` (fault-domain-isolated). Bounded rounds (`trm_max_rounds`). |
| Energy objective / IoT device battery model | Dropped entirely. |

## HFWPF / Hybrid-FT (`hfwpf.py`)

Q. Ren, G. Yao, *"A Hybrid Fault-Tolerant Workflow Scheduling With Performance
Fluctuated Cloud Resources,"* IEEE TSC 19(1):44-57, 2026.
DOI:10.1109/TSC.2025.3635177.

| Paper element | Our adaptation |
|---|---|
| Preprocessing (sub-deadline slicing) | Shared `_shared.sub_deadlines`. |
| Initial scheduling FT choice (eq.8): `slack >= 2*(Tboot + max_pred(TT) + psi)` -> resubmission else replication | Reproduced with **deterministic** times: `psi` = min exec time over nodes, `max_pred(TT)` = max predecessor transfer (avg link). `slack < threshold` -> active (replication, 2 replicas); else passive (resubmission). |
| VM selection (elastic, unbounded VMs) | Capacity-biased earliest-finish placement on our **fixed** node set: replication -> small capacity (cost), resubmission -> large capacity (headroom). Active replicas via `add_active_replicas`. |
| `Tboot=15s` | Framework has no VM boot; `t_boot=0` (configurable). |
| Execution-time **uncertainty** (the paper's core), online re-execution, slack back-fill, replication->resubmission down-switch, elastic VM supply | **Removed** -> deterministic one-shot schedule + our recovery verifier. **This削弱s HFWPF's relative advantage vs GCN-ALNS by design; state this in the report.** |

## GA-Placement (`ga_placement.py`)

Stand-in for M. Xu et al., *"Genetic Programming for Dynamic Workflow Scheduling
in Fog Computing,"* IEEE TSC 16(4):2657-2671, 2023.
DOI:10.1109/TSC.2023.3249160. Code: github.com/MengXu95/Fog-Computing.

MTGP's original problem (dynamic arrivals, makespan minimisation, **no
fault-tolerance / no SLA feasibility**) is not directly comparable. Per the
handoff (option A), we provide a **standard GA** as an *evolutionary-metaheuristic
reference* against ALNS / GCN-ALNS, **not** a GP hyper-heuristic reproduction.

| Element | Design |
|---|---|
| Individual | per task: (node choice, active/passive bit). |
| Decode | primary node from gene; active tasks get fault-domain-isolated replicas via `add_active_replicas`. |
| Fitness | our `compute_cost` total + deadline/recovery penalties with the fast `estimate_rmax` gate (same quick objective ALNS uses); final incumbent judged by exact `verify`. |
| Operators | tournament selection, uniform crossover, per-gene mutation, elitism; optional greedy-EFT seed. |

Report wording: *"standard GA represents evolutionary metaheuristics; MTGP's
original dynamic-makespan problem cannot be compared directly."*

## RA-MOMA (`ra_moma.py`)

S. Qin, D. Pi, Z. Shao, Y. Xu, Y. Chen, *"Reliability-Aware Multi-Objective
Memetic Algorithm for Workflow Scheduling Problem in Multi-Cloud System,"* IEEE
TPDS 34(4):1343-1361, 2023. DOI:10.1109/TPDS.2023.3245089. **CCF-A.**

RA-MOMA solves MOWSP-MCS: multi-objective (makespan, cost, reliability) DAG
scheduling where each task may get **one backup copy**. It is a memetic
algorithm = NSGA-II global search + hybrid HEFT initialization + a local search
of four problem-specific neighborhood operators over the elite non-dominated
archive.

| Paper element | Our adaptation |
|---|---|
| Encoding `<alpha, beta, gamma, pi>` | Keep **alpha (primary node)** + **beta (backup node / none)**; drop **gamma** (VM resource-pool/leasing -- our continuum is a *fixed* node set) and **pi** (execution order -- our evaluator derives node order topologically, so pi has no effect). |
| backup = primary + 1 copy on a distinct resource | **active backup, m=2**, requiring **distinct fault domains** (framework isolation rule); otherwise the task stays passive. |
| Objectives (makespan, cost, reliability) | NSGA-II over **(c_base, compute_cost total, recovery-risk = `estimate_rmax`)**; minimising recovery-risk is our analogue of maximising reliability. |
| "User picks a schedule" from the Pareto front | From the final non-dominated archive we return the **min-cost solution our estimate deems SLA-feasible** (deadline + recovery budget as the user constraint); final judged by exact `verify`. |
| Hybrid init: HEFT / backup-HEFT / cheapest / cheapest+backup + random | Reproduced (HEFT via `_greedy_eft_place`; cheapest = min `cost_coef*exec`; all-backup adds distinct-domain replicas). |
| Diversification: order crossover (pi) + multi-point crossover (alpha,beta) + mutation | Multi-point (per-gene) crossover on alpha/beta + per-gene mutation (order crossover dropped with pi). Paper-calibrated `cr=0.1, mr=0.02`. |
| Intensification: CP_perturb / CP_remove / VM_release / VM_backup | Reproduced on our schedule: critical path = tasks with ~0 `sched_slack`; low-utilisation nodes from CPU footprint. |
| NSGA-II non-dominated sort + crowding distance | Reproduced for environmental selection + archive. |
| Poisson/CTMC reliability, hourly/minute multi-cloud billing, performance degradation, fuzzy times | Replaced by framework deterministic exec + `compute_cost` + `verify`. |

Report wording: *"RA-MOMA is a CCF-A reliability-aware multi-objective memetic
method; we take its (makespan, cost, reliability) Pareto search and select the
min-cost SLA-feasible schedule under our recovery gate."*

---

### Common fairness protocol

* Energy: dropped (framework has none).
* Execution-time uncertainty: replaced by deterministic exec.
* Online arrival / multi-workflow: single-instance one-shot scheduling.
* Fault primitives: replication -> active backup; resubmission/re-execution -> passive.
* Objective: each method outputs `placement + mode`; **our** `verify` judges
  feasibility and **our** `compute_cost` scores it.
