# GCN-ALNS: Recovery-Latency-Constrained Hybrid Fault-Tolerant Scheduling

Reference implementation of **Recovery-Latency-Constrained Hybrid Fault-Tolerant
Scheduling for DAG Workflows in the Computing Continuum (RLC-HFTSP)** and the
**GCN-ALNS** solver, together with the full experimental pipeline (baselines,
ablations, sensitivity and scalability studies).

The system jointly decides, for every DAG task: the deployment node(s), the
fault-tolerance mode (active backup vs. passive recompute), active-replica
placement, and the post-fault dynamic rescheduling policy, so that the
worst-case recovery delay of any controllable single-point fault never exceeds
the SLA effective recovery budget `Delta_bar(chi)`, while minimizing total cost.

## Layout

```
config/default.yaml          all parameters & seeds (single source of truth)
src/gcnalns/
  config.py                  config loader
  core/                      data model, DAG analysis, schedule eval, fault set, recovery verifier, cost
  data/                      synthetic generators, Pegasus/Alibaba parsers, layering, links, SLA, instance builder
  algorithms/                HEFT/PEFT + baselines + ALNS + GCN-ALNS (all share one recovery verifier)
  gcn/                       feature extraction, dataset, GCN model, training, inference
  eval/                      metrics + CSV aggregation
  experiments/              orchestration (main / ablation / sensitivity / scalability)
  viz/                       plotting (reads CSV only)
tests/                       recovery-verifier golden-case tests
tools/                       offline analysis, report, and debug scripts
scripts/                     cost-fix rerun drivers (run from anywhere; they cd to repo root)
docs/                        progress notes and handoff writeups
data/{raw,instances}         raw datasets & generated instance JSON
results/{raw_json,csv,figures,logs}
```

## Install

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# torch / torch-geometric: install the build matching your platform first, e.g.
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install torch-geometric
pip install -e .
```

> The scheduling pipeline (data, baselines, recovery verifier, ALNS, metrics,
> plots) runs **without** torch. Only GCN training / GCN-ALNS require
> torch + torch-geometric.

## Datasets

- **Workflows (DAG input):** Pegasus scientific workflows
  (Montage, CyberShake, Epigenomics, LIGO, SIPHT) via
  [WfCommons / WorkflowHub](https://wfcommons.org). Place JSON under `data/raw/workflows/`.
- **Resources:** [Alibaba cluster-trace-v2018](https://github.com/alibaba/clusterdata)
  `machine_meta` / `machine_usage`. Place under `data/raw/alibaba/`.

Until real data is present, synthetic generators (`gcnalns.data.synthetic`)
produce inputs with the same schema, so the entire pipeline is runnable
end-to-end immediately.

## Quick start

```bash
# 1. generate experiment instances (synthetic by default)
python -m gcnalns.data.instance_builder --out data/instances --n 20

# 2. run baselines + ALNS on the instances, write raw JSON + CSV
python -m gcnalns.experiments.run_main --instances data/instances --out results

# 3. (optional) build GCN training data, train, run GCN-ALNS
python -m gcnalns.gcn.dataset --instances data/instances --out data/gcn
python -m gcnalns.gcn.train --data data/gcn --out results/gcn_model.pt

# 4. plots
python -m gcnalns.viz.plots --csv results/csv --out results/figures

# 5. interactive dashboard (filters the runs in results/csv/runs.csv)
streamlit run src/gcnalns/viz/dashboard.py
```

## Correctness gate

Per the experiment spec, the **recovery verifier** is implemented and unit
tested first; every algorithm calls the *same* verifier so SLA feasibility is
judged identically across methods. See `tests/test_recovery_verifier.py`.

## Independent evaluation

The frozen protocol in `data/protocols/independent_v1/protocol.json` separates
60 learning inputs, 15 validation inputs and 90 newly generated solver-test
instances. The GCN teacher sees only train/val; infeasible solver-test instances
are retained. Train a new model with auditable provenance before using `test`:

```powershell
$env:PYTHONPATH="src"
python -m gcnalns.gcn.dataset --protocol data/protocols/independent_v1/protocol.json --out data/gcn_independent_v1 --workers 8
python -m gcnalns.gcn.train --data data/gcn_independent_v1 --out results/independent_v1/gcn_model.pt --device cuda:0
python -m gcnalns.experiments.run_main --protocol data/protocols/independent_v1/protocol.json --split val --out results/independent_v1/validation --model-path results/independent_v1/gcn_model.pt --algorithms ALNS GCN-ALNS --workers 8
# After fixing the model and parameters:
python -m gcnalns.experiments.run_main --protocol data/protocols/independent_v1/protocol.json --split test --out results/independent_v1/test --model-path results/independent_v1/gcn_model.pt --workers 8
```

Each new experiment stores a version manifest and deployment snapshots. Resuming
requires the same inputs, configuration, model, code and study parameters.
Historical `results/raw_json` records lack those manifests; use a fresh output
directory for new runs. The earlier quick-start commands describe development
runs, not held-out evaluation. Full details: [evaluation handoff](docs/INDEPENDENT_EVALUATION_HANDOFF.md).
