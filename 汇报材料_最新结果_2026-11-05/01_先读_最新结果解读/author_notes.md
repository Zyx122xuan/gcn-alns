# Scientific editing notes

These notes are for the authors and are not included in the compiled paper.
The LaTeX text describes the registered implementation rather than a proposed
algorithm that has not actually been evaluated.

## Agreed paper scope (2026-10-28)

The author confirmed that the mode head is not an independent research point.
Present one final GCN-ALNS variant in the manuscript; retain the original
diagnostic runs and alternative checkpoints in the development archive.
The running experiment queue is unchanged and should finish as scheduled.
The manuscript now has three research questions and no dedicated model A/B
results subsection. Its main/component/budget studies total 19,200 planned
records; the pipeline additionally retains 1,350 model A/B diagnostic records
for a total of 20,550. Keep these two scopes distinct when reporting progress.

The validation-selected A2-off checkpoint is already frozen for the main
studies. Do not substitute a checkpoint selected after inspecting held-out
results. After all runs finish, the final deliverable should expose this
single selected model, with other versions archived for reproducibility.

## Before submission

Post-run audit on 2026-11-05: see
`analysis/independent_v1/INTERPRETATION_ZH.md` in the repository. The two
legacy all-active/all-passive ablation labels do not describe globally
enforced policies. Among their feasible outputs, 0/319 satisfy strict
all-active and 210/268 satisfy strict all-passive; 58 passive-labelled
outputs still contain active tasks. The manuscript now calls these
repair-mode preference variants. Do not use them to claim hybrid superiority
over strictly homogeneous protection. A clean comparison needs targeted
new-version runs, or the narrower interpretation must be retained.

The paired primary results have been recomputed from raw records. Medium
and large matched-success cost reductions are approximately 3.00% and
2.58%; feasibility and first-feasible iteration effects remain inconclusive.
The original cost-breakdown plot double-counts informational backup cost,
and the original cost-per-task/scalability plots read mixed study tags.
Use explicitly filtered plots from the analysis directory when drafting.

1. Choose the target venue and verify its class options, author block, page
   limit, anonymity policy, reference format, and algorithm-float guidance.
2. Replace the author and affiliation placeholder. No authorship, funding,
   ethics statement, or data availability commitment has been invented.
3. Use the completed new-protocol runs. Never insert the older 52.1%/49.9%
   aggregate or the older 595/621 cost comparison into this independent study.
4. Import final tables and review effect directions scale by scale. Update the
   abstract and conclusion only after checking paired results and uncertainty.
5. Include the full baseline suite and the paired ALNS/GCN comparison. The
   initial six-row table is an authoring slot, not evidence.
6. Audit all quantitative prose after inserting results; the draft currently
   makes no claim that GCN is universally better.

## Important implementation boundaries

- The GCN operates on a resource kNN graph. Task structure is encoded through
  features and an MLP, not a workflow-DAG GCN.
- The scorer is supervised on feasible demonstrations, with a contrastive
  ranking loss. This is not reinforcement learning or optimal-solution
  supervision.
- CPU/memory limits use cumulative deployed footprint, not instantaneous
  time-sharing capacity. Nominal task order is fixed by the evaluator.
- Default faults cover deployed copies, occupied nodes, and effective
  cross-node workflow transfers at specified times. No arbitrary-time,
  simultaneous-failure, or universal physical fault guarantee is established.
- Domain separation is enforced during construction, not by a general domain
  constraint check at the final verifier. Its removal is a constraint
  relaxation, not an ordinary solver heuristic ablation.
- `w/o Dynamic Resched` restricts recovery to original domains; it does not
  remove every recovery action and it changes the evaluated recovery policy.
- The recovery-risk surrogate is not a proven upper bound, and candidate
  screening is not an exhaustive feasibility search.
- Communication cost counts all provisioned replica pairs, while readiness
  selects an effective producer. This is a modeling convention.
- The recovery charge is a scenario-average delay proxy. It is not a failure-
  probability-weighted expected monetary loss. Do not interpret the current
  average-recovery metric on unrecoverable runs as an expectation.
- All active-copy compute cost already belongs to `compute`; the informational
  `backup` field must not be added to total cost again.
- HEFT and No-FT share passive placement logic in this implementation. Do not
  call them independent evidence of superiority over two distinct algorithms.
- External RECR-D, HFWPF, and RA-MOMA configurations are static adaptations.
  GA-Placement is a conventional GA reference, not an MTGP reproduction. Do
  not claim that their source papers have been defeated on their original
  problems or that all adaptation bias has been ruled out.

## Feasible register and quick/full objectives

In `algorithms/alns/solver.py`, quick cost excludes the recovery-cost proxy.
The admission screen compares this quick cost with the stored full feasible
score; after full feasibility succeeds, the register is replaced without a
separate full-cost improvement test. The draft therefore says **verified
feasible register**, not a monotonically improving minimum-cost incumbent.
This implementation detail is explicitly represented in the search pseudocode.

Do not change the currently frozen experiment code to resolve this while the
registered rerun is active. Any future change needs a separate version and
fresh affected experiments. The same rule applies to expanded failure sets,
different capacity semantics, or changes to the cost model.

## Model selection and statistical language

- Both A2-on/off checkpoints were retrained from the new teacher dataset.
- Only 26 train and 4 validation inputs contributed labels, but all 15 original
  validation inputs were retained for 75 solver runs per model.
- A2-off was selected from validation (23/75 feasible versus 18/75 for A2-on).
  These are model-selection facts, not a significance claim or held-out result.
- The A2-off checkpoint lacks the trained mode head; it can still use learned
  pair scores in placement and replica ordering. The code switch named
  `gcn_mode` must not automatically be equated with changing only the model
  head. Check its replica-ranking branch when interpreting the ablation.
- Pair instance and seed for conditional comparisons, then use instances as
  the independent units. Keep the number of shared feasible instances visible.
- A higher conditional safety margin is not automatically a lower violation
  rate. Raw conditional cost means can compare different successful subsets.
- First-feasible iteration is not measured wall-clock time-to-feasibility.
  Final `verify_time` excludes internal search-time verification calls.
- Individually nonsignificant subcomponent effects do not prove redundancy.
  Keep negative, null, or reversed effects in the final account.

## Result sections still requiring empirical text

- RQ1: three-scale main comparison and paired ALNS/GCN effects.
- RQ2: heuristic ablations separated from policy/constraint changes.
- RQ3: full budget curves and measured solver runtime.
- Internal archive only: A2-on/A2-off/ALNS diagnostic comparison. It is not
  a manuscript research question or an independent contribution.
- Abstract and conclusion: supported magnitudes, uncertainty, and failures to
  generalize. No unfinished run should be silently promoted to a final finding.

The mathematical proposition proves only the stated finite-scenario budget
equivalence. No NP-hardness reduction, surrogate admissibility theorem,
global-optimality guarantee, or convergence theorem has been established.
