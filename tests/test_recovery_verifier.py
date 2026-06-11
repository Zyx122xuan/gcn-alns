"""Hand-computable golden cases for the recovery verifier (correctness gate).

Instances are built directly (not via synthetic generators) so all timings can
be derived by hand and asserted exactly.
"""
import math

import pytest

from gcnalns.config import load_config
from gcnalns.core.model import Continuum, Instance, Node, Solution, Task, Workflow
from gcnalns.core.recovery import simulate_fault, verify
from gcnalns.core.schedule_eval import compute_schedule
from gcnalns.core.fault_set import Fault, build_fault_set
from gcnalns.data.links import LinkFactory

CFG = load_config()
DET = CFG.get("recovery.delta_detect")      # 1.0
REPLAN = CFG.get("recovery.delta_replan")    # 0.5
SW = CFG.get("recovery.delta_switch")        # 0.5
SYNC = CFG.get("recovery.delta_sync")        # 0.5


def _node(nid, fd, compute=10.0, layer="cloud"):
    return Node(nid=nid, compute=compute, cpu_cap=100.0, mem_cap=100.0, load=0.0,
                load_var=0.0, failure_rate=0.0, cost_coef=1.0, layer=layer,
                fault_domain=fd)


def _single_task_instance(deadline=100.0, rwin=100.0):
    # one task, workload 10, two identical nodes => p = 1.0 on each
    tasks = {0: Task(tid=0, workload=10.0, resource=1.0, mem=1.0)}
    wf = Workflow(name="unit", tasks=tasks, edges=[], data={})
    nodes = {0: _node(0, "cloud-0"), 1: _node(1, "cloud-1")}
    cont = Continuum(nodes=nodes, links={})
    return Instance(iid="unit", workflow=wf, continuum=cont,
                    deadline=deadline, recovery_window=rwin)


def test_passive_single_task_recovery():
    inst = _single_task_instance()
    factory = LinkFactory(inst.continuum.nodes, CFG)
    sol = Solution(placement={0: [0]}, mode={0: False})
    base = compute_schedule(inst, sol, factory, CFG)
    assert math.isclose(base.cmax, 1.0)

    faults = build_fault_set(inst, sol, base)
    # 1 task-instance fault + 1 node fault, no links
    kinds = sorted(f.kind for f in faults)
    assert kinds == ["node", "task"]

    # task-instance fault: t_omega = 1.0; recompute floor = 1.0+det+replan = 2.5; finish = 3.5
    tf = next(f for f in faults if f.kind == "task")
    c = simulate_fault(inst, sol, base, tf, factory, CFG)
    assert math.isclose(c, 1.0 + DET + REPLAN + 1.0)  # 3.5

    # node fault: t_omega = 0 (start); recompute on other node, floor 0+det+replan=1.5; finish 2.5
    nf = next(f for f in faults if f.kind == "node")
    c = simulate_fault(inst, sol, base, nf, factory, CFG)
    assert math.isclose(c, DET + REPLAN + 1.0)  # 2.5

    res = verify(inst, sol, factory, CFG, base)
    assert math.isclose(res.r_max, 2.5)
    assert res.feasible  # 2.5 <= min(100, 99)


def test_active_backup_switch():
    inst = _single_task_instance()
    factory = LinkFactory(inst.continuum.nodes, CFG)
    sol = Solution(placement={0: [0, 1]}, mode={0: True})
    base = compute_schedule(inst, sol, factory, CFG)
    assert math.isclose(base.cmax, 1.0)  # min of two replicas, both 1.0

    res = verify(inst, sol, factory, CFG, base)
    # worst is a task/node fault: survivor floored by t_omega + det + sw + sync
    # task fault t_omega=1.0 -> floor 1.0+2.0=3.0 -> R=2.0
    assert math.isclose(res.r_max, 2.0)


def test_infeasible_when_budget_too_small():
    # tight deadline so the recompute blows the budget
    inst = _single_task_instance(deadline=1.5, rwin=1.0)
    factory = LinkFactory(inst.continuum.nodes, CFG)
    sol = Solution(placement={0: [0]}, mode={0: False})
    res = verify(inst, sol, factory, CFG)
    # r_eff = min(1.0, 1.5-1.0)=0.5 ; r_max=2.5 -> infeasible
    assert math.isclose(res.r_eff, 0.5)
    assert not res.feasible


def test_chain_propagation_passive():
    # t0 -> t1 chain; both passive, placed on n0; recompute shifts downstream.
    tasks = {0: Task(tid=0, workload=10.0), 1: Task(tid=1, workload=10.0)}
    wf = Workflow(name="chain", tasks=tasks, edges=[(0, 1)], data={(0, 1): 0.0})
    nodes = {0: _node(0, "cloud-0"), 1: _node(1, "cloud-1")}
    inst = Instance(iid="chain", workflow=wf, continuum=Continuum(nodes, {}),
                    deadline=100.0, recovery_window=100.0)
    factory = LinkFactory(nodes, CFG)
    local = CFG.get("links.local_latency")  # 0.001s same-node access latency
    sol = Solution(placement={0: [0], 1: [0]}, mode={0: False, 1: False})
    base = compute_schedule(inst, sol, factory, CFG)
    # n0 serial: t0 [0,1], t1 [1+local, 2+local] => cmax 2 + local
    assert math.isclose(base.cmax, 2.0 + local)

    # fault on t0 instance: t_omega=1.0; t0 recompute finish = max(0,1.0+det+replan)+1 = 3.5
    # then t1 (passive, unaffected) reruns after t0: start = 3.5 + local, finish = 4.5 + local
    tf = Fault(kind="task", i=0, k=0)
    c = simulate_fault(inst, sol, base, tf, factory, CFG)
    assert c > base.cmax  # downstream propagation occurred
    assert math.isclose(c, 4.5 + local)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
