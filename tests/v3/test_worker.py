"""Tests for Worker — claims todo items and executes interventions."""
import pytest
from autoresearcher2.v3.workspace import Workspace
from autoresearcher2.v3.worker import Worker
from autoresearcher2.v3.proposal import Proposal


@pytest.fixture
def ws(tmp_path):
    workspace = Workspace(tmp_path / "research")
    workspace.init()
    return workspace


def _add_todo(ws, intent="test experiment", rank=1):
    p = Proposal(
        intent=intent, rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    p.set_critic_decision("accept", rank=rank, rationale="ok")
    p.promote("todo")
    ws.save_proposal(p)
    return p


def test_worker_claims_and_executes(ws):
    _add_todo(ws)
    execute_called = []

    def mock_execute(proposal):
        execute_called.append(proposal.id)
        return {"metrics": {"val_bpb": 1.5}, "compute_cost": 0.50}

    worker = Worker(ws, execute_fn=mock_execute)
    result = worker.tick()

    assert result is not None
    assert result["outcome_success"] is True
    assert result["outcome_metrics"]["val_bpb"] == 1.5
    assert len(execute_called) == 1
    assert ws.count_proposals("done") == 1
    assert ws.count_proposals("todo") == 0


def test_worker_returns_none_when_empty(ws):
    worker = Worker(ws, execute_fn=lambda p: {})
    result = worker.tick()
    assert result is None


def test_worker_handles_execution_failure(ws):
    _add_todo(ws)

    def failing_execute(proposal):
        raise RuntimeError("GPU out of memory")

    worker = Worker(ws, execute_fn=failing_execute)
    result = worker.tick()

    assert result is not None
    assert result["outcome_success"] is False
    assert "GPU out of memory" in result["error"]
    assert ws.count_proposals("done") == 1


def test_worker_executes_highest_priority_first(ws):
    _add_todo(ws, intent="low priority", rank=5)
    _add_todo(ws, intent="high priority", rank=1)

    executed = []

    def track_execute(proposal):
        executed.append(proposal.intent)
        return {"metrics": {}}

    worker = Worker(ws, execute_fn=track_execute)
    worker.tick()

    assert executed[0] == "high priority"


def test_worker_run_with_max_ticks(ws):
    _add_todo(ws, intent="task 1")
    _add_todo(ws, intent="task 2")

    def mock_execute(proposal):
        return {"metrics": {"done": True}}

    worker = Worker(ws, execute_fn=mock_execute)
    worker.run(poll_interval=0.01, max_ticks=3)

    # 2 tasks executed, 1 tick returns None
    assert ws.count_proposals("done") == 2
    assert ws.count_proposals("todo") == 0
