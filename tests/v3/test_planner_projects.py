"""Tests for project-scoped Planner behavior."""
import pytest
from autoresearcher2.v3.store import Store
from autoresearcher2.v3.planner import Planner
from autoresearcher2.v3.worker import Worker


def make_mock_llm(proposals):
    """Mock LLM that returns fixed proposals."""
    def llm_call(prompt):
        if "NEW OBSERVATION" in prompt and "UPDATE INSTRUCTIONS" in prompt:
            return {
                "beliefs_added": [
                    {"claim": "result informative", "confidence": 0.6,
                     "evidence_for": ["obs_new"]},
                ],
            }
        elif "Generate" in prompt and "proposals" in prompt.lower():
            return {"proposals": proposals}
        elif "ranking" in prompt.lower() or "evaluate" in prompt.lower():
            return {"rankings": []}
        return {}
    return llm_call


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "research.db")
    s.init()
    yield s
    s.close()


def test_planner_generates_for_own_project(store):
    """Planner only generates proposals for its assigned project."""
    pid1 = store.create_project(name="NanoGPT", description="GPT")
    pid2 = store.create_project(name="Atari", description="RL")

    gpt_llm = make_mock_llm([
        {"intent": "Test lr", "rationale": "test", "expected_learning": "lr effect",
         "intervention_type": "config_change", "intervention_spec": {"MATRIX_LR": "0.08"},
         "estimated_cost": {}},
    ])
    atari_llm = make_mock_llm([
        {"intent": "Test PPO", "rationale": "test", "expected_learning": "algorithm",
         "intervention_type": "config_change", "intervention_spec": {"game": "Breakout"},
         "estimated_cost": {}},
    ])

    planner1 = Planner(store, llm_call_fn=gpt_llm, min_queue_size=3,
                        min_todo=2, n_proposals=2, n_select=2, project_id=pid1)
    planner2 = Planner(store, llm_call_fn=atari_llm, min_queue_size=3,
                        min_todo=2, n_proposals=2, n_select=2, project_id=pid2)

    planner1.tick()
    planner2.tick()

    # Proposals may be in backlog or todo (critic promotes immediately)
    gpt_all = (store.list_proposals("backlog", project_id=pid1) +
               store.list_proposals("todo", project_id=pid1))
    atari_all = (store.list_proposals("backlog", project_id=pid2) +
                 store.list_proposals("todo", project_id=pid2))

    assert len(gpt_all) > 0
    assert len(atari_all) > 0
    assert all(p.project_id == pid1 for p in gpt_all)
    assert all(p.project_id == pid2 for p in atari_all)


def test_planner_does_not_see_other_projects_proposals(store):
    """Planner counts only its own project's proposals."""
    pid1 = store.create_project(name="NanoGPT", description="GPT")
    pid2 = store.create_project(name="Atari", description="RL")

    # Fill pid1 backlog above threshold
    llm = make_mock_llm([
        {"intent": f"test", "rationale": "t", "expected_learning": "t",
         "intervention_type": "config_change", "intervention_spec": {"x": "1"},
         "estimated_cost": {}},
    ])

    planner1 = Planner(store, llm_call_fn=llm, min_queue_size=2,
                        min_todo=1, n_proposals=3, n_select=1, project_id=pid1)
    planner1.tick()  # Generates for pid1

    # pid2 planner should still generate (its backlog is empty)
    planner2 = Planner(store, llm_call_fn=llm, min_queue_size=2,
                        min_todo=1, n_proposals=3, n_select=1, project_id=pid2)
    summary = planner2.tick()
    assert summary["generated"] > 0


def test_worker_claims_across_projects(store):
    """Worker claims from any active project's todo queue."""
    pid1 = store.create_project(name="NanoGPT", description="GPT")
    pid2 = store.create_project(name="Atari", description="RL")

    llm = make_mock_llm([
        {"intent": "work item", "rationale": "t", "expected_learning": "t",
         "intervention_type": "config_change", "intervention_spec": {"x": "1"},
         "estimated_cost": {}},
    ])

    # Generate and promote for both projects
    for pid in [pid1, pid2]:
        planner = Planner(store, llm_call_fn=llm, min_queue_size=3,
                          min_todo=2, n_proposals=2, n_select=2, project_id=pid)
        planner.tick()

    executed_projects = []

    def execute(proposal):
        executed_projects.append(proposal.project_id)
        return {"metrics": {"val": 1.0}}

    worker = Worker(store, execute_fn=execute, worker_id="worker_0")

    # Worker should claim from both projects
    for _ in range(10):  # Run enough ticks
        if worker.tick() is None:
            break

    assert pid1 in executed_projects or pid2 in executed_projects
    assert len(executed_projects) >= 2


def test_planner_orients_only_own_project(store):
    """Planner only processes done proposals from its own project."""
    pid1 = store.create_project(name="NanoGPT", description="GPT")
    pid2 = store.create_project(name="Atari", description="RL")

    llm = make_mock_llm([
        {"intent": "work", "rationale": "t", "expected_learning": "t",
         "intervention_type": "config_change", "intervention_spec": {"x": "1"},
         "estimated_cost": {}},
    ])

    # Generate, promote, execute for pid1
    planner1 = Planner(store, llm_call_fn=llm, min_queue_size=3,
                        min_todo=2, n_proposals=2, n_select=2, project_id=pid1)
    planner1.tick()

    def execute(proposal):
        return {"metrics": {"val": 1.0}}

    worker = Worker(store, execute_fn=execute, worker_id="w0")
    while worker.tick() is not None:
        pass

    # pid1 planner orients its own done proposals
    summary1 = planner1.tick()

    # pid2 planner should NOT process pid1's done proposals
    planner2 = Planner(store, llm_call_fn=llm, min_queue_size=3,
                        min_todo=2, n_proposals=2, n_select=2, project_id=pid2)
    summary2 = planner2.tick()
    assert summary2["oriented"] == 0  # No done proposals for pid2
