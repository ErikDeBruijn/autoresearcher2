"""Test concurrent worker claiming — verifies no double-claiming of proposals."""
import threading
from autoresearcher2.v3.store import Store
from autoresearcher2.v3.proposal import Proposal


def test_concurrent_claim_no_double_assignment(tmp_path):
    """Multiple workers claiming simultaneously should never get the same proposal."""
    db_path = tmp_path / "concurrent.db"
    store = Store(db_path)
    store.init()

    # Create 5 todo proposals
    for i in range(5):
        p = Proposal(
            intent=f"Test proposal {i}",
            rationale="concurrent test",
            intervention_type="config_change",
            intervention_spec={"idx": i},
        expected_learning="test",
        )
        p.set_critic_decision("accept", rank=i + 1, rationale="test")
        p.promote("todo")
        store.save_proposal(p)
    store.close()

    # 10 workers try to claim concurrently
    claims: dict[str, list[str]] = {}  # worker_id -> list of proposal ids
    lock = threading.Lock()

    def worker_fn(worker_id: str):
        # Each thread gets its own Store (and thus its own connection)
        s = Store(db_path)
        claimed = []
        while True:
            p = s.claim_next_todo(worker_id)
            if p is None:
                break
            claimed.append(p.id)
        s.close()
        with lock:
            claims[worker_id] = claimed

    threads = []
    for i in range(10):
        t = threading.Thread(target=worker_fn, args=(f"worker-{i}",))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify: each proposal claimed by exactly one worker
    all_claimed = []
    for worker_id, ids in claims.items():
        all_claimed.extend(ids)

    assert len(all_claimed) == 5, f"Expected 5 claims, got {len(all_claimed)}: {all_claimed}"
    assert len(set(all_claimed)) == 5, f"Duplicate claims: {all_claimed}"


def test_concurrent_claim_empty_queue(tmp_path):
    """Workers on empty queue should all get None without errors."""
    db_path = tmp_path / "empty.db"
    store = Store(db_path)
    store.init()
    store.close()

    results = []
    lock = threading.Lock()

    def worker_fn(worker_id):
        s = Store(db_path)
        p = s.claim_next_todo(worker_id)
        s.close()
        with lock:
            results.append(p)

    threads = [threading.Thread(target=worker_fn, args=(f"w-{i}",)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r is None for r in results)


def test_concurrent_claim_single_item(tmp_path):
    """Only one worker should get the single available proposal."""
    db_path = tmp_path / "single.db"
    store = Store(db_path)
    store.init()

    p = Proposal(
        intent="The only proposal",
        rationale="single item test",
        intervention_type="probe",
        intervention_spec={},
        expected_learning="test",
    )
    p.set_critic_decision("accept", rank=1, rationale="test")
    p.promote("todo")
    store.save_proposal(p)
    store.close()

    claims = []
    lock = threading.Lock()

    def worker_fn(worker_id):
        s = Store(db_path)
        result = s.claim_next_todo(worker_id)
        s.close()
        with lock:
            if result is not None:
                claims.append((worker_id, result.id))

    threads = [threading.Thread(target=worker_fn, args=(f"w-{i}",)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(claims) == 1, f"Expected exactly 1 claim, got {len(claims)}: {claims}"
