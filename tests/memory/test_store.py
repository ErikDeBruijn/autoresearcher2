# tests/memory/test_store.py
from autoresearcher2.memory.store import MemoryStore


def test_store_and_retrieve():
    store = MemoryStore()
    store.add(
        cell_index=0,
        config={"optimizer": "adam", "lr": "low"},
        outcome=0.85,
        appraisal={"surprise": 0.3, "learntropy": 0.2},
    )
    records = store.all()
    assert len(records) == 1
    assert records[0]["outcome"] == 0.85


def test_dedup_detects_repeat():
    store = MemoryStore()
    store.add(cell_index=0, config={"optimizer": "adam"}, outcome=0.85)
    assert store.has_tried(cell_index=0)
    assert not store.has_tried(cell_index=1)


def test_retrieve_by_cell():
    store = MemoryStore()
    store.add(cell_index=0, config={}, outcome=0.8)
    store.add(cell_index=0, config={}, outcome=0.85)
    store.add(cell_index=1, config={}, outcome=0.5)

    results = store.get_by_cell(0)
    assert len(results) == 2


def test_top_by_appraisal():
    store = MemoryStore()
    store.add(cell_index=0, config={}, outcome=0.8, appraisal={"learntropy": 0.1})
    store.add(cell_index=1, config={}, outcome=0.5, appraisal={"learntropy": 0.9})
    store.add(cell_index=2, config={}, outcome=0.6, appraisal={"learntropy": 0.5})

    top = store.top_by_appraisal("learntropy", n=2)
    assert len(top) == 2
    assert top[0]["cell_index"] == 1


def test_summary():
    store = MemoryStore()
    store.add(cell_index=0, config={"opt": "adam"}, outcome=0.9)
    store.add(cell_index=1, config={"opt": "sgd"}, outcome=0.3)
    store.add(cell_index=0, config={"opt": "adam"}, outcome=0.85)

    summary = store.summary()
    assert summary["n_experiments"] == 3
    assert summary["n_unique_cells"] == 2
    assert summary["best_outcome"] == 0.9
