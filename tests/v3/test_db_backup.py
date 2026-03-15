"""Tests for automatic DB backup on init()."""
import sqlite3
import pytest
from autoresearcher2.v3.store import Store


@pytest.fixture
def store_with_data(tmp_path):
    """Create a store with some data to trigger backup."""
    s = Store(tmp_path / "research.db")
    s.init()
    # Insert a fake observation directly
    s.conn.execute(
        "INSERT INTO observations (id, created_at, intervention_type, intervention_spec, outcome_success) "
        "VALUES (?, ?, ?, ?, ?)",
        ("obs_001", 1000.0, "probe", '{"x": "1"}', 1),
    )
    s.conn.commit()
    s.close()
    return tmp_path / "research.db"


def test_init_creates_backup_when_data_exists(store_with_data, tmp_path):
    """Re-initializing a store with data should auto-backup."""
    db_path = store_with_data
    s = Store(db_path)
    s.init()
    s.close()

    backup_dir = db_path.parent / "db_backups"
    assert backup_dir.exists()
    backups = list(backup_dir.glob("*.db"))
    assert len(backups) == 1

    # Verify backup contains the data
    conn = sqlite3.connect(str(backups[0]))
    count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    conn.close()
    assert count == 1


def test_init_no_backup_for_empty_db(tmp_path):
    """Empty DB should not create a backup."""
    s = Store(tmp_path / "research.db")
    s.init()
    s.close()

    backup_dir = tmp_path / "db_backups"
    assert not backup_dir.exists()


def test_init_no_backup_for_new_db(tmp_path):
    """Brand new DB should not create a backup."""
    db_path = tmp_path / "research.db"
    assert not db_path.exists()
    s = Store(db_path)
    s.init()
    s.close()

    backup_dir = tmp_path / "db_backups"
    assert not backup_dir.exists()
