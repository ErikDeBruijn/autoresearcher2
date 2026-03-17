"""Shared test fixtures for v3 tests."""
import pytest

from autoresearcher2.v3.store import Store


@pytest.fixture
def store(tmp_path):
    """Create and initialize a temporary Store for testing."""
    s = Store(tmp_path / "research.db")
    s.init()
    yield s
    s.close()
