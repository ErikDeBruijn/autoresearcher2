"""Tests for TrainPyEnvironment dataset support."""

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.research.trainpy_environment import TrainPyEnvironment


SCHEMA = InterventionSchema(
    factors={
        "DEPTH": ["6", "8", "10"],
        "MATRIX_LR": ["0.02", "0.04", "0.08"],
        "WEIGHT_DECAY": ["0.1", "0.2", "0.4"],
    }
)


def test_default_dataset_is_climbmix():
    env = TrainPyEnvironment(schema=SCHEMA)
    assert env.dataset == "climbmix"


def test_custom_dataset():
    env = TrainPyEnvironment(schema=SCHEMA, dataset="wikipedia")
    assert env.dataset == "wikipedia"


def test_code_dataset():
    env = TrainPyEnvironment(schema=SCHEMA, dataset="code")
    assert env.dataset == "code"
