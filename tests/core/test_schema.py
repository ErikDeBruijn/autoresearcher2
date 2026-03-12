# tests/core/test_schema.py
from autoresearcher2.core.schema import InterventionSchema


def test_schema_creation():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "adamw", "sgd", "lion"],
            "lr_bucket": ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2"],
            "pos_encoding": ["rope", "learned", "alibi", "none"],
        }
    )
    assert schema.n_factors == 3
    assert schema.n_cells == 4 * 5 * 4  # 80


def test_schema_cell_to_config():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )
    config = schema.cell_to_config(0)
    assert config == {"optimizer": "adam", "lr": "low"}

    config = schema.cell_to_config(3)
    assert config == {"optimizer": "sgd", "lr": "high"}


def test_schema_config_to_cell():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )
    cell = schema.config_to_cell({"optimizer": "sgd", "lr": "low"})
    assert cell == 2


def test_schema_one_hot():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )
    x = schema.one_hot(0)  # adam, low
    # adam=[1,0], low=[1,0]
    assert list(x) == [1.0, 0.0, 1.0, 0.0]


def test_schema_feature_vector_with_interactions():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )
    x = schema.feature_vector(0, include_interactions=True)
    # 4 main effects + 4 pairwise interactions = 8
    assert len(x) == 8


def test_schema_cell_roundtrip():
    """cell_to_config -> config_to_cell should be identity for all cells."""
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
            "batch": ["small", "large"],
        }
    )
    for cell in range(schema.n_cells):
        config = schema.cell_to_config(cell)
        assert schema.config_to_cell(config) == cell
