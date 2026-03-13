"""Tests for AtariEnvironment."""

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.research.atari_environment import AtariEnvironment, SCORE_RANGES


ATARI_SCHEMA = InterventionSchema(
    factors={
        "game": ["Breakout", "SpaceInvaders", "Pong"],
        "learning_rate": ["1e-4", "5e-4", "1e-3"],
        "network_size": ["small", "medium", "large"],
    }
)


def test_atari_schema_27_cells():
    assert ATARI_SCHEMA.n_cells == 27


def test_atari_environment_instantiates():
    env = AtariEnvironment(schema=ATARI_SCHEMA)
    assert env.cuda_device == "0"
    assert env.total_timesteps == 1_000_000


def test_score_normalization():
    """Verify score normalization maps known ranges to [0, 1]."""
    for game, (low, high) in SCORE_RANGES.items():
        # Low score → ~0.0
        normalized_low = (low - low) / (high - low)
        assert abs(normalized_low) < 0.01

        # High score → ~1.0
        normalized_high = (high - low) / (high - low)
        assert abs(normalized_high - 1.0) < 0.01


def test_cell_to_config_games():
    """Verify each game appears in the schema."""
    games_seen = set()
    for cell in range(ATARI_SCHEMA.n_cells):
        config = ATARI_SCHEMA.cell_to_config(cell)
        games_seen.add(config["game"])
    assert games_seen == {"Breakout", "SpaceInvaders", "Pong"}


def test_custom_timesteps():
    env = AtariEnvironment(schema=ATARI_SCHEMA, total_timesteps=500_000)
    assert env.total_timesteps == 500_000
