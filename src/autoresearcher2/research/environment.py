from abc import ABC, abstractmethod


class Environment(ABC):
    """Abstract interface for experiment environments."""

    @abstractmethod
    def run(self, cell_index: int) -> float:
        """Run an experiment and return the outcome (higher = better)."""
        ...
