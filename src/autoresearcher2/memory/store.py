import time


class MemoryStore:
    """Simple episodic memory for experiment results."""

    def __init__(self):
        self._records: list[dict] = []
        self._tried_cells: set[int] = set()

    def add(
        self,
        cell_index: int,
        config: dict | None = None,
        outcome: float = 0.0,
        appraisal: dict | None = None,
    ) -> None:
        self._records.append(
            {
                "cell_index": cell_index,
                "config": config or {},
                "outcome": outcome,
                "appraisal": appraisal or {},
                "timestamp": time.time(),
            }
        )
        self._tried_cells.add(cell_index)

    def all(self) -> list[dict]:
        return list(self._records)

    def has_tried(self, cell_index: int) -> bool:
        return cell_index in self._tried_cells

    def get_by_cell(self, cell_index: int) -> list[dict]:
        return [r for r in self._records if r["cell_index"] == cell_index]

    def top_by_appraisal(self, signal: str, n: int = 5) -> list[dict]:
        scored = [r for r in self._records if signal in r.get("appraisal", {})]
        scored.sort(key=lambda r: r["appraisal"][signal], reverse=True)
        return scored[:n]

    def summary(self) -> dict:
        if not self._records:
            return {"n_experiments": 0, "n_unique_cells": 0, "best_outcome": None}
        return {
            "n_experiments": len(self._records),
            "n_unique_cells": len(self._tried_cells),
            "best_outcome": max(r["outcome"] for r in self._records),
        }
