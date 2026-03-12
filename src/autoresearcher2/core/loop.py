from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel
from autoresearcher2.core.controller import Controller
from autoresearcher2.memory.store import MemoryStore
from autoresearcher2.research.environment import Environment
from autoresearcher2.appraisal.signals import compute_appraisal


class ResearchLoop:
    """Main research loop: select -> run -> appraise -> store -> repeat."""

    def __init__(
        self,
        schema: InterventionSchema,
        model: BayesianLinearModel,
        controller: Controller,
        memory: MemoryStore,
        env: Environment,
    ):
        self.schema = schema
        self.model = model
        self.controller = controller
        self.memory = memory
        self.env = env

    def run(self, n_experiments: int) -> list[dict]:
        results = []
        for i in range(n_experiments):
            cell = self.controller.select_next()
            config = self.schema.cell_to_config(cell)
            scores = self.controller.score_cell(cell)

            outcome = self.env.run(cell)

            snapshot_before = self.model.snapshot()
            self.model.update(cell, outcome)
            snapshot_after = self.model.snapshot()

            appraisal = compute_appraisal(
                self.schema, cell, outcome, snapshot_before, snapshot_after
            )

            self.memory.add(
                cell_index=cell,
                config=config,
                outcome=outcome,
                appraisal=appraisal,
            )

            results.append(
                {
                    "experiment": i,
                    "cell": cell,
                    "config": config,
                    "outcome": outcome,
                    "appraisal": appraisal,
                    "scores": scores,
                }
            )

        return results
