import numpy as np
from scipy.special import softmax, digamma

from autoresearcher2.toy.pomdp import ToyPOMDP


class ActiveInferenceAgent:
    """Canonical active inference agent for the toy POMDP.

    Maintains Dirichlet beliefs about the observation likelihood A(o|s).
    Selects actions by minimizing expected free energy G.
    """

    def __init__(self, pomdp: ToyPOMDP, gamma: float = 1.0):
        self.pomdp = pomdp
        self.gamma = gamma
        self.beliefs = np.ones((pomdp.n_states, pomdp.n_observations))

    def _expected_A(self) -> np.ndarray:
        return self.beliefs / self.beliefs.sum(axis=1, keepdims=True)

    def compute_efe_all_actions(self) -> dict[int, dict[str, float]]:
        efe = {}
        expected_A = self._expected_A()

        for a in range(self.pomdp.n_states):
            q_o = expected_A[a]
            pragmatic = -float(q_o @ self.pomdp.C)

            h_expected = -float(np.sum(q_o * np.log(q_o + 1e-16)))

            alpha = self.beliefs[a]
            alpha_0 = alpha.sum()
            expected_entropy = float(
                -np.sum(
                    (alpha / alpha_0) * (digamma(alpha + 1) - digamma(alpha_0 + 1))
                )
            )

            epistemic = h_expected - expected_entropy

            efe[a] = {
                "pragmatic": pragmatic,
                "epistemic": epistemic,
                "total": pragmatic + epistemic,
            }

        return efe

    def select_action(self) -> int:
        efe = self.compute_efe_all_actions()
        g_values = np.array([efe[a]["total"] for a in range(self.pomdp.n_states)])
        probs = softmax(-self.gamma * g_values)
        return int(np.random.choice(self.pomdp.n_states, p=probs))

    def update(self, action: int, observation: int) -> None:
        self.beliefs[action, observation] += 1.0
