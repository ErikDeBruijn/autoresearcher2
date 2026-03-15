"""Generator: rationale-first proposal generation.

OODA: Observe → Orient → Decide → Act
This is part of the Decide step. The generator reads the world model
and proposes research actions, reasoning from epistemic intent first.

Cognitive order per proposal:
1. epistemic intent — which belief/tension/question
2. rationale — why now
3. expected learning — what we learn regardless of outcome
4. intervention — concrete action
5. estimated cost — cost_to_test, cheaper probes
"""

import json
import logging
from dataclasses import dataclass, field

from autoresearcher2.v3.world_model import WorldModel
from autoresearcher2.v3.proposal import Proposal

logger = logging.getLogger(__name__)

PROPOSAL_SCHEMA = {
    "proposals": [
        {
            "intent": "which belief/tension/question does this address",
            "rationale": "why is this valuable now",
            "expected_learning": "what we learn regardless of outcome",
            "intervention_type": "config_change|probe",
            "intervention_spec": {"key": "value"},
            "estimated_cost": {
                "cost_to_test": "description of compute cost",
                "cheaper_probe": "null or description of a cheaper alternative",
            },
        }
    ]
}


@dataclass
class DomainConfig:
    """Domain description for the generator prompt."""
    name: str = "NanoGPT training"
    description: str = "We run NanoGPT training experiments."
    intervention_types: str = "config_change (modify training hyperparameters) or probe (short training run to test hypothesis cheaply)"
    parameters: str = "DEPTH, MATRIX_LR, WEIGHT_DECAY, num_steps, batch_size"
    diversity_hint: str = "Mix of config_change (full run) and probe (short run, include \"run_steps\" in spec)"
    hardware: str = ""


NANOGPT_DOMAIN = DomainConfig(
    hardware="2x NVIDIA RTX PRO 6000 Blackwell Max-Q (96GB VRAM each). Experiments run on one GPU at a time. VRAM is NOT a bottleneck — 96GB is enormous for NanoGPT scale models. Focus on compute efficiency (wall time, val_bpb per GPU-hour) rather than memory constraints."
)

ATARI_DOMAIN = DomainConfig(
    name="Atari RL",
    description="We optimize Atari game agents using reinforcement learning.",
    intervention_types="config_change (modify RL hyperparameters) or probe (short training run)",
    parameters="game, learning_rate, network_size, algorithm, n_envs, total_timesteps",
    diversity_hint="Mix of config_change and probe. Try different games, algorithms (PPO, DQN), and network sizes",
)

GENERIC_DOMAIN = DomainConfig(
    name="generic optimization",
    description="We run experiments to optimize a target metric.",
    intervention_types="config_change (modify parameters) or probe (cheap test)",
    parameters="any key-value pairs relevant to the domain",
    diversity_hint="Mix of config_change (full run) and probe (quick test)",
)


def build_generator_prompt(world_model: WorldModel, n_proposals: int = 5, domain: DomainConfig = None) -> str:
    """Build prompt for the generator LLM role."""
    domain = domain or NANOGPT_DOMAIN
    wm_json = json.dumps(world_model.to_dict(), indent=2, default=str)
    schema_json = json.dumps(PROPOSAL_SCHEMA, indent=2)

    return f"""You are a curious researcher generating experiment proposals for a research system.

## CURRENT WORLD MODEL (what we currently believe and don't know)

{wm_json}

## YOUR ROLE

Generate {n_proposals} diverse research proposals. You are the creative, divergent thinker:
- Look for untested assumptions and shaky beliefs
- Propose experiments that would resolve tensions
- Consider cheap probes before expensive full experiments
- {domain.description} Available intervention types are ONLY: {domain.intervention_types}
{f"- Hardware: {domain.hardware}" if domain.hardware else ""}
- Prioritize proposals that teach us something regardless of outcome

## COGNITIVE ORDER (follow this for each proposal)

For each proposal, reason in this order:
1. **Epistemic intent** — which belief, tension, or open question are you addressing?
2. **Rationale** — why is this the right thing to investigate now?
3. **Expected learning** — what would we learn even if the result is negative?
4. **Intervention** — what concrete action to take
5. **Cost estimate** — how expensive is this, and is there a cheaper probe?

## COST AWARENESS

Our measured cost data (from real power monitoring — energy_kwh, cost_eur, avg_power_w, wall_time_s are actual measurements):
{json.dumps(world_model.cost_beliefs, indent=2, default=str)}

Every experiment has real costs: electricity (€), GPU time (minutes), and wall clock time (which blocks other experiments). A cheap probe that tests a shaky belief is often better than an expensive experiment confirming what we already know. Consider:
- cost_to_test: how much energy (kWh) and money (€) does this require based on similar past experiments?
- wall_time: how long will this block the GPU? Shorter experiments allow more learning per hour.
- cost_of_being_wrong: what if our current belief is wrong and we don't test it?
- Is there a shorter/faster way to get approximate evidence?

## DIVERSITY REQUIREMENTS

Your {n_proposals} proposals should include:
- At least one that challenges the highest-confidence belief
- At least one cheap probe (lowest cost_to_test)
- At least one that addresses an unresolved tension
- {domain.diversity_hint}
- intervention_spec must contain valid parameters like: {domain.parameters}

## OUTPUT FORMAT

Respond with ONLY a JSON object (no other text):

{schema_json}"""


def generate_proposals(
    world_model: WorldModel,
    n_proposals: int = 5,
    llm_call_fn=None,
    domain: DomainConfig = None,
) -> list[Proposal]:
    """Generate research proposals from the world model.

    Args:
        world_model: Current epistemic state
        n_proposals: How many proposals to generate
        llm_call_fn: Function that takes prompt string and returns parsed JSON dict
        domain: Domain configuration for prompt generation

    Returns:
        List of Proposal objects
    """
    prompt = build_generator_prompt(world_model, n_proposals, domain=domain)

    try:
        response = llm_call_fn(prompt)
    except Exception:
        logger.warning("Generator LLM call failed", exc_info=True)
        return []

    return _parse_proposals(response)


def _parse_proposals(response: dict) -> list[Proposal]:
    """Parse LLM response into Proposal objects."""
    proposals_data = response.get("proposals", [])
    if not isinstance(proposals_data, list):
        logger.warning("Generator response 'proposals' is not a list")
        return []

    proposals = []
    for item in proposals_data:
        try:
            p = Proposal(
                intent=item.get("intent", ""),
                rationale=item.get("rationale", ""),
                expected_learning=item.get("expected_learning", ""),
                intervention_type=item.get("intervention_type", "other"),
                intervention_spec=item.get("intervention_spec", {}),
                estimated_cost=item.get("estimated_cost", {}),
            )
            proposals.append(p)
        except Exception as e:
            logger.warning("Failed to parse proposal: %s", e)

    return proposals
