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
            "intervention_type": "config_change|probe|code_change",
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
    intervention_types: str = "config_change (modify training hyperparameters), probe (short training run to test hypothesis cheaply), or code_change (rewrite train.py with structural changes — use file_changes key in intervention_spec)"
    parameters: str = "DEPTH, MATRIX_LR, WEIGHT_DECAY, num_steps, batch_size. For code_change: {\"file_changes\": {\"train.py\": \"full file content\"}}"
    diversity_hint: str = "Mix of config_change (full run), probe (short run, include \"run_steps\" in spec), and code_change (when the hypothesis requires structural code changes, not just parameter tweaks)"
    hardware: str = ""
    base_script: str = ""  # Current training script content — shown to LLM for code_change proposals
    base_script_name: str = "train.py"  # Filename for the base script


NANOGPT_DOMAIN = DomainConfig(
    hardware="2x NVIDIA RTX PRO 6000 Blackwell Max-Q (96GB VRAM each). Experiments run on one GPU at a time. VRAM is NOT a bottleneck — 96GB is enormous for NanoGPT scale models. Focus on compute efficiency (wall time, val_bpb per GPU-hour) rather than memory constraints."
)

ATARI_DOMAIN = DomainConfig(
    name="Atari RL",
    description="We train RL agents to play Atari Breakout. The base script uses SB3 PPO as boilerplate. The real research is in code_change proposals that rewrite the training script with structural improvements: custom reward shaping, novel network architectures, different algorithms, preprocessing tricks, curriculum learning, etc.",
    intervention_types="config_change (tweak hyperparameters — LEAST interesting), probe (quick diagnostic run), or code_change (MOST VALUABLE: rewrite train_atari.py with structural changes via file_changes key). PREFER code_change — parameter tuning alone won't achieve breakthroughs.",
    parameters="For config_change: learning_rate, n_envs, total_timesteps, gamma, clip_range. For code_change: {\"file_changes\": {\"train_atari.py\": \"full file content\"}} — you can completely rewrite the training script.",
    diversity_hint="STRONGLY prefer code_change proposals. The training script is boilerplate SB3 — real progress comes from structural code changes: reward shaping, custom networks, preprocessing, action masking, curriculum learning, exploration strategies. config_change is only useful for establishing baselines.",
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

    base_script_section = ""
    if domain.base_script:
        base_script_section = f"""
## CURRENT TRAINING SCRIPT ({domain.base_script_name})

This is the script that gets executed. For code_change proposals, you MUST provide the COMPLETE modified version of this script in intervention_spec.file_changes.

```python
{domain.base_script}
```
"""

    code_change_rules = ""
    if "code_change" in domain.intervention_types:
        code_change_rules = f"""
## CODE_CHANGE RULES (CRITICAL)

For code_change proposals, intervention_spec MUST contain actual code, NOT descriptions.

Two formats are supported (pick one):

**Option A — Unified diff (preferred for small changes):**
{{"diff": "--- a/{domain.base_script_name}\\n+++ b/{domain.base_script_name}\\n@@ -37,6 +37,8 @@\\n def main():\\n     n_envs = 4\\n+    # Add reward shaping\\n+    reward_scale = 2.0\\n"}}

**Option B — Full file replacement (for major rewrites):**
{{"file_changes": {{"{domain.base_script_name}": "#!/usr/bin/env python3\\n...COMPLETE FILE CONTENT..."}}}}

Rules:
- NEVER put text descriptions in intervention_spec — only working code (diff or full file)
- The script must print metrics in "key: value" format so they can be parsed
- Required output: mean_reward, std_reward, wall_time_s, fps
- You can change ANYTHING: algorithm, network architecture, reward shaping, preprocessing, etc.
"""

    return f"""You are a curious researcher generating experiment proposals for a research system.

## CURRENT WORLD MODEL (what we currently believe and don't know)

{wm_json}
{base_script_section}
## YOUR ROLE

Generate {n_proposals} diverse research proposals. You are the creative, divergent thinker:
- Look for untested assumptions and shaky beliefs
- Propose experiments that would resolve tensions
- Consider cheap probes before expensive full experiments
- {domain.description} Available intervention types are ONLY: {domain.intervention_types}
{f"- Hardware: {domain.hardware}" if domain.hardware else ""}
- Prioritize proposals that teach us something regardless of outcome
{code_change_rules}
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
