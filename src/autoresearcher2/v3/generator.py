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
from dataclasses import dataclass

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
    name: str = "research experiment"
    description: str = "We run experiments to optimize a target metric."
    intervention_types: str = "config_change (modify parameters), probe (cheap test to validate hypothesis), or code_change (rewrite experiment script with structural changes — use file_changes key in intervention_spec)"
    parameters: str = "any key-value pairs relevant to the experiment domain"
    diversity_hint: str = "Mix of config_change (full run), probe (quick test), and code_change (when the hypothesis requires structural changes)"
    hardware: str = ""
    base_script: str = ""  # Current training script content — shown to LLM for code_change proposals
    base_script_name: str = "script.py"  # Filename for the base script

_DOMAIN_DEFAULTS = DomainConfig()


def domain_config_from_project(project: dict) -> DomainConfig:
    """Build a DomainConfig from project metadata.

    Reads the domain_config dict stored in the project's metadata.
    Falls back to generic defaults if the project has no domain_config.

    If domain_config contains a base_script_path, the file is read from disk
    and its contents are set as base_script (used for code_change proposals).
    """
    dc = project.get("domain_config")
    if not dc:
        return DomainConfig()

    base_script = dc.get("base_script", "")
    base_script_path = dc.get("base_script_path", "")
    if base_script_path and not base_script:
        try:
            with open(base_script_path) as f:
                base_script = f.read()
        except FileNotFoundError:
            logger.warning("Base script not found: %s", base_script_path)

    return DomainConfig(
        name=dc.get("name", project.get("name", "research experiment")),
        description=dc.get("description", project.get("description", _DOMAIN_DEFAULTS.description)),
        intervention_types=dc.get("intervention_types", _DOMAIN_DEFAULTS.intervention_types),
        parameters=dc.get("parameters", _DOMAIN_DEFAULTS.parameters),
        diversity_hint=dc.get("diversity_hint", _DOMAIN_DEFAULTS.diversity_hint),
        hardware=dc.get("hardware", ""),
        base_script=base_script,
        base_script_name=dc.get("base_script_name", "script.py"),
    )


def build_generator_prompt(world_model: WorldModel, n_proposals: int = 5, domain: DomainConfig = None) -> str:
    """Build prompt for the generator LLM role."""
    domain = domain or DomainConfig()
    wm_json = json.dumps(world_model.to_dict(), indent=2, default=str)
    schema_json = json.dumps(PROPOSAL_SCHEMA, indent=2)

    base_script_section = ""
    if domain.base_script:
        base_script_section = f"""
## CURRENT EXPERIMENT SCRIPT ({domain.base_script_name})

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
- Required output: the target metric(s) defined in the domain config, plus wall_time_s
- You can change ANYTHING within the experiment's parameter space
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

Our measured cost data (wall_time_s and any domain-specific cost fields are actual measurements{f" from {domain.hardware}" if domain.hardware else ""}):
{json.dumps(world_model.cost_beliefs, indent=2, default=str)}

Every experiment has real costs: time and resources. A cheap probe that tests a shaky belief is often better than an expensive experiment confirming what we already know. Consider:
- cost_to_test: how much time and resources does this require based on similar past experiments?
- wall_time: how long will this block the worker? Shorter experiments allow more learning per hour.
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
