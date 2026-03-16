"""Orientation step: LLM-led world model update.

OODA: Observe → Orient → Decide → Act
This is the Orient step. After a new observation arrives (Observe),
the LLM inspects facts + current state and produces a structured
update-delta for the world model.

Evidence-first prompt order:
1. New observation (facts)
2. Current world model (state)
3. Update instructions
4. Required output schema (delta format)
"""

import json
import logging

from autoresearcher2.v3.world_model import WorldModel
from autoresearcher2.v3.observation import Observation
from autoresearcher2.v3.generator import DomainConfig

logger = logging.getLogger(__name__)

DELTA_SCHEMA = {
    "beliefs_added": [{"claim": "str", "confidence": "0.0-1.0", "evidence_for": ["obs_id"]}],
    "beliefs_revised": [{"id": "str", "new_confidence": "0.0-1.0", "new_evidence_for": ["obs_id"], "reason": "str"}],
    "beliefs_retired": [{"id": "str", "reason": "str"}],
    "expectations_added": [{"if": "condition", "then": "prediction", "confidence": "0.0-1.0", "basis": ["ids"]}],
    "expectations_revised": [],
    "tensions_added": [{"beliefs": ["B_ids"], "nature": "str", "salience": "high|medium|low"}],
    "tensions_resolved": [{"id": "str", "resolution": "str", "reasoning": "str"}],
    "salience_updated": {"high_learntropy": ["obs_ids"], "stale_beliefs": ["B_ids"]},
    "cost_beliefs_updated": {"intervention_type": {"wall_time_s": "float", "cost_unit": "float"}},
}


def build_orientation_prompt(
    world_model: WorldModel,
    observation: Observation,
    domain: DomainConfig = None,
) -> str:
    """Build evidence-first prompt for the orientation step.

    Args:
        world_model: Current epistemic state
        observation: New reality contact
        domain: Optional domain configuration for domain-appropriate language
    """
    domain = domain or DomainConfig()
    obs_json = json.dumps(observation.to_dict(), indent=2, default=str)
    wm_json = json.dumps(world_model.to_dict(), indent=2, default=str)
    schema_json = json.dumps(DELTA_SCHEMA, indent=2)

    hardware_hint = ""
    if domain.hardware:
        hardware_hint = f"\n- Check the observation's real cost measurements (from {domain.hardware}). Update cost_beliefs with these measured values — they are ground truth, not estimates."
    else:
        hardware_hint = "\n- Check the observation's cost measurements (wall_time_s and any domain-specific cost fields). Update cost_beliefs with measured values where available."

    return f"""You are updating a research system's world model after a new experimental observation.
Domain: {domain.name} — {domain.description}

## 1. NEW OBSERVATION (facts — what just happened)

{obs_json}

## 2. CURRENT WORLD MODEL (epistemic state — what we currently believe)

{wm_json}

## 3. UPDATE INSTRUCTIONS

Reason step by step about what this observation means for our beliefs, expectations, tensions, salience, and cost estimates.

Consider:
- Does this observation confirm or contradict any existing beliefs? Adjust confidence accordingly.
- Does it create new tensions between beliefs?
- Does it resolve existing tensions?
- Does it reveal something surprising (high learntropy)?{hardware_hint}
- Are any beliefs now stale (not tested recently)?
- Should we add new beliefs or expectations based on this evidence?

Important:
- Be conservative with confidence changes. A single observation shifts confidence modestly.
- Only retire beliefs with strong contradictory evidence, not just one data point.
- Tensions are valuable — they drive future experiments. Don't resolve them prematurely.

## 4. REQUIRED OUTPUT

Respond with ONLY a JSON object matching this schema (no other text):

{schema_json}

Include only fields that actually changed. Empty arrays for unchanged categories."""


def orient(world_model: WorldModel, observation: Observation, llm_call_fn, domain: DomainConfig = None) -> dict:
    """Run the orientation step: update world model based on new observation.

    Args:
        world_model: Current epistemic state (will be mutated)
        observation: New reality contact
        llm_call_fn: Function that takes a prompt string and returns parsed JSON dict
        domain: Optional domain configuration for domain-appropriate language

    Returns:
        The delta that was applied (includes 'learntropy' score)
    """
    from autoresearcher2.v3.learntropy import compute_learntropy

    prompt = build_orientation_prompt(world_model, observation, domain=domain)

    try:
        response = llm_call_fn(prompt)
    except Exception:
        logger.warning("Orientation LLM call failed, skipping world model update", exc_info=True)
        return {}

    # Extract delta — handle various response formats
    delta = _extract_delta(response)

    if delta:
        wm_before_dict = world_model.to_dict()
        from autoresearcher2.v3.world_model import WorldModel as WM
        wm_snapshot = WM.from_dict(wm_before_dict)

        world_model.apply_delta(delta)

        # Compute learntropy retrospectively
        learntropy_score = compute_learntropy(wm_snapshot, world_model, delta)
        delta["learntropy"] = learntropy_score
        logger.info("World model updated to v%d (learntropy=%.3f)", world_model.version, learntropy_score)

    return delta


def _extract_delta(response) -> dict:
    """Extract the delta dict from LLM response, handling various formats."""
    if isinstance(response, dict):
        # If the response has a "delta" wrapper, unwrap it
        if "delta" in response:
            return response["delta"]
        # If it directly contains delta fields, use as-is
        if any(k in response for k in ("beliefs_added", "beliefs_revised", "beliefs_retired",
                                         "tensions_added", "tensions_resolved", "cost_beliefs_updated")):
            return response
    return {}
