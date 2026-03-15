"""Critic: ordinal ranking and selection of proposals.

OODA: Observe → Orient → Decide → Act
This is part of the Decide step. The critic evaluates proposals from the
generator, ranking them ordinally (this > that) and selecting the most
valuable ones for execution.

The critic is a skeptical reviewer: it deduplicates, rejects weak proposals,
and prioritizes those that challenge shaky beliefs for the lowest cost.
"""

import json
import logging

from autoresearcher2.v3.world_model import WorldModel
from autoresearcher2.v3.proposal import Proposal

logger = logging.getLogger(__name__)

CRITIC_SCHEMA = {
    "rankings": [
        {
            "proposal_id": "prop_xxx",
            "decision": "accept|reject|deprioritize",
            "rank": "integer (1 = highest priority, null for rejected)",
            "rationale": "why this ranking",
        }
    ]
}


def build_critic_prompt(
    world_model: WorldModel,
    proposals: list[Proposal],
    n_select: int = 2,
) -> str:
    """Build prompt for the critic LLM role."""
    wm_json = json.dumps(world_model.to_dict(), indent=2, default=str)
    proposals_json = json.dumps(
        [p.to_dict() for p in proposals], indent=2, default=str
    )
    schema_json = json.dumps(CRITIC_SCHEMA, indent=2)

    return f"""You are a skeptical research reviewer evaluating experiment proposals.

## CURRENT WORLD MODEL

{wm_json}

## PROPOSALS TO EVALUATE

{proposals_json}

## YOUR ROLE

You are the convergent thinker — the editor who decides what's worth our limited compute budget.

Evaluate each proposal and:
1. **Accept** the {n_select} most valuable proposals for execution
2. **Reject** proposals that are redundant, uninformative, or too expensive relative to expected learning
3. **Deprioritize** proposals that have merit but shouldn't run now

## RANKING CRITERIA (in order of importance)

1. **Addresses weakest assumptions**: Does this test a belief we're not confident about?
2. **Cost-effectiveness**: What's the ratio of expected learning to cost_to_test?
3. **Resolves tensions**: Does this help resolve a known contradiction?
4. **Novel information**: Would this teach us something new vs confirming what we know?
5. **Diversity**: Don't select proposals that all test the same thing

## IMPORTANT

- Rank ORDINALLY: proposal A > proposal B > proposal C. Not absolute scores.
- A cheap probe that tests a shaky belief beats an expensive experiment confirming a strong one.
- Reject proposals that duplicate recent observations or test already-confirmed beliefs.
- Select exactly {n_select} proposals to accept.

## OUTPUT FORMAT

Respond with ONLY a JSON object (no other text):

{schema_json}"""


def critique_proposals(
    world_model: WorldModel,
    proposals: list[Proposal],
    n_select: int = 2,
    llm_call_fn=None,
) -> list[Proposal]:
    """Evaluate and rank proposals, returning the accepted ones.

    Args:
        world_model: Current epistemic state
        proposals: Proposals from the generator
        n_select: How many to accept
        llm_call_fn: Function that takes prompt string and returns parsed JSON dict

    Returns:
        Accepted proposals, sorted by rank (best first)
    """
    if not proposals:
        return []

    prompt = build_critic_prompt(world_model, proposals, n_select)

    try:
        response = llm_call_fn(prompt)
    except Exception:
        logger.warning("Critic LLM call failed, accepting first %d proposals", n_select, exc_info=True)
        return proposals[:n_select]

    return _apply_rankings(proposals, response, n_select)


def _apply_rankings(
    proposals: list[Proposal],
    response: dict,
    n_select: int,
) -> list[Proposal]:
    """Apply critic rankings to proposals, return accepted ones sorted by rank."""
    rankings = response.get("rankings", [])
    if not isinstance(rankings, list):
        logger.warning("Critic response 'rankings' is not a list")
        return proposals[:n_select]

    # Build lookup by proposal ID
    proposal_map = {p.id: p for p in proposals}

    for ranking in rankings:
        pid = ranking.get("proposal_id", "")
        if pid in proposal_map:
            proposal_map[pid].set_critic_decision(
                decision=ranking.get("decision", "deprioritize"),
                rank=ranking.get("rank"),
                rationale=ranking.get("rationale", ""),
            )

    # Return accepted proposals sorted by rank
    accepted = [
        p for p in proposals
        if p.critic and p.critic.get("decision") == "accept"
    ]
    accepted.sort(key=lambda p: (p.critic.get("rank") or 999))

    if not accepted:
        # Fallback: if no proposals were explicitly accepted (e.g. empty rankings,
        # IDs didn't match), accept the first n_select by position
        logger.info("No proposals explicitly accepted, falling back to first %d", n_select)
        for i, p in enumerate(proposals[:n_select]):
            p.set_critic_decision("accept", rank=i + 1, rationale="fallback: no explicit ranking")
        return proposals[:n_select]

    return accepted[:n_select]
