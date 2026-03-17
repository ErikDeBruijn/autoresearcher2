"""Learntropy: retrospective measure of how much an observation reshaped the world model.

Learntropy (from Wozniak) measures the subjective informativeness of an observation.
High learntropy = the observation significantly changed beliefs, resolved tensions,
or created new ones. Low learntropy = the observation confirmed what we already knew.

This is calculated AFTER orientation, by comparing world model versions N and N+1.
"""
from autoresearcher2.v3.world_model import WorldModel


def compute_learntropy(wm_before: WorldModel, wm_after: WorldModel, delta: dict) -> float:
    """Compute learntropy score for a world model update.

    Returns a score in [0.0, 1.0] where:
    - 0.0 = observation changed nothing
    - 1.0 = observation completely transformed the model

    The score is a weighted sum of:
    - Belief changes (added, revised, retired)
    - Tension changes (added, resolved)
    - Cost belief changes
    - Confidence shifts magnitude
    """
    if not delta:
        return 0.0

    score = 0.0

    # Belief changes
    beliefs_added = len(delta.get("beliefs_added", []))
    beliefs_revised = len(delta.get("beliefs_revised", []))
    beliefs_retired = len(delta.get("beliefs_retired", []))

    n_beliefs = max(len(wm_before.beliefs), 1)
    belief_change_ratio = (beliefs_added + beliefs_revised + beliefs_retired) / n_beliefs
    score += min(belief_change_ratio, 1.0) * 0.4

    # Confidence shift magnitude
    confidence_shift = 0.0
    for revision in delta.get("beliefs_revised", []):
        old_conf = _find_belief_confidence(wm_before, revision.get("id", ""))
        new_conf = revision.get("new_confidence", old_conf)
        try:
            old_conf = float(old_conf) if old_conf is not None else None
            new_conf = float(new_conf) if new_conf is not None else None
        except (ValueError, TypeError):
            continue
        if old_conf is not None and new_conf is not None:
            confidence_shift += abs(new_conf - old_conf)
    if beliefs_revised > 0:
        avg_shift = confidence_shift / beliefs_revised
        score += min(avg_shift, 1.0) * 0.2

    # Tension changes
    tensions_added = len(delta.get("tensions_added", []))
    tensions_resolved = len(delta.get("tensions_resolved", []))
    n_tensions = max(len(wm_before.tensions), 1)
    tension_change_ratio = (tensions_added + tensions_resolved) / n_tensions
    score += min(tension_change_ratio, 1.0) * 0.2

    # Cost belief changes
    cost_updates = delta.get("cost_beliefs_updated", {})
    if cost_updates:
        score += 0.1

    # Retirement is especially informative (we were wrong)
    if beliefs_retired > 0:
        score += 0.1

    return min(score, 1.0)


def _find_belief_confidence(wm: WorldModel, belief_id: str) -> float | None:
    """Find a belief's confidence by ID."""
    for b in wm.beliefs:
        if b.get("id") == belief_id:
            val = b.get("confidence")
            try:
                return float(val) if val is not None else None
            except (ValueError, TypeError):
                return None
    return None
