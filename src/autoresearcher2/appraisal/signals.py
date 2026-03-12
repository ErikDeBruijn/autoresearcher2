import numpy as np

from autoresearcher2.core.schema import InterventionSchema


def compute_appraisal(
    schema: InterventionSchema,
    cell_index: int,
    outcome: float,
    snapshot_before: dict,
    snapshot_after: dict,
    prediction_change_threshold: float = 0.01,
) -> dict[str, float]:
    """Compute learntropy-inspired appraisal signals for an observation.

    Appraisal does NOT update the model. The caller is responsible for:
      1. Taking snapshot_before = model.snapshot()
      2. Calling model.update(cell_index, outcome)
      3. Taking snapshot_after = model.snapshot()
      4. Passing both snapshots here
    """
    mu_before, sigma_before = snapshot_before["mu_w"], snapshot_before["sigma_w"]
    mu_after = snapshot_after["mu_w"]

    # Infer include_interactions from snapshot dimensions
    include_interactions = len(mu_before) > schema.n_main_effects
    x = schema.feature_vector(cell_index, include_interactions=include_interactions)
    mean_before = float(x @ mu_before)
    var_before = float(x @ sigma_before @ x) + snapshot_before["noise_variance"]

    prediction_error = abs(outcome - mean_before)
    std = max(np.sqrt(var_before), 1e-8)
    surprise = float(prediction_error / std)
    surprise_norm = float(1 - np.exp(-0.5 * surprise**2))

    epistemic_var = float(x @ sigma_before @ x)
    confidence = float(1.0 / (1.0 + epistemic_var))

    theory_conflict = float(surprise_norm * confidence)

    impact_count = 0
    for c in range(schema.n_cells):
        xc = schema.feature_vector(c, include_interactions=include_interactions)
        pred_before = float(xc @ mu_before)
        pred_after = float(xc @ mu_after)
        if abs(pred_after - pred_before) > prediction_change_threshold:
            impact_count += 1
    prediction_impact_breadth = float(impact_count)

    # learntropy diverges from theory_conflict by incorporating how broadly
    # the update reshaped predictions — "confidently wrong AND it changed the map"
    normalized_breadth = np.sqrt(prediction_impact_breadth / schema.n_cells)
    learntropy = float(theory_conflict * normalized_breadth)

    return {
        "surprise": surprise_norm,
        "theory_conflict": theory_conflict,
        "prediction_impact_breadth": prediction_impact_breadth,
        "learntropy": learntropy,
    }
