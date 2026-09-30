"""Score aggregation for the model-level expansion in CoVer+."""

import torch


def add_score(totals, name, score):
    """Accumulate a model's per-image, view-averaged scores without reordering."""
    score = torch.as_tensor(score).detach().cpu()
    if score.ndim != 1 or score.numel() == 0:
        raise ValueError(f'{name}: expected a nonempty one-dimensional score array')
    if name in totals:
        if totals[name].shape != score.shape:
            raise ValueError(f'{name}: different image count across models')
        totals[name] += score
    else:
        totals[name] = score.clone()


def finish_scores(totals, model_count):
    """Return equal-weight mean over all models and their equal-weight view means."""
    if model_count < 1 or not totals:
        raise ValueError('At least one model and dataset score are required')
    return {name: total / model_count for name, total in totals.items()}
