"""Piecewise-linear schedule for the delta coefficient (0 = shared-only, 1 = full)."""


def delta_coeff_at_epoch(epoch: int, hold_until: int, ramp_end: int) -> float:
    """Linear coeff in ``[0, 1]`` for a 0-based ``epoch``.

    Stays 0 through ``hold_until`` (inclusive of the first ramp epoch, where the
    value is 0), reaches 1 at ``ramp_end``, and stays 1 afterwards.

    ``ramp_end <= hold_until`` is a step: 0 before ``hold_until``, 1 from then on.
    ``hold_until = ramp_end = 0`` is "always 1".
    """
    if epoch < hold_until:
        return 0.0
    if ramp_end <= hold_until:
        return 1.0
    if epoch >= ramp_end:
        return 1.0
    return (epoch - hold_until) / (ramp_end - hold_until)
