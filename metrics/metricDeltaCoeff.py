from metrics.metric import Metric


class MetricDeltaCoeff(Metric):
    """Tracks the model's delta coefficient (the scale on every δ in forward)."""

    def __init__(self):
        super().__init__()
        self.total_coeff = 0.0
        self.last_coeff = None

    def accumulate(self, model=None, **kwargs):
        if model is None or not hasattr(model, "get_delta_coeff"):
            return
        self.last_coeff = float(model.get_delta_coeff())
        self.total_coeff += self.last_coeff
        self.total_samples += 1

    def compute_metric(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.total_coeff / self.total_samples

    def pbar_value(self) -> str | None:
        if self.last_coeff is None:
            return None
        return f"{self.last_coeff:.3f}"
