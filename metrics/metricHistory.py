import csv
from typing import Dict
from pathlib import Path
import pickle
from metrics.metric import Metric
import matplotlib.pyplot as plt


class DictHistoryMetrics:
    def __init__(self, path: Path | str, split: str, saveCheckpoints:bool=True):
        # Coerced rather than trusted: a str would survive a whole epoch of training and
        # only blow up on the first save, at the / operator.
        self.path = Path(path)
        self.split = split
        self.histories: Dict[str, MetricHistory] = {}

    def addHistoryMetric(
        self,
        name: str,
        metricClass: type[Metric],
        higher_is_better: bool = True,
        optional: bool = False,
        **metric_kwargs,
    ):
        full_name = f"{self.split}_{name}"
        self.histories[name] = MetricHistory(
            self.path,
            metricClass,
            full_name,
            higher_is_better=higher_is_better,
            optional=optional,
            metric_kwargs=metric_kwargs,
        )

    def accumulate(self, **kwargs):
        for histmetric in self.histories.values():
            histmetric.accumulate_last_point(**kwargs)

    def create_point(self):
        for histmetric in self.histories.values():
            histmetric.create_point()

    def get(self, name: str) -> "MetricHistory":
        return self.histories[name]

    def pbar(self) -> Dict[str, str]:
        """Progress bar entries of every metric that asked for one."""
        entries = {}
        for name, histmetric in self.histories.items():
            value = histmetric.pbar_value()
            if value is not None:
                entries[name] = value
        return entries

    def num_epochs(self) -> int:
        # Metrics added after a run started have no history yet, and they should not drag
        # the resume point back to epoch 0.
        lengths = [len(h.metricHistory) for h in self.histories.values() if h.metricHistory]
        if not lengths:
            return 0
        return min(lengths)

    def plot_history(self, names: list[str] | None = None):
        names = names if names is not None else list(self.histories.keys())
        Path(self.path / "plots").mkdir(parents=True, exist_ok=True)
        for name in names:
            self.histories[name].plot_history()

    def save(self):
        for histmetric in self.histories.values():
            histmetric.save()

    def load(self) -> bool:
        if not self.histories:
            return False
        loaded = True
        for histmetric in self.histories.values():
            if not histmetric.load() and not histmetric.optional:
                loaded = False
        return loaded

    def drop_empty_trailing(self):
        for histmetric in self.histories.values():
            histmetric.drop_empty_trailing()

    def truncate(self, n: int):
        for histmetric in self.histories.values():
            histmetric.truncate(n)

    def pad_to(self, n: int):
        for histmetric in self.histories.values():
            histmetric.pad_to(n)

    def restore_best(self, name: str | None = None):
        if name is not None:
            self.histories[name].restore_best()
            return
        for histmetric in self.histories.values():
            histmetric.restore_best()

    def last_is_best(self, name: str) -> bool:
        return self.histories[name].last_is_best()

    def print_last(self):
        for histmetric in self.histories.values():
            histmetric.print_last()


class MetricHistory:
    def __init__(
        self,
        path: Path,
        metric_cls: type[Metric],
        name: str,
        higher_is_better: bool = True,
        optional: bool = False,
        metric_kwargs: dict | None = None,
    ):
        self.metricHistory: list[Metric] = []
        self.path = Path(path)
        self.metric_cls = metric_cls
        self.name = name
        self.higher_is_better = higher_is_better
        self.optional = optional
        self.metric_kwargs = metric_kwargs or {}
        self.best_metric = None
        self.best_metric_index = None

    def append(self, metric: Metric):
        self.metricHistory.append(metric)

    def compute_last(self):
        return self.metricHistory[-1].compute_metric()

    def create_point(self):
        self.append(self.metric_cls(**self.metric_kwargs))

    def accumulate_last_point(self, **kwargs):
        self.metricHistory[-1].accumulate(**kwargs)

    def pbar_value(self) -> str | None:
        if not self.metricHistory:
            return None
        return self.metricHistory[-1].pbar_value()

    def save(self):
        Path(self.path / "history").mkdir(parents=True, exist_ok=True)
        Path(self.path / "results").mkdir(parents=True, exist_ok=True)
        # Atomic write: a killed job mid-dump left a truncated .pkl that made
        # RETAKE crash with EOFError on the next segment.
        final = self.history_path()
        tmp = final.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.metricHistory, f)
        tmp.replace(final)
        self.toCSV()

    def toCSV(self):
        rows = []
        columns: list[str] = []
        for i, metric in enumerate(self.metricHistory):
            if metric.total_samples == 0:
                continue
            components = metric.compute_components()
            for key in components:
                if key not in columns:
                    columns.append(key)
            rows.append((i, float(metric.compute_metric()), components))
        csv_path = self.path / "results" / f"{self.name}_history.csv"
        tmp = csv_path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "value", *columns])
            for i, value, components in rows:
                writer.writerow([i, value, *(components.get(key, "") for key in columns)])
        tmp.replace(csv_path)

    def history_path(self) -> Path:
        return self.path / "history" / f"{self.name}_history.pkl"

    def csv_path(self) -> Path:
        return self.path / "results" / f"{self.name}_history.csv"

    def load(self) -> bool:
        path = self.history_path()
        if path.is_file():
            try:
                with open(path, "rb") as f:
                    self.metricHistory = pickle.load(f)
                return True
            except (EOFError, pickle.UnpicklingError, AttributeError) as exc:
                print(f"Warning: corrupt history pickle {path} ({exc}); trying CSV")
        return self._load_from_csv()

    def _load_from_csv(self) -> bool:
        """Rebuild a minimal history from the CSV so RETAKE can continue after a truncated pickle."""
        csv_path = self.csv_path()
        if not csv_path.is_file():
            return False
        rows: list[tuple[int, float]] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if "Epoch" not in (reader.fieldnames or []) or "value" not in (reader.fieldnames or []):
                return False
            for row in reader:
                rows.append((int(float(row["Epoch"])), float(row["value"])))
        if not rows:
            return False
        by_epoch = {epoch: value for epoch, value in rows}
        n = max(by_epoch) + 1
        history: list[Metric] = []
        for epoch in range(n):
            metric = self.metric_cls(**self.metric_kwargs)
            if epoch in by_epoch:
                self._seed_metric_value(metric, by_epoch[epoch])
            history.append(metric)
        self.metricHistory = history
        print(f"Rebuilt {self.name} history from CSV ({len(by_epoch)} epochs)")
        return True

    @staticmethod
    def _seed_metric_value(metric: Metric, value: float) -> None:
        """Make ``compute_metric()`` return ``value`` without replaying batches."""
        metric.total_samples = 1
        if hasattr(metric, "total_loss"):
            metric.total_loss = float(value)
            metric.last_loss = float(value)
        elif hasattr(metric, "total_correct"):
            metric.total_correct = float(value)
        elif hasattr(metric, "total_grad_norm"):
            metric.total_grad_norm = float(value)
            metric.last_grad_norm = float(value)
        elif hasattr(metric, "total_lr"):
            metric.total_lr = float(value)
            metric.last_lr = float(value)
        elif hasattr(metric, "total_coeff"):
            metric.total_coeff = float(value)
            metric.last_coeff = float(value)
        elif hasattr(metric, "total_ratios"):
            # Delta-ratio plots need the per-part series; CSV rebuild only keeps the mean.
            metric.total_ratios = {"mean": float(value)}
            metric.last_mean = float(value)
        else:
            # Unknown metric shape: leave empty (pad/truncate still keep epoch alignment).
            metric.total_samples = 0


    def drop_empty_trailing(self):
        while self.metricHistory and self.metricHistory[-1].total_samples == 0:
            self.metricHistory.pop()

    def truncate(self, n: int):
        self.metricHistory = self.metricHistory[:n]

    def pad_to(self, n: int):
        while len(self.metricHistory) < n:
            self.create_point()

    def _is_better(self, value, best) -> bool:
        if self.higher_is_better:
            return value > best
        return value < best

    def restore_best(self):
        if not self.metricHistory:
            self.best_metric = None
            self.best_metric_index = None
            return
        best_i = 0
        best_v = self.metricHistory[0].compute_metric()
        for i, metric in enumerate(self.metricHistory):
            value = metric.compute_metric()
            if self._is_better(value, best_v):
                best_v = value
                best_i = i
        self.best_metric = best_v
        self.best_metric_index = best_i

    def plot_history(self):
        Path(self.path / "plots").mkdir(parents=True, exist_ok=True)
        # A metric whose epoch value is an aggregate of many series draws its own figures.
        plot_series = getattr(self.metric_cls, "plot_series", None)
        if plot_series is not None:
            plot_series(self.metricHistory, self.path / "plots", self.name)
            return
        values = [float(metric.compute_metric()) for metric in self.metricHistory]
        plt.plot(values)
        plt.xlabel("Epoch")
        plt.ylabel(self.metric_cls.__name__)
        plt.title(f"{self.name} History")
        plt.savefig(self.path / "plots" / f"{self.name}_{self.metric_cls.__name__}_history.png")
        plt.close()

    def last_is_best(self) -> bool:
        if not self.metricHistory:
            return False
        current = self.metricHistory[-1].compute_metric()
        if self.best_metric is None:
            self.best_metric = current
            self.best_metric_index = len(self.metricHistory) - 1
            return True
        if self._is_better(current, self.best_metric):
            self.best_metric = current
            self.best_metric_index = len(self.metricHistory) - 1
        return self.best_metric_index == len(self.metricHistory) - 1

    def print_last(self):
        print(f"{self.name}: {self.metricHistory[-1].compute_metric()}")
