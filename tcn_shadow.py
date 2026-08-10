"""Pure shadow-research mathematics for the frozen TCN/graph protocol.

Nothing in this module has live allocation authority.  The public functions
either construct causal research inputs, calculate the graph risk haircut, or
fit expanding walk-forward shadow probabilities under the rules frozen in
``TCN_SHADOW_PROTOCOL.md``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

import alpha_core as core


FEATURE_TICKERS = (
    "QQQ",
    "SMH",
    "SPY",
    "HYG",
    "TLT",
    "GLD",
    "UUP",
    "^VIX",
)
REQUIRED_TICKERS = (*FEATURE_TICKERS, "QLD", "SOXL")
FEATURE_NAMES = (
    "qqq_log_return",
    "smh_log_return",
    "spy_log_return",
    "hyg_log_return",
    "tlt_log_return",
    "gld_log_return",
    "uup_log_return",
    "vix_log_change",
    "qqq_log_sma200_gap",
    "residual_z",
    "qqq_log_vol21_to_63",
    "smh_log_vol21_to_63",
)
GRAPH_NODES = FEATURE_TICKERS

LOOKBACK = 64
LABEL_HORIZON = 21
LABEL_OPEN_OFFSET = LABEL_HORIZON + 1
LABEL_HURDLE = 0.002

GRAPH_WINDOW = 63
GRAPH_SHRINKAGE = 0.25
GRAPH_HISTORY = 756
GRAPH_DIFFUSION_ONE = 0.50
GRAPH_DIFFUSION_TWO = 0.25
GRAPH_HAIRCUT_PERCENTILE = 0.90

TRAIN_MINIMUM = 756
VALIDATION_SIZE = 252
CALIBRATION_SIZE = 252
PARTITION_PURGE = 21
REFIT_SESSIONS = 21

TCN_CHANNELS = 8
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = (1, 2, 4, 8)
TCN_DROPOUT = 0.10
TCN_LEARNING_RATE = 0.001
TCN_WEIGHT_DECAY = 0.001
TCN_BATCH_SIZE = 64
TCN_MAX_EPOCHS = 100
TCN_PATIENCE = 10
TCN_SEEDS = (17, 29, 43)

DEADBAND_LOW = 0.55
DEADBAND_FULL = 0.65
_EPSILON = 1e-12


@dataclass(frozen=True)
class PartitionDates:
    train: tuple[pd.Timestamp, ...]
    validation: tuple[pd.Timestamp, ...]
    calibration: tuple[pd.Timestamp, ...]


@dataclass(frozen=True)
class SeedFit:
    seed: int
    best_epoch: int
    validation_loss: float
    parameter_count: int
    model: object


@dataclass(frozen=True)
class WalkForwardFit:
    fit_date: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    train_count: int
    validation_count: int
    calibration_count: int
    calibration_base_rate: float
    calibration_brier: float
    calibration_brier_baseline: float
    calibration_brier_skill: float
    calibration_log_loss: float
    calibration_ece: float
    platt_slope: float
    platt_intercept: float
    qualified: bool
    seed_epochs: tuple[int, ...]
    seed_validation_losses: tuple[float, ...]
    parameter_count: int


@dataclass(frozen=True)
class WalkForwardResult:
    predictions: pd.DataFrame
    fits: tuple[WalkForwardFit, ...]


def _clean_positive_frame(
    frame: pd.DataFrame,
    tickers: Sequence[str],
    *,
    name: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{name} must be a DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must use a DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} dates must be unique and increasing")
    missing = set(tickers) - set(str(item) for item in frame.columns)
    if missing:
        raise ValueError(f"{name} is missing tickers: {sorted(missing)}")
    clean = frame.loc[:, list(tickers)].apply(pd.to_numeric, errors="coerce")
    values = clean.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError(f"{name} contains invalid prices")
    clean.index = pd.DatetimeIndex(clean.index).normalize()
    return clean


def _rolling_volatility(log_returns: pd.Series, window: int) -> pd.Series:
    return log_returns.rolling(window, min_periods=window).std(ddof=1) * math.sqrt(
        252.0
    )


def _residual_z_frame(closes: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=closes.index, name="residual_z", dtype=float)
    required_closes = core.BETA_WINDOW + core.RESIDUAL_SCORE_WINDOW + 1
    for position in range(required_closes - 1, len(closes)):
        window = closes.iloc[position - required_closes + 1 : position + 1]
        try:
            signal = core.calculate_residual_signal(window)
        except ValueError:
            continue
        result.iloc[position] = signal.residual_z
    return result


def build_feature_frame(closes: pd.DataFrame) -> pd.DataFrame:
    """Build the 12 frozen daily channels without filling missing history."""
    clean = _clean_positive_frame(
        closes,
        REQUIRED_TICKERS,
        name="Adjusted closes",
    )
    log_returns = np.log(clean).diff()
    qqq = clean["QQQ"]
    qqq_sma = qqq.rolling(core.SMA_WINDOW, min_periods=core.SMA_WINDOW).mean()
    qqq_vol21 = _rolling_volatility(log_returns["QQQ"], 21)
    qqq_vol63 = _rolling_volatility(log_returns["QQQ"], 63)
    smh_vol21 = _rolling_volatility(log_returns["SMH"], 21)
    smh_vol63 = _rolling_volatility(log_returns["SMH"], 63)

    frame = pd.DataFrame(index=clean.index)
    for ticker, column in zip(FEATURE_TICKERS, FEATURE_NAMES[:8]):
        frame[column] = log_returns[ticker]
    frame["qqq_log_sma200_gap"] = np.log(qqq / qqq_sma)
    frame["residual_z"] = _residual_z_frame(clean.loc[:, ["QQQ", "SMH"]])
    frame["qqq_log_vol21_to_63"] = np.log(qqq_vol21 / qqq_vol63)
    frame["smh_log_vol21_to_63"] = np.log(smh_vol21 / smh_vol63)
    frame = frame.loc[:, list(FEATURE_NAMES)].replace([np.inf, -np.inf], np.nan)
    return frame


def build_relative_label_frame(opens: pd.DataFrame) -> pd.DataFrame:
    """Return the frozen t+1-open to t+22-open relative-growth label."""
    clean = _clean_positive_frame(opens, ("QLD", "SOXL"), name="Adjusted opens")
    future_qld = clean["QLD"].shift(-LABEL_OPEN_OFFSET)
    future_soxl = clean["SOXL"].shift(-LABEL_OPEN_OFFSET)
    entry_qld = clean["QLD"].shift(-1)
    entry_soxl = clean["SOXL"].shift(-1)
    advantage = (
        np.log(future_soxl / entry_soxl)
        - np.log(future_qld / entry_qld)
        - LABEL_HURDLE
    )
    end_dates = pd.Series(pd.NaT, index=clean.index, dtype="datetime64[ns]")
    if len(clean) > LABEL_OPEN_OFFSET:
        end_dates.iloc[:-LABEL_OPEN_OFFSET] = clean.index[LABEL_OPEN_OFFSET:]
    label = (advantage > 0.0).astype(float).where(advantage.notna())
    return pd.DataFrame(
        {
            "relative_log_growth": advantage,
            "label": label,
            "label_end_date": end_dates,
        },
        index=clean.index,
    )


def graph_shock_score(return_window: pd.DataFrame) -> float:
    """Calculate one frozen partial-correlation diffusion shock score."""
    if not isinstance(return_window, pd.DataFrame):
        raise ValueError("Graph returns must be a DataFrame")
    if tuple(str(item) for item in return_window.columns) != GRAPH_NODES:
        raise ValueError("Graph node order differs from the frozen protocol")
    if len(return_window) != GRAPH_WINDOW:
        raise ValueError(f"Graph window must contain {GRAPH_WINDOW} returns")
    values = return_window.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Graph returns contain invalid values")
    standard_deviations = values.std(axis=0, ddof=1)
    if not np.isfinite(standard_deviations).all() or (
        standard_deviations <= _EPSILON
    ).any():
        raise ValueError("Graph node volatility is zero or invalid")

    correlation = np.corrcoef(values, rowvar=False)
    shrunk = (1.0 - GRAPH_SHRINKAGE) * correlation + GRAPH_SHRINKAGE * np.eye(
        len(GRAPH_NODES)
    )
    precision = np.linalg.pinv(shrunk, hermitian=True)
    diagonal = np.diag(precision)
    denominator = np.sqrt(np.outer(diagonal, diagonal))
    partial = -precision / denominator
    np.fill_diagonal(partial, 0.0)
    adjacency = np.abs(partial)
    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = np.divide(
        adjacency,
        row_sums,
        out=np.zeros_like(adjacency),
        where=row_sums > _EPSILON,
    )

    current = np.abs(values[-1]) / standard_deviations
    node_shocks = np.maximum(0.0, current - 1.0)
    propagated = (
        node_shocks
        + GRAPH_DIFFUSION_ONE * adjacency @ node_shocks
        + GRAPH_DIFFUSION_TWO * adjacency @ adjacency @ node_shocks
    )
    qqq_index = GRAPH_NODES.index("QQQ")
    smh_index = GRAPH_NODES.index("SMH")
    score = float(max(propagated[qqq_index], propagated[smh_index]))
    if not np.isfinite(score) or score < 0:
        raise ValueError("Graph shock score is invalid")
    return score


def graph_haircut_from_percentile(percentile: float) -> float:
    numeric = float(percentile)
    if not np.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError("Graph percentile must be in [0, 1]")
    if numeric <= GRAPH_HAIRCUT_PERCENTILE:
        return 1.0
    return float(
        np.clip(
            (1.0 - numeric) / (1.0 - GRAPH_HAIRCUT_PERCENTILE),
            0.0,
            1.0,
        )
    )


def build_graph_haircut_frame(closes: pd.DataFrame) -> pd.DataFrame:
    """Build causal graph scores, past-only percentiles, and SOXL haircuts."""
    clean = _clean_positive_frame(closes, GRAPH_NODES, name="Graph closes")
    returns = np.log(clean).diff()
    scores = pd.Series(np.nan, index=clean.index, dtype=float, name="graph_score")
    for position in range(GRAPH_WINDOW, len(clean)):
        window = returns.iloc[position - GRAPH_WINDOW + 1 : position + 1]
        try:
            scores.iloc[position] = graph_shock_score(window)
        except ValueError:
            continue

    percentiles = pd.Series(
        np.nan,
        index=clean.index,
        dtype=float,
        name="graph_percentile",
    )
    haircuts = pd.Series(0.0, index=clean.index, dtype=float, name="graph_haircut")
    available = pd.Series(False, index=clean.index, dtype=bool, name="graph_available")
    for position, current in enumerate(scores.to_numpy(dtype=float)):
        if not np.isfinite(current):
            continue
        prior = scores.iloc[max(0, position - GRAPH_HISTORY) : position].dropna()
        if len(prior) < GRAPH_HISTORY:
            continue
        percentile = float(np.mean(prior.to_numpy(dtype=float) <= current))
        percentiles.iloc[position] = percentile
        haircuts.iloc[position] = graph_haircut_from_percentile(percentile)
        available.iloc[position] = True
    return pd.concat([scores, percentiles, haircuts, available], axis=1)


def deadband_confidence(probability: float, *, qualified: bool = True) -> float:
    if not isinstance(qualified, (bool, np.bool_)):
        raise ValueError("Qualified flag must be boolean")
    if not qualified:
        return 0.0
    numeric = float(probability)
    if not np.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError("Probability must be in [0, 1]")
    return float(
        np.clip(
            (numeric - DEADBAND_LOW) / (DEADBAND_FULL - DEADBAND_LOW),
            0.0,
            1.0,
        )
    )


def _sigmoid(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    positive = array >= 0
    result = np.empty_like(array, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("Labels and probabilities must be nonempty aligned vectors")
    if not np.isfinite(y).all() or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("Binary labels are invalid")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities are invalid")
    clipped = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if bins < 2:
        raise ValueError("Calibration bin count is too small")
    binary_log_loss(y, p)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (p >= lower) & (p <= upper if index == bins - 1 else p < upper)
        if not mask.any():
            continue
        error += float(mask.sum() / total) * abs(float(p[mask].mean() - y[mask].mean()))
    return float(error)


def fit_platt_map(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit a monotone two-parameter Platt map with deterministic Newton steps."""
    z = np.asarray(logits, dtype=float)
    y = np.asarray(labels, dtype=float)
    if z.shape != y.shape or z.ndim != 1 or len(z) < 3:
        raise ValueError("Platt inputs must be aligned vectors with at least 3 rows")
    if not np.isfinite(z).all() or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("Platt inputs are invalid")
    base = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    theta = np.array([1.0, math.log(base / (1.0 - base))], dtype=float)
    design = np.column_stack([z, np.ones(len(z), dtype=float)])
    ridge = np.diag([1e-3, 1e-6])

    def objective(candidate: np.ndarray) -> float:
        probabilities = _sigmoid(design @ candidate)
        return binary_log_loss(y, probabilities) + float(
            0.5 * 1e-3 * (candidate[0] - 1.0) ** 2
            + 0.5 * 1e-6 * candidate[1] ** 2
        )

    for _ in range(100):
        probabilities = _sigmoid(design @ theta)
        gradient = design.T @ (probabilities - y) / len(y)
        gradient += np.array([1e-3 * (theta[0] - 1.0), 1e-6 * theta[1]])
        weights = probabilities * (1.0 - probabilities)
        hessian = (design.T * weights) @ design / len(y) + ridge
        step = np.linalg.solve(hessian, gradient)
        if float(np.linalg.norm(step)) < 1e-9:
            break
        prior = objective(theta)
        scale = 1.0
        accepted = False
        for _ in range(20):
            candidate = theta - scale * step
            candidate[0] = max(0.0, candidate[0])
            if objective(candidate) <= prior + 1e-12:
                theta = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
    if theta[0] <= _EPSILON:
        theta = np.array([0.0, math.log(base / (1.0 - base))])
    return float(theta[0]), float(theta[1])


def apply_platt_map(
    logits: np.ndarray | float,
    slope: float,
    intercept: float,
) -> np.ndarray:
    a = float(slope)
    b = float(intercept)
    values = np.asarray(logits, dtype=float)
    if not np.isfinite(values).all() or not np.isfinite((a, b)).all() or a < 0:
        raise ValueError("Platt map inputs are invalid")
    return _sigmoid(a * values + b)


def available_sequence_dates(features: pd.DataFrame) -> pd.DatetimeIndex:
    if tuple(str(item) for item in features.columns) != FEATURE_NAMES:
        raise ValueError("Feature columns differ from the frozen protocol")
    values = features.to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    rolling = pd.Series(finite.astype(int), index=features.index).rolling(
        LOOKBACK,
        min_periods=LOOKBACK,
    ).sum()
    return pd.DatetimeIndex(features.index[rolling == LOOKBACK])


def split_walk_forward_partitions(
    origins: Sequence[pd.Timestamp],
) -> PartitionDates | None:
    dates = tuple(pd.Timestamp(item).normalize() for item in origins)
    calibration_start = len(dates) - CALIBRATION_SIZE
    validation_end = calibration_start - PARTITION_PURGE
    validation_start = validation_end - VALIDATION_SIZE
    train_end = validation_start - PARTITION_PURGE
    if train_end < TRAIN_MINIMUM:
        return None
    train = dates[:train_end]
    validation = dates[validation_start:validation_end]
    calibration = dates[calibration_start:]
    if not (
        len(train) >= TRAIN_MINIMUM
        and len(validation) == VALIDATION_SIZE
        and len(calibration) == CALIBRATION_SIZE
    ):
        raise RuntimeError("Walk-forward partition sizes are inconsistent")
    return PartitionDates(train, validation, calibration)


def _standardizer(
    features: pd.DataFrame,
    final_training_origin: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    training_rows = features.loc[:final_training_origin].dropna()
    if training_rows.empty:
        raise RuntimeError("Feature standardizer has no model-training rows")
    mean = training_rows.to_numpy(dtype=float).mean(axis=0)
    standard_deviation = training_rows.to_numpy(dtype=float).std(axis=0, ddof=1)
    if not np.isfinite(mean).all() or not np.isfinite(standard_deviation).all():
        raise RuntimeError("Feature standardizer is invalid")
    if (standard_deviation <= _EPSILON).any():
        raise RuntimeError("A model-training feature has zero variance")
    return mean, standard_deviation


def make_sequence_array(
    features: pd.DataFrame,
    origins: Iterable[pd.Timestamp],
    mean: np.ndarray,
    standard_deviation: np.ndarray,
) -> np.ndarray:
    positions = {pd.Timestamp(date): position for position, date in enumerate(features.index)}
    sequences: list[np.ndarray] = []
    for origin in origins:
        date = pd.Timestamp(origin)
        if date not in positions:
            raise RuntimeError("Sequence origin is absent from the feature frame")
        position = positions[date]
        if position < LOOKBACK - 1:
            raise RuntimeError("Sequence origin lacks the frozen lookback")
        window = features.iloc[position - LOOKBACK + 1 : position + 1].to_numpy(
            dtype=float
        )
        standardized = np.clip((window - mean) / standard_deviation, -5.0, 5.0)
        if not np.isfinite(standardized).all():
            raise RuntimeError("A standardized TCN sequence is invalid")
        sequences.append(standardized)
    if not sequences:
        return np.empty((0, LOOKBACK, len(FEATURE_NAMES)), dtype=np.float32)
    return np.asarray(sequences, dtype=np.float32)


def _load_torch():
    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "TCN research requires the optional PyTorch research dependency"
        ) from exc
    return torch, nn, functional


def build_tcn_model(input_channels: int = len(FEATURE_NAMES)):
    """Create the frozen small causal residual TCN."""
    torch, nn, functional = _load_torch()

    class _ChannelLayerNorm(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.normalization = nn.LayerNorm(channels)

        def forward(self, values):
            return self.normalization(values.transpose(1, 2)).transpose(1, 2)

    class _Block(nn.Module):
        def __init__(self, incoming: int, outgoing: int, dilation: int) -> None:
            super().__init__()
            self.pad = (TCN_KERNEL_SIZE - 1) * dilation
            self.conv1 = nn.Conv1d(
                incoming,
                outgoing,
                TCN_KERNEL_SIZE,
                dilation=dilation,
            )
            self.norm1 = _ChannelLayerNorm(outgoing)
            self.conv2 = nn.Conv1d(
                outgoing,
                outgoing,
                TCN_KERNEL_SIZE,
                dilation=dilation,
            )
            self.norm2 = _ChannelLayerNorm(outgoing)
            self.dropout = nn.Dropout(TCN_DROPOUT)
            self.projection = (
                nn.Identity() if incoming == outgoing else nn.Conv1d(incoming, outgoing, 1)
            )

        def forward(self, values):
            residual = self.projection(values)
            output = self.conv1(functional.pad(values, (self.pad, 0)))
            output = self.dropout(functional.gelu(self.norm1(output)))
            output = self.conv2(functional.pad(output, (self.pad, 0)))
            output = self.dropout(functional.gelu(self.norm2(output)))
            return output + residual

    class _TCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks = []
            incoming = input_channels
            for dilation in TCN_DILATIONS:
                blocks.append(_Block(incoming, TCN_CHANNELS, dilation))
                incoming = TCN_CHANNELS
            self.blocks = nn.Sequential(*blocks)
            self.head = nn.Linear(TCN_CHANNELS, 1)

        def forward(self, values):
            output = values.transpose(1, 2)
            output = self.blocks(output)
            output = output.mean(dim=2)
            return self.head(output).squeeze(-1)

    return _TCN()


def tcn_parameter_count() -> int:
    model = build_tcn_model()
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _model_logits(model: object, sequences: np.ndarray) -> np.ndarray:
    torch, _, _ = _load_torch()
    model.eval()
    with torch.no_grad():
        values = torch.as_tensor(sequences, dtype=torch.float32)
        return model(values).detach().cpu().numpy().astype(float)


def fit_tcn_seed(
    train_sequences: np.ndarray,
    train_labels: np.ndarray,
    validation_sequences: np.ndarray,
    validation_labels: np.ndarray,
    *,
    seed: int,
) -> SeedFit:
    """Fit one frozen seed with validation-only early stopping."""
    torch, nn, _ = _load_torch()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(4, int(torch.get_num_threads()))))
    model = build_tcn_model(train_sequences.shape[2])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TCN_LEARNING_RATE,
        weight_decay=TCN_WEIGHT_DECAY,
    )
    loss_function = nn.BCEWithLogitsLoss()
    x_train = torch.as_tensor(train_sequences, dtype=torch.float32)
    y_train = torch.as_tensor(train_labels, dtype=torch.float32)
    x_validation = torch.as_tensor(validation_sequences, dtype=torch.float32)
    y_validation = torch.as_tensor(validation_labels, dtype=torch.float32)
    generator = torch.Generator().manual_seed(int(seed))

    best_state = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, TCN_MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), TCN_BATCH_SIZE):
            batch = order[start : start + TCN_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_function(model(x_validation), y_validation).detach().cpu()
            )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= TCN_PATIENCE:
                break
    model.load_state_dict(best_state)
    return SeedFit(
        seed=int(seed),
        best_epoch=best_epoch,
        validation_loss=float(best_loss),
        parameter_count=int(sum(parameter.numel() for parameter in model.parameters())),
        model=model,
    )


def walk_forward_tcn(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    progress: Callable[[WalkForwardFit], None] | None = None,
) -> WalkForwardResult:
    """Generate frozen expanding, purged, calibrated shadow probabilities."""
    if not features.index.equals(labels.index):
        raise ValueError("Feature and label dates differ")
    sequence_dates = available_sequence_dates(features)
    output = pd.DataFrame(
        {
            "raw_logit": np.nan,
            "probability": np.nan,
            "confidence": 0.0,
            "qualified": False,
            "fit_date": pd.NaT,
            "calibration_brier_skill": np.nan,
        },
        index=features.index,
    )
    fits: list[WalkForwardFit] = []
    label_end = pd.to_datetime(labels["label_end_date"])

    for block_start in range(0, len(sequence_dates), REFIT_SESSIONS):
        fit_date = pd.Timestamp(sequence_dates[block_start])
        eligible = [
            pd.Timestamp(date)
            for date in sequence_dates[:block_start]
            if pd.notna(labels.loc[date, "label"])
            and pd.notna(label_end.loc[date])
            and pd.Timestamp(label_end.loc[date]) < fit_date
        ]
        partitions = split_walk_forward_partitions(eligible)
        if partitions is None:
            continue

        mean, standard_deviation = _standardizer(features, partitions.train[-1])
        train_x = make_sequence_array(
            features,
            partitions.train,
            mean,
            standard_deviation,
        )
        validation_x = make_sequence_array(
            features,
            partitions.validation,
            mean,
            standard_deviation,
        )
        calibration_x = make_sequence_array(
            features,
            partitions.calibration,
            mean,
            standard_deviation,
        )
        train_y = labels.loc[list(partitions.train), "label"].to_numpy(dtype=float)
        validation_y = labels.loc[
            list(partitions.validation), "label"
        ].to_numpy(dtype=float)
        calibration_y = labels.loc[
            list(partitions.calibration), "label"
        ].to_numpy(dtype=float)

        seed_fits = tuple(
            fit_tcn_seed(
                train_x,
                train_y,
                validation_x,
                validation_y,
                seed=seed,
            )
            for seed in TCN_SEEDS
        )
        calibration_logits = np.mean(
            [_model_logits(fit.model, calibration_x) for fit in seed_fits],
            axis=0,
        )
        slope, intercept = fit_platt_map(calibration_logits, calibration_y)
        calibration_probability = apply_platt_map(
            calibration_logits,
            slope,
            intercept,
        )
        brier = float(np.mean((calibration_probability - calibration_y) ** 2))
        base_rate = float(calibration_y.mean())
        brier_baseline = float(np.mean((base_rate - calibration_y) ** 2))
        brier_skill = (
            float(1.0 - brier / brier_baseline)
            if brier_baseline > _EPSILON
            else -math.inf
        )
        qualified = bool(np.isfinite(brier_skill) and brier_skill > 0.0)
        calibration_loss = binary_log_loss(calibration_y, calibration_probability)
        calibration_ece = expected_calibration_error(
            calibration_y,
            calibration_probability,
        )

        block_dates = sequence_dates[
            block_start : block_start + REFIT_SESSIONS
        ]
        block_x = make_sequence_array(
            features,
            block_dates,
            mean,
            standard_deviation,
        )
        block_logits = np.mean(
            [_model_logits(fit.model, block_x) for fit in seed_fits],
            axis=0,
        )
        block_probability = apply_platt_map(block_logits, slope, intercept)
        output.loc[block_dates, "raw_logit"] = block_logits
        output.loc[block_dates, "probability"] = block_probability
        output.loc[block_dates, "confidence"] = [
            deadband_confidence(value, qualified=qualified)
            for value in block_probability
        ]
        output.loc[block_dates, "qualified"] = qualified
        output.loc[block_dates, "fit_date"] = fit_date
        output.loc[block_dates, "calibration_brier_skill"] = brier_skill

        fit_record = WalkForwardFit(
                fit_date=fit_date,
                train_start=partitions.train[0],
                train_end=partitions.train[-1],
                validation_start=partitions.validation[0],
                validation_end=partitions.validation[-1],
                calibration_start=partitions.calibration[0],
                calibration_end=partitions.calibration[-1],
                train_count=len(partitions.train),
                validation_count=len(partitions.validation),
                calibration_count=len(partitions.calibration),
                calibration_base_rate=base_rate,
                calibration_brier=brier,
                calibration_brier_baseline=brier_baseline,
                calibration_brier_skill=brier_skill,
                calibration_log_loss=calibration_loss,
                calibration_ece=calibration_ece,
                platt_slope=slope,
                platt_intercept=intercept,
                qualified=qualified,
                seed_epochs=tuple(item.best_epoch for item in seed_fits),
                seed_validation_losses=tuple(
                    item.validation_loss for item in seed_fits
                ),
                parameter_count=seed_fits[0].parameter_count,
            )
        fits.append(fit_record)
        if progress is not None:
            progress(fit_record)
    output["qualified"] = output["qualified"].astype(bool)
    return WalkForwardResult(predictions=output, fits=tuple(fits))
