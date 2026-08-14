"""Pure quantitative primitives for the QLD-core/SOXL-overlay strategy.

This module has no filesystem, network, email, environment-variable, or
production-state side effects.  It is intentionally shared by research and
production so the audited equations cannot silently diverge.

The volatility model uses daily squared returns.  It is therefore described as
HAR-style; it is not the intraday realized-volatility HAR-RV estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


QQQ = "QQQ"
SMH = "SMH"
QLD = "QLD"
SOXL = "SOXL"
CASH = "CASH"

# Bump this value only after reviewing a decision-semantic change. Source-file
# bytes belong in implementation lineage, not in the live allocation identity.
DECISION_SEMANTIC_REVISION = "qld-soxl-tiered-core-v2"
MODEL_HISTORY_START = "2010-03-11"

SMA_WINDOW = 200
BETA_WINDOW = 126
RESIDUAL_SCORE_WINDOW = 63
ALPHA_REVIEW_SESSIONS = 21
BULLISH_REENTRY_CLOSES = 2
VOLATILITY_UPSHIFT_CLOSES = 5

VARIANCE_FAST_WINDOW = 5
VARIANCE_MONTH_WINDOW = 21
VARIANCE_QUARTER_WINDOW = 63
VARIANCE_HORIZON = 21
VARIANCE_MIN_TRAINING = 756
RIDGE_ALPHA = 10.0

VOLATILITY_BUDGET = 0.55
SOXL_WEIGHT_GRID = (0.0, 0.15, 0.25, 0.35)
MAX_SOXL_WEIGHT = 0.35
MAX_ADVERTISED_DAILY_EXPOSURE = 2.35

_EPSILON = 1e-12
_WEIGHT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ResidualSignal:
    """Separated-window SMH residual-strength estimate."""

    intercept: float
    beta: float
    residual_momentum: float
    residual_sigma: float
    residual_z: float
    estimation_start: pd.Timestamp
    estimation_end: pd.Timestamp
    scoring_start: pd.Timestamp
    scoring_end: pd.Timestamp


@dataclass(frozen=True)
class RidgeFit:
    """Auditable standardized ridge fit and current prediction."""

    feature_names: tuple[str, ...]
    # ``training_end`` is the latest information date available to the fit.
    # ``last_label_origin`` is earlier because its forward label ends on the
    # information date.
    training_end: pd.Timestamp
    last_label_origin: pd.Timestamp
    sample_count: int
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    current_features: tuple[float, ...]
    prediction: float
    smearing_factor: float


@dataclass(frozen=True)
class VarianceForecast:
    """One asset's causal 21-session annualized volatility forecast."""

    model_volatility: float
    trailing_volatility_21: float
    trailing_volatility_63: float
    sizing_volatility: float
    ridge_fit: RidgeFit


@dataclass(frozen=True)
class PortfolioVolatility:
    """Volatility inputs used to select the incremental SOXL weight."""

    qld: VarianceForecast
    soxl: VarianceForecast
    correlation_21: float | None
    correlation_63: float | None
    sizing_correlation: float
    raw_soxl_weight: float
    raw_portfolio_volatility: float


@dataclass(frozen=True)
class OverlayState:
    """Minimal state for causal overlay transitions."""

    overlay_active: bool = False
    eligible_streak: int = 0
    soxl_weight: float = 0.0
    soxl_weight_date: str = ""
    pending_soxl_weight: float = 0.0
    pending_scale_days: int = 0
    last_alpha_review_date: str = ""
    last_processed_signal_date: str = ""


@dataclass(frozen=True)
class OverlayTransition:
    """Result of processing one distinct completed session."""

    state: OverlayState
    reason: str
    alpha_reviewed: bool
    structural_change: bool


def _as_clean_close_frame(
    price_data: pd.DataFrame,
    tickers: Sequence[str],
) -> pd.DataFrame:
    if not isinstance(price_data, pd.DataFrame):
        raise ValueError("Price data must be a pandas DataFrame")
    if not isinstance(price_data.index, pd.DatetimeIndex):
        raise ValueError("Price data must use a DatetimeIndex")
    if price_data.index.has_duplicates or not price_data.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and increasing")
    missing = set(tickers) - set(str(item) for item in price_data.columns)
    if missing:
        raise ValueError(f"Price data is missing tickers: {sorted(missing)}")
    result = price_data.loc[:, list(tickers)].apply(pd.to_numeric, errors="coerce")
    values = result.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError("Required price history contains invalid values")
    return result


def _finite_positive(value: float, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return numeric


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def advertised_daily_exposure(soxl_weight: float) -> float:
    """Return advertised daily exposure for QLD=(1-w), SOXL=w."""
    weight = float(soxl_weight)
    if not np.isfinite(weight) or weight < 0 or weight > MAX_SOXL_WEIGHT + _EPSILON:
        raise ValueError("SOXL weight is outside the strategic range")
    return 2.0 * (1.0 - weight) + 3.0 * weight


def calculate_residual_signal(
    price_data: pd.DataFrame,
    *,
    beta_window: int = BETA_WINDOW,
    score_window: int = RESIDUAL_SCORE_WINDOW,
) -> ResidualSignal:
    """Fit SMH on QQQ, then score a strictly separated residual window.

    ``beta_window`` log returns immediately precede ``score_window`` log
    returns.  The residual standard error uses ``n - 2`` degrees of freedom
    because both an intercept and slope are estimated.
    """
    if beta_window < 3 or score_window < 1:
        raise ValueError("Residual windows are too short")
    clean = _as_clean_close_frame(price_data, (QQQ, SMH))
    returns = np.log(clean).diff().dropna()
    required = beta_window + score_window
    if len(returns) < required:
        raise ValueError(
            f"Residual signal needs {required + 1} closes; received {len(clean)}"
        )

    estimation = returns.iloc[-required:-score_window]
    scoring = returns.iloc[-score_window:]
    x = estimation[QQQ].to_numpy(dtype=float)
    y = estimation[SMH].to_numpy(dtype=float)
    x_centered = x - x.mean()
    denominator = float(x_centered @ x_centered)
    if not np.isfinite(denominator) or denominator <= _EPSILON:
        raise ValueError("QQQ estimation variance is zero or invalid")

    beta = float(x_centered @ (y - y.mean()) / denominator)
    intercept = float(y.mean() - beta * x.mean())
    fitted_residuals = y - (intercept + beta * x)
    residual_dof = beta_window - 2
    residual_variance = float(fitted_residuals @ fitted_residuals / residual_dof)
    if not np.isfinite(residual_variance) or residual_variance <= _EPSILON:
        raise ValueError("Residual estimation variance is zero or invalid")
    residual_sigma = float(np.sqrt(residual_variance))

    score_x = scoring[QQQ].to_numpy(dtype=float)
    score_y = scoring[SMH].to_numpy(dtype=float)
    score_residuals = score_y - (intercept + beta * score_x)
    residual_momentum = float(score_residuals.sum())
    residual_z = float(
        residual_momentum / (residual_sigma * np.sqrt(float(score_window)))
    )
    values = (intercept, beta, residual_momentum, residual_sigma, residual_z)
    if not np.isfinite(values).all():
        raise ValueError("Residual signal is invalid")

    return ResidualSignal(
        intercept=intercept,
        beta=beta,
        residual_momentum=residual_momentum,
        residual_sigma=residual_sigma,
        residual_z=residual_z,
        estimation_start=pd.Timestamp(estimation.index[0]).normalize(),
        estimation_end=pd.Timestamp(estimation.index[-1]).normalize(),
        scoring_start=pd.Timestamp(scoring.index[0]).normalize(),
        scoring_end=pd.Timestamp(scoring.index[-1]).normalize(),
    )


def calculate_residual_signal_frame(
    price_data: pd.DataFrame,
    *,
    beta_window: int = BETA_WINDOW,
    score_window: int = RESIDUAL_SCORE_WINDOW,
) -> pd.DataFrame:
    """Calculate the exact residual signal causally for every eligible date."""
    clean = _as_clean_close_frame(price_data, (QQQ, SMH))
    required_closes = beta_window + score_window + 1
    columns = (
        "residual_intercept",
        "smh_beta",
        "residual_momentum",
        "residual_sigma",
        "residual_z",
    )
    result = pd.DataFrame(np.nan, index=clean.index, columns=columns)
    for position in range(required_closes - 1, len(clean)):
        signal = calculate_residual_signal(
            clean.iloc[position - required_closes + 1 : position + 1],
            beta_window=beta_window,
            score_window=score_window,
        )
        result.iloc[position] = (
            signal.intercept,
            signal.beta,
            signal.residual_momentum,
            signal.residual_sigma,
            signal.residual_z,
        )
    return result


def _trailing_downside_variance(returns: pd.Series, window: int) -> pd.Series:
    """Annualized mean squared negative daily return.

    Positive-return observations contribute zero. A truly zero window remains
    invalid rather than being replaced with a synthetic value.
    """
    negative_squared = returns.where(returns < 0.0, 0.0).pow(2)
    return negative_squared.rolling(window, min_periods=window).mean() * 252.0


def build_variance_learning_frame(close: pd.Series) -> pd.DataFrame:
    """Build HAR-style features and causal future-variance labels."""
    numeric = pd.to_numeric(close, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if (
        not isinstance(numeric.index, pd.DatetimeIndex)
        or numeric.index.has_duplicates
        or not numeric.index.is_monotonic_increasing
    ):
        raise ValueError("Close series must use unique increasing dates")
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError("Close series contains invalid prices")

    returns = np.log(numeric).diff()
    squared = returns.pow(2)
    frame = pd.DataFrame(index=numeric.index)
    variance_5 = (
        squared.rolling(
            VARIANCE_FAST_WINDOW,
            min_periods=VARIANCE_FAST_WINDOW,
        ).mean()
        * 252.0
    )
    variance_21 = (
        squared.rolling(
            VARIANCE_MONTH_WINDOW,
            min_periods=VARIANCE_MONTH_WINDOW,
        ).mean()
        * 252.0
    )
    variance_63 = (
        squared.rolling(
            VARIANCE_QUARTER_WINDOW,
            min_periods=VARIANCE_QUARTER_WINDOW,
        ).mean()
        * 252.0
    )
    downside_variance_21 = _trailing_downside_variance(
        returns,
        VARIANCE_MONTH_WINDOW,
    )
    frame["log_var_5"] = np.log(variance_5.where(variance_5 > 0))
    frame["log_var_21"] = np.log(variance_21.where(variance_21 > 0))
    frame["log_var_63"] = np.log(variance_63.where(variance_63 > 0))
    frame["log_downside_var_21"] = np.log(
        downside_variance_21.where(downside_variance_21 > 0)
    )

    future = np.full(len(frame), np.nan, dtype=float)
    squared_values = squared.to_numpy(dtype=float)
    for position in range(len(frame) - VARIANCE_HORIZON):
        window = squared_values[
            position + 1 : position + VARIANCE_HORIZON + 1
        ]
        if len(window) == VARIANCE_HORIZON and np.isfinite(window).all():
            variance = float(np.mean(window) * 252.0)
            if variance > 0:
                future[position] = np.log(variance)
    frame["future_log_var_21"] = future
    return frame.replace([np.inf, -np.inf], np.nan)


def standardized_ridge_fit_predict(
    features: pd.DataFrame,
    target: pd.Series,
    current_features: pd.Series,
    *,
    alpha: float = RIDGE_ALPHA,
    min_samples: int = VARIANCE_MIN_TRAINING,
) -> RidgeFit:
    """Fit a deterministic ridge model with train-only standardization."""
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("Training features are empty")
    if "_target" in features.columns:
        raise ValueError("'_target' is reserved and cannot be a feature")
    if not features.index.equals(target.index):
        raise ValueError("Feature and target dates differ")
    if tuple(str(item) for item in features.columns) != tuple(
        str(item) for item in current_features.index
    ):
        raise ValueError("Current feature order differs from training features")
    penalty = _finite_positive(alpha, "Ridge alpha")
    if min_samples < 3:
        raise ValueError("Minimum ridge sample count is too small")

    combined = features.copy()
    combined["_target"] = pd.to_numeric(target, errors="coerce")
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
    if len(combined) < min_samples:
        raise ValueError(
            f"Ridge model needs {min_samples} labeled rows; received {len(combined)}"
        )

    x = combined.loc[:, features.columns].to_numpy(dtype=float)
    y = combined["_target"].to_numpy(dtype=float)
    current = pd.to_numeric(current_features, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(current).all():
        raise ValueError("Current ridge features are invalid")

    means = x.mean(axis=0)
    scales = x.std(axis=0, ddof=1)
    if not np.isfinite(scales).all() or (scales <= _EPSILON).any():
        raise ValueError("A ridge training feature has zero or invalid variance")
    standardized = (x - means) / scales
    current_standardized = (current - means) / scales
    y_mean = float(y.mean())
    centered_y = y - y_mean

    gram = standardized.T @ standardized
    rhs = standardized.T @ centered_y
    coefficients = np.linalg.solve(
        gram + penalty * np.eye(standardized.shape[1]),
        rhs,
    )
    fitted = y_mean + standardized @ coefficients
    residuals = y - fitted
    smearing_factor = float(np.mean(np.exp(residuals)))
    prediction = float(y_mean + current_standardized @ coefficients)
    if (
        not np.isfinite(coefficients).all()
        or not np.isfinite(prediction)
        or not np.isfinite(smearing_factor)
        or smearing_factor <= 0
    ):
        raise ValueError("Ridge fit produced invalid coefficients")
    if current_features.name is None or isinstance(
        current_features.name,
        (bool, int, float, np.integer, np.floating),
    ):
        raise ValueError("Current ridge features require a dated information cutoff")
    try:
        information_cutoff = pd.Timestamp(current_features.name).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Current ridge features require a dated information cutoff"
        ) from exc
    if pd.isna(information_cutoff):
        raise ValueError("Current ridge information cutoff is invalid")

    return RidgeFit(
        feature_names=tuple(str(item) for item in features.columns),
        training_end=information_cutoff,
        last_label_origin=pd.Timestamp(combined.index[-1]).normalize(),
        sample_count=len(combined),
        feature_means=tuple(float(item) for item in means),
        feature_scales=tuple(float(item) for item in scales),
        intercept=y_mean,
        coefficients=tuple(float(item) for item in coefficients),
        current_features=tuple(float(item) for item in current),
        prediction=prediction,
        smearing_factor=smearing_factor,
    )


def forecast_daily_bar_volatility(
    close: pd.Series,
    *,
    alpha: float = RIDGE_ALPHA,
    min_samples: int = VARIANCE_MIN_TRAINING,
) -> VarianceForecast:
    """Forecast 21-session volatility without using an unrealized label."""
    learning = build_variance_learning_frame(close)
    feature_names = (
        "log_var_5",
        "log_var_21",
        "log_var_63",
        "log_downside_var_21",
    )
    current = learning.loc[learning.index[-1], list(feature_names)]
    training = learning.iloc[:-VARIANCE_HORIZON]
    ridge = standardized_ridge_fit_predict(
        training.loc[:, list(feature_names)],
        training["future_log_var_21"],
        current,
        alpha=alpha,
        min_samples=min_samples,
    )
    # The ridge predicts log variance. Duan's train-only smearing factor
    # corrects the systematic downward bias from exponentiating a conditional
    # log mean.
    model_volatility = float(
        np.sqrt(np.exp(ridge.prediction) * ridge.smearing_factor)
    )

    returns = np.log(pd.to_numeric(close, errors="coerce")).diff()
    volatility_21 = float(
        returns.iloc[-VARIANCE_MONTH_WINDOW:].std(ddof=1) * np.sqrt(252.0)
    )
    volatility_63 = float(
        returns.iloc[-VARIANCE_QUARTER_WINDOW:].std(ddof=1) * np.sqrt(252.0)
    )
    for name, value in (
        ("Model volatility", model_volatility),
        ("21-session volatility", volatility_21),
        ("63-session volatility", volatility_63),
    ):
        _finite_positive(value, name)
    sizing = max(model_volatility, volatility_21, volatility_63)
    return VarianceForecast(
        model_volatility=model_volatility,
        trailing_volatility_21=volatility_21,
        trailing_volatility_63=volatility_63,
        sizing_volatility=sizing,
        ridge_fit=ridge,
    )


def forecast_portfolio_volatility(
    qld_volatility: float,
    soxl_volatility: float,
    correlation: float,
    soxl_weight: float,
) -> float:
    """Calculate annualized volatility of QLD=(1-w), SOXL=w."""
    qld_vol = _finite_positive(qld_volatility, "QLD volatility")
    soxl_vol = _finite_positive(soxl_volatility, "SOXL volatility")
    corr = float(correlation)
    weight = float(soxl_weight)
    if not np.isfinite(corr) or corr < -1.0 or corr > 1.0:
        raise ValueError("Correlation must be finite and in [-1, 1]")
    if not np.isfinite(weight) or weight < 0.0 or weight > 1.0:
        raise ValueError("SOXL weight must be in [0, 1]")
    qld_weight = 1.0 - weight
    variance = (
        (qld_weight * qld_vol) ** 2
        + (weight * soxl_vol) ** 2
        + 2.0
        * qld_weight
        * weight
        * corr
        * qld_vol
        * soxl_vol
    )
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Forecast portfolio variance is zero or invalid")
    return float(np.sqrt(variance))


def conservative_finite_correlation(
    *correlations: float,
) -> float:
    """Return the maximum finite correlation, clipped to its valid range."""
    finite = [
        float(value)
        for value in correlations
        if np.isfinite(float(value))
    ]
    if not finite:
        raise ValueError("No finite QLD/SOXL correlation is available")
    return float(np.clip(max(finite), -1.0, 1.0))


def choose_soxl_weight(
    qld_volatility: float,
    soxl_volatility: float,
    correlation: float,
    *,
    budget: float = VOLATILITY_BUDGET,
    grid: Iterable[float] = SOXL_WEIGHT_GRID,
) -> tuple[float, float]:
    """Choose the largest frozen grid weight within the volatility budget."""
    limit = _finite_positive(budget, "Volatility budget")
    candidates = tuple(float(item) for item in grid)
    if not candidates or any(
        not np.isfinite(item) or item < 0 or item > MAX_SOXL_WEIGHT + _EPSILON
        for item in candidates
    ):
        raise ValueError("SOXL weight grid is invalid")
    if tuple(sorted(set(candidates))) != candidates or candidates[0] != 0.0:
        raise ValueError("SOXL weight grid must be sorted, unique, and start at zero")

    chosen_weight = 0.0
    chosen_volatility = forecast_portfolio_volatility(
        qld_volatility,
        soxl_volatility,
        correlation,
        0.0,
    )
    # If even QLD exceeds the overlay budget, retain QLD rather than selling
    # the permanent core.
    if chosen_volatility > limit + _EPSILON:
        return chosen_weight, chosen_volatility

    for candidate in candidates[1:]:
        candidate_volatility = forecast_portfolio_volatility(
            qld_volatility,
            soxl_volatility,
            correlation,
            candidate,
        )
        if candidate_volatility <= limit + _EPSILON:
            chosen_weight = candidate
            chosen_volatility = candidate_volatility
    return chosen_weight, chosen_volatility


def is_soxl_tier(value: object) -> bool:
    """Return whether ``value`` is one of the frozen strategic SOXL tiers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return bool(
        np.isfinite(numeric)
        and any(
            abs(numeric - candidate) <= _WEIGHT_TOLERANCE
            for candidate in SOXL_WEIGHT_GRID
        )
    )


def floor_soxl_tier(value: float) -> float:
    """Map a legacy SOXL weight down to the nearest current risk tier."""
    numeric = float(value)
    if (
        not np.isfinite(numeric)
        or numeric < -_WEIGHT_TOLERANCE
        or numeric > MAX_SOXL_WEIGHT + _WEIGHT_TOLERANCE
    ):
        raise ValueError("SOXL weight is outside the strategic range")
    bounded = min(max(numeric, 0.0), MAX_SOXL_WEIGHT)
    return max(
        candidate
        for candidate in SOXL_WEIGHT_GRID
        if candidate <= bounded + _WEIGHT_TOLERANCE
    )


def calculate_portfolio_volatility(
    price_data: pd.DataFrame,
    *,
    budget: float = VOLATILITY_BUDGET,
    min_samples: int = VARIANCE_MIN_TRAINING,
) -> PortfolioVolatility:
    """Forecast QLD/SOXL risk and return the largest permitted SOXL weight."""
    clean = _as_clean_close_frame(price_data, (QLD, SOXL))
    qld = forecast_daily_bar_volatility(
        clean[QLD],
        min_samples=min_samples,
    )
    soxl = forecast_daily_bar_volatility(
        clean[SOXL],
        min_samples=min_samples,
    )
    returns = np.log(clean).diff().dropna()
    correlation_21 = float(
        returns.iloc[-VARIANCE_MONTH_WINDOW:][QLD].corr(
            returns.iloc[-VARIANCE_MONTH_WINDOW:][SOXL]
        )
    )
    correlation_63 = float(
        returns.iloc[-VARIANCE_QUARTER_WINDOW:][QLD].corr(
            returns.iloc[-VARIANCE_QUARTER_WINDOW:][SOXL]
        )
    )
    sizing_correlation = conservative_finite_correlation(
        correlation_21,
        correlation_63,
    )
    weight, portfolio_volatility = choose_soxl_weight(
        qld.sizing_volatility,
        soxl.sizing_volatility,
        sizing_correlation,
        budget=budget,
    )
    return PortfolioVolatility(
        qld=qld,
        soxl=soxl,
        correlation_21=(
            correlation_21 if np.isfinite(correlation_21) else None
        ),
        correlation_63=(
            correlation_63 if np.isfinite(correlation_63) else None
        ),
        sizing_correlation=sizing_correlation,
        raw_soxl_weight=weight,
        raw_portfolio_volatility=portfolio_volatility,
    )


def target_weights(soxl_weight: float) -> dict[str, float]:
    """Return exact strategic QLD/SOXL weights."""
    weight = float(soxl_weight)
    if not is_soxl_tier(weight):
        raise ValueError("SOXL target weight is not a strategic tier")
    return {QLD: 1.0 - weight, SOXL: weight}


def advance_overlay_state(
    state: OverlayState,
    *,
    signal_date: pd.Timestamp,
    trend_positive: bool,
    residual_positive: bool,
    raw_soxl_weight: float,
    alpha_review_due: bool,
    reentry_closes: int = BULLISH_REENTRY_CLOSES,
    upshift_closes: int = VOLATILITY_UPSHIFT_CLOSES,
) -> OverlayTransition:
    """Process one distinct completed session with asymmetric risk changes."""
    session = pd.Timestamp(signal_date).normalize()
    session_text = session.date().isoformat()
    if state.last_processed_signal_date:
        previous = pd.Timestamp(state.last_processed_signal_date).normalize()
        if session < previous:
            raise ValueError("Overlay state is ahead of the signal date")
        if session == previous:
            return OverlayTransition(
                state=state,
                reason="SAME_DATE",
                alpha_reviewed=False,
                structural_change=False,
            )
    if reentry_closes < 1 or upshift_closes < 1:
        raise ValueError("Confirmation counts must be positive")
    raw_weight = float(raw_soxl_weight)
    if not is_soxl_tier(raw_weight):
        raise ValueError("Raw SOXL weight is not a strategic tier")

    trend = _strict_bool(trend_positive, "trend_positive")
    residual = _strict_bool(residual_positive, "residual_positive")
    review_due = _strict_bool(alpha_review_due, "alpha_review_due")
    eligible = trend and residual
    eligible_streak = state.eligible_streak + 1 if eligible else 0
    next_state = replace(
        state,
        eligible_streak=eligible_streak,
        last_processed_signal_date=session_text,
    )
    prior_active = state.overlay_active
    prior_weight = state.soxl_weight
    reason = "HOLD"
    reviewed = False

    if not trend:
        reason = "TREND_EXIT" if (prior_active or prior_weight > 0) else "TREND_BLOCK"
        reviewed = review_due
        next_state = replace(
            next_state,
            overlay_active=False,
            soxl_weight=0.0,
            soxl_weight_date=(
                session_text if prior_weight > _WEIGHT_TOLERANCE else state.soxl_weight_date
            ),
            pending_soxl_weight=0.0,
            pending_scale_days=0,
            last_alpha_review_date=(
                session_text if review_due else state.last_alpha_review_date
            ),
        )
    elif prior_active:
        if review_due:
            reviewed = True
            next_state = replace(next_state, last_alpha_review_date=session_text)
            if not residual:
                reason = "RESIDUAL_EXIT"
                next_state = replace(
                    next_state,
                    overlay_active=False,
                    soxl_weight=0.0,
                    soxl_weight_date=session_text,
                    pending_soxl_weight=0.0,
                    pending_scale_days=0,
                )

        if next_state.overlay_active:
            if raw_weight < prior_weight - _WEIGHT_TOLERANCE:
                reason = "VOLATILITY_DOWNSHIFT"
                next_state = replace(
                    next_state,
                    soxl_weight=raw_weight,
                    soxl_weight_date=session_text,
                    pending_soxl_weight=0.0,
                    pending_scale_days=0,
                )
            elif raw_weight > prior_weight + _WEIGHT_TOLERANCE:
                pending_days = (
                    state.pending_scale_days + 1
                    if abs(state.pending_soxl_weight - raw_weight)
                    <= _WEIGHT_TOLERANCE
                    else 1
                )
                if pending_days >= upshift_closes:
                    reason = "VOLATILITY_UPSHIFT"
                    next_state = replace(
                        next_state,
                        soxl_weight=raw_weight,
                        soxl_weight_date=session_text,
                        pending_soxl_weight=0.0,
                        pending_scale_days=0,
                    )
                else:
                    reason = "VOLATILITY_UPSHIFT_PENDING"
                    next_state = replace(
                        next_state,
                        pending_soxl_weight=raw_weight,
                        pending_scale_days=pending_days,
                    )
            else:
                next_state = replace(
                    next_state,
                    pending_soxl_weight=0.0,
                    pending_scale_days=0,
                )
    else:
        if review_due:
            reviewed = True
            next_state = replace(next_state, last_alpha_review_date=session_text)
            if eligible and eligible_streak >= reentry_closes:
                reason = "OVERLAY_REENTRY"
                next_state = replace(
                    next_state,
                    overlay_active=True,
                    soxl_weight=raw_weight,
                    soxl_weight_date=session_text,
                    pending_soxl_weight=0.0,
                    pending_scale_days=0,
                )
            else:
                reason = "ALPHA_BLOCK"

    structural_change = (
        next_state.overlay_active != prior_active
        or abs(next_state.soxl_weight - prior_weight) > _WEIGHT_TOLERANCE
    )
    return OverlayTransition(
        state=next_state,
        reason=reason,
        alpha_reviewed=reviewed,
        structural_change=structural_change,
    )
