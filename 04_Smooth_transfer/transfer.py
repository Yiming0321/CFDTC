"""
Smooth-transfer ensemble between MLP and XGBoost.

Single-driver convex combination:

    P_final = w_mlp * P_MLP + w_xgb * P_XGB        (w_mlp + w_xgb == 1)
    w_mlp   = 1 / (1 + exp((log10(P_est) - log10(P0)) / s))

where
    P_est : power-level estimate driving the transition (default: P_XGB)
    P0    : transition centre, expressed in uW (default 10.0 uW)
    s     : transition width, in decades / log10 units (default 0.5)

The logistic is evaluated in log10 space so that the transition spans a
meaningful fraction of the multi-decade measurement range (the data spans
roughly 0.056 - 991 uW, i.e. ~4.2 decades), instead of being squeezed into a
narrow band near P0.
"""
import math

_LOG_FLOOR = 1e-12   # guards log10(0) / non-positive predictions
P0_UW = 10.0         # transition centre in uW (matches the original P0 = 10.00 uW)
S_DECADES = 0.5      # transition width in decades (log10 units)


def _log10_safe(x: float) -> float:
    """log10 with a floor, so non-positive predictions cannot break the gate."""
    return math.log10(x) if x > _LOG_FLOOR else math.log10(_LOG_FLOOR)


def logistic(x: float) -> float:
    """Numerically stable logistic 1 / (1 + exp(-x)); does not overflow for large |x|."""
    if x >= 0.0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def omega(p_est: float, p0_uw: float = P0_UW, s_decades: float = S_DECADES) -> float:
    """
    MLP weight omega in (0, 1), monotonically decreasing with power (log10 space).

        omega = 1 / (1 + exp((log10(p_est) - log10(p0)) / s))

    - p_est << p0 (weak signal)   -> omega -> 1  (MLP dominates)
    - p_est >> p0 (strong signal) -> omega -> 0  (XGB dominates)
    - p_est == p0                 -> omega = 0.5
    """
    z = (_log10_safe(p_est) - math.log10(p0_uw)) / s_decades
    return logistic(-z)


def ensemble_predict(mlp_pred: float, xgb_pred: float,
                     p0_uw: float = P0_UW, s_decades: float = S_DECADES) -> float:
    """
    Combine MLP and XGBoost predictions with a smooth, convex transition.

        P_est   = xgb_pred
        w_mlp   = omega(P_est)
        P_final = w_mlp * mlp_pred + (1 - w_mlp) * xgb_pred

    Guarantees:
      - convex combination: weights sum to 1, so the output always lies between
        mlp_pred and xgb_pred;
      - continuous everywhere (no hard branches / jumps);
      - saturates to pure MLP at low power and pure XGB at high power.
    """
    w_mlp = omega(xgb_pred, p0_uw, s_decades)
    return w_mlp * mlp_pred + (1.0 - w_mlp) * xgb_pred