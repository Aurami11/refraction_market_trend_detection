import json
import os
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

# PARAMETRES UTILISATEUR

UNIVERSE_GROUPS = {
    # US core / styles / factors
    "SPY": "us_core",
    "QQQ": "us_core",
    "DIA": "us_core",
    "IWM": "us_core",
    "MDY": "us_core",
    "VTI": "us_core",
    "RSP": "us_core",
    "MTUM": "us_core",
    "QUAL": "us_core",
    "USMV": "us_core",
    # US sectors
    "XLB": "us_sector",
    "XLE": "us_sector",
    "XLF": "us_sector",
    "XLI": "us_sector",
    "XLK": "us_sector",
    "XLP": "us_sector",
    "XLU": "us_sector",
    "XLV": "us_sector",
    "XLY": "us_sector",
    "XLRE": "us_sector",
    "XLC": "us_sector",
    # International equity
    "EFA": "intl_equity",
    "EEM": "intl_equity",
    "VWO": "intl_equity",
    "VGK": "intl_equity",
    "EWJ": "intl_equity",
    "EWU": "intl_equity",
    "EWC": "intl_equity",
    "EWA": "intl_equity",
    # Rates / credit
    "BIL": "rates_credit",
    "SHY": "rates_credit",
    "IEF": "rates_credit",
    "TLT": "rates_credit",
    "TIP": "rates_credit",
    "LQD": "rates_credit",
    "HYG": "rates_credit",
    "EMB": "rates_credit",
    "BND": "rates_credit",
    # Real assets
    "VNQ": "real_assets",
    "GLD": "real_assets",
    "SLV": "real_assets",
    "DBC": "real_assets",
    "USO": "real_assets",
}

TICKERS = list(UNIVERSE_GROUPS.keys())

# sleeve défensive
DEFENSIVE_SLEEVE = ["BIL", "SHY", "IEF", "TIP"]

START = "2020-01-01"
END: Optional[str] = None
INTERVAL = "1d"

CACHE_DIR = "cache"
OUTPUT_DIR = "outputs"
FORCE_REFRESH = False

INITIAL_CAPITAL = 1_000_000.0

VOL_WINDOW = 20
ATR_WINDOW = 20
EPS = 1e-8
SIN_ACCEPT_MIN = 1e-3

ANGLE_TOL = 0.12
AMP_TOL_Z = 1.25
SMALL_CORRECTION_Z = 0.75

TRAIN_LOOKBACK = 756
BACKTEST_WARMUP = 252
BACKTEST_STEP = 1

CHANGE_CLASS_WEIGHT = "balanced"
SAME_CLASS_WEIGHT = None
LOGIT_MAX_ITER = 2000

PLATT_CALIBRATION_FRAC = 0.25
PLATT_MIN_CALIBRATION_SIZE = 120
PLATT_MIN_CORE_TRAIN_SIZE = 250

ENTRY_P_CHANGE_MAX = 0.10
EXIT_P_CHANGE_MIN = 0.20
CHANGE_THRESHOLD = 0.1

TOP_K = 10
ASSET_CAP = 0.10
GROUP_CAPS = {
    "us_core": 0.55,
    "us_sector": 0.25,
    "intl_equity": 0.20,
    "rates_credit": 0.30,
    "real_assets": 0.15,
}

LAMBDA_VISC = 0.25
LAMBDA_AGE = 0.10

HARD_STOP_ATR = 2.0
TRAIL_STOP_ATR = 3.0

REBALANCE_MODE = "monthly"  # "weekly" ou "monthly"
DEFENSIVE_ALLOCATION_MODE = "inverse_vol"  # "inverse_vol" ou "equal"

# coûts
IBKR_FIXED_FEE_PER_ORDER = 1.00
SEC_FEE_PER_DOLLAR_SOLD = 20.60 / 1_000_000.0
SLIPPAGE_BPS_PER_SIDE = 0.0
MIN_TRADE_DOLLARS = 100.0

# CONFIG

@dataclass
class Config:
    tickers: List[str]
    universe_groups: Dict[str, str]
    defensive_sleeve: List[str]

    start: str = START
    end: Optional[str] = END
    interval: str = INTERVAL

    cache_dir: str = CACHE_DIR
    output_dir: str = OUTPUT_DIR
    force_refresh: bool = FORCE_REFRESH

    initial_capital: float = INITIAL_CAPITAL

    vol_window: int = VOL_WINDOW
    atr_window: int = ATR_WINDOW
    eps: float = EPS
    sin_accept_min: float = SIN_ACCEPT_MIN

    angle_tol: float = ANGLE_TOL
    amp_tol_z: float = AMP_TOL_Z
    small_correction_z: float = SMALL_CORRECTION_Z

    train_lookback: int = TRAIN_LOOKBACK
    backtest_warmup: int = BACKTEST_WARMUP
    backtest_step: int = BACKTEST_STEP

    change_class_weight: Optional[str] = CHANGE_CLASS_WEIGHT
    same_class_weight: Optional[str] = SAME_CLASS_WEIGHT
    logit_max_iter: int = LOGIT_MAX_ITER

    platt_calibration_frac: float = PLATT_CALIBRATION_FRAC
    platt_min_calibration_size: int = PLATT_MIN_CALIBRATION_SIZE
    platt_min_core_train_size: int = PLATT_MIN_CORE_TRAIN_SIZE

    entry_p_change_max: float = ENTRY_P_CHANGE_MAX
    exit_p_change_min: float = EXIT_P_CHANGE_MIN
    change_threshold: float = CHANGE_THRESHOLD

    top_k: int = TOP_K
    asset_cap: float = ASSET_CAP
    group_caps: Dict[str, float] = None

    lambda_visc: float = LAMBDA_VISC
    lambda_age: float = LAMBDA_AGE

    hard_stop_atr: float = HARD_STOP_ATR
    trail_stop_atr: float = TRAIL_STOP_ATR

    rebalance_mode: str = REBALANCE_MODE
    defensive_allocation_mode: str = DEFENSIVE_ALLOCATION_MODE

    ibkr_fixed_fee_per_order: float = IBKR_FIXED_FEE_PER_ORDER
    sec_fee_per_dollar_sold: float = SEC_FEE_PER_DOLLAR_SOLD
    slippage_bps_per_side: float = SLIPPAGE_BPS_PER_SIDE
    min_trade_dollars: float = MIN_TRADE_DOLLARS

# ETAT REDUIT

BASE_STATE_COLS = [
    "theta",
    "amp_z",
    "sigma_local",
    "viscosity",
    "causal_run_length",
    "causal_theta_so_far",
    "causal_current_sign",
]

DELTA_STATE_COLS = [
    "d_theta",
    "d_amp_z",
    "d_sigma_local",
    "d_viscosity",
    "d_causal_theta_so_far",
]

STATE_COLS = BASE_STATE_COLS + DELTA_STATE_COLS

# UTILITAIRES

def safe_name(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace("=", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def today_end_exclusive() -> str:
    return str((pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)).date())


def rebalance_flags(dates: pd.DatetimeIndex, mode: str) -> pd.Series:
    s = pd.Series(False, index=dates)

    if mode == "weekly":
        periods = dates.to_period("W-FRI")
    elif mode == "monthly":
        periods = dates.to_period("M")
    else:
        raise ValueError(f"Mode de rebalancement inconnu: {mode}")

    last_in_period = periods != periods.shift(-1)
    s.loc[:] = last_in_period
    return s


def calibration_curve_df(df: pd.DataFrame, prob_col: str, actual_col: str, n_bins: int = 10) -> pd.DataFrame:
    tmp = df[[prob_col, actual_col]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame(columns=["mean_pred", "mean_actual", "count"])

    try:
        tmp["bin"] = pd.qcut(tmp[prob_col], q=n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=["mean_pred", "mean_actual", "count"])

    out = tmp.groupby("bin", observed=False).agg(
        mean_pred=(prob_col, "mean"),
        mean_actual=(actual_col, "mean"),
        count=(actual_col, "size"),
    ).reset_index(drop=True)
    return out


def threshold_sweep(df: pd.DataFrame, prob_col: str, actual_col: str, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    y_true = df[actual_col].to_numpy(dtype=float)
    y_prob = df[prob_col].to_numpy(dtype=float)

    for thr in thresholds:
        y_hat = (y_prob >= thr).astype(float)
        tp = float(((y_hat == 1.0) & (y_true == 1.0)).sum())
        fp = float(((y_hat == 1.0) & (y_true == 0.0)).sum())
        fn = float(((y_hat == 0.0) & (y_true == 1.0)).sum())
        tn = float(((y_hat == 0.0) & (y_true == 0.0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        rate_pred_positive = float(y_hat.mean())

        rows.append(
            {
                "threshold": float(thr),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "predicted_positive_rate": rate_pred_positive,
            }
        )

    return pd.DataFrame(rows)


def max_drawdown(series: pd.Series) -> float:
    cummax = series.cummax()
    dd = series / cummax - 1.0
    return float(dd.min())


def ulcer_index(series: pd.Series) -> float:
    cummax = series.cummax()
    dd_pct = 100.0 * (series / cummax - 1.0)
    return float(np.sqrt(np.mean(dd_pct ** 2)))


def performance_metrics(equity: pd.Series, daily_returns: pd.Series, name: str) -> Dict[str, Any]:
    daily_returns = daily_returns.dropna()
    if daily_returns.empty or len(equity) < 2:
        return {"name": name, "status": "empty"}

    ann_factor = 252.0
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = max(len(daily_returns) / ann_factor, 1e-8)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)

    vol = float(daily_returns.std(ddof=0) * np.sqrt(ann_factor))
    mean_daily = float(daily_returns.mean())
    sharpe = float((mean_daily * ann_factor) / vol) if vol > 0 else np.nan

    downside = daily_returns[daily_returns < 0]
    downside_vol = float(downside.std(ddof=0) * np.sqrt(ann_factor)) if len(downside) > 0 else np.nan
    sortino = float((mean_daily * ann_factor) / downside_vol) if pd.notna(downside_vol) and downside_vol > 0 else np.nan

    mdd = max_drawdown(equity)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan
    ui = ulcer_index(equity)

    return {
        "name": name,
        "status": "ok",
        "total_return": total_return,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "ulcer_index": ui,
        "hit_ratio": float((daily_returns > 0).mean()),
    }


def apply_caps(raw_scores: pd.Series, group_map: Dict[str, str], asset_cap: float, group_caps: Dict[str, float]) -> Dict[str, float]:
    scores = raw_scores[raw_scores > 0].copy()
    if scores.empty:
        return {}

    w = scores / scores.sum()
    assets = list(w.index)

    for _ in range(40):
        changed = False

        # caps par actif
        over_asset = w > asset_cap + 1e-12
        if over_asset.any():
            excess = float((w[over_asset] - asset_cap).sum())
            w.loc[over_asset] = asset_cap

            under = w < asset_cap - 1e-12
            if excess > 0 and under.any():
                redist = w.loc[under]
                if redist.sum() > 0:
                    w.loc[under] += excess * redist / redist.sum()
            changed = True

        # caps par groupe
        for g, cap in group_caps.items():
            g_assets = [a for a in assets if group_map.get(a) == g and a in w.index]
            if not g_assets:
                continue

            g_sum = float(w.loc[g_assets].sum())
            if g_sum > cap + 1e-12:
                deficit = g_sum - cap
                w.loc[g_assets] *= cap / g_sum

                others = [a for a in assets if a not in g_assets and a in w.index]
                room_assets = [a for a in others if w.loc[a] < asset_cap - 1e-12]
                if deficit > 0 and room_assets:
                    redist = w.loc[room_assets]
                    if redist.sum() > 0:
                        w.loc[room_assets] += deficit * redist / redist.sum()
                changed = True

        w = w.clip(lower=0.0)
        total = float(w.sum())
        if total <= 0:
            break

        if total > 1.0 + 1e-12:
            w /= total
            changed = True

        if not changed:
            break

    w = w.clip(lower=0.0, upper=asset_cap)

    for g, cap in group_caps.items():
        g_assets = [a for a in w.index if group_map.get(a) == g]
        if g_assets:
            g_sum = float(w.loc[g_assets].sum())
            if g_sum > cap + 1e-12:
                w.loc[g_assets] *= cap / g_sum

    total = float(w.sum())
    if total > 1.0 + 1e-12:
        w /= total

    return {k: float(v) for k, v in w.items() if v > 0}

# DONNEES

def load_ohlc(ticker: str, cfg: Config) -> pd.DataFrame:
    ensure_dir(cfg.cache_dir)
    path = os.path.join(cfg.cache_dir, f"{safe_name(ticker)}_{cfg.interval}.csv")

    if os.path.exists(path) and not cfg.force_refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[df.index >= START].copy()
        return df.sort_index()

    yf.set_tz_cache_location(str(Path(cfg.cache_dir) / "yfinance_tz"))

    end = cfg.end if cfg.end is not None else today_end_exclusive()
    df = yf.download(
        ticker,
        start=cfg.start,
        end=end,
        interval=cfg.interval,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    if df is None or df.empty:
        raise ValueError(f"Aucune donnée pour {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour {ticker}: {missing}")

    df = df[cols].dropna().copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df.to_csv(path)
    return df

# FEATURES

def build_features(ohlc: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = ohlc.copy()

    df["log_close"] = np.log(df["Close"])
    df["r"] = df["log_close"].diff()
    df["amplitude"] = df["r"].abs()
    df["sign"] = np.sign(df["r"]).fillna(0.0)
    df["theta"] = np.arctan(df["r"].fillna(0.0))

    min_periods = max(5, cfg.vol_window // 2)
    df["sigma_local"] = df["r"].rolling(cfg.vol_window, min_periods=min_periods).std()

    dr_abs = df["r"].diff().abs()
    df["viscosity"] = df["amplitude"] / (dr_abs + cfg.eps)
    df["amp_z"] = df["amplitude"] / (df["sigma_local"] + cfg.eps)

    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR20"] = tr.rolling(cfg.atr_window, min_periods=max(10, cfg.atr_window // 2)).mean()

    df = df.dropna(subset=["r"]).copy()

    df["sin_theta_abs"] = np.abs(np.sin(df["theta"]))
    sin_ref = float(df["sin_theta_abs"].iloc[0] + cfg.eps)

    if df["sin_theta_abs"].iloc[0] < cfg.sin_accept_min:
        warnings.warn("Premier angle très faible: la jauge n_1 = 1 est conservée, mais l'échelle de n_eff peut être fragile.")

    df["x_log_n"] = np.log(sin_ref) - np.log(df["sin_theta_abs"] + cfg.eps)
    df["n_eff"] = np.exp(df["x_log_n"])
    df["n_eff_acceptable"] = df["sin_theta_abs"] >= cfg.sin_accept_min

    n_eff_max = sin_ref / (cfg.sin_accept_min + cfg.eps)
    df["n_eff_for_use"] = np.where(df["n_eff_acceptable"], df["n_eff"], n_eff_max)

    return df

# SEGMENTS

def locally_compatible(prev_row: pd.Series, cur_row: pd.Series, cfg: Config) -> bool:
    angle_close = abs(float(cur_row["theta"]) - float(prev_row["theta"])) <= cfg.angle_tol

    prev_amp_z = float(prev_row["amp_z"]) if pd.notna(prev_row["amp_z"]) else 0.0
    cur_amp_z = float(cur_row["amp_z"]) if pd.notna(cur_row["amp_z"]) else 0.0
    amp_close = abs(cur_amp_z - prev_amp_z) <= cfg.amp_tol_z

    prev_sign = float(prev_row["sign"])
    cur_sign = float(cur_row["sign"])
    same_sign = (prev_sign == cur_sign) or (prev_sign == 0.0) or (cur_sign == 0.0)

    absorbable = (cur_amp_z <= cfg.small_correction_z) or (prev_amp_z <= cfg.small_correction_z)

    return angle_close and (amp_close or absorbable) and (same_sign or absorbable)


def build_causal_state(features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = features.copy()
    n = len(df)

    seg_id = np.zeros(n, dtype=int)
    run_length = np.zeros(n, dtype=int)
    theta_so_far = np.zeros(n, dtype=float)
    current_sign = np.zeros(n, dtype=float)

    current_seg = 0
    run_return = 0.0
    run_amplitude = 0.0
    theta_num = 0.0

    for i in range(n):
        row = df.iloc[i]

        if i == 0:
            new_segment = True
        else:
            new_segment = not locally_compatible(df.iloc[i - 1], row, cfg)

        if new_segment:
            if i > 0:
                current_seg += 1
            run_len = 1
            run_return = float(row["r"])
            run_amplitude = float(row["amplitude"])
            theta_num = float(row["theta"]) * float(row["amplitude"])
        else:
            run_len += 1
            run_return += float(row["r"])
            run_amplitude += float(row["amplitude"])
            theta_num += float(row["theta"]) * float(row["amplitude"])

        seg_id[i] = current_seg
        run_length[i] = run_len
        theta_so_far[i] = theta_num / (run_amplitude + cfg.eps)
        current_sign[i] = float(np.sign(run_return)) if run_return != 0 else 0.0

    df["causal_segment_id"] = seg_id
    df["causal_run_length"] = run_length
    df["causal_theta_so_far"] = theta_so_far
    df["causal_current_sign"] = current_sign

    df["d_theta"] = df["theta"].diff()
    df["d_amp_z"] = df["amp_z"].diff()
    df["d_sigma_local"] = df["sigma_local"].diff()
    df["d_viscosity"] = df["viscosity"].diff()
    df["d_causal_theta_so_far"] = df["causal_theta_so_far"].diff()

    return df

# LABELS EVENEMENTIELS

def build_event_labels(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    n = len(df)

    labels = pd.DataFrame(index=df.index)
    labels["continue_next"] = False
    labels["change_next"] = False
    labels["next_segment_completed_by"] = np.nan
    labels["next_segment_same_direction"] = np.nan
    labels["next_segment_opp_direction"] = np.nan
    labels["next_segment_theta"] = np.nan
    labels["next_segment_amp"] = np.nan
    labels["next_segment_length"] = np.nan
    labels["next_segment_net_return"] = np.nan

    seg_ids = df["causal_segment_id"].to_numpy()

    labels.iloc[:-1, labels.columns.get_loc("continue_next")] = (seg_ids[1:] == seg_ids[:-1])
    labels.iloc[:-1, labels.columns.get_loc("change_next")] = ~labels["continue_next"].iloc[:-1]

    seg_info: Dict[int, Dict[str, Any]] = {}
    for sid in np.unique(seg_ids):
        idx = np.where(seg_ids == sid)[0]
        start, end = int(idx[0]), int(idx[-1])
        seg = df.iloc[start:end + 1]

        net_return = float(seg["r"].sum())
        sign_global = float(np.sign(net_return)) if net_return != 0 else 0.0

        weights = seg["amplitude"].to_numpy()
        theta_vals = seg["theta"].to_numpy()
        theta_mean = float(np.average(theta_vals, weights=weights)) if weights.sum() > 0 else float(np.nanmean(theta_vals))

        seg_info[sid] = {
            "start": start,
            "end": end,
            "sign_global": sign_global,
            "theta_mean": theta_mean,
            "amp_sum": float(seg["amplitude"].sum()),
            "length": int(len(seg)),
            "net_return": net_return,
        }

    for t in range(n - 1):
        sid_t = int(seg_ids[t])
        sid_t1 = int(seg_ids[t + 1])

        if sid_t1 != sid_t:
            prev_sign = seg_info[sid_t]["sign_global"]
            next_sign = seg_info[sid_t1]["sign_global"]

            labels.iloc[t, labels.columns.get_loc("next_segment_completed_by")] = seg_info[sid_t1]["end"]
            labels.iloc[t, labels.columns.get_loc("next_segment_theta")] = seg_info[sid_t1]["theta_mean"]
            labels.iloc[t, labels.columns.get_loc("next_segment_amp")] = seg_info[sid_t1]["amp_sum"]
            labels.iloc[t, labels.columns.get_loc("next_segment_length")] = seg_info[sid_t1]["length"]
            labels.iloc[t, labels.columns.get_loc("next_segment_net_return")] = seg_info[sid_t1]["net_return"]

            labels.iloc[t, labels.columns.get_loc("next_segment_same_direction")] = float(next_sign == prev_sign)
            labels.iloc[t, labels.columns.get_loc("next_segment_opp_direction")] = float(next_sign == -prev_sign)

    return labels

# MODELES

def fit_logistic_binary(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    class_weight: Optional[str],
    cfg: Config,
) -> Tuple[Optional[Any], float, pd.DataFrame]:
    data = pd.concat([X[feature_cols], y.rename("target")], axis=1).dropna()
    empty_coef = pd.DataFrame(columns=["feature", "coef_std", "odds_ratio_1sd", "abs_coef_std"])

    if data.empty:
        return None, np.nan, empty_coef

    Xv = data[feature_cols]
    yv = data["target"].astype(int)
    base_rate = float(yv.mean())

    if yv.nunique() < 2:
        return None, base_rate, empty_coef

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=cfg.logit_max_iter,
            class_weight=class_weight,
            solver="lbfgs",
        ),
    )
    model.fit(Xv, yv)

    logit = model.named_steps["logisticregression"]
    coef = logit.coef_[0]

    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coef_std": coef,
            "odds_ratio_1sd": np.exp(coef),
        }
    )
    coef_df["abs_coef_std"] = coef_df["coef_std"].abs()
    coef_df = coef_df.sort_values("abs_coef_std", ascending=False).reset_index(drop=True)

    return model, base_rate, coef_df


def predict_raw_proba_binary(
    model: Optional[Any],
    base_rate: float,
    x_row: pd.Series,
    feature_cols: List[str],
) -> float:
    if model is None:
        return float(base_rate) if pd.notna(base_rate) else np.nan

    Xq = pd.DataFrame([x_row[feature_cols].to_dict()])
    return float(model.predict_proba(Xq)[0, 1])


def fit_logistic_with_platt(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    class_weight: Optional[str],
    cfg: Config,
) -> Tuple[Optional[Any], Optional[Any], float, pd.DataFrame, Dict[str, Any]]:
    data = pd.concat([X[feature_cols], y.rename("target")], axis=1).dropna().reset_index(drop=True)
    empty_coef = pd.DataFrame(columns=["feature", "coef_std", "odds_ratio_1sd", "abs_coef_std"])

    if data.empty:
        return None, None, np.nan, empty_coef, {"used": False, "reason": "empty_training_data"}

    X_all = data[feature_cols]
    y_all = data["target"].astype(int)
    base_rate = float(y_all.mean())

    if y_all.nunique() < 2:
        return None, None, base_rate, empty_coef, {
            "used": False,
            "reason": "single_class",
            "n_total": int(len(data)),
        }

    n_total = len(data)
    n_cal = max(cfg.platt_min_calibration_size, int(round(cfg.platt_calibration_frac * n_total)))
    n_core = n_total - n_cal

    if n_core < cfg.platt_min_core_train_size:
        model_core, base_rate_core, coef_df = fit_logistic_binary(X_all, y_all, feature_cols, class_weight, cfg)
        return model_core, None, base_rate_core, coef_df, {
            "used": False,
            "reason": "not_enough_points_for_causal_calibration",
            "n_total": int(n_total),
            "n_core": int(n_core),
            "n_calibration": int(n_cal),
        }

    X_core = X_all.iloc[:n_core]
    y_core = y_all.iloc[:n_core]
    X_cal = X_all.iloc[n_core:]
    y_cal = y_all.iloc[n_core:]

    if y_core.nunique() < 2 or y_cal.nunique() < 2:
        model_core, base_rate_core, coef_df = fit_logistic_binary(X_all, y_all, feature_cols, class_weight, cfg)
        return model_core, None, base_rate_core, coef_df, {
            "used": False,
            "reason": "core_or_calibration_single_class",
            "n_total": int(n_total),
            "n_core": int(n_core),
            "n_calibration": int(n_cal),
        }

    model_core, base_rate_core, coef_df = fit_logistic_binary(X_core, y_core, feature_cols, class_weight, cfg)
    if model_core is None:
        return None, None, base_rate_core, coef_df, {
            "used": False,
            "reason": "core_model_unavailable",
            "n_total": int(n_total),
            "n_core": int(n_core),
            "n_calibration": int(n_cal),
        }

    raw_scores_cal = model_core.decision_function(X_cal)

    calibrator = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight=None,
    )
    calibrator.fit(raw_scores_cal.reshape(-1, 1), y_cal.astype(int))

    return model_core, calibrator, base_rate_core, coef_df, {
        "used": True,
        "reason": "ok",
        "n_total": int(n_total),
        "n_core": int(n_core),
        "n_calibration": int(n_cal),
        "base_rate_core": float(y_core.mean()),
        "base_rate_calibration": float(y_cal.mean()),
    }


def predict_platt_proba_binary(
    model: Optional[Any],
    calibrator: Optional[Any],
    base_rate: float,
    x_row: pd.Series,
    feature_cols: List[str],
) -> float:
    raw_prob = predict_raw_proba_binary(model, base_rate, x_row, feature_cols)

    if model is None or calibrator is None or pd.isna(raw_prob):
        return raw_prob

    Xq = pd.DataFrame([x_row[feature_cols].to_dict()])
    raw_score = float(model.decision_function(Xq)[0])
    return float(calibrator.predict_proba(np.array([[raw_score]]))[0, 1])


def summarize_next_segment_stats(train_labels: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for class_name, class_col in [("same", "next_segment_same_direction"), ("opp", "next_segment_opp_direction")]:
        subset = train_labels.loc[train_labels[class_col] == 1.0].copy()

        if subset.empty:
            out[class_name] = {
                "theta_mean": np.nan,
                "amp_mean": np.nan,
                "length_mean": np.nan,
                "ret_mean": np.nan,
            }
            continue

        out[class_name] = {
            "theta_mean": float(subset["next_segment_theta"].mean()),
            "amp_mean": float(subset["next_segment_amp"].mean()),
            "length_mean": float(subset["next_segment_length"].mean()),
            "ret_mean": float(subset["next_segment_net_return"].mean()),
        }

    return out

# SIGNALS PAR ACTIF

def predict_one_step(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    query_idx: int,
    cfg: Config,
    return_coef: bool = False,
) -> Dict[str, Any]:
    if query_idx <= cfg.backtest_warmup:
        return {"status": "insufficient_history"}

    start_idx = max(0, query_idx - cfg.train_lookback)
    x_query = features.iloc[query_idx]

    X1 = features.iloc[start_idx:query_idx][STATE_COLS]
    y1 = labels.iloc[start_idx:query_idx]["change_next"].astype(float)

    model_change, calibrator_change, base_change, coef_change, platt_info_change = fit_logistic_with_platt(
        X1, y1, STATE_COLS, cfg.change_class_weight, cfg
    )

    p_change_raw = predict_raw_proba_binary(model_change, base_change, x_query, STATE_COLS)
    p_change = predict_platt_proba_binary(model_change, calibrator_change, base_change, x_query, STATE_COLS)
    p_continue = 1.0 - p_change if pd.notna(p_change) else np.nan

    completed_mask = labels["next_segment_completed_by"] <= query_idx
    train_change_mask = (labels["change_next"] == 1.0) & completed_mask
    train_change_mask.iloc[:start_idx] = False

    X2 = features.loc[train_change_mask, STATE_COLS]
    y2 = labels.loc[train_change_mask, "next_segment_same_direction"].astype(float)

    model_same, base_same, coef_same = fit_logistic_binary(X2, y2, STATE_COLS, cfg.same_class_weight, cfg)

    q_same = predict_raw_proba_binary(model_same, base_same, x_query, STATE_COLS)
    q_opp = 1.0 - q_same if pd.notna(q_same) else np.nan

    stats = summarize_next_segment_stats(labels.loc[train_change_mask].copy())

    q_same_for_score = q_same
    if pd.isna(q_same_for_score):
        q_same_for_score = base_same if pd.notna(base_same) else 0.5

    pi_bull = (1.0 - p_change) + p_change * q_same_for_score if pd.notna(p_change) else np.nan

    eligible_long = (
        (float(x_query["causal_current_sign"]) > 0.0)
        and (float(x_query["causal_theta_so_far"]) > 0.0)
        and (pd.notna(p_change) and p_change <= cfg.entry_p_change_max)
    )

    denom = float(x_query["sigma_local"]) + cfg.eps
    age_penalty = 1.0 + cfg.lambda_age * np.log1p(float(x_query["causal_run_length"]))
    visc_boost = 1.0 + cfg.lambda_visc * max(float(x_query["viscosity"]), 0.0)

    alpha = 0.0
    if eligible_long and pd.notna(pi_bull):
        alpha = (
            pi_bull
            * (max(float(x_query["causal_theta_so_far"]), 0.0) / denom)
            * (visc_boost / age_penalty)
        )

    expected_theta = np.nan
    expected_amp = np.nan
    expected_length = np.nan
    expected_ret = np.nan

    if pd.notna(q_same):
        expected_theta = q_same * stats["same"]["theta_mean"] + q_opp * stats["opp"]["theta_mean"]
        expected_amp = q_same * stats["same"]["amp_mean"] + q_opp * stats["opp"]["amp_mean"]
        expected_length = q_same * stats["same"]["length_mean"] + q_opp * stats["opp"]["length_mean"]
        expected_ret = q_same * stats["same"]["ret_mean"] + q_opp * stats["opp"]["ret_mean"]

    out = {
        "status": "ok",
        "timestamp_t": features.index[query_idx],
        "p_change_raw": p_change_raw,
        "p_change": p_change,
        "p_continue": p_continue,
        "q_same_given_change": q_same,
        "q_opp_given_change": q_opp,
        "pi_bull": pi_bull,
        "eligible_long": float(eligible_long),
        "alpha": float(alpha),
        "expected_next_segment_theta": expected_theta,
        "expected_next_segment_amp": expected_amp,
        "expected_next_segment_length": expected_length,
        "expected_next_segment_net_return": expected_ret,
        "base_rate_change_train": base_change,
        "base_rate_same_train": base_same,
        "n_completed_changes_train": int(train_change_mask.sum()),
        "platt_used_change": platt_info_change["used"],
        "platt_reason_change": platt_info_change["reason"],
    }

    if return_coef:
        out["coef_change"] = coef_change
        out["coef_same"] = coef_same

    return out


def build_asset_signal_table(ticker: str, ohlc: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    features = build_features(ohlc, cfg)
    features = build_causal_state(features, cfg)
    features = features.dropna(subset=STATE_COLS + ["ATR20"]).copy()

    labels = build_event_labels(features)

    signal_rows = []
    last_coef_change = pd.DataFrame()
    last_coef_same = pd.DataFrame()

    for q in range(cfg.backtest_warmup, len(features)):
        pred = predict_one_step(features, labels, q, cfg, return_coef=(q == len(features) - 1))
        if pred.get("status") != "ok":
            continue

        row = {
            "date": features.index[q],
            "p_change_raw": pred["p_change_raw"],
            "p_change": pred["p_change"],
            "p_continue": pred["p_continue"],
            "q_same_given_change": pred["q_same_given_change"],
            "q_opp_given_change": pred["q_opp_given_change"],
            "pi_bull": pred["pi_bull"],
            "eligible_long": pred["eligible_long"],
            "alpha": pred["alpha"],
            "causal_current_sign": features["causal_current_sign"].iloc[q],
            "causal_theta_so_far": features["causal_theta_so_far"].iloc[q],
            "causal_run_length": features["causal_run_length"].iloc[q],
            "theta": features["theta"].iloc[q],
            "amp_z": features["amp_z"].iloc[q],
            "sigma_local": features["sigma_local"].iloc[q],
            "viscosity": features["viscosity"].iloc[q],
            "ATR20": features["ATR20"].iloc[q],
            "Open": features["Open"].iloc[q],
            "High": features["High"].iloc[q],
            "Low": features["Low"].iloc[q],
            "Close": features["Close"].iloc[q],
        }
        signal_rows.append(row)

        if q == len(features) - 1:
            last_coef_change = pred.get("coef_change", pd.DataFrame())
            last_coef_same = pred.get("coef_same", pd.DataFrame())

    signal_table = pd.DataFrame(signal_rows).set_index("date").sort_index()

    return {
        "ticker": ticker,
        "ohlc": ohlc,
        "features": features,
        "labels": labels,
        "signals": signal_table,
        "coef_change_latest": last_coef_change,
        "coef_same_latest": last_coef_same,
    }

# ALLOCATION DEFENSIVE

def build_defensive_weights(asset_data: Dict[str, Dict[str, Any]], date_t: pd.Timestamp, next_date: pd.Timestamp, residual_weight: float, cfg: Config) -> Dict[str, float]:
    if residual_weight <= 0:
        return {}

    scores = {}
    for t in cfg.defensive_sleeve:
        if t not in asset_data:
            continue
        sig = asset_data[t]["signals"]
        if date_t not in sig.index or next_date not in sig.index:
            continue

        open_next = float(sig.loc[next_date, "Open"])
        sigma_t = float(sig.loc[date_t, "sigma_local"])
        if pd.isna(open_next) or open_next <= 0:
            continue

        if cfg.defensive_allocation_mode == "equal":
            score = 1.0
        else:
            score = 1.0 / max(sigma_t, 1e-8)

        if np.isfinite(score) and score > 0:
            scores[t] = score

    if not scores:
        return {}

    s = pd.Series(scores)
    w = residual_weight * (s / s.sum())
    return {k: float(v) for k, v in w.items() if v > 0}

# PORTEFEUILLE

def simulate_portfolio(asset_data: Dict[str, Dict[str, Any]], cfg: Config) -> Dict[str, Any]:
    common_dates = None
    for pack in asset_data.values():
        idx = pack["signals"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)

    common_dates = common_dates.sort_values()
    if len(common_dates) < 3:
        raise ValueError("Pas assez de dates communes pour simuler le portefeuille.")

    rebal_flags = rebalance_flags(common_dates, cfg.rebalance_mode)

    tickers = list(asset_data.keys())
    risk_universe = [t for t in tickers if t not in cfg.defensive_sleeve]

    weights = {t: 0.0 for t in tickers}
    entry_price = {t: np.nan for t in tickers}
    peak_close = {t: np.nan for t in tickers}
    position_kind = {t: None for t in tickers}  # "risk" ou "defensive"

    value = cfg.initial_capital

    equity_rows = [{
        "date": common_dates[0],
        "portfolio_value": value,
        "gross_return": 0.0,
        "costs": 0.0,
        "net_return": 0.0,
        "gross_exposure": 0.0,
        "n_positions": 0,
        "turnover": 0.0,
        "cash_weight": 1.0,
        "defensive_weight": 0.0,
        "risk_weight": 0.0,
    }]

    trade_rows = []
    weight_rows = []

    for k in range(0, len(common_dates) - 1):
        date_t = common_dates[k]
        next_date = common_dates[k + 1]
        value_start = value

        # rendement open_t -> open_t1 avec les poids existants
        asset_interval_returns = {}
        for t in tickers:
            sig = asset_data[t]["signals"]
            if date_t not in sig.index or next_date not in sig.index:
                asset_interval_returns[t] = 0.0
                continue

            open_t = float(sig.loc[date_t, "Open"])
            open_t1 = float(sig.loc[next_date, "Open"])
            if pd.isna(open_t) or pd.isna(open_t1) or open_t <= 0:
                asset_interval_returns[t] = 0.0
            else:
                asset_interval_returns[t] = open_t1 / open_t - 1.0

        gross_return = float(sum(weights[t] * asset_interval_returns[t] for t in tickers))
        value_pretrade = value_start * (1.0 + gross_return)

        if abs(1.0 + gross_return) > 1e-12:
            pretrade_weights = {
                t: (weights[t] * (1.0 + asset_interval_returns[t])) / (1.0 + gross_return)
                for t in tickers
            }
        else:
            pretrade_weights = {t: 0.0 for t in tickers}

        target_weights = pretrade_weights.copy()
        exited_today = set()

        # sorties quotidiennes: seulement sur les positions "risk"
        for t in tickers:
            if pretrade_weights[t] <= 0:
                continue
            if position_kind[t] != "risk":
                continue

            sig = asset_data[t]["signals"]
            if date_t not in sig.index:
                continue

            close_t = float(sig.loc[date_t, "Close"])
            atr_t = float(sig.loc[date_t, "ATR20"])
            p_change_t = float(sig.loc[date_t, "p_change"])
            sign_t = float(sig.loc[date_t, "causal_current_sign"])
            theta_seg_t = float(sig.loc[date_t, "causal_theta_so_far"])

            hard_stop = False
            trail_stop = False

            if pd.notna(entry_price[t]) and pd.notna(atr_t):
                hard_stop = close_t <= entry_price[t] - cfg.hard_stop_atr * atr_t

            if pd.notna(peak_close[t]) and pd.notna(atr_t):
                trail_stop = close_t <= peak_close[t] - cfg.trail_stop_atr * atr_t

            model_exit = (sign_t <= 0.0) or (theta_seg_t <= 0.0) or (p_change_t >= cfg.exit_p_change_min)

            if hard_stop or trail_stop or model_exit:
                target_weights[t] = 0.0
                exited_today.add(t)

        # mise à jour du peak uniquement pour les positions risk encore ouvertes
        for t in tickers:
            if pretrade_weights[t] > 0 and position_kind[t] == "risk":
                sig = asset_data[t]["signals"]
                if date_t in sig.index:
                    close_t = float(sig.loc[date_t, "Close"])
                    if pd.isna(peak_close[t]):
                        peak_close[t] = close_t
                    else:
                        peak_close[t] = max(float(peak_close[t]), close_t)

        # rebalancement
        if bool(rebal_flags.loc[date_t]):
            # bloc risk
            risk_scores = {}
            for t in risk_universe:
                if t in exited_today:
                    continue

                sig = asset_data[t]["signals"]
                if date_t not in sig.index or next_date not in sig.index:
                    continue

                alpha = float(sig.loc[date_t, "alpha"])
                eligible = float(sig.loc[date_t, "eligible_long"]) > 0.5
                next_open = float(sig.loc[next_date, "Open"])

                if eligible and pd.notna(alpha) and alpha > 0 and pd.notna(next_open) and next_open > 0:
                    risk_scores[t] = alpha

            if risk_scores:
                raw_scores = pd.Series(risk_scores).sort_values(ascending=False).head(cfg.top_k)
                risk_target = apply_caps(
                    raw_scores=raw_scores,
                    group_map=cfg.universe_groups,
                    asset_cap=cfg.asset_cap,
                    group_caps=cfg.group_caps,
                )
            else:
                risk_target = {}

            # sleeve défensive pour le reliquat
            risk_sum = float(sum(risk_target.values()))
            residual = max(0.0, 1.0 - risk_sum)

            defensive_target = build_defensive_weights(
                asset_data=asset_data,
                date_t=date_t,
                next_date=next_date,
                residual_weight=residual,
                cfg=cfg,
            )

            target_weights = {t: 0.0 for t in tickers}
            target_weights.update(risk_target)

            for t, w in defensive_target.items():
                target_weights[t] = target_weights.get(t, 0.0) + w

        # coûts
        diffs = {t: target_weights.get(t, 0.0) - pretrade_weights.get(t, 0.0) for t in tickers}
        order_assets = [t for t in tickers if abs(diffs[t]) * value_pretrade >= cfg.min_trade_dollars]
        num_orders = len(order_assets)

        turnover = float(sum(abs(diffs[t]) for t in tickers))
        sold_notional = float(sum(max(pretrade_weights.get(t, 0.0) - target_weights.get(t, 0.0), 0.0) for t in tickers) * value_pretrade)
        traded_notional = float(sum(abs(diffs[t]) for t in tickers) * value_pretrade)

        fixed_cost = cfg.ibkr_fixed_fee_per_order * num_orders
        sec_cost = cfg.sec_fee_per_dollar_sold * sold_notional
        slippage_cost = (cfg.slippage_bps_per_side / 10000.0) * traded_notional
        total_cost = fixed_cost + sec_cost + slippage_cost

        value = value_pretrade - total_cost
        if value <= 0:
            raise ValueError("La valeur du portefeuille est devenue non positive.")

        # journal des trades
        for t in order_assets:
            trade_notional = abs(diffs[t]) * value_pretrade
            side = "BUY" if diffs[t] > 0 else "SELL"
            trade_rows.append(
                {
                    "date_signal": date_t,
                    "date_execution": next_date,
                    "ticker": t,
                    "side": side,
                    "pretrade_weight": pretrade_weights.get(t, 0.0),
                    "target_weight": target_weights.get(t, 0.0),
                    "trade_weight": diffs[t],
                    "trade_notional": trade_notional,
                }
            )

        # mise à jour des états de position au next open
        risk_target_names = {t for t, w in target_weights.items() if w > 0 and t not in cfg.defensive_sleeve}
        defensive_target_names = {t for t, w in target_weights.items() if w > 0 and t in cfg.defensive_sleeve}

        for t in tickers:
            prev_open = pretrade_weights[t] > 1e-12
            new_open = target_weights.get(t, 0.0) > 1e-12

            next_open_price = np.nan
            sig = asset_data[t]["signals"]
            if next_date in sig.index:
                next_open_price = float(sig.loc[next_date, "Open"])

            if (not prev_open) and new_open:
                if t in risk_target_names:
                    position_kind[t] = "risk"
                    entry_price[t] = next_open_price
                    peak_close[t] = next_open_price
                elif t in defensive_target_names:
                    position_kind[t] = "defensive"
                    entry_price[t] = np.nan
                    peak_close[t] = np.nan

            elif prev_open and (not new_open):
                position_kind[t] = None
                entry_price[t] = np.nan
                peak_close[t] = np.nan

            elif prev_open and new_open:
                if t in risk_target_names and position_kind[t] != "risk":
                    position_kind[t] = "risk"
                    entry_price[t] = next_open_price
                    peak_close[t] = next_open_price
                elif t in defensive_target_names and position_kind[t] != "defensive":
                    position_kind[t] = "defensive"
                    entry_price[t] = np.nan
                    peak_close[t] = np.nan

        weights = {t: float(target_weights.get(t, 0.0)) for t in tickers}

        gross_exposure = float(sum(weights.values()))
        net_return = value / value_start - 1.0
        defensive_weight = float(sum(weights[t] for t in cfg.defensive_sleeve if t in weights))
        risk_weight = float(sum(weights[t] for t in risk_universe if t in weights))
        cash_weight = max(0.0, 1.0 - gross_exposure)

        equity_rows.append(
            {
                "date": next_date,
                "portfolio_value": value,
                "gross_return": gross_return,
                "costs": total_cost,
                "net_return": net_return,
                "gross_exposure": gross_exposure,
                "n_positions": int(sum(w > 0 for w in weights.values())),
                "turnover": turnover,
                "cash_weight": cash_weight,
                "defensive_weight": defensive_weight,
                "risk_weight": risk_weight,
            }
        )

        row = {"date": next_date}
        row.update(weights)
        weight_rows.append(row)

    equity = pd.DataFrame(equity_rows).set_index("date").sort_index()
    weights_df = pd.DataFrame(weight_rows).set_index("date").sort_index()
    trades_df = pd.DataFrame(trade_rows)

    return {
        "equity": equity,
        "weights": weights_df,
        "trades": trades_df,
        "common_dates": common_dates,
    }

# BENCHMARKS

def benchmark_buy_and_hold_open_to_open(asset_data: Dict[str, Dict[str, Any]], ticker: str, dates: pd.DatetimeIndex, name: str, initial_capital: float) -> Dict[str, Any]:
    sig = asset_data[ticker]["signals"]
    idx = dates.intersection(sig.index).sort_values()

    if len(idx) < 2:
        return {"equity": pd.Series(dtype=float), "returns": pd.Series(dtype=float), "metrics": {"name": name, "status": "empty"}}

    rets = []
    for k in range(len(idx) - 1):
        d0 = idx[k]
        d1 = idx[k + 1]
        r = float(sig.loc[d1, "Open"] / sig.loc[d0, "Open"] - 1.0)
        rets.append((d1, r))

    ret_s = pd.Series({d: r for d, r in rets}).sort_index()
    equity = initial_capital * (1.0 + ret_s).cumprod()
    equity = pd.concat([pd.Series({idx[0]: initial_capital}), equity]).sort_index()

    metrics = performance_metrics(equity, ret_s, name)
    return {"equity": equity, "returns": ret_s, "metrics": metrics}


def benchmark_equal_weight_universe(asset_data: Dict[str, Dict[str, Any]], dates: pd.DatetimeIndex, name: str, initial_capital: float) -> Dict[str, Any]:
    idx = dates.sort_values()
    if len(idx) < 2:
        return {"equity": pd.Series(dtype=float), "returns": pd.Series(dtype=float), "metrics": {"name": name, "status": "empty"}}

    returns = []
    for k in range(len(idx) - 1):
        d0 = idx[k]
        d1 = idx[k + 1]
        rs = []
        for t, pack in asset_data.items():
            sig = pack["signals"]
            if d0 in sig.index and d1 in sig.index:
                open0 = float(sig.loc[d0, "Open"])
                open1 = float(sig.loc[d1, "Open"])
                if open0 > 0 and pd.notna(open1):
                    rs.append(open1 / open0 - 1.0)

        r = float(np.mean(rs)) if rs else 0.0
        returns.append((d1, r))

    ret_s = pd.Series({d: r for d, r in returns}).sort_index()
    equity = initial_capital * (1.0 + ret_s).cumprod()
    equity = pd.concat([pd.Series({idx[0]: initial_capital}), equity]).sort_index()

    metrics = performance_metrics(equity, ret_s, name)
    return {"equity": equity, "returns": ret_s, "metrics": metrics}


def benchmark_6040(asset_data: Dict[str, Dict[str, Any]], dates: pd.DatetimeIndex, initial_capital: float) -> Dict[str, Any]:
    spy = asset_data["SPY"]["signals"]
    ief = asset_data["IEF"]["signals"]

    idx = dates.intersection(spy.index).intersection(ief.index).sort_values()
    if len(idx) < 2:
        return {"equity": pd.Series(dtype=float), "returns": pd.Series(dtype=float), "metrics": {"name": "SPY/IEF 60/40", "status": "empty"}}

    returns = []
    for k in range(len(idx) - 1):
        d0 = idx[k]
        d1 = idx[k + 1]
        r_spy = float(spy.loc[d1, "Open"] / spy.loc[d0, "Open"] - 1.0)
        r_ief = float(ief.loc[d1, "Open"] / ief.loc[d0, "Open"] - 1.0)
        returns.append((d1, 0.6 * r_spy + 0.4 * r_ief))

    ret_s = pd.Series({d: r for d, r in returns}).sort_index()
    equity = initial_capital * (1.0 + ret_s).cumprod()
    equity = pd.concat([pd.Series({idx[0]: initial_capital}), equity]).sort_index()

    metrics = performance_metrics(equity, ret_s, "SPY/IEF 60/40")
    return {"equity": equity, "returns": ret_s, "metrics": metrics}


def benchmark_defensive_blend(asset_data: Dict[str, Dict[str, Any]], dates: pd.DatetimeIndex, defensive_tickers: List[str], initial_capital: float) -> Dict[str, Any]:
    idx = dates.sort_values()
    if len(idx) < 2:
        return {"equity": pd.Series(dtype=float), "returns": pd.Series(dtype=float), "metrics": {"name": "Defensive Sleeve EW", "status": "empty"}}

    returns = []
    for k in range(len(idx) - 1):
        d0 = idx[k]
        d1 = idx[k + 1]
        rs = []
        for t in defensive_tickers:
            sig = asset_data[t]["signals"]
            if d0 in sig.index and d1 in sig.index:
                open0 = float(sig.loc[d0, "Open"])
                open1 = float(sig.loc[d1, "Open"])
                if open0 > 0 and pd.notna(open1):
                    rs.append(open1 / open0 - 1.0)
        r = float(np.mean(rs)) if rs else 0.0
        returns.append((d1, r))

    ret_s = pd.Series({d: r for d, r in returns}).sort_index()
    equity = initial_capital * (1.0 + ret_s).cumprod()
    equity = pd.concat([pd.Series({idx[0]: initial_capital}), equity]).sort_index()

    metrics = performance_metrics(equity, ret_s, "Defensive Sleeve EW")
    return {"equity": equity, "returns": ret_s, "metrics": metrics}

# HTML

def make_portfolio_dashboard_html(
    portfolio: Dict[str, Any],
    benchmarks: Dict[str, Dict[str, Any]],
    metrics_table: pd.DataFrame,
    out_path: str,
) -> str:
    equity = portfolio["equity"].copy()

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.42, 0.18, 0.20, 0.20],
        subplot_titles=(
            "Courbes d'équité",
            "Drawdown du portefeuille",
            "Exposition: risque / défensif / cash",
            "Nombre de positions et turnover",
        ),
    )

    base = equity["portfolio_value"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity["portfolio_value"] / base,
            mode="lines",
            name="Régime Portfolio",
            line=dict(width=2.4, color="#1d4ed8"),
        ),
        row=1,
        col=1,
    )

    for name, pack in benchmarks.items():
        eq = pack["equity"]
        if len(eq) > 0:
            fig.add_trace(
                go.Scatter(
                    x=eq.index,
                    y=eq / eq.iloc[0],
                    mode="lines",
                    name=name,
                    line=dict(width=1.4),
                ),
                row=1,
                col=1,
            )

    dd = equity["portfolio_value"] / equity["portfolio_value"].cummax() - 1.0
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd,
            mode="lines",
            name="DD Portfolio",
            line=dict(color="#dc2626", width=1.8),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity["risk_weight"],
            mode="lines",
            name="Risk book",
            line=dict(color="#2563eb", width=1.8),
            stackgroup="one",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity["defensive_weight"],
            mode="lines",
            name="Defensive sleeve",
            line=dict(color="#16a34a", width=1.8),
            stackgroup="one",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity["cash_weight"],
            mode="lines",
            name="Cash",
            line=dict(color="#a855f7", width=1.8),
            stackgroup="one",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity["n_positions"],
            mode="lines",
            name="# positions",
            line=dict(color="#7c3aed", width=1.8),
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity["turnover"],
            mode="lines",
            name="Turnover",
            line=dict(color="#f97316", width=1.4),
        ),
        row=4,
        col=1,
    )

    txt = metrics_table.round(4).to_string(index=False)
    fig.add_annotation(
        x=0.995,
        y=0.99,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        text=f"<b>Metrics</b><br><span style='font-family:monospace; white-space:pre'>{txt}</span>",
        bordercolor="#cbd5e1",
        borderwidth=1,
        bgcolor="white",
        font=dict(size=10),
    )

    fig.update_layout(
        title=dict(text="Backtest Portefeuille Régime Long-Only + Sleeve Défensive", x=0.5),
        template="plotly_white",
        height=1300,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0.01),
        margin=dict(l=60, r=40, t=90, b=50),
    )

    fig.update_yaxes(title_text="Equity / Equity0", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    fig.update_yaxes(title_text="Allocation", row=3, col=1, range=[0, 1.02])
    fig.update_yaxes(title_text="Count / Turnover", row=4, col=1)
    fig.update_xaxes(rangeslider_visible=True, row=4, col=1)

    pio.write_html(fig, file=out_path, include_plotlyjs="cdn", full_html=True)
    return out_path

# EXPORTS

def export_outputs(
    asset_data: Dict[str, Dict[str, Any]],
    portfolio: Dict[str, Any],
    benchmarks: Dict[str, Dict[str, Any]],
    metrics_table: pd.DataFrame,
    cfg: Config,
) -> Dict[str, str]:
    ensure_dir(cfg.output_dir)
    paths: Dict[str, str] = {}

    paths["portfolio_equity"] = os.path.join(cfg.output_dir, f"portfolio_equity_{cfg.interval}.csv")
    portfolio["equity"].to_csv(paths["portfolio_equity"])

    paths["portfolio_weights"] = os.path.join(cfg.output_dir, f"portfolio_weights_{cfg.interval}.csv")
    portfolio["weights"].to_csv(paths["portfolio_weights"])

    paths["portfolio_trades"] = os.path.join(cfg.output_dir, f"portfolio_trades_{cfg.interval}.csv")
    portfolio["trades"].to_csv(paths["portfolio_trades"], index=False)

    paths["portfolio_metrics"] = os.path.join(cfg.output_dir, f"portfolio_metrics_{cfg.interval}.csv")
    metrics_table.to_csv(paths["portfolio_metrics"], index=False)

    coef_rows = []
    for t, pack in asset_data.items():
        c = pack["coef_change_latest"].copy()
        if not c.empty:
            c.insert(0, "ticker", t)
            coef_rows.append(c)
    if coef_rows:
        coef_df = pd.concat(coef_rows, ignore_index=True)
        paths["coef_change_latest_all"] = os.path.join(cfg.output_dir, f"coef_change_latest_all_{cfg.interval}.csv")
        coef_df.to_csv(paths["coef_change_latest_all"], index=False)

    with open(os.path.join(cfg.output_dir, f"portfolio_config_{cfg.interval}.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2, default=str)

    return paths

# MAIN

def main() -> None:
    cfg = Config(
        tickers=TICKERS,
        universe_groups=UNIVERSE_GROUPS,
        defensive_sleeve=DEFENSIVE_SLEEVE,
        group_caps=GROUP_CAPS,
    )

    ensure_dir(cfg.cache_dir)
    ensure_dir(cfg.output_dir)

    print("=" * 108)
    print("BACKTEST PORTEFEUILLE REGIME LONG-ONLY + SLEEVE DEFENSIVE")
    print("=" * 108)
    print(f"TICKERS                    : {len(cfg.tickers)} actifs")
    print(f"DEFENSIVE_SLEEVE           : {cfg.defensive_sleeve}")
    print(f"START                      : {cfg.start}")
    print(f"END                        : {cfg.end}")
    print(f"INTERVAL                   : {cfg.interval}")
    print(f"TRAIN_LOOKBACK             : {cfg.train_lookback}")
    print(f"BACKTEST_WARMUP            : {cfg.backtest_warmup}")
    print(f"ENTRY_P_CHANGE_MAX         : {cfg.entry_p_change_max}")
    print(f"EXIT_P_CHANGE_MIN          : {cfg.exit_p_change_min}")
    print(f"CHANGE_THRESHOLD           : {cfg.change_threshold}")
    print(f"TOP_K                      : {cfg.top_k}")
    print(f"ASSET_CAP                  : {cfg.asset_cap:.2%}")
    print(f"REBALANCE_MODE             : {cfg.rebalance_mode}")
    print(f"DEFENSIVE_ALLOCATION_MODE  : {cfg.defensive_allocation_mode}")
    print(f"HARD_STOP_ATR              : {cfg.hard_stop_atr}")
    print(f"TRAIL_STOP_ATR             : {cfg.trail_stop_atr}")
    print(f"IBKR_FIXED_FEE_PER_ORDER   : {cfg.ibkr_fixed_fee_per_order}")
    print("=" * 108)

    asset_data: Dict[str, Dict[str, Any]] = {}

    for i, ticker in enumerate(cfg.tickers, start=1):
        print(f"[{i:02d}/{len(cfg.tickers):02d}] {ticker}")
        ohlc = load_ohlc(ticker, cfg)
        pack = build_asset_signal_table(ticker, ohlc, cfg)
        asset_data[ticker] = pack

    portfolio = simulate_portfolio(asset_data, cfg)

    equity = portfolio["equity"]
    portfolio_metrics = performance_metrics(equity["portfolio_value"], equity["net_return"].dropna(), "Regime Portfolio")
    portfolio_metrics["avg_exposure"] = float(equity["gross_exposure"].mean())
    portfolio_metrics["avg_risk_weight"] = float(equity["risk_weight"].mean())
    portfolio_metrics["avg_defensive_weight"] = float(equity["defensive_weight"].mean())
    portfolio_metrics["avg_cash_weight"] = float(equity["cash_weight"].mean())
    portfolio_metrics["avg_positions"] = float(equity["n_positions"].mean())
    portfolio_metrics["turnover_annualized"] = float(equity["turnover"].mean() * 252.0)

    benchmarks = {
        "SPY Buy&Hold": benchmark_buy_and_hold_open_to_open(asset_data, "SPY", portfolio["common_dates"], "SPY Buy&Hold", cfg.initial_capital),
        "EqualWeight Universe": benchmark_equal_weight_universe(asset_data, portfolio["common_dates"], "EqualWeight Universe", cfg.initial_capital),
        "SPY/IEF 60/40": benchmark_6040(asset_data, portfolio["common_dates"], cfg.initial_capital),
        "Defensive Sleeve EW": benchmark_defensive_blend(asset_data, portfolio["common_dates"], cfg.defensive_sleeve, cfg.initial_capital),
    }

    metric_rows = [portfolio_metrics]
    for _, pack in benchmarks.items():
        metric_rows.append(pack["metrics"])

    metrics_table = pd.DataFrame(metric_rows)

    dashboard_path = make_portfolio_dashboard_html(
        portfolio=portfolio,
        benchmarks=benchmarks,
        metrics_table=metrics_table,
        out_path=os.path.join(cfg.output_dir, f"portfolio_dashboard_{cfg.interval}.html"),
    )

    export_paths = export_outputs(
        asset_data=asset_data,
        portfolio=portfolio,
        benchmarks=benchmarks,
        metrics_table=metrics_table,
        cfg=cfg,
    )

    print("\n[RESULTATS PORTEFEUILLE]")
    print(metrics_table.round(4).to_string(index=False))

    print("\n[ALLOCATIONS MOYENNES]")
    print(f" - avg gross exposure   : {portfolio_metrics['avg_exposure']:.2%}")
    print(f" - avg risk weight      : {portfolio_metrics['avg_risk_weight']:.2%}")
    print(f" - avg defensive weight : {portfolio_metrics['avg_defensive_weight']:.2%}")
    print(f" - avg cash weight      : {portfolio_metrics['avg_cash_weight']:.2%}")
    print(f" - avg positions        : {portfolio_metrics['avg_positions']:.2f}")
    print(f" - turnover annualisé   : {portfolio_metrics['turnover_annualized']:.2f}")

    if not portfolio["trades"].empty:
        print(f" - nombre de trades     : {len(portfolio['trades'])}")

    print("\n[HTML]")
    print(f" - dashboard : {dashboard_path}")

    print("\n[EXPORTS]")
    for k, v in export_paths.items():
        print(f" - {k:24s}: {v}")

    print("\nTerminé.")


if __name__ == "__main__":
    main()