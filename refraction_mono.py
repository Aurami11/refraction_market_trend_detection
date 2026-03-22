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
TICKERS = ["AAPL", "MSFT", "NVDA"]
START = "2020-01-01"
END: Optional[str] = None
INTERVAL = "1d"
CACHE_DIR = "cache"
OUTPUT_DIR = "outputs"
FORCE_REFRESH = False
VOL_WINDOW = 20
EPS = 1e-8
SIN_ACCEPT_MIN = 1e-3
ANGLE_TOL = 0.12
AMP_TOL_Z = 1.25
SMALL_CORRECTION_Z = 0.75
BACKTEST_WARMUP = 150
BACKTEST_STEP = 1
BACKTEST_MAX_STEPS: Optional[int] = None

# Régressions logistiques
CHANGE_CLASS_WEIGHT = "balanced"   # None ou "balanced"
SAME_CLASS_WEIGHT = None           # None ou "balanced"
LOGIT_MAX_ITER = 2000

# Seuil décisionnel pour "change"
CHANGE_THRESHOLD = 0.50

# Platt scaling causal
PLATT_CALIBRATION_FRAC = 0.25
PLATT_MIN_CALIBRATION_SIZE = 120
PLATT_MIN_CORE_TRAIN_SIZE = 250

# CONFIG
@dataclass
class Config:
    cache_dir: str = CACHE_DIR
    output_dir: str = OUTPUT_DIR
    start: str = START
    end: Optional[str] = END
    interval: str = INTERVAL
    force_refresh: bool = FORCE_REFRESH
    vol_window: int = VOL_WINDOW
    eps: float = EPS
    sin_accept_min: float = SIN_ACCEPT_MIN
    angle_tol: float = ANGLE_TOL
    amp_tol_z: float = AMP_TOL_Z
    small_correction_z: float = SMALL_CORRECTION_Z
    backtest_warmup: int = BACKTEST_WARMUP
    backtest_step: int = BACKTEST_STEP
    backtest_max_steps: Optional[int] = BACKTEST_MAX_STEPS
    change_class_weight: Optional[str] = CHANGE_CLASS_WEIGHT
    same_class_weight: Optional[str] = SAME_CLASS_WEIGHT
    logit_max_iter: int = LOGIT_MAX_ITER
    change_threshold: float = CHANGE_THRESHOLD
    platt_calibration_frac: float = PLATT_CALIBRATION_FRAC
    platt_min_calibration_size: int = PLATT_MIN_CALIBRATION_SIZE
    platt_min_core_train_size: int = PLATT_MIN_CORE_TRAIN_SIZE

# ETAT REDUIT + DELTAS
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

def ticker_dir(base_dir: str, ticker: str) -> str:
    path = os.path.join(base_dir, safe_name(ticker))
    ensure_dir(path)
    return path

def today_end_exclusive() -> str:
    return str((pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)).date())

def infer_next_timestamp(index: pd.DatetimeIndex) -> pd.Timestamp:
    if len(index) < 2:
        return index[-1] + pd.Timedelta(days=1)
    diffs = pd.Series(index[1:] - index[:-1])
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return index[-1] + pd.Timedelta(days=1)
    return index[-1] + diffs.median()

def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.sum(x[mask] * w[mask]) / np.sum(w[mask]))

def weighted_quantile(values: np.ndarray, qs: List[float], weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return np.array([np.nan] * len(qs), dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return np.interp(qs, cdf, values)

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

# DONNEES

def load_ohlc(ticker: str, cfg: Config) -> pd.DataFrame:
    ensure_dir(cfg.cache_dir)
    path = os.path.join(cfg.cache_dir, f"{safe_name(ticker)}_{cfg.interval}.csv")
    if os.path.exists(path) and not cfg.force_refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
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
    df = df.dropna(subset=["r"]).copy()
    df["sin_theta_abs"] = np.abs(np.sin(df["theta"]))
    sin_ref = float(df["sin_theta_abs"].iloc[0] + cfg.eps)
    if df["sin_theta_abs"].iloc[0] < cfg.sin_accept_min:
        warnings.warn(
            "Premier angle très faible: la jauge n_1 = 1 est conservée, "
            "mais l'échelle globale de n_eff peut être fragile."
        )
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

def summarize_segment(feat: pd.DataFrame, start: int, end: int, segment_id: int) -> Dict[str, Any]:
    seg = feat.iloc[start:end + 1]
    net_return = float(seg["r"].sum())
    sign_global = float(np.sign(net_return)) if net_return != 0 else 0.0
    weights = seg["amplitude"].to_numpy()
    theta_vals = seg["theta"].to_numpy()
    theta_mean = float(np.average(theta_vals, weights=weights)) if weights.sum() > 0 else float(np.nanmean(theta_vals))
    return {
        "segment_id": segment_id,
        "start_pos": start,
        "end_pos": end,
        "start_time": seg.index[0],
        "end_time": seg.index[-1],
        "length": int(len(seg)),
        "net_return": net_return,
        "theta_mean": theta_mean,
        "amp_z_mean": float(seg["amp_z"].mean()),
        "sigma_mean": float(seg["sigma_local"].mean()),
        "viscosity_mean": float(seg["viscosity"].mean()),
        "sign_global": sign_global,
    }

def build_expost_segments(features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    segments = []
    start = 0
    seg_id = 0
    for i in range(1, len(features)):
        if not locally_compatible(features.iloc[i - 1], features.iloc[i], cfg):
            segments.append(summarize_segment(features, start, i - 1, seg_id))
            start = i
            seg_id += 1
    segments.append(summarize_segment(features, start, len(features) - 1, seg_id))
    return pd.DataFrame(segments)

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
    """
    Calibration causale:
    - coeur ancien -> apprend le score brut
    - queue récente -> apprend le Platt scaling sur ce score
    - prédiction finale faite avec le modèle coeur + calibrateur
      pour rester causal et cohérent
    """
    data = pd.concat([X[feature_cols], y.rename("target")], axis=1).dropna().reset_index(drop=True)
    empty_coef = pd.DataFrame(columns=["feature", "coef_std", "odds_ratio_1sd", "abs_coef_std"])
    if data.empty:
        return None, None, np.nan, empty_coef, {
            "used": False,
            "reason": "empty_training_data",
        }
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
        model_core, base_rate_core, coef_df = fit_logistic_binary(
            X_all, y_all, feature_cols, class_weight, cfg
        )
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
        model_core, base_rate_core, coef_df = fit_logistic_binary(
            X_all, y_all, feature_cols, class_weight, cfg
        )
        return model_core, None, base_rate_core, coef_df, {
            "used": False,
            "reason": "core_or_calibration_single_class",
            "n_total": int(n_total),
            "n_core": int(n_core),
            "n_calibration": int(n_cal),
        }
    model_core, base_rate_core, coef_df = fit_logistic_binary(
        X_core, y_core, feature_cols, class_weight, cfg
    )
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
                "length_q10": np.nan,
                "length_q50": np.nan,
                "length_q90": np.nan,
            }
            continue
        len_q10, len_q50, len_q90 = np.nanquantile(subset["next_segment_length"], [0.10, 0.50, 0.90])
        out[class_name] = {
            "theta_mean": float(subset["next_segment_theta"].mean()),
            "amp_mean": float(subset["next_segment_amp"].mean()),
            "length_mean": float(subset["next_segment_length"].mean()),
            "ret_mean": float(subset["next_segment_net_return"].mean()),
            "length_q10": float(len_q10),
            "length_q50": float(len_q50),
            "length_q90": float(len_q90),
        }
    return out

# PREDICTION HIERARCHIQUE

def predict_one_step(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    query_idx: int,
    cfg: Config,
    return_coef: bool = False,
) -> Dict[str, Any]:
    if query_idx <= cfg.backtest_warmup:
        return {"status": "insufficient_history"}
    x_query = features.iloc[query_idx]

    # Etape 1: hazard change / continue avec Platt scaling causal
    X1 = features.iloc[:query_idx][STATE_COLS]
    y1 = labels.iloc[:query_idx]["change_next"].astype(float)
    model_change, calibrator_change, base_change, coef_change, platt_info_change = fit_logistic_with_platt(
        X1,
        y1,
        STATE_COLS,
        cfg.change_class_weight,
        cfg,
    )
    p_change_raw = predict_raw_proba_binary(
        model_change,
        base_change,
        x_query,
        STATE_COLS,
    )
    p_change = predict_platt_proba_binary(
        model_change,
        calibrator_change,
        base_change,
        x_query,
        STATE_COLS,
    )
    p_continue = 1.0 - p_change if pd.notna(p_change) else np.nan

    # Etape 2: same / opp | change
    completed_mask = labels["next_segment_completed_by"] <= query_idx
    train_change_mask = (labels["change_next"] == 1.0) & completed_mask
    X2 = features.loc[train_change_mask, STATE_COLS]
    y2 = labels.loc[train_change_mask, "next_segment_same_direction"].astype(float)
    model_same, base_same, coef_same = fit_logistic_binary(
        X2,
        y2,
        STATE_COLS,
        cfg.same_class_weight,
        cfg,
    )
    if len(X2.dropna()) == 0:
        q_same = np.nan
        q_opp = np.nan
        stats = summarize_next_segment_stats(pd.DataFrame())
    else:
        q_same = predict_raw_proba_binary(model_same, base_same, x_query, STATE_COLS)
        q_opp = 1.0 - q_same if pd.notna(q_same) else np.nan
        train_change_labels = labels.loc[train_change_mask].copy()
        stats = summarize_next_segment_stats(train_change_labels)
    p_same_uncond = np.nan
    p_opp_uncond = np.nan
    if pd.notna(p_change) and pd.notna(q_same):
        p_same_uncond = (1.0 - p_change) + p_change * q_same
        p_opp_uncond = p_change * q_opp
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
        "timestamp_t1_expected": infer_next_timestamp(features.index[:query_idx + 1]),
        "p_change_raw": p_change_raw,
        "p_change": p_change,
        "p_continue": p_continue,
        "q_same_given_change": q_same,
        "q_opp_given_change": q_opp,
        "p_same_unconditional": p_same_uncond,
        "p_opp_unconditional": p_opp_uncond,
        "expected_next_segment_theta": expected_theta,
        "expected_next_segment_amp": expected_amp,
        "expected_next_segment_length": expected_length,
        "expected_next_segment_net_return": expected_ret,
        "same_length_q10": stats["same"]["length_q10"],
        "same_length_q50": stats["same"]["length_q50"],
        "same_length_q90": stats["same"]["length_q90"],
        "opp_length_q10": stats["opp"]["length_q10"],
        "opp_length_q50": stats["opp"]["length_q50"],
        "opp_length_q90": stats["opp"]["length_q90"],
        "base_rate_change_train": base_change,
        "base_rate_same_train": base_same,
        "n_completed_changes_train": int(train_change_mask.sum()),
        "platt_used_change": platt_info_change["used"],
        "platt_reason_change": platt_info_change["reason"],
        "platt_n_total_change": platt_info_change.get("n_total", np.nan),
        "platt_n_core_change": platt_info_change.get("n_core", np.nan),
        "platt_n_calibration_change": platt_info_change.get("n_calibration", np.nan),
    }
    if return_coef:
        out["coef_change"] = coef_change
        out["coef_same"] = coef_same
    return out

# BACKTEST

def run_backtest(features: pd.DataFrame, labels: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    n = len(features)
    if n < cfg.backtest_warmup + 2:
        return pd.DataFrame()
    last_query_idx = n - 2
    query_indices = list(range(cfg.backtest_warmup, last_query_idx + 1, cfg.backtest_step))
    if cfg.backtest_max_steps is not None:
        query_indices = query_indices[-cfg.backtest_max_steps:]
    rows: List[Dict[str, Any]] = []
    for q in query_indices:
        pred = predict_one_step(features, labels, q, cfg, return_coef=False)
        if pred.get("status") != "ok":
            continue
        actual_change = float(labels["change_next"].iloc[q])
        actual_continue = float(labels["continue_next"].iloc[q])
        pred_change_label = float(pred["p_change"] >= cfg.change_threshold) if pd.notna(pred["p_change"]) else np.nan
        row = {
            "timestamp_t": features.index[q],
            "timestamp_t1": features.index[q + 1],
            "p_change_raw": pred["p_change_raw"],
            "p_change": pred["p_change"],
            "p_continue": pred["p_continue"],
            "actual_change": actual_change,
            "actual_continue": actual_continue,
            "pred_change_label": pred_change_label,
            "hit_change": float(pred_change_label == actual_change) if pd.notna(pred_change_label) else np.nan,
            "brier_change": float((pred["p_change"] - actual_change) ** 2) if pd.notna(pred["p_change"]) else np.nan,
            "q_same_given_change": pred["q_same_given_change"],
            "q_opp_given_change": pred["q_opp_given_change"],
            "actual_same_given_change": labels["next_segment_same_direction"].iloc[q],
            "actual_opp_given_change": labels["next_segment_opp_direction"].iloc[q],
            "expected_next_segment_theta": pred["expected_next_segment_theta"],
            "expected_next_segment_amp": pred["expected_next_segment_amp"],
            "expected_next_segment_length": pred["expected_next_segment_length"],
            "expected_next_segment_net_return": pred["expected_next_segment_net_return"],
            "actual_next_segment_theta": labels["next_segment_theta"].iloc[q],
            "actual_next_segment_amp": labels["next_segment_amp"].iloc[q],
            "actual_next_segment_length": labels["next_segment_length"].iloc[q],
            "actual_next_segment_net_return": labels["next_segment_net_return"].iloc[q],
        }
        if actual_change == 1.0 and pd.notna(row["actual_same_given_change"]) and pd.notna(row["q_same_given_change"]):
            row["brier_same_given_change"] = float((row["q_same_given_change"] - row["actual_same_given_change"]) ** 2)
            row["hit_same_given_change"] = float((row["q_same_given_change"] >= 0.5) == bool(row["actual_same_given_change"]))
        else:
            row["brier_same_given_change"] = np.nan
            row["hit_same_given_change"] = np.nan
        rows.append(row)
    bt = pd.DataFrame(rows)
    if not bt.empty:
        bt["rolling_hit_change_50"] = bt["hit_change"].rolling(50, min_periods=5).mean()
        bt["rolling_brier_change_50"] = bt["brier_change"].rolling(50, min_periods=5).mean()
    return bt

def backtest_metrics(bt: pd.DataFrame) -> Dict[str, Any]:
    if bt.empty:
        return {"status": "empty"}
    y_true = bt["actual_change"].to_numpy(dtype=float)
    y_prob = bt["p_change"].to_numpy(dtype=float)
    y_prob_raw = bt["p_change_raw"].to_numpy(dtype=float)
    y_hat = bt["pred_change_label"].to_numpy(dtype=float)
    tp = float(((y_hat == 1.0) & (y_true == 1.0)).sum())
    fp = float(((y_hat == 1.0) & (y_true == 0.0)).sum())
    fn = float(((y_hat == 0.0) & (y_true == 1.0)).sum())
    base_rate = float(np.mean(y_true))
    baseline_brier = base_rate * (1.0 - base_rate)
    baseline_accuracy = max(base_rate, 1.0 - base_rate)
    metrics: Dict[str, Any] = {
        "status": "ok",
        "n_predictions": int(len(bt)),
        "base_rate_change": base_rate,
        "predicted_rate_change": float(np.mean(y_hat)),
        "avg_p_change_raw": float(np.mean(y_prob_raw)),
        "avg_p_change": float(np.mean(y_prob)),
        "brier_change_raw": float(np.mean((y_prob_raw - y_true) ** 2)),
        "brier_change": float(np.mean(bt["brier_change"])),
        "baseline_brier_change": float(baseline_brier),
        "accuracy_change": float(np.mean(bt["hit_change"])),
        "baseline_accuracy_change": float(baseline_accuracy),
        "precision_change": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "recall_change": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc_change_raw"] = float(roc_auc_score(y_true, y_prob_raw))
        metrics["avg_precision_change_raw"] = float(average_precision_score(y_true, y_prob_raw))
        metrics["roc_auc_change"] = float(roc_auc_score(y_true, y_prob))
        metrics["avg_precision_change"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc_change_raw"] = np.nan
        metrics["avg_precision_change_raw"] = np.nan
        metrics["roc_auc_change"] = np.nan
        metrics["avg_precision_change"] = np.nan
    valid_change = bt["brier_same_given_change"].notna()
    metrics["n_real_changes"] = int(valid_change.sum())
    if valid_change.any():
        actual_same = bt.loc[valid_change, "actual_same_given_change"].to_numpy(dtype=float)
        q_same = bt.loc[valid_change, "q_same_given_change"].to_numpy(dtype=float)
        metrics["brier_same_given_change"] = float(bt.loc[valid_change, "brier_same_given_change"].mean())
        metrics["accuracy_same_given_change"] = float(bt.loc[valid_change, "hit_same_given_change"].mean())
        metrics["base_rate_same_given_change"] = float(np.mean(actual_same))
        metrics["avg_q_same_given_change"] = float(np.mean(q_same))
    else:
        metrics["brier_same_given_change"] = np.nan
        metrics["accuracy_same_given_change"] = np.nan
        metrics["base_rate_same_given_change"] = np.nan
        metrics["avg_q_same_given_change"] = np.nan
    return metrics

# HTML

def make_dashboard_html(
    ticker: str,
    features: pd.DataFrame,
    segments_expost: pd.DataFrame,
    latest_pred: Dict[str, Any],
    bt: pd.DataFrame,
    metrics: Dict[str, Any],
    cfg: Config,
) -> str:
    out_dir = ticker_dir(cfg.output_dir, ticker)
    out_path = os.path.join(out_dir, f"{safe_name(ticker)}_{cfg.interval}_dashboard.html")
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.42, 0.18, 0.20, 0.20],
        subplot_titles=(
            f"{ticker} • Prix + segments ex post",
            "n_eff brut / n_eff utilisable",
            "Backtest: P(change) brut vs calibré",
            "Backtest: q(same | change) sur changements réalisés",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=features.index,
            y=features["Close"],
            mode="lines",
            name="Close",
            line=dict(color="#0f172a", width=1.5),
        ),
        row=1,
        col=1,
    )
    for _, seg in segments_expost.iterrows():
        s = int(seg["start_pos"])
        e = int(seg["end_pos"])
        idx = features.index[s:e + 1]
        color = "#10b981" if seg["sign_global"] > 0 else "#ef4444" if seg["sign_global"] < 0 else "#94a3b8"
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=features.loc[idx, "Close"],
                mode="lines",
                line=dict(color=color, width=3),
                showlegend=False,
                hovertemplate="Date=%{x}<br>Close=%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=features.index,
            y=features["n_eff"],
            mode="lines",
            name="n_eff brut",
            line=dict(color="#2563eb", width=1.1),
            opacity=0.35,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=features.index,
            y=features["n_eff_for_use"],
            mode="lines",
            name="n_eff utilisable",
            line=dict(color="#0f766e", width=1.8),
        ),
        row=2,
        col=1,
    )
    bad_mask = ~features["n_eff_acceptable"]
    if bad_mask.any():
        fig.add_trace(
            go.Scatter(
                x=features.index[bad_mask],
                y=features.loc[bad_mask, "n_eff_for_use"],
                mode="markers",
                name="Non acceptable",
                marker=dict(color="#b91c1c", size=6, symbol="x"),
            ),
            row=2,
            col=1,
        )
    if not bt.empty:
        fig.add_trace(
            go.Scatter(
                x=bt["timestamp_t"],
                y=bt["p_change_raw"],
                mode="lines",
                name="P(change) brut",
                line=dict(color="#f59e0b", width=1.4, dash="dot"),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bt["timestamp_t"],
                y=bt["p_change"],
                mode="lines",
                name="P(change) calibré",
                line=dict(color="#dc2626", width=1.8),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bt["timestamp_t"],
                y=bt["actual_change"],
                mode="lines",
                name="Change réalisé",
                line=dict(color="#111827", width=1.2, dash="dot"),
            ),
            row=3,
            col=1,
        )
        change_only = bt["actual_change"] == 1.0
        if change_only.any():
            fig.add_trace(
                go.Scatter(
                    x=bt.loc[change_only, "timestamp_t"],
                    y=bt.loc[change_only, "q_same_given_change"],
                    mode="lines+markers",
                    name="q(same | change)",
                    line=dict(color="#16a34a", width=1.8),
                    marker=dict(size=6),
                ),
                row=4,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=bt.loc[change_only, "timestamp_t"],
                    y=bt.loc[change_only, "actual_same_given_change"],
                    mode="lines+markers",
                    name="Same réalisé | change",
                    line=dict(color="#1f2937", width=1.2, dash="dot"),
                    marker=dict(size=5),
                ),
                row=4,
                col=1,
            )
    live_text = "Prévision indisponible"
    if latest_pred.get("status") == "ok":
        live_text = (
            "<b>Prévision live</b><br>"
            f"P(change) brut = {latest_pred['p_change_raw']:.2%}<br>"
            f"P(change) calibré = {latest_pred['p_change']:.2%}<br>"
            f"P(continue) = {latest_pred['p_continue']:.2%}<br>"
            f"q(same | change) = {latest_pred['q_same_given_change']:.2%}<br>"
            f"q(opp | change) = {latest_pred['q_opp_given_change']:.2%}<br>"
            f"E[length next seg] = {latest_pred['expected_next_segment_length']:.2f}<br>"
            f"E[ret next seg] = {latest_pred['expected_next_segment_net_return']:.5f}"
        )
    metric_text = "Backtest indisponible"
    if metrics.get("status") == "ok":
        metric_text = (
            "<b>Backtest</b><br>"
            f"N = {metrics['n_predictions']}<br>"
            f"Base rate change = {metrics['base_rate_change']:.2%}<br>"
            f"Avg P(change) brut = {metrics['avg_p_change_raw']:.2%}<br>"
            f"Avg P(change) calibré = {metrics['avg_p_change']:.2%}<br>"
            f"Brier brut = {metrics['brier_change_raw']:.4f}<br>"
            f"Brier calibré = {metrics['brier_change']:.4f}<br>"
            f"ROC AUC calibré = {metrics['roc_auc_change']:.4f}"
        )
    fig.add_annotation(
        x=0.995,
        y=0.99,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        text=live_text,
        bordercolor="#cbd5e1",
        borderwidth=1,
        bgcolor="white",
        font=dict(size=11),
    )
    fig.add_annotation(
        x=0.995,
        y=0.63,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        text=metric_text,
        bordercolor="#86efac",
        borderwidth=1,
        bgcolor="#ecfdf5",
        font=dict(size=11, color="#14532d"),
    )
    fig.update_layout(
        title=dict(text=f"{ticker} • Dashboard régimes + Platt scaling", x=0.5),
        template="plotly_white",
        height=1250,
        hovermode="x unified",
        margin=dict(l=60, r=30, t=90, b=60),
        legend=dict(orientation="h", y=1.02, x=0.01),
    )
    fig.update_yaxes(type="log", row=2, col=1, title_text="n_eff")
    fig.update_yaxes(range=[-0.02, 1.02], row=3, col=1, title_text="P(change)")
    fig.update_yaxes(range=[-0.02, 1.02], row=4, col=1, title_text="q(same|change)")
    fig.update_yaxes(title_text="Close", row=1, col=1)
    fig.update_xaxes(rangeslider_visible=True, row=4, col=1)
    pio.write_html(fig, file=out_path, include_plotlyjs="cdn", full_html=True)
    return out_path

def make_probability_diagnostics_html(
    ticker: str,
    bt: pd.DataFrame,
    cfg: Config,
) -> str:
    out_dir = ticker_dir(cfg.output_dir, ticker)
    out_path = os.path.join(out_dir, f"{safe_name(ticker)}_{cfg.interval}_probability_diagnostics.html")
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Histogramme P(change) brut vs calibré",
            "Boxplot P(change) calibré selon réalité",
            "Calibration brute vs calibrée",
            "Precision / Recall selon seuil calibré",
            "Histogramme q(same | change)",
            "Calibration de q(same | change)",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
    )
    if not bt.empty:
        fig.add_trace(
            go.Histogram(
                x=bt["p_change_raw"],
                nbinsx=30,
                marker_color="#f59e0b",
                opacity=0.55,
                name="P(change) brut",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=bt["p_change"],
                nbinsx=30,
                marker_color="#dc2626",
                opacity=0.55,
                name="P(change) calibré",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Box(
                x=bt["actual_change"].astype(str),
                y=bt["p_change"],
                boxmean=True,
                marker_color="#2563eb",
                name="P(change) calibré",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        calib_raw = calibration_curve_df(bt, "p_change_raw", "actual_change", n_bins=10)
        calib_cal = calibration_curve_df(bt, "p_change", "actual_change", n_bins=10)
        if not calib_raw.empty:
            fig.add_trace(
                go.Scatter(
                    x=calib_raw["mean_pred"],
                    y=calib_raw["mean_actual"],
                    mode="lines+markers",
                    line=dict(color="#f59e0b", width=2),
                    marker=dict(size=8),
                    name="Calibration brute",
                ),
                row=1,
                col=3,
            )
        if not calib_cal.empty:
            fig.add_trace(
                go.Scatter(
                    x=calib_cal["mean_pred"],
                    y=calib_cal["mean_actual"],
                    mode="lines+markers",
                    line=dict(color="#16a34a", width=2),
                    marker=dict(size=8),
                    name="Calibration calibrée",
                ),
                row=1,
                col=3,
            )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="#9ca3af", dash="dash"),
                name="x = y",
            ),
            row=1,
            col=3,
        )
        thresholds = np.linspace(0.0, 1.0, 101)
        sweep = threshold_sweep(bt, "p_change", "actual_change", thresholds)
        fig.add_trace(
            go.Scatter(
                x=sweep["threshold"],
                y=sweep["precision"],
                mode="lines",
                name="Precision calibrée",
                line=dict(color="#2563eb", width=2),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sweep["threshold"],
                y=sweep["recall"],
                mode="lines",
                name="Recall calibré",
                line=dict(color="#dc2626", width=2),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[cfg.change_threshold, cfg.change_threshold],
                y=[0, 1],
                mode="lines",
                line=dict(color="#111827", dash="dash"),
                name="threshold courant",
            ),
            row=2,
            col=1,
        )
        valid_same = bt["actual_change"] == 1.0
        if valid_same.any():
            fig.add_trace(
                go.Histogram(
                    x=bt.loc[valid_same, "q_same_given_change"],
                    nbinsx=20,
                    marker_color="#16a34a",
                    opacity=0.75,
                    showlegend=False,
                ),
                row=2,
                col=2,
            )
            calib_same = calibration_curve_df(
                bt.loc[valid_same].dropna(subset=["q_same_given_change", "actual_same_given_change"]),
                "q_same_given_change",
                "actual_same_given_change",
                n_bins=8,
            )
            if not calib_same.empty:
                fig.add_trace(
                    go.Scatter(
                        x=calib_same["mean_pred"],
                        y=calib_same["mean_actual"],
                        mode="lines+markers",
                        line=dict(color="#16a34a", width=2),
                        marker=dict(size=8),
                        showlegend=False,
                    ),
                    row=2,
                    col=3,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        line=dict(color="#9ca3af", dash="dash"),
                        showlegend=False,
                    ),
                    row=2,
                    col=3,
                )
    fig.update_layout(
        title=dict(text=f"{ticker} • Diagnostics des probabilités", x=0.5),
        template="plotly_white",
        height=950,
        margin=dict(l=50, r=30, t=80, b=40),
        legend=dict(orientation="h", y=1.04, x=0.01),
        barmode="overlay",
    )
    fig.update_xaxes(title_text="P(change)", row=1, col=1)
    fig.update_xaxes(title_text="Actual change", row=1, col=2)
    fig.update_xaxes(title_text="Probabilité prédite", row=1, col=3)
    fig.update_xaxes(title_text="Threshold", row=2, col=1)
    fig.update_xaxes(title_text="q(same | change)", row=2, col=2)
    fig.update_xaxes(title_text="Probabilité prédite", row=2, col=3)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="P(change)", row=1, col=2)
    fig.update_yaxes(title_text="Fréquence observée", row=1, col=3)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    fig.update_yaxes(title_text="Fréquence observée", row=2, col=3)
    pio.write_html(fig, file=out_path, include_plotlyjs="cdn", full_html=True)
    return out_path

# EXPORTS

def export_files(
    ticker: str,
    ohlc: pd.DataFrame,
    features: pd.DataFrame,
    segments_expost: pd.DataFrame,
    labels: pd.DataFrame,
    backtest: pd.DataFrame,
    latest_prediction: pd.DataFrame,
    coef_change: pd.DataFrame,
    coef_same: pd.DataFrame,
    metrics: Dict[str, Any],
    cfg: Config,
) -> Dict[str, str]:
    out_dir = ticker_dir(cfg.output_dir, ticker)
    base = f"{safe_name(ticker)}_{cfg.interval}"
    paths: Dict[str, str] = {}
    paths["ohlc"] = os.path.join(out_dir, f"{base}_ohlc.csv")
    ohlc.to_csv(paths["ohlc"])
    paths["features"] = os.path.join(out_dir, f"{base}_features.csv")
    features.to_csv(paths["features"])
    paths["segments_expost"] = os.path.join(out_dir, f"{base}_segments_expost.csv")
    segments_expost.to_csv(paths["segments_expost"], index=False)
    paths["labels"] = os.path.join(out_dir, f"{base}_event_labels.csv")
    labels.to_csv(paths["labels"])
    paths["backtest"] = os.path.join(out_dir, f"{base}_backtest.csv")
    backtest.to_csv(paths["backtest"], index=False)
    paths["latest_prediction"] = os.path.join(out_dir, f"{base}_latest_prediction.csv")
    latest_prediction.to_csv(paths["latest_prediction"], index=False)
    paths["coef_change"] = os.path.join(out_dir, f"{base}_coef_change.csv")
    coef_change.to_csv(paths["coef_change"], index=False)
    paths["coef_same"] = os.path.join(out_dir, f"{base}_coef_same.csv")
    coef_same.to_csv(paths["coef_same"], index=False)
    paths["metrics"] = os.path.join(out_dir, f"{base}_metrics.json")
    with open(paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    paths["config"] = os.path.join(out_dir, f"{base}_config.json")
    with open(paths["config"], "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2, default=str)
    return paths

# PIPELINE PAR TICKER

def run_for_ticker(ticker: str, cfg: Config) -> Dict[str, Any]:
    print(f"[TICKER] {ticker}")
    ohlc = load_ohlc(ticker, cfg)
    print(f"[DATA] {ticker}: OHLCV shape = {ohlc.shape}")
    features = build_features(ohlc, cfg)
    features = build_causal_state(features, cfg)
    features = features.dropna(subset=STATE_COLS).copy()
    print(f"[FEATURES] {ticker}: shape = {features.shape}")
    segments_expost = build_expost_segments(features, cfg)
    print(f"[SEGMENTS] {ticker}: ex post = {len(segments_expost)}")
    labels = build_event_labels(features)
    backtest = run_backtest(features, labels, cfg)
    metrics = backtest_metrics(backtest)
    latest_pred = predict_one_step(features, labels, len(features) - 1, cfg, return_coef=True)
    latest_prediction = pd.DataFrame([{
        "ticker": ticker,
        "last_timestamp": features.index[-1],
        "p_change_raw": latest_pred.get("p_change_raw", np.nan),
        "p_change": latest_pred.get("p_change", np.nan),
        "p_continue": latest_pred.get("p_continue", np.nan),
        "q_same_given_change": latest_pred.get("q_same_given_change", np.nan),
        "q_opp_given_change": latest_pred.get("q_opp_given_change", np.nan),
        "p_same_unconditional": latest_pred.get("p_same_unconditional", np.nan),
        "p_opp_unconditional": latest_pred.get("p_opp_unconditional", np.nan),
        "expected_next_segment_theta": latest_pred.get("expected_next_segment_theta", np.nan),
        "expected_next_segment_amp": latest_pred.get("expected_next_segment_amp", np.nan),
        "expected_next_segment_length": latest_pred.get("expected_next_segment_length", np.nan),
        "expected_next_segment_net_return": latest_pred.get("expected_next_segment_net_return", np.nan),
        "n_completed_changes_train": latest_pred.get("n_completed_changes_train", np.nan),
        "base_rate_change_train": latest_pred.get("base_rate_change_train", np.nan),
        "base_rate_same_train": latest_pred.get("base_rate_same_train", np.nan),
        "platt_used_change": latest_pred.get("platt_used_change", np.nan),
        "platt_reason_change": latest_pred.get("platt_reason_change", np.nan),
        "platt_n_total_change": latest_pred.get("platt_n_total_change", np.nan),
        "platt_n_core_change": latest_pred.get("platt_n_core_change", np.nan),
        "platt_n_calibration_change": latest_pred.get("platt_n_calibration_change", np.nan),
    }])
    coef_change = latest_pred.get("coef_change", pd.DataFrame())
    coef_same = latest_pred.get("coef_same", pd.DataFrame())
    dashboard_html = make_dashboard_html(
        ticker=ticker,
        features=features,
        segments_expost=segments_expost,
        latest_pred=latest_pred,
        bt=backtest,
        metrics=metrics,
        cfg=cfg,
    )
    prob_diag_html = make_probability_diagnostics_html(
        ticker=ticker,
        bt=backtest,
        cfg=cfg,
    )
    exported = export_files(
        ticker=ticker,
        ohlc=ohlc,
        features=features,
        segments_expost=segments_expost,
        labels=labels,
        backtest=backtest,
        latest_prediction=latest_prediction,
        coef_change=coef_change,
        coef_same=coef_same,
        metrics=metrics,
        cfg=cfg,
    )
    return {
        "ticker": ticker,
        "metrics": metrics,
        "latest_prediction": latest_prediction,
        "coef_change": coef_change,
        "coef_same": coef_same,
        "dashboard_html": dashboard_html,
        "prob_diag_html": prob_diag_html,
        "exported": exported,
    }

# PRINT COEFFICIENTS

def print_coefficients(title: str, coef_df: pd.DataFrame, top_n: Optional[int] = None) -> None:
    print(title)
    if coef_df is None or coef_df.empty:
        print(" - modèle indisponible ou fallback sur base rate")
        return
    view = coef_df.copy()
    if top_n is not None:
        view = view.head(top_n)
    for _, row in view.iterrows():
        print(
            f" - {row['feature']:22s} "
            f"coef_std={row['coef_std']:+.4f} "
            f"odds_ratio_1sd={row['odds_ratio_1sd']:.4f}"
        )

# MAIN

def main() -> None:
    cfg = Config()
    ensure_dir(cfg.cache_dir)
    ensure_dir(cfg.output_dir)
    print("=" * 96)
    print("MOTEUR DE REGIMES - HAZARD SEQUENTIEL + PLATT SCALING")
    print("=" * 96)
    print(f"TICKERS                    : {TICKERS}")
    print(f"START                      : {cfg.start}")
    print(f"END                        : {cfg.end}")
    print(f"INTERVAL                   : {cfg.interval}")
    print(f"BACKTEST_WARMUP            : {cfg.backtest_warmup}")
    print(f"STATE_COLS                 : {STATE_COLS}")
    print(f"CHANGE_CLASS_WEIGHT        : {cfg.change_class_weight}")
    print(f"SAME_CLASS_WEIGHT          : {cfg.same_class_weight}")
    print(f"CHANGE_THRESHOLD           : {cfg.change_threshold}")
    print(f"PLATT_CALIBRATION_FRAC     : {cfg.platt_calibration_frac}")
    print(f"PLATT_MIN_CALIBRATION_SIZE : {cfg.platt_min_calibration_size}")
    print(f"PLATT_MIN_CORE_TRAIN_SIZE  : {cfg.platt_min_core_train_size}")
    print("=" * 96)
    master_preds = []
    master_metrics = []
    for ticker in TICKERS:
        print("-" * 96)
        try:
            result = run_for_ticker(ticker, cfg)
            master_preds.append(result["latest_prediction"])
            metric_row = {"ticker": ticker}
            metric_row.update(result["metrics"])
            master_metrics.append(metric_row)
            print("[HTML]")
            print(f" - dashboard               : {result['dashboard_html']}")
            print(f" - probability diagnostics : {result['prob_diag_html']}")
            print("[BACKTEST]")
            if result["metrics"].get("status") == "ok":
                m = result["metrics"]
                print(f" - n_predictions              : {m['n_predictions']}")
                print(f" - base_rate_change           : {m['base_rate_change']:.2%}")
                print(f" - predicted_rate_change      : {m['predicted_rate_change']:.2%}")
                print(f" - avg_p_change_raw           : {m['avg_p_change_raw']:.2%}")
                print(f" - avg_p_change               : {m['avg_p_change']:.2%}")
                print(f" - brier_change_raw           : {m['brier_change_raw']:.6f}")
                print(f" - brier_change               : {m['brier_change']:.6f}")
                print(f" - baseline_brier_change      : {m['baseline_brier_change']:.6f}")
                print(f" - accuracy_change            : {m['accuracy_change']:.2%}")
                print(f" - baseline_accuracy_change   : {m['baseline_accuracy_change']:.2%}")
                print(f" - precision_change           : {m['precision_change']:.2%}")
                print(f" - recall_change              : {m['recall_change']:.2%}")
                print(f" - roc_auc_change_raw         : {m['roc_auc_change_raw']:.4f}")
                print(f" - roc_auc_change             : {m['roc_auc_change']:.4f}")
                print(f" - avg_precision_change_raw   : {m['avg_precision_change_raw']:.4f}")
                print(f" - avg_precision_change       : {m['avg_precision_change']:.4f}")
                print(f" - n_real_changes             : {m['n_real_changes']}")
                print(f" - brier_same_given_change    : {m['brier_same_given_change']}")
                print(f" - accuracy_same_given_change : {m['accuracy_same_given_change']}")
            else:
                print(" - Backtest vide")
            print("[LIVE]")
            lp = result["latest_prediction"].iloc[0]
            print(f" - P(change) brut             : {lp['p_change_raw']:.2%}")
            print(f" - P(change) calibré          : {lp['p_change']:.2%}")
            print(f" - P(continue)                : {lp['p_continue']:.2%}")
            print(f" - q(same | change)           : {lp['q_same_given_change']:.2%}")
            print(f" - q(opp | change)            : {lp['q_opp_given_change']:.2%}")
            print(f" - P(same) unconditionnelle   : {lp['p_same_unconditional']:.2%}")
            print(f" - P(opp)  unconditionnelle   : {lp['p_opp_unconditional']:.2%}")
            print(f" - E[length next seg]         : {lp['expected_next_segment_length']}")
            print(f" - E[ret next seg]            : {lp['expected_next_segment_net_return']}")
            print(f" - completed changes in train : {lp['n_completed_changes_train']}")
            print(f" - platt used                 : {lp['platt_used_change']}")
            print(f" - platt reason               : {lp['platt_reason_change']}")
            print(f" - platt n_core / n_cal       : {lp['platt_n_core_change']} / {lp['platt_n_calibration_change']}")
            print_coefficients("[COEFFICIENTS - CHANGE HAZARD] (coefficients standardisés)", result["coef_change"])
            print_coefficients("[COEFFICIENTS - SAME | CHANGE] (coefficients standardisés)", result["coef_same"])
        except Exception as e:
            print(f"[ERREUR] {ticker}: {e}")
    if master_preds:
        master_pred_df = pd.concat(master_preds, ignore_index=True)
        master_pred_path = os.path.join(cfg.output_dir, f"master_latest_predictions_{cfg.interval}.csv")
        master_pred_df.to_csv(master_pred_path, index=False)
        print("\n[MASTER LATEST PREDICTIONS]")
        print(master_pred_df)
        print(master_pred_path)
    if master_metrics:
        master_metrics_df = pd.DataFrame(master_metrics)
        master_metrics_path = os.path.join(cfg.output_dir, f"master_backtest_metrics_{cfg.interval}.csv")
        master_metrics_df.to_csv(master_metrics_path, index=False)
        print("\n[MASTER BACKTEST METRICS]")
        print(master_metrics_df)
        print(master_metrics_path)
    print("\nTerminé.")

if __name__ == "__main__":
    main()