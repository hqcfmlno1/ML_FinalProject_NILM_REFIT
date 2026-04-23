from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIG
# =========================================================

@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parents[2]

    processed_data_dir: Path = None
    checkpoint_dir: Path = None
    output_dir: Path = None
    plots_dir: Path = None

    n_appliances: int = 9
    window_size: int = 20
    train_ratio: float = 0.8
    use_scaler: bool = True

    # số điểm cho overview
    overview_points: int = 1200

    # độ dài vùng zoom để vẽ focused plot
    focus_window: int = 600

    def __post_init__(self):
        if self.processed_data_dir is None:
            self.processed_data_dir = self.project_root / "data" / "processed_data"
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.project_root / "checkpoints" / "basic models" / "linear_regression_window_shift"
        if self.output_dir is None:
            self.output_dir = self.project_root / "outputs" / "linear_regression_window_shift"
        if self.plots_dir is None:
            self.plots_dir = self.output_dir / "plots"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# DATA LOADING
# =========================================================

def load_house2_data(processed_data_dir: Path) -> pd.DataFrame:
    full_path = processed_data_dir / "House2_full.csv"
    if full_path.exists():
        print(f"[INFO] Loading full dataset: {full_path}")
        return pd.read_csv(full_path)

    part_files = sorted(processed_data_dir.glob("House2_part*.csv"))
    if not part_files:
        raise FileNotFoundError(
            f"Không tìm thấy House2_full.csv hoặc House2_part*.csv trong {processed_data_dir}"
        )

    print("[INFO] House2_full.csv không có. Sẽ ghép từ:")
    for f in part_files:
        print("   -", f.name)

    dfs = [pd.read_csv(f) for f in part_files]
    return pd.concat(dfs, axis=0, ignore_index=True)


def clean_and_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = {"Time", "Aggregate"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Aggregate"] = pd.to_numeric(df["Aggregate"], errors="coerce")
    df = df.dropna(subset=["Time", "Aggregate"]).reset_index(drop=True)
    df = df.sort_values("Time").reset_index(drop=True)

    return df


def detect_appliance_columns(df: pd.DataFrame, n_appliances: int = 9) -> List[str]:
    excluded = {"Time", "Unix", "Aggregate"}
    candidates = [c for c in df.columns if c not in excluded]

    numeric_cols = []
    for col in candidates:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 0:
            numeric_cols.append(col)

    if len(numeric_cols) < n_appliances:
        raise ValueError(
            f"Chỉ phát hiện được {len(numeric_cols)} cột numeric cho appliance, ít hơn {n_appliances}."
        )

    return numeric_cols[:n_appliances]


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["Time"].dt.hour
    out["minute"] = out["Time"].dt.minute
    out["dayofweek"] = out["Time"].dt.dayofweek
    out["month"] = out["Time"].dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    return out


def build_window_dataset(
    df: pd.DataFrame,
    appliance_cols: List[str],
    window_size: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    work_df = add_time_features(df)

    for lag in range(window_size):
        work_df[f"agg_lag_{lag}"] = work_df["Aggregate"].shift(lag)

    feature_cols = [f"agg_lag_{lag}" for lag in range(window_size)] + [
        "hour", "minute", "dayofweek", "month", "is_weekend"
    ]

    target_cols = appliance_cols

    work_df = work_df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)

    X = work_df[feature_cols].copy()
    y = work_df[target_cols].copy()
    time_current = work_df["Time"].copy()
    aggregate_current = work_df["Aggregate"].copy()

    return X, y, time_current, aggregate_current


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    time_current: pd.Series,
    aggregate_current: pd.Series,
    train_ratio: float = 0.8
):
    n = len(X)
    split_idx = int(n * train_ratio)

    X_train = X.iloc[:split_idx].reset_index(drop=True)
    X_test = X.iloc[split_idx:].reset_index(drop=True)

    y_train = y.iloc[:split_idx].reset_index(drop=True)
    y_test = y.iloc[split_idx:].reset_index(drop=True)

    time_train = time_current.iloc[:split_idx].reset_index(drop=True)
    time_test = time_current.iloc[split_idx:].reset_index(drop=True)

    agg_train = aggregate_current.iloc[:split_idx].reset_index(drop=True)
    agg_test = aggregate_current.iloc[split_idx:].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, time_train, time_test, agg_train, agg_test


# =========================================================
# METRICS
# =========================================================

def energy_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(y_pred)
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(np.minimum(y_true, y_pred)) / denom)


def energy_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(y_true)
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(np.minimum(y_true, y_pred)) / denom)


def energy_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = energy_precision(y_true, y_pred)
    r = energy_recall(y_true, y_pred)
    if p + r <= 1e-12:
        return 0.0
    return float(2 * p * r / (p + r))


def nep(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(y_true)
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "energy_precision": energy_precision(y_true, y_pred),
        "energy_recall": energy_recall(y_true, y_pred),
        "energy_f1": energy_f1(y_true, y_pred),
        "nep": nep(y_true, y_pred),
    }


# =========================================================
# MODEL TRAINING
# =========================================================

def train_individual_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame
) -> Dict[str, LinearRegression]:
    models = {}
    for appliance in y_train.columns:
        model = LinearRegression()
        model.fit(X_train, y_train[appliance])
        models[appliance] = model
    return models


def predict_individual_models(
    models: Dict[str, LinearRegression],
    X_test: pd.DataFrame
) -> pd.DataFrame:
    preds = {}
    for appliance, model in models.items():
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, None)
        preds[appliance] = y_pred
    return pd.DataFrame(preds)


def compute_others(aggregate_series: pd.Series, y_df: pd.DataFrame) -> np.ndarray:
    others = aggregate_series.to_numpy() - y_df.sum(axis=1).to_numpy()
    return np.clip(others, 0, None)


# =========================================================
# VISUALIZATION HELPERS
# =========================================================

def _downsample_for_overview(
    time_values: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_points: int
):
    n = len(y_true)
    if n <= max_points:
        return time_values, y_true, y_pred

    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return time_values.iloc[idx], y_true[idx], y_pred[idx]


def _find_focus_window(
    y_true: np.ndarray,
    window: int = 600
) -> Tuple[int, int]:
    """
    Tìm đoạn có nhiều tín hiệu nhất để plot cho meaningful hơn.
    Dùng rolling sum trên |y_true|.
    """
    n = len(y_true)
    if n <= window:
        return 0, n

    s = pd.Series(np.abs(y_true))
    scores = s.rolling(window=window, min_periods=1).sum()
    end_idx = int(scores.idxmax())
    start_idx = max(0, end_idx - window + 1)
    end_idx = min(n, start_idx + window)

    return start_idx, end_idx


def _nice_plot(
    time_values,
    y_true,
    y_pred,
    title: str,
    save_path: Path
):
    plt.figure(figsize=(14, 5))
    plt.plot(time_values, y_true, linewidth=2.0, label="Ground Truth")
    plt.plot(time_values, y_pred, linewidth=2.0, alpha=0.9, label="Prediction")
    plt.title(title, fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Power", fontsize=12)
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.25)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_overview_plot(
    time_values: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    appliance_name: str,
    save_path: Path,
    max_points: int = 1200
):
    t, yt, yp = _downsample_for_overview(time_values, y_true, y_pred, max_points)
    _nice_plot(
        t,
        yt,
        yp,
        title=f"{appliance_name} - Overview: Ground Truth vs Prediction",
        save_path=save_path
    )


def save_focused_plot(
    time_values: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    appliance_name: str,
    save_path: Path,
    focus_window: int = 600
):
    start, end = _find_focus_window(y_true, focus_window)
    t = time_values.iloc[start:end]
    yt = y_true[start:end]
    yp = y_pred[start:end]

    _nice_plot(
        t,
        yt,
        yp,
        title=f"{appliance_name} - Focused Active Window",
        save_path=save_path
    )


# =========================================================
# SAVE
# =========================================================

def save_metrics(metrics_df: pd.DataFrame, output_dir: Path):
    metrics_df.to_csv(output_dir / "metrics_per_appliance.csv", index=False)

    numeric_cols = [c for c in metrics_df.columns if c != "appliance"]
    summary_data = {col: [metrics_df[col].mean()] for col in numeric_cols}
    summary_df = pd.DataFrame(summary_data)
    summary_df.insert(0, "appliance", "OVERALL_MEAN")
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)


def save_predictions(
    time_test: pd.Series,
    y_test: pd.DataFrame,
    y_pred: pd.DataFrame,
    others_true: np.ndarray,
    others_pred: np.ndarray,
    output_dir: Path
):
    out = pd.DataFrame({"Time": time_test.astype(str)})

    for col in y_test.columns:
        out[f"{col}_true"] = y_test[col].to_numpy()
        out[f"{col}_pred"] = y_pred[col].to_numpy()

    out["others_true"] = others_true
    out["others_pred"] = others_pred

    out.to_csv(output_dir / "predictions.csv", index=False)


def save_bundle(
    config: Config,
    appliance_cols: List[str],
    feature_columns: List[str],
    scaler: StandardScaler | None,
    models: Dict[str, LinearRegression]
):
    bundle = {
        "config": config,
        "appliance_cols": appliance_cols,
        "feature_columns": feature_columns,
        "scaler": scaler,
        "models": models,
    }
    joblib.dump(bundle, config.checkpoint_dir / "linear_regression_window_shift_bundle.pkl")


# =========================================================
# MAIN
# =========================================================

def main():
    config = Config()

    print("========== STEP 1: LOAD DATA ==========")
    raw_df = load_house2_data(config.processed_data_dir)
    df = clean_and_prepare_dataframe(raw_df)
    print("Data shape after cleaning:", df.shape)

    appliance_cols = detect_appliance_columns(df, config.n_appliances)
    print("Detected appliance columns:")
    for i, col in enumerate(appliance_cols, start=1):
        print(f"  {i}. {col}")

    print("\n========== STEP 2: BUILD WINDOW DATASET ==========")
    X, y, time_current, aggregate_current = build_window_dataset(
        df=df,
        appliance_cols=appliance_cols,
        window_size=config.window_size
    )
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\n========== STEP 3: TRAIN / TEST SPLIT ==========")
    X_train, X_test, y_train, y_test, time_train, time_test, agg_train, agg_test = temporal_train_test_split(
        X, y, time_current, aggregate_current, train_ratio=config.train_ratio
    )
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    scaler = None
    if config.use_scaler:
        print("\n========== STEP 4: SCALE FEATURES ==========")
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
    else:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

    print("\n========== STEP 5: TRAIN 9 LINEAR REGRESSION MODELS ==========")
    models = train_individual_models(X_train_scaled, y_train)

    print("\n========== STEP 6: PREDICT ==========")
    y_pred = predict_individual_models(models, X_test_scaled)

    print("\n========== STEP 7: COMPUTE OTHERS ==========")
    others_true = compute_others(agg_test, y_test)
    others_pred = compute_others(agg_test, y_pred)

    print("\n========== STEP 8: EVALUATE ==========")
    rows = []
    for col in y_test.columns:
        row = {"appliance": col}
        row.update(calc_metrics(y_test[col].to_numpy(), y_pred[col].to_numpy()))
        rows.append(row)

    row_others = {"appliance": "others"}
    row_others.update(calc_metrics(others_true, others_pred))
    rows.append(row_others)

    metrics_df = pd.DataFrame(rows)
    print(metrics_df)
    save_metrics(metrics_df, config.output_dir)

    print("\n========== STEP 9: SAVE PREDICTIONS ==========")
    save_predictions(
        time_test=time_test,
        y_test=y_test,
        y_pred=y_pred,
        others_true=others_true,
        others_pred=others_pred,
        output_dir=config.output_dir
    )

    print("\n========== STEP 10: PLOT ==========")
    overview_dir = config.plots_dir / "overview"
    focused_dir = config.plots_dir / "focused"
    overview_dir.mkdir(parents=True, exist_ok=True)
    focused_dir.mkdir(parents=True, exist_ok=True)

    for col in y_test.columns:
        yt = y_test[col].to_numpy()
        yp = y_pred[col].to_numpy()

        save_overview_plot(
            time_values=time_test,
            y_true=yt,
            y_pred=yp,
            appliance_name=col,
            save_path=overview_dir / f"{col}_overview.png",
            max_points=config.overview_points
        )

        save_focused_plot(
            time_values=time_test,
            y_true=yt,
            y_pred=yp,
            appliance_name=col,
            save_path=focused_dir / f"{col}_focused.png",
            focus_window=config.focus_window
        )

    save_overview_plot(
        time_values=time_test,
        y_true=others_true,
        y_pred=others_pred,
        appliance_name="others",
        save_path=overview_dir / "others_overview.png",
        max_points=config.overview_points
    )

    save_focused_plot(
        time_values=time_test,
        y_true=others_true,
        y_pred=others_pred,
        appliance_name="others",
        save_path=focused_dir / "others_focused.png",
        focus_window=config.focus_window
    )

    print("\n========== STEP 11: SAVE MODEL BUNDLE ==========")
    save_bundle(
        config=config,
        appliance_cols=appliance_cols,
        feature_columns=list(X.columns),
        scaler=scaler,
        models=models
    )

    print("\nDONE.")
    print("Checkpoint dir :", config.checkpoint_dir)
    print("Output dir     :", config.output_dir)
    print("Plots dir      :", config.plots_dir)


if __name__ == "__main__":
    main()