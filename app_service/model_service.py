import functools
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

FEATURE_COLUMNS = [
    "volatility_7",
    "ma_7",
    "ma_14",
    "hl_range",
    "volume_mcap_ratio",
]


@dataclass
class ModelArtifacts:
    model: object
    feature_columns: list


def _resolve_market_cap_column(df: pd.DataFrame) -> str:
    if "market_cap" in df.columns:
        return "market_cap"
    if "marketCap" in df.columns:
        return "marketCap"
    raise KeyError(
        "Neither 'market_cap' nor 'marketCap' found in dataset columns"
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features matching the training logic in crypto.py."""
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility_7"] = df["log_return"].rolling(7).std()
    df["volatility_14"] = df["log_return"].rolling(14).std()
    df["ma_7"] = df["close"].rolling(7).mean()
    df["ma_14"] = df["close"].rolling(14).mean()
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]

    market_col = _resolve_market_cap_column(df)
    df["volume_mcap_ratio"] = df["volume"] / df[market_col]

    return df


def prepare_dataset(dataset_path: str = "dataset.csv"):
    df = pd.read_csv(dataset_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = build_features(df)

    x = df[FEATURE_COLUMNS]
    y = df["volatility_14"]

    mask = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x = x.loc[mask]
    y = y.loc[mask]

    return x, y


@functools.lru_cache(maxsize=1)
def get_trained_artifacts(
    dataset_path: str = "dataset.csv",
) -> ModelArtifacts:
    """Train an XGBoost regressor and cache the result in-process."""
    from sklearn.model_selection import train_test_split

    x, y = prepare_dataset(dataset_path)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        missing=np.inf,
    )

    model.fit(x_train, y_train)
    return ModelArtifacts(model=model, feature_columns=FEATURE_COLUMNS)


def predict_from_raw_row(
    row: Dict[str, float],
    dataset_path: str = "dataset.csv",
    crypto_name: Optional[str] = None,
) -> float:
    """Predict volatility_14 from engineered features.

    For stable API usage, callers must provide the engineered features
    matching FEATURE_COLUMNS.

    crypto_name is accepted for future extension but not used by the model.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in row]
    if missing:
        raise ValueError(
            f"Missing engineered features: {missing}. "
            f"Provide {FEATURE_COLUMNS} in request body."
        )

    artifacts = get_trained_artifacts(dataset_path)
    x = np.array([[row[c] for c in artifacts.feature_columns]], dtype=float)
    pred = artifacts.model.predict(x)
    return float(pred[0])


def predict_from_engineered_features(
    engineered: Dict[str, float],
    dataset_path: str = "dataset.csv",
) -> float:
    return predict_from_raw_row(engineered, dataset_path=dataset_path)