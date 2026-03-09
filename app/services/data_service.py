"""Data loading, validation, and profiling utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataSummary:
    n_rows: int
    n_cols: int
    missing_by_column: dict[str, int]
    dtypes: dict[str, str]
    duplicated_rows: int
    numeric_outliers_iqr: dict[str, int]


def load_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def summarize_dataframe(df: pd.DataFrame) -> DataSummary:
    missing = df.isna().sum().to_dict()
    dtypes = {k: str(v) for k, v in df.dtypes.to_dict().items()}
    duplicated = int(df.duplicated().sum())

    outliers: dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outliers[col] = 0
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers[col] = int(((df[col] < lower) | (df[col] > upper)).sum())

    return DataSummary(
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        missing_by_column={k: int(v) for k, v in missing.items()},
        dtypes=dtypes,
        duplicated_rows=duplicated,
        numeric_outliers_iqr=outliers,
    )
