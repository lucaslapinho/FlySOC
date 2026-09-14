"""File-only ingestion. Telemetry values are never executable instructions."""

from pathlib import Path

import pandas as pd


def read_alerts(path: str | Path) -> pd.DataFrame:
    path = Path(path).resolve(strict=True)
    if path.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError("Only CSV and TSV alert exports are supported")
    frame = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",",
                        low_memory=False)
    if frame.empty:
        raise ValueError("Alert file is empty")
    if "alert_id" not in frame:
        frame["alert_id"] = [f"ROW{i:07d}" for i in range(len(frame))]
    if frame.alert_id.isna().any() or frame.alert_id.astype(str).duplicated().any():
        raise ValueError("alert_id must be non-null and unique")
    frame["alert_id"] = frame.alert_id.astype(str)
    if "verdict" in frame:
        frame["verdict"] = frame.verdict.astype("string").str.strip().str.upper()
    return frame


def chronological_split(frame: pd.DataFrame, train_ratio: float = .7,
                        validation_ratio: float = .15) -> dict[str, pd.DataFrame]:
    """Split by time without dividing a group of identical timestamps.

    Ratios are approximate if boundary timestamps repeat. Invalid/missing times
    fail explicitly: inventing chronology would invalidate the experiment.
    """
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Invalid temporal split ratios")
    if "timestamp" not in frame or len(frame) < 10:
        raise ValueError("Chronological evaluation needs timestamp and at least 10 rows")
    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered.timestamp, utc=True, errors="coerce")
    if ordered.timestamp.isna().any():
        raise ValueError("Chronological evaluation requires valid timestamps for every row")
    ordered = ordered.sort_values("timestamp", kind="stable").reset_index(drop=True)
    times = ordered.timestamp
    first = int(times.searchsorted(times.iloc[int(len(frame) * train_ratio)], side="left"))
    second = int(times.searchsorted(times.iloc[int(len(frame) * (train_ratio + validation_ratio))], side="left"))
    splits = {"train": ordered.iloc[:first].copy(),
              "validation": ordered.iloc[first:second].copy(),
              "test": ordered.iloc[second:].copy()}
    if any(part.empty for part in splits.values()):
        raise ValueError("Timestamp ties leave an empty split; use more distinct timestamps")
    return {name: part.reset_index(drop=True) for name, part in splits.items()}
