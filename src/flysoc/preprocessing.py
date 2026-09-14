"""Sparse PN representation with explicit, label-free feature selection."""

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def clean_text(value: str) -> str:
    """Normalize volatile tokens without decoding or executing telemetry."""
    value = value.lower()[:16384]
    value = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", " ipaddr ", value)
    value = re.sub(r"\b[0-9a-f]{16,}\b", " hexvalue ", value)
    return re.sub(r"\d+", " num ", value)


def column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series("", index=frame.index, dtype="string")
    return frame[name].astype("string").fillna("")


class AlertPreprocessor:
    """Hash categoricals, learn TF-IDF on training only, and add eight numerics.

    The output has a fixed width, padding an undersized learned vocabulary with
    zeros. Missing fields are allowed, including an entirely empty alert.
    """

    def __init__(self, features: dict[str, Any]):
        self.features = dict(features)
        self.n_features = features["input_features"]
        self.hasher = FeatureHasher(n_features=features["categorical_hash_dim"],
                                    input_type="dict", alternate_sign=False, dtype=np.float32)
        self.vectorizer = TfidfVectorizer(max_features=features["text_max_features"],
                                          ngram_range=(1, 2), sublinear_tf=True,
                                          dtype=np.float32, preprocessor=clean_text,
                                          token_pattern=r"(?u)\b[a-z_][a-z_]+\b")
        self.fitted = False

    def _texts(self, frame: pd.DataFrame) -> list[str]:
        result = pd.Series("", index=frame.index, dtype="string")
        for name in self.features["text_fields"]:
            result = result + " " + column(frame, name).str.slice(0, 16384)
        return result.tolist()

    def fit(self, frame: pd.DataFrame) -> "AlertPreprocessor":
        if len(frame) == 0:
            raise ValueError("Cannot fit preprocessing on zero alerts")
        texts = self._texts(frame)
        analyzer = self.vectorizer.build_analyzer()
        self.has_text = any(analyzer(text) for text in texts)
        if self.has_text:
            self.vectorizer.fit(texts)
        self.fitted = True
        return self

    def _numeric(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        values = np.zeros((len(frame), 8), dtype=np.float32)
        for i, name in enumerate(("source_port", "destination_port")):
            ports = pd.to_numeric(column(frame, name), errors="coerce")
            valid = ports.between(0, 65535).fillna(False).to_numpy(dtype=bool)
            array = ports.fillna(0).to_numpy(dtype=float)
            values[:, i] = np.where(valid, np.log1p(np.clip(array, 0, 65535)) / np.log(65536), 0)
            values[:, i + 2] = valid
        times = pd.to_datetime(column(frame, "timestamp"), utc=True, errors="coerce")
        valid_time = times.notna().to_numpy()
        for offset, cycle, period in ((4, times.dt.hour, 24), (6, times.dt.dayofweek, 7)):
            angle = cycle.fillna(0).to_numpy() * (2 * np.pi / period)
            values[:, offset] = np.where(valid_time, (np.sin(angle) + 1) / 2, 0)
            values[:, offset + 1] = np.where(valid_time, (np.cos(angle) + 1) / 2, 0)
        return sparse.csr_matrix(values * self.features["numeric_weight"])

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        if len(frame) == 0:
            return sparse.csr_matrix((0, self.n_features), dtype=np.float32)
        fields = self.features["categorical_fields"] + self.features["identity_fields"]
        columns = {name: column(frame, name).str.lower().str.slice(0, 512).tolist() for name in fields}
        records = []
        for i in range(len(frame)):
            records.append({f"{name}={columns[name][i]}":
                            self.features["identity_weight"] if name in self.features["identity_fields"] else 1.0
                            for name in fields if columns[name][i]})
        categorical = normalize(self.hasher.transform(records), norm="l2") * self.features["categorical_weight"]
        width = self.features["text_max_features"]
        if self.has_text:
            text = self.vectorizer.transform(self._texts(frame))
            text = sparse.hstack([text, sparse.csr_matrix((len(frame), width - text.shape[1]))], format="csr")
        else:
            text = sparse.csr_matrix((len(frame), width), dtype=np.float32)
        result = sparse.hstack([categorical, text * self.features["text_weight"], self._numeric(frame)], format="csr", dtype=np.float32)
        result = normalize(result, norm="l2").tocsr()
        result.eliminate_zeros()
        return result

    def fit_transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        return self.fit(frame).transform(frame)
