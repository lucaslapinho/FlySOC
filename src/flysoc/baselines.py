"""Conventional feature-space and downstream fingerprint estimators."""

from typing import Any

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def classifiers(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["classification"]
    return {
        "logistic_regression": LogisticRegression(max_iter=settings["logistic_max_iter"], random_state=config["seed"]),
        "random_forest": RandomForestClassifier(n_estimators=settings["forest_trees"],
                                                random_state=config["seed"], n_jobs=1),
        "knn": KNeighborsClassifier(n_neighbors=settings["knn_neighbors"], metric="cosine",
                                     algorithm="brute", weights="distance", n_jobs=1),
    }


def isolation_forest(config: dict[str, Any]) -> IsolationForest:
    return IsolationForest(n_estimators=config["classification"]["forest_trees"], max_samples=256,
                           random_state=config["seed"], n_jobs=1)
