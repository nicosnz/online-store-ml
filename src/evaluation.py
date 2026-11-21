"""Métricas internas, búsqueda de hiperparámetros y utilidades de reporte."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

from .config import (FIGURES_DIR, GMM_COV_TYPES, K_MIN_SELECTION, K_RANGE,
                     MIN_CLUSTER_FRAC, SEED)


def internal_metrics(X, labels) -> dict:
    """Silhouette (↑), Davies-Bouldin (↓) y Calinski-Harabasz (↑)."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return {
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
    }


def _min_frac(labels) -> float:
    return np.bincount(labels).min() / len(labels)


def make_kmeans(k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=SEED)


def make_gmm(k: int, cov: str) -> GaussianMixture:
    return GaussianMixture(n_components=k, covariance_type=cov, n_init=3, random_state=SEED)


def search_kmeans(X, k_range=K_RANGE) -> pd.DataFrame:
    rows = []
    for k in k_range:
        model = make_kmeans(k).fit(X)
        lab = model.labels_
        rows.append({"k": k, "inertia": model.inertia_, "min_cluster_frac": _min_frac(lab),
                     **internal_metrics(X, lab)})
    return pd.DataFrame(rows)


def search_gmm(X, k_range=K_RANGE, cov_types=GMM_COV_TYPES) -> pd.DataFrame:
    rows = []
    for cov in cov_types:
        for k in k_range:
            model = make_gmm(k, cov).fit(X)
            lab = model.predict(X)
            rows.append({"k": k, "covariance_type": cov, "bic": model.bic(X), "aic": model.aic(X),
                         "n_used_clusters": len(np.unique(lab)),
                         "min_cluster_frac": _min_frac(lab) if len(np.unique(lab)) == k else 0.0,
                         **internal_metrics(X, lab)})
    return pd.DataFrame(rows)


def select_best(results: pd.DataFrame, min_frac: float = MIN_CLUSTER_FRAC,
                k_min: int = K_MIN_SELECTION,
                extra_rank_cols: dict | None = None) -> pd.Series:
    """Regla explícita de selección (solo métricas internas sobre train).

    0. Solo se consideran k >= `k_min`.
    1. Se descartan soluciones con algún cluster < `min_frac` del train
       (segmentos inservibles en la práctica).
    2. Se rankea cada métrica (Silhouette ↑, DB ↓, CH ↑ y, opcionalmente,
       BIC ↓) y se elige el menor rango promedio.
    """
    cand = results[(results["k"] >= k_min) & (results["min_cluster_frac"] >= min_frac)].copy()
    if cand.empty:
        cand = results.copy()
    ranks = {
        "silhouette": cand["silhouette"].rank(ascending=False),
        "davies_bouldin": cand["davies_bouldin"].rank(ascending=True),
        "calinski_harabasz": cand["calinski_harabasz"].rank(ascending=False),
    }
    for col, asc in (extra_rank_cols or {}).items():
        ranks[col] = cand[col].rank(ascending=asc)
    cand["mean_rank"] = pd.DataFrame(ranks).mean(axis=1)
    return cand.sort_values("mean_rank").iloc[0]


def evaluate_model(model, X_train, X_test, eval_spaces: dict | None = None) -> pd.DataFrame:
    """Métricas en train (labels de ajuste) y test (predict).

    `eval_spaces` permite calcular además las métricas de las mismas etiquetas
    en otro espacio, p. ej. {"original": (Xtr_orig, Xte_orig)}.
    """
    lab_tr = model.predict(X_train)
    lab_te = model.predict(X_test)
    spaces = {"propio": (X_train, X_test), **(eval_spaces or {})}
    rows = []
    for space, (A, B) in spaces.items():
        rows.append({"espacio_métrica": space, "conjunto": "train", **internal_metrics(A, lab_tr)})
        rows.append({"espacio_métrica": space, "conjunto": "test", **internal_metrics(B, lab_te)})
    return pd.DataFrame(rows)


def relabel_by_score(labels, score) -> np.ndarray:
    """Renumera clusters por la media de `score` (descendente): 0 = mayor score.

    Las métricas son invariantes a permutaciones de etiquetas; esto solo hace
    que el mismo segmento tenga el mismo número/color en todas las figuras.
    """
    labels = np.asarray(labels)
    order = pd.Series(score).groupby(labels).mean().sort_values(ascending=False).index
    mapping = {old: new for new, old in enumerate(order)}
    return np.vectorize(mapping.get)(labels)


def profile_clusters(features: pd.DataFrame, labels, cols) -> pd.DataFrame:
    """Mediana de cada variable original por cluster + tamaño."""
    prof = features[cols].groupby(np.asarray(labels)).median()
    prof.insert(0, "n_clientes", pd.Series(labels).value_counts().sort_index().values)
    prof.insert(1, "%", (prof["n_clientes"] / prof["n_clientes"].sum() * 100).round(1))
    prof.index.name = "cluster"
    return prof


def save_fig(fig, name: str) -> None:
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=120, bbox_inches="tight")
