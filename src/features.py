"""Carga, limpieza e ingeniería de variables por cliente (RFM extendido)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import NON_PRODUCT_REGEX, RAW_CSV

FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_ticket",
    "n_products",
    "avg_items_per_invoice",
    "tenure_days",
    "avg_days_between_purchases",
    "return_rate",
]


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    """Lee el CSV con tipos explícitos y nombres de columna normalizados."""
    df = pd.read_csv(
        path,
        dtype={"Invoice": str, "StockCode": str, "Description": str, "Country": str},
        parse_dates=["InvoiceDate"],
    )
    return df.rename(columns={"Customer ID": "CustomerID"})


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de problemas de calidad del dataset transaccional."""
    n = len(df)
    is_cancel = df["Invoice"].str.startswith("C")
    non_product = df["StockCode"].str.match(NON_PRODUCT_REGEX)
    rows = {
        "Filas totales": n,
        "Filas duplicadas": int(df.duplicated().sum()),
        "Sin CustomerID": int(df["CustomerID"].isna().sum()),
        "Sin Description": int(df["Description"].isna().sum()),
        "Cancelaciones (Invoice 'C…')": int(is_cancel.sum()),
        "Quantity < 0": int((df["Quantity"] < 0).sum()),
        "Quantity < 0 y NO cancelación": int(((df["Quantity"] < 0) & ~is_cancel).sum()),
        "Price == 0": int((df["Price"] == 0).sum()),
        "Price < 0": int((df["Price"] < 0).sum()),
        "StockCode no-producto": int(non_product.sum()),
    }
    out = pd.DataFrame({"filas": rows})
    out["%"] = (out["filas"] / n * 100).round(2)
    return out


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica la limpieza y devuelve (transacciones limpias, log de pasos)."""
    log = [("Inicial", len(df))]

    df = df.drop_duplicates()
    log.append(("- duplicados", len(df)))

    df = df.dropna(subset=["CustomerID"])
    log.append(("- sin CustomerID", len(df)))

    df = df[df["Price"] > 0]
    log.append(("- Price <= 0", len(df)))

    df = df[~df["StockCode"].str.match(NON_PRODUCT_REGEX)]
    log.append(("- StockCode no-producto", len(df)))

    df = df.assign(
        CustomerID=df["CustomerID"].astype(int),
        is_cancel=df["Invoice"].str.startswith("C"),
        amount=df["Quantity"] * df["Price"],
    )
    # Coherencia: ventas con Quantity > 0, cancelaciones con Quantity < 0
    df = df[(df["is_cancel"] & (df["Quantity"] < 0)) | (~df["is_cancel"] & (df["Quantity"] > 0))]
    log.append(("- signo Quantity incoherente", len(df)))

    log_df = pd.DataFrame(log, columns=["paso", "filas_restantes"])
    log_df["eliminadas"] = (-log_df["filas_restantes"].diff()).fillna(0).astype(int)
    return df.reset_index(drop=True), log_df


def build_customer_features(tx: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """Agrega transacciones limpias a una fila por cliente.

    Cada cliente se describe solo con sus propias transacciones, por lo que
    dividir después en train/test por cliente no produce fuga de información.
    """
    sales = tx[~tx["is_cancel"]]
    cancels = tx[tx["is_cancel"]]

    g = sales.groupby("CustomerID")
    feats = pd.DataFrame(
        {
            "first_purchase": g["InvoiceDate"].min(),
            "last_purchase": g["InvoiceDate"].max(),
            "frequency": g["Invoice"].nunique(),
            "gross_sales": g["amount"].sum(),
            "n_products": g["StockCode"].nunique(),
            "total_items": g["Quantity"].sum(),
            "country": g["Country"].agg(lambda s: s.mode().iat[0]),
        }
    )

    cg = cancels.groupby("CustomerID")
    feats["cancel_amount"] = cg["amount"].sum().reindex(feats.index, fill_value=0.0)
    feats["n_cancel_invoices"] = cg["Invoice"].nunique().reindex(feats.index, fill_value=0)

    feats["recency_days"] = (ref_date - feats["last_purchase"]).dt.days
    feats["tenure_days"] = (ref_date - feats["first_purchase"]).dt.days
    feats["monetary"] = feats["gross_sales"] + feats["cancel_amount"]  # gasto neto
    feats["avg_ticket"] = feats["gross_sales"] / feats["frequency"]
    feats["avg_items_per_invoice"] = feats["total_items"] / feats["frequency"]

    active_span = (feats["last_purchase"] - feats["first_purchase"]).dt.days
    between = active_span / (feats["frequency"] - 1).replace(0, np.nan)
    # Clientes de una sola compra: intervalo no observado -> se usa la antigüedad
    # como cota inferior (observación censurada).
    feats["avg_days_between_purchases"] = between.fillna(feats["tenure_days"])

    feats["return_rate"] = feats["n_cancel_invoices"] / (
        feats["frequency"] + feats["n_cancel_invoices"]
    )

    # Solo clientes con gasto neto positivo
    feats = feats[feats["monetary"] > 0]
    return feats


def log_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """log1p sobre columnas asimétricas (todas son >= 0)."""
    out = df.copy()
    out[cols] = np.log1p(out[cols])
    return out
