"""Configuración central del proyecto (semilla, rutas, constantes)."""
from pathlib import Path

SEED = 42
TEST_SIZE = 0.20

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "dataset" / "online_retail_II.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
CUSTOMERS_PARQUET = PROCESSED_DIR / "customers.parquet"
FIGURES_DIR = ROOT / "reports" / "figures"

# StockCodes que NO son productos (portes, comisiones, ajustes, vales, pruebas).
# Los DCGS0xxx sí son productos reales y se conservan.
NON_PRODUCT_REGEX = (
    r"^(POST|DOT|M|m|C2|C3|D|S|B|BANK CHARGES|ADJUST\d*|AMAZONFEE|CRUK|PADS|"
    r"TEST\d+|GIFT|gift_.*|DCGS[SL](BOY|GIRL))$"
)

# Rango de búsqueda de hiperparámetros
K_RANGE = range(2, 11)
# k=2 suele ganar en métricas internas pero solo separa "alto valor / bajo valor";
# para que la segmentación sea útil se elige entre k >= 3 (se reporta k=2 igualmente).
K_MIN_SELECTION = 3
GMM_COV_TYPES = ("full", "diag", "tied", "spherical")
MIN_CLUSTER_FRAC = 0.03   # descarta soluciones con clusters < 3 % del train
PCA_VAR_THRESHOLD = 0.90  # varianza explicada acumulada mínima

for _d in (PROCESSED_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
