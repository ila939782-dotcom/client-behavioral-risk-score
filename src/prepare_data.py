"""Prepare the UCI credit-card dataset and build the analytical feature mart."""

from pathlib import Path
import sqlite3
import urllib.request
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SQL_FILE = ROOT / "sql" / "01_feature_mart.sql"

SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)
RAW_FILE = DATA_DIR / "uci_default.xls"
ZIP_FILE = DATA_DIR / "uci_default.zip"

PAYMENT_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


def ensure_directories() -> None:
    """Create folders used by the pipeline."""
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def download_dataset() -> None:
    """Download the UCI XLS file when it is not already available locally."""
    if RAW_FILE.exists():
        return

    print("Downloading UCI dataset...")
    urllib.request.urlretrieve(SOURCE_URL, ZIP_FILE)

    with zipfile.ZipFile(ZIP_FILE) as archive:
        xls_name = next(name for name in archive.namelist() if name.lower().endswith(".xls"))
        with archive.open(xls_name) as source, RAW_FILE.open("wb") as target:
            target.write(source.read())

    ZIP_FILE.unlink(missing_ok=True)
    print(f"Saved raw data to {RAW_FILE}")


def load_raw_data() -> pd.DataFrame:
    """Load the original UCI spreadsheet."""
    return pd.read_excel(RAW_FILE, header=1)


def build_client_tables(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create static client attributes and the target table."""
    clients = raw[["ID", "LIMIT_BAL", "AGE", "EDUCATION", "MARRIAGE"]].copy()
    clients.columns = ["client_id", "credit_limit", "age", "education", "marriage"]

    outcomes = raw[["ID", "default payment next month"]].copy()
    outcomes.columns = ["client_id", "default_flag"]

    return clients, outcomes


def build_payment_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert six monthly payment columns into a long client-month table."""
    months = []

    for i, status_column in enumerate(PAYMENT_STATUS_COLUMNS, start=1):
        month = raw[["ID", status_column, f"BILL_AMT{i}", f"PAY_AMT{i}"]].copy()
        month.columns = [
            "client_id",
            "payment_status",
            "bill_amount",
            "payment_amount",
        ]
        month["month_no"] = 10 - i  # September=9, ..., April=4
        month["delay_positive"] = month["payment_status"].clip(lower=0)
        months.append(month)

    return pd.concat(months, ignore_index=True).sort_values(["client_id", "month_no"])


def build_feature_mart(
    clients: pd.DataFrame,
    outcomes: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Write source tables to SQLite and build the SQL feature mart."""
    database_path = DATA_DIR / "risk.sqlite"

    with sqlite3.connect(database_path) as connection:
        connection.create_function("LN", 1, np.log)

        for table_name, frame in (
            ("clients", clients),
            ("outcomes", outcomes),
            ("payment_history", history),
        ):
            frame.to_sql(table_name, connection, if_exists="replace", index=False)

        connection.executescript(SQL_FILE.read_text())
        return pd.read_sql_query(
            "SELECT * FROM feature_mart ORDER BY client_id",
            connection,
        )


def save_outputs(raw: pd.DataFrame, mart: pd.DataFrame, history: pd.DataFrame) -> None:
    """Save prepared tables and grouping IDs used for train/validation/test splitting."""
    mart.to_csv(OUTPUT_DIR / "feature_mart.csv", index=False)
    history.to_csv(OUTPUT_DIR / "payment_history.csv", index=False)

    # Identical raw predictors stay in the same split. ID and target are excluded.
    groups = pd.util.hash_pandas_object(raw.iloc[:, 1:-1], index=False)
    pd.DataFrame(
        {"client_id": raw["ID"], "group_id": groups}
    ).to_csv(OUTPUT_DIR / "groups.csv", index=False)


def main() -> None:
    ensure_directories()
    download_dataset()

    raw = load_raw_data()
    clients, outcomes = build_client_tables(raw)
    history = build_payment_history(raw)
    mart = build_feature_mart(clients, outcomes, history)
    save_outputs(raw, mart, history)

    print(f"Prepared {len(mart):,} clients and {len(history):,} monthly rows.")


if __name__ == "__main__":
    main()
