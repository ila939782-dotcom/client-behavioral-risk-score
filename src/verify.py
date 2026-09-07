"""Run basic consistency checks after the analysis pipeline."""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


def main() -> None:
    scores = pd.read_csv(OUTPUT_DIR / "client_scores.csv")
    history = pd.read_csv(OUTPUT_DIR / "payment_history.csv")

    assert len(scores) == 30_000
    assert scores["client_id"].is_unique
    assert len(history) == 180_000
    assert history.groupby("client_id").size().eq(6).all()
    assert scores["risk_score"].between(0, 100).all()
    assert scores["default_flag"].sum() == 6_636
    assert scores.query("split == 'test'").shape[0] == 6_000
    assert not scores.isna().any().any()

    groups = pd.read_csv(OUTPUT_DIR / "groups.csv").merge(
        scores[["client_id", "split"]]
    )
    assert groups.groupby("group_id")["split"].nunique().max() == 1

    parameters = json.loads(
        (OUTPUT_DIR / "score_model.json").read_text(encoding="utf-8")
    )
    weights = np.array(list(parameters["weights"].values()))
    features = list(parameters["weights"])

    linear_score = (
        parameters["intercept"]
        + scores[features].to_numpy() @ weights
    )
    reconstructed_score = 100 / (1 + np.exp(-linear_score))

    assert np.allclose(
        reconstructed_score,
        scores["risk_score"],
        atol=1e-7,
    )

    print(
        "PASS: rows, IDs, history, target, splits, missing values "
        "and score formula."
    )


if __name__ == "__main__":
    main()
