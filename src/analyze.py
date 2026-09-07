"""Train risk models, calculate client scores, and export model diagnostics."""

from pathlib import Path
import json

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = ROOT / "figures"

PROFILE_FEATURES = ["log_limit", "age", "higher_education", "married"]
BEHAVIOR_FEATURES = [
    "latest_delay",
    "prior_late_share",
    "prior_max_delay",
    "delay_change",
    "utilization",
    "utilization_change",
    "log_mean_payment",
    "zero_payment_share",
]
ALL_FEATURES = PROFILE_FEATURES + BEHAVIOR_FEATURES


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the feature mart and group IDs created by prepare_data.py."""
    df = pd.read_csv(OUTPUT_DIR / "feature_mart.csv")
    groups = pd.read_csv(OUTPUT_DIR / "groups.csv")["group_id"]
    return df, groups


def make_splits(
    df: pd.DataFrame,
    groups: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create approximately 60/20/20 train, validation, and test splits."""
    y = df["default_flag"]

    train_validation, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42).split(
            df, y, groups
        )
    )

    train_relative, validation_relative = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43).split(
            df.iloc[train_validation],
            y.iloc[train_validation],
            groups.iloc[train_validation],
        )
    )

    train = train_validation[train_relative]
    validation = train_validation[validation_relative]
    return train, validation, test


def build_models() -> dict[str, object]:
    """Create two logistic-regression specifications and one boosting model."""
    logistic = lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000, tol=1e-9),
    )

    return {
        "Profile": logistic(),
        "Full": logistic(),
        "Boosting": HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=15,
            learning_rate=0.05,
            l2_regularization=1,
            early_stopping=False,
            random_state=42,
        ),
    }


def train_and_evaluate(
    df: pd.DataFrame,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Train the three models and calculate validation/test metrics."""
    y = df["default_flag"]
    models = build_models()
    model_features = {
        "Profile": PROFILE_FEATURES,
        "Full": ALL_FEATURES,
        "Boosting": ALL_FEATURES,
    }

    metrics = []

    for model_name, model in models.items():
        columns = model_features[model_name]
        model.fit(df.loc[train, columns], y.iloc[train])

        for split_name, rows in (("validation", validation), ("test", test)):
            probabilities = model.predict_proba(df.loc[rows, columns])[:, 1]

            metrics.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "n": len(rows),
                    "roc_auc": roc_auc_score(y.iloc[rows], probabilities),
                    "brier": brier_score_loss(y.iloc[rows], probabilities),
                    "log_loss": log_loss(y.iloc[rows], probabilities),
                    "precision": precision_score(
                        y.iloc[rows], probabilities >= 0.5, zero_division=0
                    ),
                    "recall": recall_score(
                        y.iloc[rows], probabilities >= 0.5, zero_division=0
                    ),
                }
            )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    return models, metrics_df


def score_clients(
    df: pd.DataFrame,
    full_model: object,
    train: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Convert Full-model default probabilities into a 0-100 risk score and bands A-E."""
    df = df.copy()
    df["pd"] = full_model.predict_proba(df[ALL_FEATURES])[:, 1]
    df["risk_score"] = 100 * df["pd"]

    band_cuts = np.quantile(df.loc[train, "pd"], [0.2, 0.4, 0.6, 0.8])
    df["risk_band"] = np.array(list("ABCDE"))[
        np.searchsorted(band_cuts, df["pd"], side="left")
    ]

    df.to_csv(OUTPUT_DIR / "client_scores.csv", index=False)
    return df, band_cuts


def estimate_logit_inference(
    df: pd.DataFrame,
    full_model: object,
    train: np.ndarray,
    band_cuts: np.ndarray,
) -> tuple[pd.DataFrame, sm.Logit, pd.DataFrame]:
    """Estimate standard statsmodels inference for the same logistic specification."""
    y = df["default_flag"]
    scaler = full_model.named_steps["standardscaler"]

    x = pd.DataFrame(
        scaler.transform(df[ALL_FEATURES]),
        columns=ALL_FEATURES,
    )

    fit = sm.Logit(y.iloc[train], sm.add_constant(x.iloc[train])).fit(
        disp=False,
        maxiter=100,
    )

    confidence_intervals = fit.conf_int()
    coefficients = pd.DataFrame(
        {
            "feature": ALL_FEATURES,
            "coef": fit.params.iloc[1:].values / scaler.scale_,
            "std_error": fit.bse.iloc[1:].values / scaler.scale_,
            "p_value": fit.pvalues.iloc[1:].values,
            "ci_low": confidence_intervals.iloc[1:, 0].values / scaler.scale_,
            "ci_high": confidence_intervals.iloc[1:, 1].values / scaler.scale_,
        }
    )

    coefficients["step"] = [1, 10, 1, 1, 1, 0.2, 1, 1, 0.1, 0.1, 1, 1 / 6]

    for source, destination in (
        ("coef", "odds_ratio"),
        ("ci_low", "or_low"),
        ("ci_high", "or_high"),
    ):
        coefficients[destination] = np.exp(
            coefficients[source] * coefficients["step"]
        )

    coefficients.to_csv(OUTPUT_DIR / "coefficients.csv", index=False)

    logistic_model = full_model.named_steps["logisticregression"]
    weights = logistic_model.coef_[0] / scaler.scale_
    intercept = float(logistic_model.intercept_[0] - weights @ scaler.mean_)

    parameters = {
        "intercept": intercept,
        "weights": dict(zip(ALL_FEATURES, weights)),
        "band_cuts_pd": band_cuts.tolist(),
    }

    (OUTPUT_DIR / "score_model.json").write_text(
        json.dumps(parameters, indent=2),
        encoding="utf-8",
    )

    return coefficients, fit, x


def evaluate_review_capacity(
    df: pd.DataFrame,
    validation: np.ndarray,
    test: np.ndarray,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Evaluate how many defaults are captured when only top-risk clients are reviewed."""
    threshold = float(df.loc[validation, "pd"].quantile(0.90))
    test_ranked = df.loc[test].sort_values(
        ["pd", "client_id"],
        ascending=[False, True],
    )

    queue = []
    for fraction in (0.05, 0.10, 0.20):
        selected = test_ranked.head(int(np.ceil(fraction * len(test_ranked))))
        precision = selected["default_flag"].mean()

        queue.append(
            {
                "capacity": fraction,
                "reviewed": len(selected),
                "defaults": int(selected["default_flag"].sum()),
                "precision": precision,
                "recall": selected["default_flag"].sum()
                / test_ranked["default_flag"].sum(),
                "lift": precision / test_ranked["default_flag"].mean(),
            }
        )

    queue_df = pd.DataFrame(queue)
    queue_df.to_csv(OUTPUT_DIR / "review_capacity.csv", index=False)
    return threshold, test_ranked, queue_df


def build_risk_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observed and predicted default rates by split and risk band."""
    bands = (
        df.groupby(["split", "risk_band"])
        .agg(
            clients=("client_id", "size"),
            defaults=("default_flag", "sum"),
            default_rate=("default_flag", "mean"),
            mean_pd=("pd", "mean"),
        )
        .reset_index()
    )
    bands.to_csv(OUTPUT_DIR / "risk_bands.csv", index=False)
    return bands


def save_summary(
    df: pd.DataFrame,
    threshold: float,
    test_ranked: pd.DataFrame,
    queue_df: pd.DataFrame,
    coefficients: pd.DataFrame,
    fit: object,
    x: pd.DataFrame,
) -> dict:
    """Save a compact JSON summary of the main analytical results."""
    summary = {
        "split": df.groupby("split")["default_flag"]
        .agg(["size", "sum", "mean"])
        .to_dict("index"),
        "validation_threshold": threshold,
        "test_reviewed_at_threshold": int((test_ranked["pd"] >= threshold).sum()),
        "test_capture_at_threshold": float(
            test_ranked.loc[
                test_ranked["pd"] >= threshold,
                "default_flag",
            ].sum()
            / test_ranked["default_flag"].sum()
        ),
        "sklearn_statsmodels_max_pd_difference": float(
            np.max(np.abs(fit.predict(sm.add_constant(x)) - df["pd"]))
        ),
        "h2": coefficients.set_index("feature").loc["prior_late_share"].to_dict(),
        "queue": queue_df.to_dict("records"),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def save_model_quality_figure(metrics: pd.DataFrame, bands: pd.DataFrame) -> None:
    """Create the model-quality chart used in the project output."""
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    test_metrics = metrics.query("split == 'test'")
    axes[0].bar(test_metrics["model"], test_metrics["roc_auc"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("ROC-AUC | test")

    for index, value in enumerate(test_metrics["roc_auc"]):
        axes[0].text(index, value + 0.02, f"{value:.4f}", ha="center")

    test_bands = bands.query("split == 'test'")
    axes[1].plot(
        test_bands["risk_band"],
        test_bands["default_rate"],
        "o-",
        label="Observed",
    )
    axes[1].plot(
        test_bands["risk_band"],
        test_bands["mean_pd"],
        "o--",
        label="Mean PD",
    )
    axes[1].set_title("Risk bands | test")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "model_quality.png", dpi=180)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    df, groups = load_data()
    train, validation, test = make_splits(df, groups)

    df["split"] = "train"
    df.loc[validation, "split"] = "validation"
    df.loc[test, "split"] = "test"

    models, metrics = train_and_evaluate(df, train, validation, test)
    df, band_cuts = score_clients(df, models["Full"], train)
    coefficients, fit, x = estimate_logit_inference(
        df,
        models["Full"],
        train,
        band_cuts,
    )
    threshold, test_ranked, queue_df = evaluate_review_capacity(
        df,
        validation,
        test,
    )
    bands = build_risk_bands(df)
    summary = save_summary(
        df,
        threshold,
        test_ranked,
        queue_df,
        coefficients,
        fit,
        x,
    )
    save_model_quality_figure(metrics, bands)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
