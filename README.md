# Client Behavioral Risk Score

Educational laboratory project that builds a client-level behavioral credit-risk score from the **UCI Default of Credit Card Clients** dataset.

The project combines **Python/Pandas**, **SQL/SQLite**, **scikit-learn**, **statsmodels**, and **Power BI/DAX** in one reproducible analytical pipeline.

## Project goal

The main question is whether recent payment behavior adds useful information for estimating next-month default risk compared with basic client profile variables alone.

The pipeline:

1. loads the original UCI spreadsheet;
2. restructures six months of payment history;
3. stores analytical tables in SQLite;
4. builds behavioral features in SQL;
5. compares baseline and behavioral risk models;
6. converts predicted default probability into a `risk_score` from 0 to 100;
7. groups clients into risk bands A-E;
8. evaluates a limited manual-review scenario;
9. exports tables for further analysis in Excel and Power BI.

## Tech stack

- Python
- Pandas / NumPy
- SQLite / SQL
- scikit-learn
- statsmodels
- Matplotlib
- Power BI / DAX

## Repository structure

```text
client-behavioral-risk-score/
├── src/
│   ├── prepare_data.py       # download and prepare source data
│   ├── analyze.py            # train models, score clients, export diagnostics
│   └── verify.py             # consistency and reproducibility checks
├── sql/
│   ├── 01_feature_mart.sql   # behavioral feature engineering
│   └── 02_portfolio_queries.sql
├── powerbi/
│   └── measures.dax          # measures for the Power BI report
├── data/                     # local raw data / SQLite database (not committed)
├── outputs/                  # generated analytical outputs (not committed)
├── figures/                  # generated figures (not committed)
├── PROJECT_EXPLANATION_RU.md # detailed explanation in Russian
├── requirements.txt
├── .gitignore
└── README.md
```

## Data and features

The source dataset contains **30,000 clients** and six monthly observations of payment status, bill amounts, and payments.

The SQL feature mart creates one analytical row per client. Behavioral features include:

- latest payment delay;
- share of previous months with a delay;
- maximum previous delay;
- change in delay;
- credit-limit utilization;
- change in utilization;
- average payment amount;
- share of zero-payment months.

Basic profile variables include credit limit, age, education, and marital status.

## Models

Three specifications are compared:

- **Profile** — logistic regression using only profile variables;
- **Full** — logistic regression using profile + behavioral features;
- **Boosting** — `HistGradientBoostingClassifier` using the full feature set.

Validation and test metrics include:

- ROC-AUC;
- Brier Score;
- Log Loss;
- Precision;
- Recall.

The **Full** logistic model is used for the final score:

```text
risk_score = probability_of_default × 100
```

`statsmodels.Logit` is also fitted with the same logistic specification to export coefficient-level statistical diagnostics such as p-values, confidence intervals, and odds ratios.

## Train / validation / test split

The data is split approximately into:

- 60% train;
- 20% validation;
- 20% test.

Identical raw predictor rows are grouped so that duplicates do not appear in different splits.

## Manual-review scenario

The test sample is ranked by predicted default probability. The project evaluates how many defaults can be captured when only the top:

- 5%;
- 10%;
- 20%

of highest-risk clients can be reviewed manually. The exported metrics include review precision, capture/recall, and lift.

## How to run

Python 3.11+ is recommended.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/prepare_data.py
python src/analyze.py
python src/verify.py
```

`prepare_data.py` downloads the original UCI dataset automatically if it is not already present in `data/`.

## Generated outputs

After a successful run, `outputs/` contains files such as:

- `feature_mart.csv`;
- `payment_history.csv`;
- `metrics.csv`;
- `client_scores.csv`;
- `coefficients.csv`;
- `risk_bands.csv`;
- `review_capacity.csv`;
- `score_model.json`;
- `summary.json`.

`figures/model_quality.png` contains a compact model-quality visualization.

## Power BI

`powerbi/measures.dax` contains measures for portfolio size, defaults, default rate, mean PD, mean risk score, model-quality metrics, and the limited review-capacity scenario.

## Data source

Yeh, I. (2009). **Default of Credit Card Clients**. UCI Machine Learning Repository.

- Dataset ID: 350
- DOI: `10.24432/C55S3H`
- Data license: CC BY 4.0

## Notes

This is an **educational analytical project**, not a production credit-scoring system. The resulting score should not be used for real lending decisions without independent validation, governance, fairness analysis, and applicable regulatory review.
