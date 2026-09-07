-- Run these queries against data/risk.sqlite after prepare_data.py.

-- Portfolio size and default rate.
SELECT
    COUNT(*) AS clients,
    SUM(default_flag) AS defaults,
    AVG(default_flag * 1.0) AS default_rate
FROM outcomes;

-- Share of clients with a positive delay by month.
SELECT
    month_no,
    COUNT(*) AS rows,
    AVG(CASE WHEN delay_positive > 0 THEN 1.0 ELSE 0.0 END) AS late_share
FROM payment_history
GROUP BY month_no
ORDER BY month_no;

-- Default rate by the latest observed delay.
-- LN is registered in Python for SQLite builds without math functions.
SELECT
    latest_delay,
    COUNT(*) AS clients,
    AVG(default_flag * 1.0) AS default_rate
FROM feature_mart
GROUP BY latest_delay
ORDER BY latest_delay;
