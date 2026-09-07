-- Build one analytical row per client from six months of payment history.
-- The target (default_flag) is joined only in the final SELECT.

DROP VIEW IF EXISTS feature_mart;

CREATE VIEW feature_mart AS
WITH ordered AS (
    SELECT
        *,
        LAG(delay_positive) OVER (
            PARTITION BY client_id
            ORDER BY month_no
        ) AS previous_delay
    FROM payment_history
),
behavior AS (
    SELECT
        client_id,
        MAX(CASE WHEN month_no = 9 THEN delay_positive END) AS latest_delay,
        AVG(
            CASE
                WHEN month_no < 9 THEN CASE WHEN delay_positive > 0 THEN 1.0 ELSE 0.0 END
            END
        ) AS prior_late_share,
        MAX(CASE WHEN month_no < 9 THEN delay_positive END) AS prior_max_delay,
        MAX(
            CASE
                WHEN month_no = 9 THEN delay_positive - previous_delay
            END
        ) AS delay_change,
        AVG(CASE WHEN bill_amount > 0 THEN bill_amount ELSE 0 END) AS mean_bill,
        AVG(CASE WHEN month_no >= 7 THEN bill_amount END)
            - AVG(CASE WHEN month_no <= 6 THEN bill_amount END) AS bill_change,
        AVG(payment_amount) AS mean_payment,
        AVG(CASE WHEN payment_amount = 0 THEN 1.0 ELSE 0.0 END) AS zero_payment_share
    FROM ordered
    GROUP BY client_id
)
SELECT
    c.client_id,
    LN(1.0 + c.credit_limit) AS log_limit,
    c.age,
    CASE WHEN c.education IN (1, 2) THEN 1 ELSE 0 END AS higher_education,
    CASE WHEN c.marriage = 1 THEN 1 ELSE 0 END AS married,
    b.latest_delay,
    b.prior_late_share,
    b.prior_max_delay,
    b.delay_change,
    b.mean_bill / c.credit_limit AS utilization,
    b.bill_change / c.credit_limit AS utilization_change,
    LN(1.0 + b.mean_payment) AS log_mean_payment,
    b.zero_payment_share,
    o.default_flag
FROM clients AS c
JOIN behavior AS b
    ON c.client_id = b.client_id
JOIN outcomes AS o
    ON c.client_id = o.client_id;
