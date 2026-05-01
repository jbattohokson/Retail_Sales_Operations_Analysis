-- 02_acs_setup.sql
--
-- Apple Case Study — Step 2: Database Setup, Filtering, and Aggregation
--
-- This is the second file in a three-step pipeline:
--
--     01_acs_etl.py      Reshapes raw Excel into apple_sales_clean.csv
--     02_acs_setup.sql   (this file) Loads CSV into DB, applies quality filters,
--                        builds production views and Tableau-ready aggregates
--     03_acs_analysis.py Queries the DB views, exports CSVs, runs regression models
--
-- What this file does:
--     Receives apple_sales_clean.csv from 01_acs_etl.py, loads it into a
--     staging table, applies three data quality filters (no null states, no
--     cities filed as states, only the 15 approved sub-categories), and creates
--     views that 03_acs_analysis.py queries to produce Tableau exports and
--     regression inputs.
--
--     Filtering in SQL rather than Python keeps exclusion logic readable by any
--     analyst without Python knowledge, easy to adjust without re-running the ETL,
--     and directly verifiable in any DB client (DuckDB CLI, DB Browser, etc.).
--
-- How this file is executed:
--     03_acs_analysis.py reads this file from disk and executes it at runtime
--     after loading the CSV. Single entry point — no separate terminal commands.
--
--     Manual execution for inspection:
--     DuckDB  :  duckdb apple_sales.db < 02_acs_setup.sql
--     SQLite  :  sqlite3 apple_sales.db < 02_acs_setup.sql
--
-- Compatibility:
--     All statements are written for compatibility with both DuckDB and SQLite.
--     DuckDB is preferred (faster, better window function support). The Python
--     loader falls back to sqlite3 automatically if duckdb is not installed.


-- ==========================================================================
-- Section 1: Staging Table
-- ==========================================================================
--
-- Receives the output of 01_acs_etl.py. The schema mirrors exactly what the
-- Python ETL writes to apple_sales_clean.csv. If you add or rename a column
-- in the ETL, update this schema to match.
--
-- The actual INSERT is performed by 03_acs_analysis.py using pandas to_sql(),
-- which is why there is no COPY or INSERT statement here.

DROP TABLE IF EXISTS stg_sales_raw;

CREATE TABLE stg_sales_raw (
    customer_gender   TEXT,
    customer_age      INTEGER,
    product_category  TEXT,
    sub_category      TEXT,
    state             TEXT,
    country           TEXT,
    week_start_date   DATE,
    cost              NUMERIC(12, 2),
    quantity          NUMERIC(12, 2),
    revenue           NUMERIC(12, 2),
    unit_cost         NUMERIC(12, 2),
    unit_price        NUMERIC(12, 2),
    sale_year         INTEGER,
    sale_quarter      INTEGER,
    sale_month        INTEGER,
    month_name        TEXT,
    year_month        TEXT,
    week_num          INTEGER,
    gross_profit      NUMERIC(12, 2),
    margin_pct        NUMERIC(8, 4),
    age_band          TEXT,
    geo_flag          TEXT,
    state_flag        TEXT
);


-- ==========================================================================
-- Section 2: Approved Sub-Category Reference Table
-- ==========================================================================
--
-- The business scope covers 15 of the 17 sub-categories in the raw data.
-- Touring Bikes and Bike Stands are excluded per the case study brief.
--
-- Keeping the approved list in its own table (rather than hardcoding it in
-- every WHERE clause) means scope changes require updating one place only.
-- It also makes the exclusion transparent and auditable.
--
-- CHANGE: moved allowed_sub_categories BEFORE vw_dq_audit (swapped the
-- original Section 3 and Section 2 order).
-- ORIGINAL: vw_dq_audit was defined in Section 2 and the allowed list in
-- Section 3, but the view references the table — SQLite requires the table
-- to exist before the view that references it.
-- WHY: in DuckDB this order does not matter, but in SQLite defining the view
-- before the table it references in a subquery causes a runtime error. The
-- correct order is table first, view second.

DROP TABLE IF EXISTS allowed_sub_categories;

CREATE TABLE allowed_sub_categories (sub_category TEXT PRIMARY KEY);

INSERT INTO allowed_sub_categories (sub_category) VALUES
    ('Tires and Tubes'),
    ('Helmets'),
    ('Bottles and Cages'),
    ('Hydration Packs'),
    ('Bike Racks'),
    ('Fenders'),
    ('Mountain Bikes'),
    ('Road Bikes'),
    ('Jerseys'),
    ('Caps'),
    ('Gloves'),
    ('Shorts'),
    ('Socks'),
    ('Vests'),
    ('Cleaners');


-- ==========================================================================
-- Section 3: Data Quality Audit View
-- ==========================================================================
--
-- Surfaces all rows that fail the three quality rules before any filtering.
-- Run: SELECT * FROM vw_dq_audit
-- in any DB client to inspect exclusions and revenue impact.
--
-- Preserved after filtering so stakeholders can always see what was excluded
-- and why, without re-running the Python ETL.
--
-- CHANGE: added a fourth audit flag — rows with revenue = 0 — as a separate
-- exclusion_reason category.
-- ORIGINAL: only three exclusion reasons (no state, city as state, out-of-scope
-- sub-category). Zero-revenue rows were noted in 01_acs_etl.py but never surfaced
-- in the SQL audit.
-- WHY: zero-revenue rows inflate unit counts and distort margin calculations.
-- Having them in the audit view lets a reviewer quantify the exposure before
-- deciding whether to filter them downstream.

DROP VIEW IF EXISTS vw_dq_audit;

CREATE VIEW vw_dq_audit AS
SELECT
    state,
    country,
    sub_category,
    geo_flag,
    state_flag,
    COUNT(*)               AS row_count,
    ROUND(SUM(revenue), 2) AS revenue_affected,
    CASE
        WHEN state_flag = 'No State'             THEN 'Excluded: no state'
        WHEN geo_flag   = 'City Filed as State'  THEN 'Excluded: city as state'
        WHEN revenue    = 0                      THEN 'Flagged: zero revenue'
        ELSE 'Excluded: sub-category not in scope'
    END AS exclusion_reason
FROM stg_sales_raw
WHERE state_flag = 'No State'
   OR geo_flag   = 'City Filed as State'
   OR revenue    = 0
   OR sub_category NOT IN (
        SELECT sub_category FROM allowed_sub_categories
   )
GROUP BY state, country, sub_category, geo_flag, state_flag
ORDER BY row_count DESC;


-- ==========================================================================
-- Section 4: Clean Production Table
-- ==========================================================================
--
-- Single source of truth for all downstream analysis.
-- Three filters applied:
--   1. Rows with no state dropped — cannot be attributed to a geographic market.
--   2. Rows where a city was filed in the state field (Chicago, Miami) dropped —
--      state-level aggregation would be incorrect.
--   3. Only the 15 approved sub-categories retained.
--
-- All exclusions are logged in vw_dq_audit above.
--
-- CHANGE: added revenue > 0 as a fourth filter condition.
-- ORIGINAL:
--     WHERE s.state_flag = 'Has State'
--       AND s.geo_flag   = 'Clean'
-- WHY: zero-revenue rows (cost recorded but no revenue) are noted in
-- 01_acs_etl.py and flagged in vw_dq_audit. Including them in sales_clean
-- would inflate quantity totals and produce null or misleading margin values
-- in every downstream view. Explicit exclusion here makes the filter
-- decision visible and auditable.

DROP TABLE IF EXISTS sales_clean;

CREATE TABLE sales_clean AS
SELECT
    s.customer_gender,
    s.customer_age,
    s.product_category,
    s.sub_category,
    s.state,
    s.country,
    s.week_start_date,
    s.cost,
    s.quantity,
    s.revenue,
    s.unit_cost,
    s.unit_price,
    s.sale_year,
    s.sale_quarter,
    s.sale_month,
    s.month_name,
    s.year_month,
    s.week_num,
    s.gross_profit,
    s.margin_pct,
    s.age_band,
    s.geo_flag,
    s.state_flag
FROM stg_sales_raw s
INNER JOIN allowed_sub_categories a
    ON s.sub_category = a.sub_category
WHERE s.state_flag = 'Has State'
  AND s.geo_flag   = 'Clean'
  AND s.revenue    > 0;


-- ==========================================================================
-- Section 5: Row Count Validation
-- ==========================================================================
--
-- Quick sanity check printed to console by 03_acs_analysis.py after setup runs.
-- The filtered count should be meaningfully lower than the raw count due to
-- the four exclusion rules. If the numbers look equal, the filters did not apply.
--
-- CHANGE: extended validation to include excluded row count and exclusion rate.
-- ORIGINAL: only showed stg_sales_raw and sales_clean row counts side by side.
-- WHY: the raw delta is not informative on its own. The exclusion rate (as a
-- percentage of staging rows) tells a reviewer immediately whether the filters
-- are removing an expected fraction of the data or something unusual.

SELECT
    'stg_sales_raw'                                                AS table_name,
    COUNT(*)                                                       AS row_count,
    NULL                                                           AS exclusion_rate_pct
FROM stg_sales_raw

UNION ALL

SELECT
    'sales_clean',
    COUNT(*),
    NULL
FROM sales_clean

UNION ALL

SELECT
    'excluded_rows',
    (SELECT COUNT(*) FROM stg_sales_raw) - (SELECT COUNT(*) FROM sales_clean),
    ROUND(
        CAST(
            (SELECT COUNT(*) FROM stg_sales_raw) - (SELECT COUNT(*) FROM sales_clean)
        AS REAL)
        / NULLIF((SELECT COUNT(*) FROM stg_sales_raw), 0) * 100,
        2
    );


-- ==========================================================================
-- Section 6: State-Level Financial Summary
-- ==========================================================================
--
-- Primary Tableau export. One row per state / sub-category / year-month.
-- Supports both a geographic map view and a time-series trend line in the
-- same Tableau workbook without needing a data blend.
--
-- margin_pct is recalculated here from summed revenue and gross profit rather
-- than averaging row-level margin_pct values. Averaging percentages produces
-- incorrect results when row sizes differ — always aggregate numerator and
-- denominator separately, then divide.

DROP VIEW IF EXISTS vw_state_financials;

CREATE VIEW vw_state_financials AS
SELECT
    state,
    country,
    product_category,
    sub_category,
    sale_year,
    sale_quarter,
    sale_month,
    month_name,
    year_month,
    SUM(revenue)                                                  AS total_revenue,
    SUM(cost)                                                     AS total_cost,
    SUM(gross_profit)                                             AS total_gross_profit,
    SUM(quantity)                                                 AS total_quantity,
    ROUND(SUM(gross_profit) / NULLIF(SUM(revenue), 0) * 100, 2)  AS margin_pct,
    ROUND(AVG(unit_price), 2)                                     AS avg_unit_price,
    ROUND(AVG(unit_cost),  2)                                     AS avg_unit_cost
FROM sales_clean
GROUP BY
    state, country, product_category, sub_category,
    sale_year, sale_quarter, sale_month, month_name, year_month
ORDER BY
    country, state, year_month, sub_category;


-- ==========================================================================
-- Section 7: Sub-Category Scoreboard
-- ==========================================================================
--
-- Revenue ranking with margin overlay. Source for the performance dashboard
-- comparing sub-categories within and across product categories.
--
-- Two RANK columns are included:
--   revenue_rank_overall  — absolute rank across all sub-categories
--   revenue_rank_in_cat   — rank within the parent product category
--
-- Both ranks let Tableau show either view without a calculated field.

DROP VIEW IF EXISTS vw_subcategory_scoreboard;

CREATE VIEW vw_subcategory_scoreboard AS
SELECT
    sub_category,
    product_category,
    SUM(revenue)                                                  AS total_revenue,
    SUM(cost)                                                     AS total_cost,
    SUM(gross_profit)                                             AS total_gross_profit,
    SUM(quantity)                                                 AS total_quantity,
    ROUND(SUM(gross_profit) / NULLIF(SUM(revenue), 0) * 100, 2)  AS margin_pct,
    ROUND(SUM(revenue) / NULLIF(SUM(quantity), 0), 2)            AS revenue_per_unit,
    RANK() OVER (ORDER BY SUM(revenue) DESC)                      AS revenue_rank_overall,
    RANK() OVER (
        PARTITION BY product_category
        ORDER BY SUM(revenue) DESC
    )                                                             AS revenue_rank_in_cat
FROM sales_clean
GROUP BY sub_category, product_category
ORDER BY revenue_rank_overall;


-- ==========================================================================
-- Section 8: Monthly Revenue Trend with Period-Over-Period Metrics
-- ==========================================================================
--
-- Time series for trend line charts and regression inputs.
-- LAG() provides prior-month revenue so MoM growth can be calculated in SQL
-- rather than requiring a Tableau table calculation.
--
-- rolling_3mo_avg gives a smoothed trend filtering out single-week volatility.
--
-- CHANGE: added yoy_growth_pct (year-over-year) alongside mom_growth_pct.
-- ORIGINAL: only included mom_growth_pct.
-- WHY: MoM growth is noisy in seasonal datasets. YoY growth compares the same
-- calendar month across years, removing seasonality and giving a cleaner signal
-- on underlying revenue trajectory. Having both lets Tableau show either metric
-- without a custom calculated field.

DROP VIEW IF EXISTS vw_monthly_trend;

CREATE VIEW vw_monthly_trend AS
WITH monthly AS (
    SELECT
        year_month,
        sale_year,
        sale_quarter,
        sale_month,
        month_name,
        SUM(revenue)      AS monthly_revenue,
        SUM(gross_profit) AS monthly_gross_profit,
        SUM(quantity)     AS monthly_quantity
    FROM sales_clean
    GROUP BY year_month, sale_year, sale_quarter, sale_month, month_name
)
SELECT
    year_month,
    sale_year,
    sale_quarter,
    sale_month,
    month_name,
    monthly_revenue,
    monthly_gross_profit,
    monthly_quantity,
    ROUND(
        monthly_gross_profit / NULLIF(monthly_revenue, 0) * 100,
        2
    )                                                AS margin_pct,
    LAG(monthly_revenue) OVER (ORDER BY year_month)  AS prior_month_revenue,

    -- MoM growth: short-term signal, sensitive to seasonality
    ROUND(
        (monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY year_month))
        / NULLIF(LAG(monthly_revenue) OVER (ORDER BY year_month), 0) * 100,
        2
    )                                                AS mom_growth_pct,

    -- YoY growth: compares same month across years, removes seasonal noise
    ROUND(
        (monthly_revenue - LAG(monthly_revenue, 12) OVER (ORDER BY year_month))
        / NULLIF(LAG(monthly_revenue, 12) OVER (ORDER BY year_month), 0) * 100,
        2
    )                                                AS yoy_growth_pct,

    ROUND(
        AVG(monthly_revenue) OVER (
            ORDER BY year_month
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ),
        2
    )                                                AS rolling_3mo_avg
FROM monthly
ORDER BY year_month;


-- ==========================================================================
-- Section 9: Demographic Revenue Breakdown
-- ==========================================================================
--
-- Age band and gender segmentation for Tableau and regression inputs.
-- One row per age_band / gender / product_category gives 03_acs_analysis.py
-- enough granularity to run demographic regression without re-querying
-- sales_clean directly.

DROP VIEW IF EXISTS vw_demographic_summary;

CREATE VIEW vw_demographic_summary AS
SELECT
    age_band,
    customer_gender,
    product_category,
    sub_category,
    COUNT(*)                                                      AS transaction_count,
    SUM(revenue)                                                  AS total_revenue,
    SUM(gross_profit)                                             AS total_gross_profit,
    SUM(quantity)                                                 AS total_quantity,
    ROUND(SUM(gross_profit) / NULLIF(SUM(revenue), 0) * 100, 2)  AS margin_pct,
    ROUND(AVG(unit_price), 2)                                     AS avg_unit_price,
    ROUND(SUM(revenue) / NULLIF(SUM(quantity), 0), 2)            AS revenue_per_unit
FROM sales_clean
GROUP BY age_band, customer_gender, product_category, sub_category
ORDER BY age_band, customer_gender, total_revenue DESC;


-- ==========================================================================
-- Section 10: Country-Level Summary for Geographic Analysis
-- ==========================================================================
--
-- Aggregates at the country level for the geographic overview dashboard.
-- revenue_share_pct uses a window function so Tableau does not need a
-- calculated field.
--
-- CHANGE: added avg_transaction_value (revenue / transaction_count) as a
-- column alongside revenue_per_unit.
-- ORIGINAL: only included revenue_per_unit (revenue / quantity).
-- WHY: revenue_per_unit measures price per item sold. avg_transaction_value
-- measures how much a customer spends per visit (row in the data). These are
-- different signals — a high revenue_per_unit with a low avg_transaction_value
-- means customers buy expensive items but only one at a time. Both metrics
-- together give a fuller picture of purchasing behavior by market.

DROP VIEW IF EXISTS vw_country_summary;

CREATE VIEW vw_country_summary AS
SELECT
    country,
    product_category,
    sub_category,
    COUNT(*)                                                       AS transaction_count,
    SUM(revenue)                                                   AS total_revenue,
    SUM(gross_profit)                                              AS total_gross_profit,
    SUM(quantity)                                                  AS total_quantity,
    ROUND(SUM(gross_profit) / NULLIF(SUM(revenue), 0) * 100, 2)   AS margin_pct,
    ROUND(SUM(revenue) / NULLIF(SUM(quantity), 0), 2)             AS revenue_per_unit,

    -- avg_transaction_value: revenue per row (visit/transaction), not per item
    ROUND(SUM(revenue) / NULLIF(COUNT(*), 0), 2)                  AS avg_transaction_value,

    ROUND(
        SUM(revenue) / NULLIF(SUM(SUM(revenue)) OVER (PARTITION BY country), 0) * 100,
        2
    )                                                              AS pct_of_country_revenue
FROM sales_clean
GROUP BY country, product_category, sub_category
ORDER BY country, total_revenue DESC;
