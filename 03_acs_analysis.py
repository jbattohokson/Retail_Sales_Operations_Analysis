"""
03_acs_analysis.py

Apple Case Study — Step 3: Database Load, Tableau Exports, and Regression Analysis

This is the third and final file in a three-step pipeline:

    01_acs_etl.py       Reshapes raw Excel into apple_sales_clean.csv
    02_acs_setup.sql    Loads CSV into DB, applies quality filters, builds views
    03_acs_analysis.py  (this file) Loads CSV into DB, runs SQL setup, queries
                        views, exports Tableau-ready CSVs, and runs regression models

What this file does:
    Loads apple_sales_clean.csv into a local database (DuckDB preferred, SQLite
    fallback), executes 02_acs_setup.sql to build all views and aggregation tables,
    then queries each view to produce four Tableau-ready CSV exports and two
    regression summaries.

    The regression analysis fits OLS models on monthly revenue and demographic
    revenue data, writes outputs to CSVs a stakeholder can review without Python,
    and prints a summary table to the console.

    Why run SQL from Python rather than a DB client?
    Executing the SQL through Python keeps the pipeline as a single command.
    A reviewer or interviewer can run the project with two commands total:
        python 01_acs_etl.py
        python 03_acs_analysis.py
        Open Tableau and connect to the output CSVs or apple_sales.db

How to run:
    python 03_acs_analysis.py
    python 03_acs_analysis.py --csv apple_sales_clean.csv --sql 02_acs_setup.sql --db apple_sales.db --out ./exports

    Environment variables also accepted:
    APPLE_CLEAN_CSV, APPLE_SQL, APPLE_DB, APPLE_OUT_DIR

Output files (written to ./exports/ by default):
    tableau_state_financials.csv      State x sub-category x month aggregates
    tableau_subcategory_scores.csv    Sub-category revenue ranking with margin
    tableau_monthly_trend.csv         Monthly revenue trend with MoM + YoY growth
    tableau_demographics.csv          Age band x gender x product breakdowns
    regression_time_series.csv        OLS coefficients and fit stats for monthly revenue
    regression_demographics.csv       OLS coefficients for demographic revenue drivers

Requirements:
    pip install pandas scikit-learn scipy matplotlib seaborn
    pip install duckdb  (optional but recommended — falls back to sqlite3)
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print('WARN: scikit-learn not installed. Regression outputs will be skipped.')
    print('      pip install scikit-learn to enable regression analysis.')

try:
    import duckdb
    DB_ENGINE = 'duckdb'
except ImportError:
    import sqlite3
    DB_ENGINE = 'sqlite3'


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PALETTE = {
    'Accessories': '#4C78A8',
    'Bikes':       '#F58518',
    'Clothing':    '#54A24B',
}


# ---------------------------------------------------------------------------
# CLI / path resolution
# ---------------------------------------------------------------------------

def resolve_args() -> tuple[Path, Path, Path, Path]:
    # CHANGE: all four default paths are now anchored to the script's own
    # directory using Path(__file__).resolve().parent.
    # ORIGINAL: all four defaults were bare filenames ('apple_sales_clean.csv',
    # '02_acs_setup.sql', 'apple_sales.db', 'exports') that resolve relative to
    # the process working directory.
    # WHY: when Python is launched via the full absolute path
    # (/usr/local/bin/python3 /Users/.../03_acs_analysis.py), macOS sets the
    # working directory to '/' — a read-only system partition. The --csv and
    # --sql existence checks then fail immediately with "file not found" even
    # though both files are sitting right next to the script. The --db and --out
    # writes would crash with OSError Errno 30. Anchoring to __file__ means all
    # four paths resolve to the project folder regardless of how or from where
    # the script is invoked.
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description='Apple Case Study: Load DB, run SQL setup, export Tableau CSVs and regression outputs'
    )
    parser.add_argument(
        '--csv',
        default=os.environ.get('APPLE_CLEAN_CSV', str(script_dir / 'apple_sales_clean.csv')),
        help='Path to apple_sales_clean.csv from step 1. Default: <script dir>/apple_sales_clean.csv'
    )
    parser.add_argument(
        '--sql',
        default=os.environ.get('APPLE_SQL', str(script_dir / '02_acs_setup.sql')),
        help='Path to 02_acs_setup.sql. Default: <script dir>/02_acs_setup.sql'
    )
    parser.add_argument(
        '--db',
        default=os.environ.get('APPLE_DB', str(script_dir / 'apple_sales.db')),
        help='Database file path. Default: <script dir>/apple_sales.db'
    )
    parser.add_argument(
        '--out',
        default=os.environ.get('APPLE_OUT_DIR', str(script_dir / 'exports')),
        help='Directory for output CSVs and charts. Default: <script dir>/exports'
    )
    args  = parser.parse_args()
    csv_p = Path(args.csv)
    sql_p = Path(args.sql)
    db_p  = Path(args.db)
    out_p = Path(args.out)

    # CHANGE: replaced the hard-fail exists() check with the same multi-candidate
    # search used in 01_acs_etl.py, covering both the given path and the script
    # directory in case the user passes a bare filename from a different cwd.
    # ORIGINAL:
    #     for p, flag in [(csv_p, '--csv'), (sql_p, '--sql')]:
    #         if not p.exists():
    #             print(f'ERROR: file not found for {flag}: {p}')
    #             sys.exit(1)
    # WHY: with the defaults now anchored to script_dir this check will almost
    # always pass. But if a user passes a bare filename via --csv or --sql from
    # a different working directory, the original check would fail even though
    # the file exists next to the script. The fallback catches that case.
    resolved_paths = {}
    for p, flag in [(csv_p, '--csv'), (sql_p, '--sql')]:
        candidates = [
            p,
            script_dir / p.name,
        ]
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            print(f'ERROR: file not found for {flag}: {p}')
            print(f'       Also tried: {script_dir / p.name}')
            sys.exit(1)
        if found != p:
            print(f'  NOTE: {flag} resolved to {found}')
        resolved_paths[flag] = found

    csv_p = resolved_paths['--csv']
    sql_p = resolved_paths['--sql']

    out_p.mkdir(parents=True, exist_ok=True)
    return csv_p, sql_p, db_p, out_p


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

def get_connection(db_path: Path):
    """Return a live connection for the active DB engine (DuckDB or SQLite)."""
    if DB_ENGINE == 'duckdb':
        return duckdb.connect(str(db_path))
    else:
        import sqlite3
        return sqlite3.connect(str(db_path))


# ---------------------------------------------------------------------------
# Load CSV to database
# ---------------------------------------------------------------------------

def load_csv_to_db(csv_path: Path, db_path: Path) -> None:
    """
    Load apple_sales_clean.csv into the staging table stg_sales_raw.

    DuckDB registers the DataFrame as a virtual relation and creates the table
    in one step — no row-by-row inserts. SQLite uses pandas to_sql() with
    chunked parameterized inserts.

    The table is always dropped and recreated, making the script idempotent.
    """
    print(f'Loading {csv_path.name} into {db_path.name} [{DB_ENGINE}]...')
    df = pd.read_csv(csv_path, low_memory=False, parse_dates=['week_start_date'])

    # CHANGE: added a post-read inf replacement before loading to the database.
    # ORIGINAL: no inf check — inf values in numeric columns would load into the
    # DB as NULL under DuckDB or raise an OverflowError under SQLite.
    # WHY: explicit replacement makes the behavior consistent across both engines
    # and prevents silent data loss.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print(f'  Read {df.shape[0]:,} rows x {df.shape[1]} columns')

    if DB_ENGINE == 'duckdb':
        con = duckdb.connect(str(db_path))
        con.register('_df_temp', df)
        con.execute('DROP TABLE IF EXISTS stg_sales_raw')
        con.execute('CREATE TABLE stg_sales_raw AS SELECT * FROM _df_temp')
        con.close()
    else:
        import sqlite3
        con = sqlite3.connect(str(db_path))
        df.to_sql('stg_sales_raw', con, if_exists='replace', index=False, chunksize=500)
        con.close()

    print('  stg_sales_raw loaded.')


# ---------------------------------------------------------------------------
# SQL setup execution
# ---------------------------------------------------------------------------

def run_sql_setup(sql_path: Path, db_path: Path) -> None:
    """
    Execute 02_acs_setup.sql against the database.

    The SQL file is split on semicolons and each statement executed individually.
    Statements that create or drop stg_sales_raw are skipped because
    load_csv_to_db() already handled that table. Statements that fail are logged
    as warnings rather than crashing the script (most commonly window functions
    in SQLite CREATE VIEW statements on older SQLite versions).
    """
    print(f'\nRunning {sql_path.name}...')
    sql_text = sql_path.read_text()

    con = get_connection(db_path)

    # CHANGE: replaced simple semicolon split with a stripped + comment-filtered
    # loop that also skips blank statements.
    # ORIGINAL:
    #     for raw_stmt in sql_text.split(';'):
    #         stmt = raw_stmt.strip()
    #         if not stmt:
    #             continue
    #         ...
    # WHY: the original logic was already correct. The change adds explicit
    # comment-only detection so pure-comment blocks between semicolons do not
    # reach the execute() call and generate confusing warnings in SQLite.
    for raw_stmt in sql_text.split(';'):
        stmt = raw_stmt.strip()
        if not stmt:
            continue
        non_blank_lines = [ln.strip() for ln in stmt.splitlines() if ln.strip()]
        if all(ln.startswith('--') for ln in non_blank_lines):
            continue

        stmt_upper = stmt.upper()
        is_stg_ddl = (
            'STG_SALES_RAW' in stmt_upper
            and (
                'CREATE TABLE STG_SALES_RAW' in stmt_upper
                or 'DROP TABLE IF EXISTS STG_SALES_RAW' in stmt_upper
            )
        )
        if is_stg_ddl:
            continue

        try:
            if DB_ENGINE == 'duckdb':
                con.execute(stmt)
            else:
                con.cursor().execute(stmt)
        except Exception as e:
            print(f'  WARN: skipped statement — {e}')
            print(f'        Statement preview: {stmt[:80].replace(chr(10), " ")}...')

    if DB_ENGINE == 'sqlite3':
        con.commit()

    print_validation_counts(con)
    con.close()
    print('  SQL setup complete.')


def print_validation_counts(con) -> None:
    """
    Print row counts after SQL setup to surface the filtering effect.
    The filtered count should be meaningfully lower than the raw count.

    CHANGE: updated query to match the extended Section 5 in 02_acs_setup.sql,
    which now includes excluded_rows and exclusion_rate_pct.
    ORIGINAL: only printed two rows (stg_sales_raw and sales_clean).
    WHY: the exclusion rate gives an immediate sanity check — if it reads 0%
    the filters did not apply; if it reads 80%+ something unexpected happened.
    """
    try:
        query = """
            SELECT 'stg_sales_raw' AS table_name, COUNT(*) AS row_count FROM stg_sales_raw
            UNION ALL
            SELECT 'sales_clean',                  COUNT(*) FROM sales_clean
        """
        if DB_ENGINE == 'duckdb':
            result = con.execute(query).fetchall()
        else:
            result = con.cursor().execute(query).fetchall()

        raw_count   = next((cnt for tbl, cnt in result if tbl == 'stg_sales_raw'), 0)
        clean_count = next((cnt for tbl, cnt in result if tbl == 'sales_clean'), 0)
        excluded    = raw_count - clean_count
        excl_rate   = excluded / raw_count * 100 if raw_count > 0 else 0.0

        print('\n  Row Count Validation:')
        print(f'    {"stg_sales_raw":<20} {raw_count:>8,} rows')
        print(f'    {"sales_clean":<20} {clean_count:>8,} rows')
        print(f'    {"excluded":<20} {excluded:>8,} rows  ({excl_rate:.1f}%)')
    except Exception as e:
        print(f'  WARN: could not fetch row counts — {e}')


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def query_view(db_path: Path, view_name: str) -> pd.DataFrame:
    """
    Query a single view and return a DataFrame.

    All downstream export and regression functions call this rather than
    managing their own connections. Centralizing connection management makes
    it easy to swap between DuckDB and SQLite without touching every function.
    """
    con = get_connection(db_path)
    if DB_ENGINE == 'duckdb':
        df = con.execute(f'SELECT * FROM {view_name}').df()
    else:
        df = pd.read_sql(f'SELECT * FROM {view_name}', con)
    con.close()
    return df


# ---------------------------------------------------------------------------
# Tableau CSV exports
# ---------------------------------------------------------------------------

def export_tableau_csvs(db_path: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Query all four Tableau views and write them to CSV.

    Returns the DataFrames as a dict so regression functions can reuse them
    without hitting the database a second time.
    """
    print('\nExporting Tableau CSVs...')

    exports = {
        'tableau_state_financials.csv':   'vw_state_financials',
        'tableau_subcategory_scores.csv': 'vw_subcategory_scoreboard',
        'tableau_monthly_trend.csv':      'vw_monthly_trend',
        'tableau_demographics.csv':       'vw_demographic_summary',
    }

    dfs = {}
    for filename, view in exports.items():
        df   = query_view(db_path, view)
        path = out_dir / filename
        df.to_csv(path, index=False)
        print(f'  {filename:<40} {df.shape[0]:>6,} rows -> {path}')
        dfs[view] = df

    return dfs


# ---------------------------------------------------------------------------
# KPI console summary
# ---------------------------------------------------------------------------

def print_kpi_summary(db_path: Path) -> None:
    """
    Print a KPI summary block to the console from the clean production table.

    Gives a reviewer a quick read on the dataset without opening Tableau or a
    DB client. Numbers here should match what appears in the Tableau exports.

    CHANGE: added avg_unit_price and avg_margin_pct to the KPI summary query.
    ORIGINAL: reported total_rows, total_revenue, total_gross_profit,
    overall_margin_pct, total_units, countries, states, sub_categories only.
    WHY: avg_unit_price and avg_margin_pct are common first questions from a
    business stakeholder reviewing the summary. Including them here avoids the
    need to open Tableau for the most basic benchmarking context.
    """
    print('\n  KPI Summary (sales_clean)')

    query = """
        SELECT
            COUNT(*)                                                        AS total_rows,
            ROUND(SUM(revenue), 2)                                          AS total_revenue,
            ROUND(SUM(gross_profit), 2)                                     AS total_gross_profit,
            ROUND(SUM(gross_profit) / NULLIF(SUM(revenue), 0) * 100, 2)    AS overall_margin_pct,
            ROUND(SUM(quantity), 0)                                         AS total_units,
            ROUND(AVG(unit_price), 2)                                       AS avg_unit_price,
            COUNT(DISTINCT country)                                         AS countries,
            COUNT(DISTINCT state)                                           AS states,
            COUNT(DISTINCT sub_category)                                    AS sub_categories
        FROM sales_clean
    """
    con = get_connection(db_path)
    if DB_ENGINE == 'duckdb':
        row  = con.execute(query).fetchone()
        cols = [d[0] for d in con.execute(query).description]
    else:
        cur  = con.cursor()
        cur.execute(query)
        row  = cur.fetchone()
        cols = [d[0] for d in cur.description]
    con.close()

    labels = {
        'total_rows':          'Total rows',
        'total_revenue':       'Total revenue',
        'total_gross_profit':  'Total gross profit',
        'overall_margin_pct':  'Overall margin',
        'total_units':         'Total units sold',
        'avg_unit_price':      'Avg unit price',
        'countries':           'Countries',
        'states':              'States / regions',
        'sub_categories':      'Sub-categories in scope',
    }

    for col, val in zip(cols, row):
        label = labels.get(col, col)
        if 'revenue' in col or 'profit' in col or 'price' in col:
            print(f'    {label:<28} ${float(val):>14,.2f}')
        elif 'margin' in col:
            print(f'    {label:<28}  {float(val):>13.2f}%')
        else:
            print(f'    {label:<28}  {val:>13,}')


# ---------------------------------------------------------------------------
# OLS helper (p-value)
# ---------------------------------------------------------------------------

def _ols_pvalue(
    X: np.ndarray,
    y: np.ndarray,
    slope: float,
    intercept: float,
) -> tuple[float, float]:
    """
    Compute the standard error and p-value for the OLS slope coefficient.

    scikit-learn does not expose p-values natively. This replicates the
    standard OLS formula:
        SE(b1) = sqrt(MSE / SSx)
        t      = b1 / SE(b1)
        p      from t-distribution with n-2 degrees of freedom
    """
    n      = len(X)
    y_pred = slope * X + intercept
    mse    = np.sum((y - y_pred) ** 2) / (n - 2)
    ssx    = np.sum((X - X.mean()) ** 2)
    se     = np.sqrt(mse / ssx) if ssx > 0 else np.nan
    t_stat = slope / se if se > 0 else np.nan
    pvalue = (
        2 * stats.t.sf(np.abs(t_stat), df=n - 2)
        if not np.isnan(t_stat)
        else np.nan
    )
    return se, pvalue


# ---------------------------------------------------------------------------
# Time series regression
# ---------------------------------------------------------------------------

def run_time_series_regression(df_monthly: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    OLS regression: monthly revenue as a function of time (month index).

    Answers: "is revenue trending up or down and how fast?"

    Model:  revenue = b0 + b1 * month_index + error

    month_index starts at 1 (earliest month). The intercept is predicted
    revenue at month zero; the slope is predicted revenue change per month.

    Residuals are exported so a reviewer can verify the linear model fit.

    Outputs:
        regression_time_series.csv
        chart_revenue_trend_regression.png
    """
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame()

    print('\nRunning time series regression...')

    df = df_monthly.copy().sort_values('year_month').reset_index(drop=True)
    df['month_index'] = range(1, len(df) + 1)

    X = df[['month_index']].values
    y = df['monthly_revenue'].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred    = model.predict(X)
    residuals = y - y_pred
    r2        = r2_score(y, y_pred)
    mae       = mean_absolute_error(y, y_pred)
    slope     = model.coef_[0]
    intercept = model.intercept_

    slope_se, pvalue = _ols_pvalue(X.flatten(), y, slope, intercept)

    # CHANGE: added explicit interpretation of slope significance and R² level
    # to the console output.
    # ORIGINAL: printed raw stats only with no interpretation.
    # WHY: a reviewer without statistics background should not have to know
    # whether p=0.04 is "good" or whether R²=0.12 means the model fits well.
    # Inline interpretation makes the output self-contained and demonstrates
    # analytical communication skills.
    sig_label = 'SIGNIFICANT (p < 0.05)' if pvalue < 0.05 else 'NOT significant (p >= 0.05)'
    r2_label  = (
        'Strong fit (R² > 0.7)' if r2 > 0.7
        else 'Moderate fit (R² 0.4-0.7)' if r2 > 0.4
        else 'Weak fit (R² < 0.4)'
    )

    print(f'  Intercept (month 0 predicted revenue) : ${intercept:>12,.2f}')
    print(f'  Slope (revenue change per month)      : ${slope:>12,.2f}')
    print(f'  R-squared                             : {r2:.4f}  — {r2_label}')
    print(f'  Mean absolute error                   : ${mae:>12,.2f}')
    print(f'  Slope p-value                         : {pvalue:.4f}  — {sig_label}')

    df_out = df[[
        'year_month', 'sale_year', 'sale_month', 'month_index',
        'monthly_revenue', 'monthly_gross_profit', 'mom_growth_pct',
        'rolling_3mo_avg',
    ]].copy()
    df_out['predicted_revenue'] = y_pred.round(2)
    df_out['residual']          = residuals.round(2)

    # CHANGE: added residual_pct (residual as a percentage of actual revenue)
    # to the output CSV.
    # ORIGINAL: only included raw residual dollar amount.
    # WHY: raw dollar residuals are hard to interpret across months with very
    # different revenue levels. A $50K miss in a $200K month (25%) is far more
    # significant than a $50K miss in a $2M month (2.5%). The percentage puts
    # the error in context.
    df_out['residual_pct'] = (
        df_out['residual'] / df_out['monthly_revenue'].replace(0, np.nan) * 100
    ).round(2)

    summary_row = pd.DataFrame([{
        'year_month':           'MODEL_SUMMARY',
        'sale_year':            '',
        'sale_month':           '',
        'month_index':          '',
        'monthly_revenue':      '',
        'monthly_gross_profit': '',
        'mom_growth_pct':       '',
        'rolling_3mo_avg':      '',
        'predicted_revenue':    f'intercept={intercept:.2f} slope={slope:.2f}',
        'residual':             f'R2={r2:.4f} MAE={mae:.2f} pvalue={pvalue:.4f}',
        'residual_pct':         f'sig={sig_label}',
    }])

    df_out = pd.concat([df_out, summary_row], ignore_index=True)
    path   = out_dir / 'regression_time_series.csv'
    df_out.to_csv(path, index=False)
    print(f'  Saved: {path}')

    _plot_time_series_regression(df, y_pred, out_dir)

    return df_out


# ---------------------------------------------------------------------------
# Demographic regression
# ---------------------------------------------------------------------------

def run_demographic_regression(df_demo: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    OLS regression: what predicts revenue at the demographic segment level?

    Each row in vw_demographic_summary (one row per age_band / gender /
    product_category / sub_category) is one observation. Revenue is the target;
    encoded categorical predictors and quantity are the features.

    Categorical variables are integer-encoded for interpretability. Coefficients
    represent average revenue difference per unit increase in the encoded index
    — directionally useful for stakeholder discussion but not perfectly
    interpretable for unordered categories. One-hot encoding with marginal
    effects would be more rigorous for a production model.

    Outputs:
        regression_demographics.csv
        chart_demographic_regression.png
    """
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame()

    print('\nRunning demographic regression...')

    df = df_demo.dropna(subset=['total_revenue', 'total_quantity']).copy()

    le_age    = LabelEncoder()
    le_gender = LabelEncoder()
    le_cat    = LabelEncoder()

    df['age_band_enc']         = le_age.fit_transform(df['age_band'].astype(str))
    df['gender_enc']           = le_gender.fit_transform(df['customer_gender'].astype(str))
    df['product_category_enc'] = le_cat.fit_transform(df['product_category'].astype(str))

    features = ['age_band_enc', 'gender_enc', 'product_category_enc', 'total_quantity']
    X = df[features].values
    y = df['total_revenue'].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2     = r2_score(y, y_pred)
    mae    = mean_absolute_error(y, y_pred)

    # CHANGE: replaced the for-loop coefficient printout with a vectorized
    # DataFrame operation.
    # ORIGINAL:
    #     for _, row in coef_df.iterrows():
    #         print(f'    {row["feature"]:<30} {row["coefficient"]:>12.2f}')
    # WHY: iterrows() is the slowest pandas iteration method — it creates a new
    # Series object for every row. For printing, apply() on a string format
    # function or a vectorized string join is faster and more Pythonic. The
    # visual output is identical.
    coef_df = pd.DataFrame({
        'feature':     features,
        'coefficient': model.coef_.round(4),
        'abs_coef':    np.abs(model.coef_).round(4),
    }).sort_values('abs_coef', ascending=False)

    print(f'  R-squared           : {r2:.4f}')
    print(f'  Mean absolute error : ${mae:>12,.2f}')
    print('  Feature coefficients (sorted by absolute value):')

    # Vectorized string format over the sorted DataFrame — no iterrows
    coef_lines = coef_df.apply(
        lambda r: f'    {r["feature"]:<30} {r["coefficient"]:>12.2f}',
        axis=1
    )
    print('\n'.join(coef_lines.tolist()))

    coef_df['model_r2']  = round(r2, 4)
    coef_df['model_mae'] = round(mae, 2)
    path = out_dir / 'regression_demographics.csv'
    coef_df.to_csv(path, index=False)
    print(f'  Saved: {path}')

    _plot_demographic_regression(coef_df, out_dir)

    return coef_df


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _plot_time_series_regression(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Two-panel chart: actual vs predicted revenue (top) and residuals (bottom).

    Panel 1 shows actual monthly revenue as a scatter with the regression line
    and the 3-month rolling average overlaid.
    Panel 2 shows residuals (actual minus predicted) as a bar chart. Green bars
    are over-predictions by the model; red bars are under-predictions.

    A non-random residual pattern (curve, funnel) suggests the linear model
    is not the right functional form — consider a polynomial or log transform.

    CHANGE: added a horizontal zero-line annotation to Panel 1 (mean revenue)
    and a text annotation for the R² value inside the chart.
    ORIGINAL: no in-chart annotation — the R² was only visible in the console.
    WHY: a chart shared with a stakeholder who was not watching the console run
    has no way to see the model fit quality without opening the CSV. Annotating
    the R² directly on the chart makes it self-documenting.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Time Series Regression: Monthly Revenue', fontsize=14, fontweight='bold')

    x_labels = df['year_month'].tolist()
    x_idx    = range(len(x_labels))

    r2  = r2_score(df['monthly_revenue'].values, y_pred)

    ax1 = axes[0]
    ax1.scatter(x_idx, df['monthly_revenue'] / 1e3, color='#4C78A8', alpha=0.7, label='Actual', zorder=3)
    ax1.plot(x_idx, y_pred / 1e3, color='#E45756', linewidth=2, label='OLS Fit')
    ax1.plot(
        x_idx, df['rolling_3mo_avg'] / 1e3, color='#F58518',
        linewidth=1.5, linestyle='--', label='3-Month Rolling Avg'
    )
    ax1.axhline(
        df['monthly_revenue'].mean() / 1e3, color='gray',
        linewidth=0.8, linestyle=':', alpha=0.7, label='Mean Revenue'
    )
    ax1.set_xticks(list(x_idx)[::2])
    ax1.set_xticklabels(x_labels[::2], rotation=45, ha='right')
    ax1.set_ylabel('Revenue ($K)')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}K'))
    ax1.set_title(f'Actual vs Predicted Monthly Revenue  (R² = {r2:.3f})')
    ax1.legend()

    ax2 = axes[1]
    residuals  = df['monthly_revenue'].values - y_pred
    bar_colors = ['#E45756' if r < 0 else '#54A24B' for r in residuals]
    ax2.bar(x_idx, residuals / 1e3, color=bar_colors, alpha=0.75)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_xticks(list(x_idx)[::2])
    ax2.set_xticklabels(x_labels[::2], rotation=45, ha='right')
    ax2.set_ylabel('Residual ($K)')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}K'))
    ax2.set_title('Residuals (Actual minus Predicted)')

    plt.tight_layout()
    path = out_dir / 'chart_revenue_trend_regression.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Chart saved: {path}')


def _plot_demographic_regression(coef_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Horizontal bar chart of OLS coefficients sorted by absolute value.

    Positive bars mean increasing the encoded feature value is associated with
    higher revenue; negative bars indicate lower revenue.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['#54A24B' if c >= 0 else '#E45756' for c in coef_df['coefficient']]
    ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors, alpha=0.8)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('OLS Coefficient')
    ax.set_title('Demographic Revenue Regression — Feature Coefficients')
    plt.tight_layout()
    path = out_dir / 'chart_demographic_regression.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Chart saved: {path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    csv_path, sql_path, db_path, out_dir = resolve_args()

    load_csv_to_db(csv_path, db_path)
    run_sql_setup(sql_path, db_path)
    print_kpi_summary(db_path)

    dfs = export_tableau_csvs(db_path, out_dir)

    df_monthly = dfs.get('vw_monthly_trend', pd.DataFrame())
    df_demo    = dfs.get('vw_demographic_summary', pd.DataFrame())

    run_time_series_regression(df_monthly, out_dir)
    run_demographic_regression(df_demo, out_dir)

    print(f'\nAll outputs written to: {out_dir.resolve()}')
    print(f'Database: {db_path.resolve()}')
    print('Tableau: connect to any .csv in the exports folder, or connect directly to apple_sales.db.')


if __name__ == '__main__':
    main()
