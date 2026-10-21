"""
03_analysis.py

Retail Case Study — Step 3: Database Load, Tableau Exports, and Regression Analysis

This is the third and final file in a three-step pipeline:

    01_etl.py       Reshapes raw Excel into retail_sales_clean.csv
    02_setup.sql    Loads CSV into DB, applies quality filters, builds views
    03_analysis.py  (this file) Loads CSV into DB, runs SQL setup, queries
                        views, exports Tableau-ready CSVs, and runs regression models

What this file does:
    Loads retail_sales_clean.csv into a local database (DuckDB preferred, SQLite
    fallback), executes 02_setup.sql to build all views and aggregation tables,
    then queries each view to produce four Tableau-ready CSV exports and two
    regression summaries.

    The regression analysis fits OLS models on monthly revenue and demographic
    revenue data, writes outputs to CSVs a stakeholder can review without Python,
    and prints a summary table to the console.

    Why run SQL from Python rather than a DB client?
    Executing the SQL through Python keeps the pipeline as a single command.
    A reviewer or interviewer can run the project with two commands total:
        python 01_etl.py
        python 03_analysis.py
        Open Tableau and connect to the output CSVs or retail_sales.db

How to run:
    python 03_analysis.py
    python 03_analysis.py --csv retail_sales_clean.csv --sql 02_setup.sql --db retail_sales.db --out ./exports

    Environment variables also accepted:
    RETAIL_CLEAN_CSV, RETAIL_SQL, RETAIL_DB, RETAIL_OUT_DIR

Output files (written to ./exports/ by default):
    tableau_state_financials.csv      State x sub-category x month aggregates
    tableau_subcategory_scores.csv    Sub-category revenue ranking with margin
    tableau_monthly_trend.csv         Monthly revenue trend with MoM + YoY growth
    tableau_demographics.csv          Age band x gender x product breakdowns
    regression_time_series.csv        full-period monthly trend fit + residuals
    regression_trend_comparison.csv   full-period vs like-for-like trend fits
    subcategory_launch_dates.csv      first/last week each sub-category sold
    regression_demographics.csv       one-hot OLS: do age/gender add to quantity + category?

Requirements:
    pip install -r requirements.txt   (pandas, statsmodels, matplotlib, duckdb)
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

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.stattools import durbin_watson
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print('WARN: statsmodels not installed. Regression outputs will be skipped.')
    print('      pip install statsmodels to enable regression analysis.')

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
    # ORIGINAL: all four defaults were bare filenames ('retail_sales_clean.csv',
    # '02_setup.sql', 'retail_sales.db', 'exports') that resolve relative to
    # the process working directory.
    # WHY: when Python is launched via the full absolute path
    # (/usr/local/bin/python3 /Users/.../03_analysis.py), macOS sets the
    # working directory to '/' — a read-only system partition. The --csv and
    # --sql existence checks then fail immediately with "file not found" even
    # though both files are sitting right next to the script. The --db and --out
    # writes would crash with OSError Errno 30. Anchoring to __file__ means all
    # four paths resolve to the project folder regardless of how or from where
    # the script is invoked.
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description='Retail Case Study: Load DB, run SQL setup, export Tableau CSVs and regression outputs'
    )
    parser.add_argument(
        '--csv',
        default=os.environ.get('RETAIL_CLEAN_CSV', str(script_dir / 'retail_sales_clean.csv')),
        help='Path to retail_sales_clean.csv from step 1. Default: <script dir>/retail_sales_clean.csv'
    )
    parser.add_argument(
        '--sql',
        default=os.environ.get('RETAIL_SQL', str(script_dir / '02_setup.sql')),
        help='Path to 02_setup.sql. Default: <script dir>/02_setup.sql'
    )
    parser.add_argument(
        '--db',
        default=os.environ.get('RETAIL_DB', str(script_dir / 'retail_sales.db')),
        help='Database file path. Default: <script dir>/retail_sales.db'
    )
    parser.add_argument(
        '--out',
        default=os.environ.get('RETAIL_OUT_DIR', str(script_dir / 'exports')),
        help='Directory for output CSVs and charts. Default: <script dir>/exports'
    )
    args  = parser.parse_args()
    csv_p = Path(args.csv)
    sql_p = Path(args.sql)
    db_p  = Path(args.db)
    out_p = Path(args.out)

    # CHANGE: replaced the hard-fail exists() check with the same multi-candidate
    # search used in 01_etl.py, covering both the given path and the script
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

# TESTED 2026-09-16: running this script against a real retail_sales.db that
# had been created by an earlier run of the SQLite fallback path (duckdb not
# installed at the time), then re-running with duckdb installed, reproduced:
#     _duckdb.TransactionException: ... Failed to execute query "COMMIT": disk I/O error
# The real cause has nothing to do with disk I/O — duckdb.connect() was opening
# a file that is in SQLite's on-disk format, not DuckDB's, and fails with a
# misleading error instead of a clear "wrong file format" message. This is a
# real reproducibility trap: anyone who installs duckdb after already having
# run this pipeline once will hit it. Fixed by checking the file's on-disk
# magic bytes before connecting and rebuilding it if it doesn't match the
# engine currently selected.

_SQLITE_MAGIC = b'SQLite format 3\x00'
_DUCKDB_MAGIC_OFFSET = 8
_DUCKDB_MAGIC = b'DUCK'


def _ensure_compatible_db_file(db_path: Path) -> None:
    """
    If db_path already exists but was written by a different DB engine than
    the one currently selected (DB_ENGINE), delete it so the correct engine
    can create a fresh file. Without this check, DuckDB opening a SQLite file
    (or vice versa) fails with a cryptic low-level error instead of a clear one.
    """
    if not db_path.exists() or db_path.stat().st_size == 0:
        return

    with open(db_path, 'rb') as f:
        header = f.read(16)

    is_sqlite_file = header.startswith(_SQLITE_MAGIC)
    is_duckdb_file = header[_DUCKDB_MAGIC_OFFSET:_DUCKDB_MAGIC_OFFSET + 4] == _DUCKDB_MAGIC

    mismatch = (
        (DB_ENGINE == 'duckdb' and is_sqlite_file) or
        (DB_ENGINE == 'sqlite3' and is_duckdb_file)
    )
    if mismatch:
        other_engine = 'sqlite3' if DB_ENGINE == 'duckdb' else 'duckdb'
        print(
            f'  NOTE: {db_path.name} was created by {other_engine} on a previous '
            f'run, but this run is using {DB_ENGINE}. Rebuilding the file with '
            f'{DB_ENGINE} so the pipeline stays idempotent regardless of which '
            f'engine is installed.'
        )
        db_path.unlink()


def load_csv_to_db(csv_path: Path, db_path: Path) -> None:
    """
    Load retail_sales_clean.csv into the staging table stg_sales_raw.

    DuckDB registers the DataFrame as a virtual relation and creates the table
    in one step — no row-by-row inserts. SQLite uses pandas to_sql() with
    chunked parameterized inserts.

    The table is always dropped and recreated, making the script idempotent.
    Before connecting, _ensure_compatible_db_file() checks that an existing
    db file actually matches the current engine's format (see note above).
    """
    _ensure_compatible_db_file(db_path)
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
    Execute 02_setup.sql against the database.

    The SQL file is split on semicolons and each statement executed individually.
    Statements that create or drop stg_sales_raw are skipped because
    load_csv_to_db() already handled that table. Statements that fail are logged
    as warnings rather than crashing the script (most commonly window functions
    in SQLite CREATE VIEW statements on older SQLite versions).
    """
    print(f'\nRunning {sql_path.name}...')
    sql_text = sql_path.read_text()

    con = get_connection(db_path)
    failed_statements = []

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
    # Drop '--' line comments before splitting on ';' so a semicolon inside a
    # comment can't cut a statement in half. (02_setup.sql has no string
    # literals containing '--'.)
    sql_text = '\n'.join(ln.split('--', 1)[0] for ln in sql_text.splitlines())
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
            failed_statements.append(stmt[:80].replace(chr(10), " "))

    if DB_ENGINE == 'sqlite3':
        con.commit()

    print_validation_counts(con)
    con.close()

    # TESTED 2026-09-16: a real DuckDB binder error in vw_dq_audit was being
    # caught by the try/except above and printed as a single WARN line easy to
    # miss in a long console log, while the script still printed "SQL setup
    # complete" right after — which reads as success even when a view failed
    # to build. This banner makes a partial failure impossible to miss.
    if failed_statements:
        print(f'\n  *** {len(failed_statements)} statement(s) FAILED to execute — see WARN lines above. ***')
        print(f'  *** Any views/tables from those statements do NOT exist in {db_path.name}. ***')
        # Stop here: every export below reads these views, so continuing would
        # write partial outputs and exit 0.
        sys.exit(1)
    else:
        print('  SQL setup complete — all statements executed successfully.')


def print_validation_counts(con) -> None:
    """
    Print row counts after SQL setup to surface the filtering effect.
    The filtered count should be meaningfully lower than the raw count.

    CHANGE: updated query to match the extended Section 5 in 02_setup.sql,
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
            ROUND(SUM(gross_profit) / NULLIF(SUM(CASE WHEN gross_profit IS NOT NULL THEN revenue END), 0) * 100, 2)
                                                                            AS overall_margin_pct,
            SUM(CASE WHEN gross_profit IS NULL THEN 1 ELSE 0 END)           AS rows_without_cost,
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
        'overall_margin_pct':  'Overall margin (costed rows)',
        'rows_without_cost':   'Rows with revenue, no cost',
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
# Time series regression
# ---------------------------------------------------------------------------

# AUDITED 2026-10-21: the single full-period trend (R² 0.67, +$30.8K/month)
# mostly measures the catalog filling in, not growth. Through the week of
# 2015-06-21 the only sub-category sold is Road Bikes; 11 more first sell
# between 2015-06-28 and 2015-07-12, and Fenders, Gloves and Bike Racks only
# start in late Dec 2015 / Jan 2016. The like-for-like fit below uses the
# first full month after the mid-2015 launches through the last full month
# before the July 2016 drop-off, and only sub-categories that sold in every
# month of that window.
TREND_WINDOW = ('2015-08', '2016-06')


def _fit_trend(months: pd.Series, revenue: pd.Series) -> dict:
    """OLS of revenue on a 1..n month index; returns slope, its SE/p, R², DW."""
    x = np.arange(1, len(revenue) + 1)
    fit = sm.OLS(np.asarray(revenue, dtype=float), sm.add_constant(x)).fit()
    return {
        'start': months.iloc[0],
        'end': months.iloc[-1],
        'n_months': int(fit.nobs),
        'intercept': round(float(fit.params[0]), 2),
        'slope_per_month': round(float(fit.params[1]), 2),
        'slope_se': round(float(fit.bse[1]), 2),
        'slope_pvalue': round(float(fit.pvalues[1]), 4),
        'r2': round(float(fit.rsquared), 4),
        'durbin_watson': round(float(durbin_watson(fit.resid)), 2),
        'fitted': fit.fittedvalues,
    }


def subcategory_launches(db_path: Path) -> pd.DataFrame:
    """First and last week each sub-category sold, plus how many weeks it sold."""
    con = get_connection(db_path)
    query = """
        SELECT sub_category, product_category,
               MIN(week_start_date) AS first_week, MAX(week_start_date) AS last_week,
               COUNT(DISTINCT week_start_date) AS weeks_with_sales
        FROM sales_clean
        GROUP BY sub_category, product_category
        ORDER BY first_week, sub_category
    """
    df = con.execute(query).df() if DB_ENGINE == 'duckdb' else pd.read_sql(query, con)
    con.close()
    return df


def like_for_like_monthly(db_path: Path, window: tuple[str, str] = TREND_WINDOW) -> tuple[pd.DataFrame, list[str]]:
    """
    Monthly revenue inside `window`, restricted to sub-categories with sales in
    every month of the window. Returns (monthly frame, sub-categories used).
    """
    con = get_connection(db_path)
    query = """
        SELECT year_month, sub_category, SUM(revenue) AS revenue
        FROM sales_clean
        GROUP BY year_month, sub_category
    """
    df = con.execute(query).df() if DB_ENGINE == 'duckdb' else pd.read_sql(query, con)
    con.close()
    df = df[(df['year_month'] >= window[0]) & (df['year_month'] <= window[1])]
    n_months = df['year_month'].nunique()
    months_per_sub = df.groupby('sub_category')['year_month'].nunique()
    core = sorted(months_per_sub[months_per_sub == n_months].index)
    monthly = (df[df['sub_category'].isin(core)]
               .groupby('year_month', as_index=False)['revenue'].sum()
               .rename(columns={'revenue': 'monthly_revenue'})
               .sort_values('year_month').reset_index(drop=True))
    return monthly, core


def run_time_series_regression(df_monthly: pd.DataFrame, db_path: Path, out_dir: Path) -> pd.DataFrame:
    """
    Answers: "is revenue trending up or down and how fast?"

    Model:  revenue = b0 + b1 * month_index + error

    Fits it twice and writes both to regression_trend_comparison.csv:
      full_period    every month in the data (the original model)
      like_for_like  TREND_WINDOW, only sub-categories sold every month in it

    Outputs:
        regression_time_series.csv        full-period fit + residuals (as before)
        regression_trend_comparison.csv   both fits side by side
        subcategory_launch_dates.csv      when each sub-category first sold
        chart_revenue_trend_regression.png
    """
    if not STATSMODELS_AVAILABLE:
        return pd.DataFrame()

    print('\nRunning time series regression...')

    df = df_monthly.copy().sort_values('year_month').reset_index(drop=True)
    df['month_index'] = range(1, len(df) + 1)
    full = _fit_trend(df['year_month'], df['monthly_revenue'])

    lfl_monthly, core = like_for_like_monthly(db_path)
    lfl = _fit_trend(lfl_monthly['year_month'], lfl_monthly['monthly_revenue'])

    launches = subcategory_launches(db_path)
    launches.to_csv(out_dir / 'subcategory_launch_dates.csv', index=False)
    first_weeks = launches.groupby('first_week')['sub_category'].apply(lambda s: ', '.join(s))
    print('  Sub-category first sale week (the catalog fills in over time):')
    for week, subs in first_weeks.items():
        print(f'    {pd.Timestamp(week).date()}  {subs}')

    def describe(label, f):
        sig = 'significant' if f['slope_pvalue'] < 0.05 else 'not significant'
        print(f'  {label:<15} {f["start"]}..{f["end"]} ({f["n_months"]} mo): '
              f'slope ${f["slope_per_month"]:>10,.0f}/mo (SE {f["slope_se"]:,.0f}, p={f["slope_pvalue"]:.4f}, {sig}), '
              f'R²={f["r2"]:.3f}, Durbin-Watson={f["durbin_watson"]:.2f}')

    describe('Full period', full)
    describe('Like-for-like', lfl)
    print(f'    like-for-like sub-categories ({len(core)}): {", ".join(core)}')

    comparison = pd.DataFrame([
        {'spec': 'full_period', 'sub_categories': 'all in scope',
         **{k: v for k, v in full.items() if k != 'fitted'}},
        {'spec': 'like_for_like', 'sub_categories': '; '.join(core),
         **{k: v for k, v in lfl.items() if k != 'fitted'}},
    ])
    comparison.to_csv(out_dir / 'regression_trend_comparison.csv', index=False)

    y_pred = full['fitted']
    df_out = df[[
        'year_month', 'sale_year', 'sale_month', 'month_index',
        'monthly_revenue', 'monthly_gross_profit', 'mom_growth_pct',
        'rolling_3mo_avg',
    ]].copy()
    df_out['predicted_revenue'] = np.round(y_pred, 2)
    df_out['residual'] = np.round(df_out['monthly_revenue'] - y_pred, 2)
    df_out['residual_pct'] = (
        df_out['residual'] / df_out['monthly_revenue'].replace(0, np.nan) * 100
    ).round(2)
    mae = float(np.mean(np.abs(df_out['residual'])))
    sig_label = 'SIGNIFICANT (p < 0.05)' if full['slope_pvalue'] < 0.05 else 'NOT significant (p >= 0.05)'
    summary_row = pd.DataFrame([{
        'year_month':           'MODEL_SUMMARY',
        'predicted_revenue':    f'intercept={full["intercept"]:.2f} slope={full["slope_per_month"]:.2f}',
        'residual':             f'R2={full["r2"]:.4f} MAE={mae:.2f} pvalue={full["slope_pvalue"]:.4f}',
        'residual_pct':         f'sig={sig_label} DW={full["durbin_watson"]:.2f}',
    }])
    df_out = pd.concat([df_out, summary_row], ignore_index=True)
    path = out_dir / 'regression_time_series.csv'
    df_out.to_csv(path, index=False)
    print(f'  Saved: {path}')
    print(f'  Saved: {out_dir / "regression_trend_comparison.csv"}')

    _plot_time_series_regression(df, full, lfl_monthly, lfl, out_dir)
    return comparison


# ---------------------------------------------------------------------------
# Demographic regression
# ---------------------------------------------------------------------------

def run_demographic_regression(df_demo: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Do age band and gender add anything to revenue once quantity and product
    category are known?

    Each row of vw_demographic_summary (age band x gender x category x
    sub-category) is one observation. Two nested OLS models, categories
    one-hot encoded (treatment coding; baselines 17-24, F, Accessories):

        base:  revenue ~ quantity + C(product_category)
        full:  base + C(age_band) + C(customer_gender)

    The F-test on full vs base is the answer. The previous version
    label-encoded the categoricals into 0..n integers, which forces an
    ordering (and equal spacing) onto gender and category that doesn't exist.

    Outputs:
        regression_demographics.csv       full-model coefficients, SE, p, CI
        chart_demographic_regression.png
    """
    if not STATSMODELS_AVAILABLE:
        return pd.DataFrame()

    print('\nRunning demographic regression...')

    df = df_demo.dropna(subset=['total_revenue', 'total_quantity', 'age_band',
                                'customer_gender', 'product_category']).copy()
    df = df[df['age_band'] != 'nan']

    base = smf.ols('total_revenue ~ total_quantity + C(product_category)', data=df).fit()
    full = smf.ols('total_revenue ~ total_quantity + C(product_category)'
                   ' + C(age_band) + C(customer_gender)', data=df).fit()
    f_stat, f_p, df_diff = full.compare_f_test(base)

    print(f'  Observations        : {int(full.nobs)}')
    print(f'  R² quantity+category: {base.rsquared:.4f}')
    print(f'  R² + age + gender   : {full.rsquared:.4f}  (adj. {full.rsquared_adj:.4f})')
    verdict = 'they add explanatory power' if f_p < 0.05 else 'they add nothing significant'
    print(f'  F-test, age+gender  : F={f_stat:.2f} on {int(df_diff)} df, p={f_p:.4f}  -> {verdict}')

    ci = full.conf_int()
    coef_df = pd.DataFrame({
        'feature': full.params.index,
        'coefficient': full.params.values.round(4),
        'std_err': full.bse.values.round(4),
        'p_value': full.pvalues.values.round(4),
        'ci_low': ci[0].values.round(4),
        'ci_high': ci[1].values.round(4),
    })
    coef_df['abs_coef'] = coef_df['coefficient'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    coef_df['model_r2'] = round(full.rsquared, 4)
    coef_df['base_model_r2'] = round(base.rsquared, 4)
    coef_df['demographics_f_pvalue'] = round(float(f_p), 4)

    print('  Coefficients (full model):')
    for _, r in coef_df.iterrows():
        print(f'    {r["feature"]:<42} {r["coefficient"]:>12,.2f}  (p={r["p_value"]:.3f})')

    path = out_dir / 'regression_demographics.csv'
    coef_df.to_csv(path, index=False)
    print(f'  Saved: {path}')

    _plot_demographic_regression(coef_df, out_dir)
    return coef_df


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _plot_time_series_regression(df, full, lfl_monthly, lfl, out_dir: Path) -> None:
    """
    Two panels: actual monthly revenue with the full-period fit and the
    like-for-like fit (top), full-period residuals (bottom). The residuals run
    negative, then positive, then negative: one straight line through two
    regimes, which is why the like-for-like fit exists.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Time Series Regression: Monthly Revenue', fontsize=14, fontweight='bold')

    x_labels = df['year_month'].tolist()
    x_idx = np.arange(len(x_labels))
    pos = {m: i for i, m in enumerate(x_labels)}

    ax1 = axes[0]
    lo, hi = pos.get(lfl['start']), pos.get(lfl['end'])
    if lo is not None and hi is not None:
        ax1.axvspan(lo - 0.5, hi + 0.5, color='#54A24B', alpha=0.08, label='Like-for-like window')
    ax1.scatter(x_idx, df['monthly_revenue'] / 1e3, color='#4C78A8', alpha=0.7, label='Actual (all sub-categories)', zorder=3)
    ax1.plot(x_idx, np.asarray(full['fitted']) / 1e3, color='#E45756', linewidth=2,
             label=f'Full-period fit (R² {full["r2"]:.2f}, ${full["slope_per_month"]/1e3:,.1f}K/mo)')
    lfl_x = [pos[m] for m in lfl_monthly['year_month'] if m in pos]
    ax1.plot(lfl_x, np.asarray(lfl['fitted']) / 1e3, color='#54A24B', linewidth=2,
             label=f'Like-for-like fit (R² {lfl["r2"]:.2f}, ${lfl["slope_per_month"]/1e3:,.1f}K/mo)')
    ax1.set_xticks(x_idx[::2])
    ax1.set_xticklabels(x_labels[::2], rotation=45, ha='right')
    ax1.set_ylabel('Revenue ($K)')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}K'))
    ax1.set_title('Actual vs fitted monthly revenue')
    ax1.legend(fontsize=8, loc='upper left')

    ax2 = axes[1]
    residuals = df['monthly_revenue'].values - np.asarray(full['fitted'])
    bar_colors = ['#E45756' if r < 0 else '#54A24B' for r in residuals]
    ax2.bar(x_idx, residuals / 1e3, color=bar_colors, alpha=0.75)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_xticks(x_idx[::2])
    ax2.set_xticklabels(x_labels[::2], rotation=45, ha='right')
    ax2.set_ylabel('Residual ($K)')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}K'))
    ax2.set_title(f'Full-period residuals (Durbin-Watson {full["durbin_watson"]:.2f}; 2 = no autocorrelation)')

    plt.tight_layout()
    path = out_dir / 'chart_revenue_trend_regression.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Chart saved: {path}')


def _plot_demographic_regression(coef_df: pd.DataFrame, out_dir: Path) -> None:
    """Age and gender coefficients (vs the 17-24 / F baseline) with 95% CIs."""
    demo = coef_df[coef_df['feature'].str.contains('age_band|customer_gender')].copy()
    demo['label'] = (demo['feature'].str.replace(r'C\((\w+)\)\[T\.(.+)\]', r'\1 = \2', regex=True))
    demo = demo.sort_values('label')
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['#54A24B' if p < 0.05 else '#9E9E9E' for p in demo['p_value']]
    ax.barh(demo['label'], demo['coefficient'], color=colors, alpha=0.85,
            xerr=[demo['coefficient'] - demo['ci_low'], demo['ci_high'] - demo['coefficient']],
            error_kw={'elinewidth': 1, 'capsize': 3})
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Revenue difference vs baseline segment ($), holding quantity and category fixed')
    p = coef_df['demographics_f_pvalue'].iloc[0]
    ax.set_title(f'Demographic effects (95% CI; grey = not significant). Joint F-test p = {p:.3f}')
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

    run_time_series_regression(df_monthly, db_path, out_dir)
    run_demographic_regression(df_demo, out_dir)

    print(f'\nAll outputs written to: {out_dir.resolve()}')
    print(f'Database: {db_path.resolve()}')
    print('Tableau: connect to any .csv in the exports folder, or connect directly to retail_sales.db.')


if __name__ == '__main__':
    main()
