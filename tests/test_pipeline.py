"""Pipeline tests on small hand-built inputs (no real data needed).

Run: python -m pytest tests/
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


etl = _load('etl', '01_etl.py')
ana = _load('ana', '03_analysis.py')

DIMS = ['Customer Gender', 'Customer Age', 'Product Category', 'Sub Category', 'State', 'Country', 'Measure']


def wide(rows, weeks=('28-Dec-14', '4-Jan-15')):
    """rows: (gender, age, cat, sub, state, country, {measure: [v per week]})"""
    out = []
    for g, a, c, sc, st, co, measures in rows:
        for m, vals in measures.items():
            out.append([g, a, c, sc, st, co, m, *vals])
    return pd.DataFrame(out, columns=DIMS + list(weeks))


def run_etl(df_raw):
    return etl.clean_and_derive(etl.pivot_measures(etl.melt_to_long(df_raw)))


ALL = lambda r, c, q: {'Revenue': r, 'Cost': c, 'Quantity': q, 'Unit Cost': [1, 1], 'Unit Price': [2, 2]}  # noqa: E731


def test_pivot_keeps_rows_with_blank_state():
    df = run_etl(wide([
        ('F', 30, 'Bikes', 'Road Bikes', 'California', 'United States', ALL([100, 200], [90, 150], [1, 2])),
        ('M', 40, 'Bikes', 'Road Bikes', None, 'United Kingdom',
         {'Revenue': [300, None], 'Cost': [250, None], 'Quantity': [3, None]}),
    ]))
    uk = df[df['country'] == 'United Kingdom']
    assert len(uk) == 1 and uk['state'].isna().all()
    assert (uk['state_flag'] == 'No State').all()
    assert df['revenue'].sum() == 600


def build_db(tmp_path, df_clean):
    csv = tmp_path / 'clean.csv'
    db = tmp_path / 'test.db'
    df_clean.to_csv(csv, index=False)
    ana.load_csv_to_db(csv, db)
    ana.run_sql_setup(ROOT / '02_setup.sql', db)
    return db


@pytest.fixture
def db(tmp_path_factory):
    df = run_etl(wide([
        # kept: revenue with cost
        ('F', 30, 'Bikes', 'Road Bikes', 'California', 'United States', ALL([100, 200], [90, 150], [1, 2])),
        # kept, but week 2 has revenue and no cost -> excluded from the margin denominator
        ('M', 30, 'Accessories', 'Helmets', 'Oregon', 'United States', ALL([50, 80], [40, None], [1, 1])),
        # no state
        ('M', 40, 'Bikes', 'Road Bikes', None, 'United Kingdom', ALL([300, 10], [250, 5], [3, 1])),
        # city filed as state
        ('F', 25, 'Bikes', 'Road Bikes', 'Chicago', 'United States', ALL([70, 70], [60, 60], [1, 1])),
        # out of scope sub-category
        ('F', 25, 'Bikes', 'Touring Bikes', 'Texas', 'United States', ALL([500, 500], [400, 400], [1, 1])),
        # quantity and cost, no revenue
        ('F', 50, 'Clothing', 'Socks', 'Texas', 'United States',
         {'Revenue': [None, None], 'Cost': [5, 5], 'Quantity': [1, 1], 'Unit Cost': [5, 5], 'Unit Price': [9, 9]}),
    ]))
    return build_db(tmp_path_factory.mktemp('db'), df)


def q(db, sql):
    con = ana.get_connection(db)
    out = con.execute(sql).df() if ana.DB_ENGINE == 'duckdb' else pd.read_sql(sql, con)
    con.close()
    return out


def test_audit_accounts_for_every_excluded_row(db):
    counts = q(db, """
        SELECT (SELECT COUNT(*) FROM stg_sales_raw) AS staged,
               (SELECT COUNT(*) FROM sales_clean) AS kept,
               (SELECT SUM(row_count) FROM vw_dq_audit) AS audited
    """).iloc[0]
    assert counts['kept'] + counts['audited'] == counts['staged']
    reasons = q(db, 'SELECT exclusion_reason, SUM(row_count) AS n FROM vw_dq_audit GROUP BY 1').set_index('exclusion_reason')['n']
    assert reasons.to_dict() == {
        'Excluded: no state': 2,
        'Excluded: city as state': 2,
        'Excluded: sub-category not in scope': 2,
        'Excluded: no revenue recorded': 2,
    }


def test_margin_uses_only_rows_with_cost(db):
    sub = q(db, 'SELECT sub_category, total_revenue, margin_pct FROM vw_subcategory_scoreboard').set_index('sub_category')
    # Helmets: revenue 130 total, but only the 50 with a cost (40) counts -> 20%
    assert sub.loc['Helmets', 'total_revenue'] == 130
    assert sub.loc['Helmets', 'margin_pct'] == pytest.approx(20.0)
    assert sub.loc['Road Bikes', 'margin_pct'] == pytest.approx(20.0)  # (300-240)/300


def test_semicolon_in_sql_comment_does_not_break_setup(tmp_path, db):
    sql = tmp_path / 'setup.sql'
    sql.write_text('-- a comment; with a semicolon\nCREATE VIEW v_test AS SELECT 1 AS x;\n')
    ana.run_sql_setup(sql, db)
    assert q(db, 'SELECT x FROM v_test')['x'].tolist() == [1]


def test_failed_sql_statement_exits_nonzero(tmp_path, db):
    sql = tmp_path / 'bad.sql'
    sql.write_text('SELECT * FROM table_that_does_not_exist;\n')
    with pytest.raises(SystemExit) as exc:
        ana.run_sql_setup(sql, db)
    assert exc.value.code == 1


def test_like_for_like_keeps_only_subcategories_sold_every_month(tmp_path):
    rows = []
    for month in pd.date_range('2015-08-02', '2016-06-26', freq='4W'):
        rows.append(('F', 30, 'Bikes', 'Road Bikes', 'California', 'United States', month, 100.0))
        if month >= pd.Timestamp('2016-01-01'):
            rows.append(('F', 30, 'Accessories', 'Bike Racks', 'California', 'United States', month, 1000.0))
    df = pd.DataFrame(rows, columns=['Customer Gender', 'Customer Age', 'Product Category', 'Sub Category',
                                     'State', 'Country', 'week_start_date', 'Revenue'])
    df['Cost'] = df['Revenue'] * 0.9
    df['Quantity'], df['Unit Cost'], df['Unit Price'] = 1.0, 1.0, 1.0
    db = build_db(tmp_path, etl.clean_and_derive(df))
    monthly, core = ana.like_for_like_monthly(db, ('2015-08', '2016-06'))
    assert core == ['Road Bikes']
    assert (monthly['monthly_revenue'] % 100 == 0).all() and monthly['monthly_revenue'].max() <= 200


def test_trend_fit_recovers_known_slope():
    months = pd.Series([f'2016-{m:02d}' for m in range(1, 13)])
    rev = pd.Series(1000 + 50 * np.arange(1, 13) + np.tile([5, -5], 6))
    fit = ana._fit_trend(months, rev)
    assert fit['slope_per_month'] == pytest.approx(50, abs=1)
    assert fit['slope_pvalue'] < 0.001 and fit['n_months'] == 12
