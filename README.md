# Apple Retail Finance Case Study — Sales & Margin Analysis

This project is a completed take-home case study for an Apple Retail Finance PART Manager interview loop (see `docs/case_study_prompt.docx` for the original brief). Stating that plainly here rather than presenting it as client work: the prompt, the dataset, and the three business constraints below all came from Apple's own interview packet, and this repo is my worked solution to it.

**Business question the case study asks:** build a system to identify financial trends over time, with three hard constraints from the interview stakeholders — (1) financial statements must be reported at the state level, (2) customer demographics (age/gender) are collected but never used in analysis today, and (3) the new system is scoped to 15 approved sub-categories, excluding 2 out of the 17 in the raw data.

## Key finding

Overall gross margin across the 15 in-scope sub-categories is 14.4% on $5.77M of qualifying revenue (14,707 clean transaction-weeks, after excluding 31.5% of raw rows for no-state, city-filed-as-state, zero-revenue, and out-of-scope sub-category issues — see `vw_dq_audit` in the database for the full breakdown by reason). A monthly OLS regression shows a statistically significant upward revenue trend (R²=0.67, p<0.001); a second regression reaches R²=0.76, but quantity alone explains R²=0.75, so age band, gender and category add less than 0.01. The categorical terms are label-encoded, which treats them as ordered numbers; refitting with one-hot encoding is the next step before reading any demographic effect from this model.

## Pipeline

1. `01_etl.py` — reshapes the raw wide-format export (84 weekly columns) into a clean flat CSV, deriving gross profit, margin, age bands, and geography quality flags.
2. `02_setup.sql` — loads the clean CSV into a database, applies the case study's own filtering rules, and builds the views used for both the audit trail and the Tableau exports.
3. `03_analysis.py` — runs the SQL setup, exports four Tableau-ready CSVs, and fits two OLS regressions (monthly revenue trend, demographic revenue drivers).

```bash
pip install -r requirements.txt
python 01_etl.py
python 03_analysis.py
```

Both DuckDB and SQLite are supported (DuckDB preferred, auto-detected). Outputs land in `exports/`.

## Tools

Python (pandas, statsmodels/scikit-learn, matplotlib), SQL (DuckDB/SQLite), Tableau Public.

## Dashboard

*(Add the public.tableau.com link here once `Retail_Sales_Operations_Analysis.twbx` is republished.)*

## Bugs found and fixed while testing this pipeline (2026-09-16)

Running the full pipeline three times end-to-end surfaced two real bugs that had never been caught before, both now fixed and re-verified across 3 clean runs with identical output on every run:

1. **Date parsing silently fell back to slow inference on every run.** The raw CSV's week columns use `28-Dec-14` format (`%d-%b-%y`), which matched neither of the two formats the ETL tried first. Every run silently dropped to pandas's per-element dateutil inference, which is 10-50x slower and prints a `UserWarning`. Fixed by adding the actual format as the first attempt in `01_etl.py`.
2. **`vw_dq_audit` never actually built under DuckDB.** The view's `CASE WHEN revenue = 0` referenced a non-aggregated column inside a `GROUP BY` query, which DuckDB correctly rejects and SQLite silently (and nondeterministically) allowed. Because `03_analysis.py` catches and warns on failed statements rather than crashing, this failure was easy to miss — the script still printed "SQL setup complete." Fixed by aggregating a `has_zero_revenue` flag with `MAX()` in `02_setup.sql`, and added a pass/fail summary banner to `03_analysis.py` so a partial SQL failure can never again look like a clean run.

## Repo hygiene note

Three identical (byte-for-byte, confirmed via checksum) Tableau workbook saves (`_D1`/`_D2`/`_D3`) were consolidated into one file, `Retail_Sales_Operations_Analysis.twbx`.
