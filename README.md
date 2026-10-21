# Retail Finance Case Study: Sales & Margin Analysis

A completed take-home case study for a retail finance interview loop (`docs/case_study_prompt.docx` is the original brief). The prompt, dataset and constraints come from the interview packet; this repo is my worked solution, not client work.

## The problem

Build a system that shows financial trends over time for a bike and accessories retailer, under three constraints from the brief:

1. Financial statements are reported **at the state level**.
2. Customer age and gender are collected but never used today. **Should they be?**
3. The system covers **15 approved sub-categories** (2 of the 17 in the raw data are out of scope).

## What I built

1. **`01_etl.py`** reshapes the raw export (one column per week, one row per measure) into one row per customer segment x product x state x week, and derives gross profit, margin, date fields, age bands and data-quality flags.
2. **`02_setup.sql`** loads that table into DuckDB (SQLite also works), applies the brief's rules, and builds the reporting views plus an audit view, `vw_dq_audit`, that gives every excluded row exactly one reason.
3. **`03_analysis.py`** runs the SQL, exports Tableau-ready CSVs, and fits the trend and demographic models.

Of 26,240 segment-weeks ($7.31M revenue), 14,707 ($5.77M) qualify:

| Exclusion (first rule that applies) | Rows | Revenue |
|---|---:|---:|
| No state: every United Kingdom row | 4,777 | $1,305,705 |
| Quantity/cost recorded but no revenue | 5,641 | — |
| Sub-category out of scope (Touring Bikes, Bike Stands) | 1,099 | $230,449 |
| City filed as state (Chicago, Miami) | 16 | $4,962 |

## Findings

1. **Gross margin is 14.8%** on $5.77M of in-scope revenue, measured on rows with a recorded cost. 175 rows ($160K) have revenue but no cost and are left out of the margin denominator.
2. **US Bikes lose money: −2.1% gross margin**, a $9.8K gross loss on the $472K of US bike revenue with a recorded cost. Germany sells bikes at 16.7%, so this is a US price or cost problem, not a category problem. Another $101K of US bike revenue (18%) has no recorded cost at all.
3. **Germany out-earns the US in every category:** Accessories 30.2% vs 14.6%, Clothing 28.5% vs 12.8%, Bikes 16.7% vs −2.1%. France is lowest (Accessories 14.0%, Bikes 1.1%).
4. **The "10x growth" in mid-2015 is the catalog filling in, not demand.** Through the week of 21 June 2015 the only sub-category with sales is Road Bikes. Eleven more first sell between 28 June and 12 July 2015; Fenders, Gloves and Bike Racks start in Dec 2015/Jan 2016. A linear trend over all 20 months says +$30.8K/month (R² 0.67, p<0.001), but its residuals are autocorrelated (Durbin-Watson 0.91). **Like-for-like**, the 12 sub-categories sold every month from Aug 2015 to Jun 2016, the trend is +$5.8K/month with **p = 0.33: no significant growth.**
5. **Age matters a little; gender doesn't.** On 174 segment cells, quantity and product category explain R² 0.78 of revenue. Adding age band and gender (one-hot) raises it to 0.80, and the joint F-test is significant (p = 0.016; p < 0.001 with robust errors), almost entirely from the 25–34 band: about +$15K per segment vs 17–24 at the same quantity and category (p = 0.013). Male vs female: p = 0.54.

## Recommendations

1. **US bike pricing review (P1).** Find the price change that reaches breakeven at current volume. Fix cost capture on the $101K of US bike revenue with no cost before setting the target.
2. **Explain the Germany margin gap (P2).** Compare unit price, unit cost and mix by SKU across markets to tell price, cost and mix apart.
3. **Confirm the mid-2015 range launch (P2).** Were those 11 sub-categories launched in June/July 2015, or only recorded from then on? Either way, forecast from August 2015 onward.
4. **Use age, skip gender (P3).** The 25–34 band buys a pricier mix. Test bike-checkout accessory offers (Fenders and Bike Racks carry the highest margins, 21–22%) on that band first.

## Run it

```bash
pip install -r requirements.txt
python 01_etl.py         # -> retail_sales_clean.csv
python 03_analysis.py    # -> retail_sales.db + exports/
python -m pytest tests/
```

`exports/` holds the Tableau CSVs (`tableau_*.csv`), `regression_trend_comparison.csv` (full-period vs like-for-like), `subcategory_launch_dates.csv`, `regression_demographics.csv` and two charts.

## Tools

Python (pandas, statsmodels, matplotlib), SQL (DuckDB / SQLite), Tableau Public, pytest.

## Dashboard

*(Add the public.tableau.com link once `Retail_Sales_Operations_Analysis.twbx` is republished. Refresh its data extract so it picks up the cost-matched margins in `exports/`.)*

## Audit log

**2026-10-21 audit and rebuild**
- **The UK was silently dropped.** `pivot_table` discards rows whose key contains a NaN, and every UK row has a blank State. $1.31M (17.9% of raw revenue) disappeared before the no-state rule or the audit view saw it. The ETL now keeps those rows through the pivot and checks that the pivot conserves every value. Kept totals don't change, because the UK is excluded either way, but the exclusion is now visible and counted.
- **The audit view missed 83% of exclusions.** `revenue = 0` never matches NULL, and the ETL writes NULL for rows with quantity/cost but no revenue (5,641 rows). Reasons were also picked per group, so a mixed group got one label. It is now row-level, first match wins, and `kept + audited = staged` is tested.
- **Margins counted revenue with no cost as zero-margin sales.** Every margin now divides by revenue on rows with a recorded cost: overall 14.43% → 14.84%, US Bikes −1.7% → −2.1%.
- **The trend headline was the catalog expanding.** Added the like-for-like fit, Durbin-Watson, and the sub-category launch table (see finding 4).
- **The demographic model label-encoded categories,** forcing an order and equal spacing onto gender and category. It is refit as nested one-hot OLS with an F-test.
- **A semicolon inside a SQL comment split a statement in two.** The runner now strips comments before splitting, and a failed statement exits non-zero instead of printing a warning and carrying on.
- **Page fixes:** script names (`01_acs_etl.py` etc. didn't exist), UK scope, cost-matched margins, the like-for-like result, the demographic result, and gender splits (17–24 is 42/58, 25–34 is 45/55).
- **Tests added** (`tests/test_pipeline.py`): blank-state rows surviving the pivot, audit reconciliation with one row per exclusion rule, cost-matched margins, a semicolon in a comment, a failed statement exiting 1, like-for-like sub-category selection, and recovery of a known trend. The full pipeline also produces identical exports on DuckDB and SQLite.

**2026-09-16**
1. Week headers are `28-Dec-14` (`%d-%b-%y`), a format the ETL didn't try, so every run fell back to slow per-element date parsing. Fixed.
2. `vw_dq_audit` never built under DuckDB (a non-aggregated column in a `GROUP BY`), and the script printed "SQL setup complete" anyway. Fixed and given a pass/fail banner.

Three byte-identical Tableau workbook saves were consolidated into `Retail_Sales_Operations_Analysis.twbx`.
