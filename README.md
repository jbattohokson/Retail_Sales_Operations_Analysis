# A 10x revenue surge doesn't explain itself.
### Here's what the data found.

> **[Live Report](https://jbattohokson.github.io/Retail_Sales_Operations_Analysis/Retail_Sales_Operations_Analysis.html)** | [GitHub Repo](https://github.com/jbattohokson/Retail_Sales_Operations_Analysis)

---

## Executive Summary

Between December 2014 and July 2016, this retail portfolio experienced a 10x revenue surge in a single month (August 2015). The analysis confirms the jump was structural — monthly revenue shifted from a sub-$50K baseline to a sustained $460–500K average and held there. The surge, however, exposed a profitability problem hiding underneath the revenue growth: Bikes, the highest-volume category in the largest market (US), operate at a −1.5% gross margin. The path to margin recovery does not require cutting Bike revenue — it requires repricing US Bike SKUs and cross-selling Accessories (up to 22% gross margin) to the highest-LTV demographic segment: female customers aged 45–54.

---

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| Python (pandas, NumPy, Matplotlib, statsmodels) | ETL, OLS regression, residuals analysis, visualization |
| MySQL | Data warehouse setup, exclusion filtering, Tableau-ready views |
| Tableau | 7 interactive views: revenue trends, margin scatter, demographic breakdown, geographic choropleth |
| HTML/CSS/JavaScript | Interactive report output |

---

## Findings: What the data shows, and what it means

### August 2015 was a structural inflection point, not a seasonal spike
Monthly revenue held below $50K from December 2014 through July 2015 with minimal variance. In August 2015, revenue jumped to approximately $435K in a single month — a 10x increase — and a 3-month rolling average confirms the shift held, averaging $460–500K per month through the rest of the observation window. The July 2016 data point (~$210K) is a data truncation artifact, not a real revenue contraction — both revenue and margin signals drop simultaneously, consistent with a partial-month data pull.

### US Bike sales are loss-generating at the gross profit line
The United States is the largest revenue market by volume, and its core high-volume category (Bikes) operates at −1.5% gross margin. Road Bikes alone generated approximately $650K in revenue — at −1.5%, that translates to roughly $9,750 in gross losses before any operating costs. Germany achieves 16.7% Bike margin on the same product category, indicating the gap is a pricing and cost structure problem, not a product-category limitation.

### Accessories are the margin engine, but they're being undersold to the highest-value customers
Accessories dominate portfolio profitability post-August 2015, with Fenders and Bike Racks carrying the highest margins in the entire product portfolio (21–22%) — yet both sit in the low-revenue/high-margin quadrant. Female customers aged 45–54 outspend male customers in their cohort, represent 53% of segment revenue, and are confirmed by OLS regression as the single largest positive revenue coefficient. They are currently purchasing into a low-margin category (Bikes) rather than a high-margin one (Accessories).

### Germany outperforms the US on margin by 15+ percentage points across every category
Germany achieves 30.3% on Accessories and 28.6% on Clothing — roughly double the US margins in both categories. Bike margin in Germany is 16.7% versus −1.5% in the US, an 18.2 percentage point gap on identical product categories across different markets. Whether Germany's advantage stems from pricing, cost structure, or product mix is the follow-on question that determines the strategic value of the gap.

| Market | Accessories Margin | Bikes Margin | Clothing Margin |
|--------|-------------------|-------------|----------------|
| France | 14.0% | 0.8% | 10.6% |
| Germany | 30.3% | 16.7% | 28.6% |
| United States | 14.7% | −1.5% ⚑ | 12.8% |

### The 25–34 cohort drives volume; the 45–54 cohort drives margin opportunity
The 25–34 age band is the dominant revenue segment at approximately $1.95M — nearly 20% above the next-highest group (35–44 at ~$1.6M). The 45–54 cohort is female-dominant (53%), already converting, and purchasing in a category where this portfolio has a margin advantage if cross-sells are executed. Marketing dollars allocated to the 45–54 female segment toward Accessories generate higher margin per acquisition dollar than the same spend on the 25–34 volume segment.

---

## Recommendations: Recommended next steps, ranked by estimated impact

### US Bike SKU pricing audit
US Bikes are the largest revenue category in the largest market, and they are loss-generating. A SKU-level price-cost review is the single highest-ROI analytical action available from this dataset. $650K Road Bike revenue × −1.5% margin = ~$9,750 gross loss at current volume.

### Accessories cross-sell to 45–54 female segment
Bundle Helmets, Fenders, and Bottles & Cages with Bike purchases for the highest-LTV, highest-margin demographic combination in the portfolio. At a 10% attach rate on ~650 Road Bike transactions at a $50 accessory price, that is approximately $715 in incremental gross margin at 22% — scalable to full portfolio volume.

### Investigate August 2015 inflection driver
The 10x revenue jump is confirmed structural, but the business event that caused it is unidentified. Apply a Chow test to formally verify the structural break, then document the cause. This converts a visual observation into a defensible forecast input for future expansion scenario modeling.

### Germany market expansion business case
Germany outperforms the US by 15+ margin points across all three product categories. A revenue-by-SKU breakdown with unit economics would quantify whether Germany's margin advantage justifies increased market investment — and separate whether the gap is pricing, cost, or product mix.

### Refit time series as piecewise regression
The current OLS model is structurally misspecified; residuals are non-random and autocorrelation is present. A piecewise regression using August 2015 as the breakpoint, or an ARIMA(1,1,0) model, would produce defensible trend forecasts. Report Durbin-Watson statistic and adjusted R² alongside current R² = 0.674.

### Expand geographic distribution analysis
California dominates US revenue by concentration. The Midwest and Southeast show minimal revenue despite likely comparable market size. A state-level revenue-per-capita analysis would surface expansion targets with supporting rationale — data already captured in the current schema.

---

## Statistical Models: Two OLS models, what each one measures

### Time Series Model: Does revenue trend upward over time?
A linear OLS regression was fitted on the monthly revenue time series using a sequential time index as the sole predictor.

- R² = 0.674 | Slope = +$29,500/month | p-value < 0.001 | n = 20 months

The R² of 0.674 overstates model quality. The residual plot shows a systematic pattern — negative in early months, consistently positive post-August 2015, sharply negative at period-end — the signature of temporal autocorrelation and structural non-stationarity. The $29,500/month slope is accurate as a whole-period average but should not be used for point forecasting without fitting a piecewise model at the August 2015 breakpoint.

### Demographic Model: What drives revenue at the transaction level?
A cross-sectional OLS regression was built at the demographic segment level using age band, gender, product category, and quantity as predictors of revenue per transaction. Gender is the single largest positive revenue coefficient: female customers, controlling for age and product category, are associated with higher per-transaction revenue. This result drives the Accessories cross-sell recommendation toward the 45–54 female cohort — it is a model-supported targeting signal, not a demographic observation.

---

## Methodology: Pipeline architecture & data scope

The analysis was produced by a three-step Python/SQL pipeline. Each stage is independently auditable.

**01_acs_etl.py** — Reshapes raw wide-format Excel (weekly columns × measure rows) into a flat analytical table. Derives gross profit, margin, date dimensions, and age bands. Flags geographic data quality issues before output.

**02_acs_setup.sql** — Loads cleaned CSV into a MySQL schema. Applies four exclusion filters: null states, cities filed as states, non-approved sub-categories, and zero-revenue rows. Builds production views and Tableau-ready aggregates.

**03_acs_analysis.py** — Queries DB views, exports Tableau CSVs, runs both OLS regression models, and produces the residuals panel. Inline interpretation of p-values and R² makes output self-contained for non-technical reviewers.

| Dimension | Value |
|-----------|-------|
| Date range | December 2014 – July 2016 (20 months) |
| Product categories | Bikes, Accessories, Clothing (3 top-level; 15 sub-categories) |
| Geographic markets | United States (multi-state), France, Germany |
| Monthly revenue range | $10K – $640K |
| Largest single market | California (highest US revenue concentration) |
