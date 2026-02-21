# Fama–French and Carhart Factor Model Empirical Analysis

This project replicates and evaluates the Fama–French 3-Factor (FF3) and Carhart 4-Factor models using the 25 Size–Book-to-Market portfolios from the Kenneth French Data Library.

The analysis goes beyond replication to assess:
- Joint pricing restrictions (GRS test)
- Time-series stability (rolling regressions)
- Regime robustness (pre/post 2009 and post-2020 splits)
- Regression diagnostics (HAC vs OLS, residual autocorrelation)

---

## Data

Source: **Kenneth French Data Library**:

- FF3 monthly factors (Mkt–RF, SMB, HML, RF)
- Momentum factor (Mom)
- 25 Size–Book-to-Market portfolio monthly returns

Data are downloaded programmatically and fully reproducible via scripts in src/. Processed data are not committed.

---

## Methodology

- Time-series regressions per portfolio
- Newey–West (HAC) standard errors (3 lags for baseline regressions; 12 lags for diagnostic checks)
- GRS joint alpha test
- 60-month rolling regressions
- Two-period and three-period regime splits
- Residual autocorrelation diagnostics

---

## Repository Structure
    .
    ├── src/ # Data pipelines, regressions, and plotting scripts
    ├── tests/ # Correctness tests
    ├── notebooks/ # Exploratory analysis and sanity checks
    ├── reports/ # Regression outputs and written interpretation
    ├── figures/ # Generated plots
    ├── data/
    │ ├── raw/ # Downloaded raw data (ignored by git)
    │ └── processed/# Processed datasets (ignored by git)
    └── README.md

---
## Key Findings

- The GRS test rejects joint zero alphas under both models, but Carhart reduces cross-sectional mispricing relative to FF3.

- The fraction of significant alphas declines from **36% (FF3)** to **24% (Carhart)**.

- Rolling regressions show no persistent structural drift in mean alpha.

- Pricing errors are largest pre-2009 and more stable post-crisis.

- Residual autocorrelation is present (~64% of portfolios), but HAC adjustments do not materially inflate standard errors.

**Conclusion**:
Momentum provides economically meaningful incremental explanatory power, improving stability and reducing pricing errors, though neither model fully eliminates mispricing.

---

## Key Outputs

- Portfolio-level regression results (reports/*_results.csv)
- FF3 vs Carhart comparison tables
- GRS joint test results
- Rolling alpha visualizations
- Regime stability analysis
- Diagnostic summary tables

<!--
---

## Robustness and Diagnostic Summary

We conduct a series of joint, time-series, and regime-based robustness checks to evaluate the pricing performance and statistical reliability of the Fama–French 3-factor (FF3) and Carhart 4-factor models across the 25 Size–Book-to-Market portfolios.

The Gibbons–Ross–Shanken (GRS) test rejects the joint null of zero alphas under both models, indicating the presence of cross-sectional pricing errors. However, the Carhart specification exhibits a lower GRS statistic and higher p-value relative to FF3, suggesting that the inclusion of the momentum factor reduces joint mispricing.

Rolling 60-month regressions show that mean cross-sectional alpha remains centered near zero over time for both models, with no persistent structural drift. Confidence bands for the cross-sectional mean alpha indicate that pricing errors fluctuate but do not exhibit sustained directional bias. Dispersion of alphas declines in the post-2009 regime relative to the pre-crisis period, consistent with improved stability in the cross-section.

Split-sample analysis further supports this pattern. Mean alphas are economically larger and more dispersed in the pre-2009 period, while both models display improved pricing performance in 2009–2019. Across regimes, the Carhart model consistently yields mean alphas closer to zero and a lower fraction of statistically significant pricing errors, indicating incremental explanatory power from momentum. In the full sample, the fraction of significant alphas declines from 36% under FF3 to 24% under Carhart.

Regression diagnostics indicate that residual autocorrelation is present in a majority of portfolios (≈64%), justifying the use of Newey–West HAC standard errors. However, HAC-adjusted standard errors are close in magnitude to OLS estimates (mean HAC/OLS ratio ≈ 0.93–0.97), suggesting that serial dependence does not materially distort inference.

Overall, while neither model fully eliminates cross-sectional pricing errors, the Carhart specification demonstrates improved stability and reduced mispricing relative to FF3, particularly in the pre-crisis regime. The combined evidence from joint tests, rolling estimation, regime splits, and diagnostic checks supports the conclusion that momentum provides economically meaningful, though incomplete, incremental explanatory power in this cross-section. -->
---

## Technical Stack
Python | pandas | NumPy | statsmodels | matplotlib

---

## Reproducibility

```
pip install -r requirements.txt
python -m src.<script_name>
```

All results can be reproduced from raw data.

---

## Notes

- This project is intended as a research replication and learning exercise.
- The focus is on clarity, reproducibility, and correct econometric interpretation rather than optimization.

---

## References

- Fama, E. F., & French, K. R. (1993). *Common risk factors in the returns on stocks and bonds.*
- Carhart, M. M. (1997). *On persistence in mutual fund performance.*
- Kenneth French Data Library