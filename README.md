# Fama–French and Carhart Factor Model Replication

This project replicates and compares the Fama–French three-factor (FF3) model and the Carhart four-factor (FF3 + Momentum) model using the 25 Size–Book-to-Market portfolios from the Kenneth French Data Library.

The goal is to examine how the inclusion of the momentum factor affects estimated abnormal returns (alphas) and model fit across portfolios.

---

## Research Question

- How do portfolio alphas change when moving from FF3 to the Carhart model?
- Does momentum explain systematic return variation not captured by FF3?
- Which portfolios are most affected by the inclusion of momentum?

---

## Data

All data are sourced from the **Kenneth French Data Library**:

- FF3 monthly factors (Mkt–RF, SMB, HML, RF)
- Momentum factor (Mom)
- 25 Size–Book-to-Market portfolio monthly returns

Raw data are downloaded programmatically and processed locally.  
Processed data files are **not committed** to the repository and are reproducible via the provided scripts.

---

## Models

### Fama–French 3-Factor Model
Excess returns are regressed on:
- Market excess return (Mkt–RF)
- Size factor (SMB)
- Value factor (HML)

### Carhart 4-Factor Model
Extends FF3 by adding:
- Momentum factor (Mom)

All regressions are estimated as time-series regressions for each portfolio using **Newey–West (HAC) standard errors** with 3 lags.

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

## Key Outputs

- `reports/ff3_25_5x5_results.csv`  
  FF3 regression results for all 25 portfolios

- `reports/carhart_25_5x5_results.csv`  
  Carhart regression results for all 25 portfolios

- `reports/ff3_vs_carhart_comparison.csv`  
  Clean comparison table with alphas and ΔR²

- `reports/comparison_ff3_vs_carhart.md`  
  Written interpretation of the results

- `figures/alpha_ff3_vs_carhart.png`  
  Scatter plot of FF3 vs. Carhart alphas

- `figures/delta_r2_by_portfolio.png`  
  Change in R² by portfolio after adding momentum

---

## Main Findings (Summary)

- FF3 alphas generally shrink toward zero after adding the momentum factor.
- Momentum loadings are frequently statistically significant in the Carhart model.
- The reduction in FF3 alphas is systematic and strongest for portfolios with high momentum exposure.
- Increases in R² are mechanical but indicate that momentum explains additional variation in returns.

These results are consistent with the core findings of Carhart (1997) and highlight the importance of momentum as an additional factor in asset pricing.

---

## Reproducibility

To reproduce the analysis:

1. Create and activate a Python virtual environment
2. Install dependencies from `requirements.txt`
3. Run scripts in `src/` in logical order:
   - factor download and processing
   - portfolio return construction
   - FF3 and Carhart regressions
   - comparison and plotting scripts

All results can be regenerated from source.

---

## Notes

- This project is intended as a research replication and learning exercise.
- The focus is on clarity, reproducibility, and correct econometric interpretation rather than optimization.

---

## References

- Fama, E. F., & French, K. R. (1993). *Common risk factors in the returns on stocks and bonds.*
- Carhart, M. M. (1997). *On persistence in mutual fund performance.*
- Kenneth French Data Library