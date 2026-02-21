# Walk-forward OOS Validation (25 Size–B/M Portfolios)

- Scheme: expanding-window, 1-step-ahead residuals (burn-in = 120 months)
- Metric: pricing error residual = realized excess return − model-implied return

## Carhart
- Avg portfolio MAE: 0.0136384
- Avg portfolio RMSE: 0.0187837
- OOS alpha dispersion (std of mean residuals): 0.0012448
- Mean |OOS alpha| (mean residual): 0.000796058

## FF3
- Avg portfolio MAE: 0.0136508
- Avg portfolio RMSE: 0.0187716
- OOS alpha dispersion (std of mean residuals): 0.00122647
- Mean |OOS alpha| (mean residual): 0.000774291

### Interpretation
Lower MAE/RMSE indicates improved out-of-sample pricing. Reduced dispersion of mean residuals suggests alphas are closer to zero across portfolios.

### Files
- oos_walkforward_25_5x5_by_portfolio.csv
- oos_walkforward_25_5x5_summary.csv
- oos_delta_mae_by_portfolio.png