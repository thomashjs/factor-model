# Model Diagnostics (25 Size-B/M Portfolios)

Cross-sectional summary statistics:

## Carhart
- Mean alpha t-stat (HAC): 0.021
- Fraction |t_alpha| > 1.96: 0.240
- Mean HAC/OLS SE ratio: 0.929
- Fraction residual autocorr (p<0.05): 0.640

## FF3
- Mean alpha t-stat (HAC): -0.508
- Fraction |t_alpha| > 1.96: 0.360
- Mean HAC/OLS SE ratio: 0.969
- Fraction residual autocorr (p<0.05): 0.640

## Interpretation
While residual autocorrelation is present across many portfolios (≈64%), HAC adjustments do not materially increase standard errors relative to OLS. Cross-sectional alpha significance is meaningfully reduced under the Carhart specification (24% vs 36%), indicating that the inclusion of momentum improves pricing performance, though non-zero alphas remain.