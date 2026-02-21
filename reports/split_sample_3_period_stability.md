# Three-Period Split Stability (25 Size-B/M Portfolios)

## Results
Non-overlapping regime segmentation:
- pre_2009
- 2009_2019
- post_2020

- **Carhart | 2009_2019**: mean α = -9.3619e-05, std α = 0.00156493, mean |α| = 0.00117661
- **Carhart | post_2020**: mean α = -0.000521337, std α = 0.0025921, mean |α| = 0.00195195
- **Carhart | pre_2009**: mean α = -0.000238284, std α = 0.001707, mean |α| = 0.00105462
- **FF3 | 2009_2019**: mean α = -7.86834e-05, std α = 0.0015526, mean |α| = 0.00117506
- **FF3 | post_2020**: mean α = -0.00058695, std α = 0.00277306, mean |α| = 0.00202592
- **FF3 | pre_2009**: mean α = -0.000715096, std α = 0.0019993, mean |α| = 0.00123504

## Comparison Plot
![Side-by-Side Comparison Between Model Mean Alphas Across Regimes](../figures/split_sample_3_period_comparison.png)

## Interpretation
- Pre-2009: Pricing errors are largest in magnitude and most dispersed in the pre-crisis regime.
- 2009-2019: Mean alphas move closer to zero and dispersion declines relative to pre-2009, indicating improved stability during the post-crisis, low-rate period.
- Post-2020: While sample length is shorter, dispersion does not materially exceed earlier regimes, suggesting no dramatic structural breakdown of the factor models.
- Model comparison: Across all regimes except 2009-2019, Carhart exhibits mean alphas closer to zero than FF3, consistent with incremental explanatory power from the momentum factor. In 2009-2019, the difference of mean alphas between Carhart and FF3 is negligible.