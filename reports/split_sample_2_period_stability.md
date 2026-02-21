# Split-Sample Stability (25 Size-B/M Portfolios)

Models estimated separately across subperiods.

## Results
- **Carhart | post_2009**: mean α = -8.81027e-06, std α = 0.00137638, mean |α| = 0.000799267
- **Carhart | post_2020**: mean α = -0.000521337, std α = 0.0025921, mean |α| = 0.00195195
- **Carhart | pre_2009**: mean α = -0.000238284, std α = 0.001707, mean |α| = 0.00105462
- **Carhart | pre_2020**: mean α = -0.00021469, std α = 0.00160781, mean |α| = 0.00100153
- **FF3 | post_2009**: mean α = -3.08077e-05, std α = 0.00139587, mean |α| = 0.000804141
- **FF3 | post_2020**: mean α = -0.00058695, std α = 0.00277306, mean |α| = 0.00202592
- **FF3 | pre_2009**: mean α = -0.000715096, std α = 0.0019993, mean |α| = 0.00123504
- **FF3 | pre_2020**: mean α = -0.000589922, std α = 0.00186208, mean |α| = 0.00117104

## Comparison
![Side-by-Side Comparison Between Model Mean Alphas Across Regimes](../figures/split_sample_2_period_comparison.png)


## Interpretation
- Pre-2009 vs Post-2009: Mean alphas are economically larger in magnitude before 2009 and move closer to zero after 2009 for both models. Cross-sectional dispersion also declines post-2009, indicating improved pricing stability in the post-crisis regime.
- Post-2020: Results should be interpreted cautiously due to shorter sample length (~60+ months), but no structural explosion in alpha dispersion is observed.
- FF3 vs Carhart: Carhart consistently produces mean alphas closer to zero than FF3, particularly in the pre-2009 subsample, suggesting that the momentum factor absorbs part of the cross-sectional pricing error.