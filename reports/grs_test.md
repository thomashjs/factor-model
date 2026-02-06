# GRS joint test of alphas (25 Size-B/M portfolios)

This report tests whether the 25 portfolio alphas are jointly zero under each model using the classic Gibbons–Ross–Shanken (GRS) F-test.

**Important:** This is the *classic* GRS test based on OLS residual covariance (no HAC). Your panel scripts use Newey–West HAC for per-portfolio t-stats; GRS is reported here in the standard form used in many papers.

## FF3

- Sample: **1927-01** to **2025-11**
- T = 1187, N = 25, K = 3
- Factors: Mkt-RF, SMB, HML

**GRS F-stat:** 3.27192  (df1 = 25, df2 = 1159)
**p-value:** 1.21812e-07

## Carhart (4-factor)

- Sample: **1927-01** to **2025-11**
- T = 1187, N = 25, K = 4
- Factors: Mkt-RF, SMB, HML, Mom

**GRS F-stat:** 2.85544  (df1 = 25, df2 = 1158)
**p-value:** 3.90717e-06

## Interpretation

Lower GRS statistics / higher p-values indicate the model's alphas are closer to jointly zero across the 25 portfolios. Comparing FF3 vs Carhart tells you whether adding Momentum materially reduces joint mispricing for these portfolios.
