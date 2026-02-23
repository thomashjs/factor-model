# 12–1 Momentum Cross-Sectional Test

## Signal Construction

Momentum is defined as cumulative returns from months $t-12$ to $t-2$, skipping the most recent month:


$$\text{MOM}_{i,t} = \prod_{j=t-12}^{t-2}(1 + r_{i,j}) - 1$$


Stocks are sorted each month on this signal to predict next-month returns.

---

## Portfolio Sort Results

* **Mean monthly W–L return:** 1.08%
* **Cumulative W–L growth:** 9.38× over the sample
* **Number of months:** 301

The winner–loser portfolio delivers economically meaningful returns, consistent with documented momentum effects.

---

## Fama–MacBeth Cross-Sectional Regression

Each month we estimate:

$$ R_{i,t+1} = \alpha_t + \gamma_t \cdot \text{rank}(MOM_{i,t}) + \varepsilon_{i,t+1} $$

or

$$ R_{i,t+1} = \alpha_t + \gamma_t \cdot MOM_{i,t} + \varepsilon_{i,t+1} $$


Results (Momentum, Not Rank):

* **Average $\gamma$:** 0.00544
* **t-statistic:** 2.35
* **Std($\gamma_t$):** 0.04023
* **Std. error of $\gamma$:** 0.00232
* **Months:** 301

The positive and statistically significant $\gamma$ indicates that higher-ranked momentum stocks earn higher next-month returns.

---

## Consistency Check

* **Implied spread from $\gamma$:** ≈ 0.54%–0.67% per month
* **Actual W–L mean:** 1.08% per month

The regression-implied spread is smaller because Fama–MacBeth estimates a linear relationship, while the decile portfolio isolates extreme tails where momentum effects are strongest.

---

## Conclusion

Within this Yahoo-based U.S. equity universe:

* The 12–1 momentum characteristic predicts next-month returns.
* The effect is economically meaningful (~1% per month long–short).
* Cross-sectional regression confirms statistically significant predictability.
* Portfolio and regression evidence are internally consistent once return treatment is aligned.

---