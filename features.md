# Feature Catalog - FictiPay Churn Prediction

This document catalog lists the 15 behavioral and demographic features engineered for the FictiPay churn prediction model. All features are constructed exclusively from the observation window (2024-01-01 to 2024-03-31).

---

### 1. `tenure_days`
- **Type:** Demographic / Relationship
- **Definition:** Number of days between `ACCOUNT_OPEN_DATE` and the end of the observation window (`2024-03-31`).
- **Hypothesis:** Customers who have been with FictiPay longer have higher loyalty and established habits, which lowers their risk of churn compared to newly acquired customers.

### 2. `gender_encoded`
- **Type:** Demographic
- **Definition:** Numeric mapping of GENDER (Male = 1, Female = 0, Unknown/Missing = -1).
- **Hypothesis:** Gender-based preferences and wallet usage patterns may differ across Emergingland, serving as a demographic control.

### 3. `region_encoded`
- **Type:** Demographic
- **Definition:** Label-encoded index of the customer's `REGION`.
- **Hypothesis:** Regional differences in economic development, merchant acceptance, and cash-out agent density affect transaction frequencies and churn.

### 4. `recency_days`
- **Type:** Recency (Behavioral)
- **Definition:** Days between the customer's last initiated (outgoing) transaction and the end of the observation window (`2024-03-31`). If no transaction occurred, defaults to 90 days.
- **Hypothesis:** A long period of inactivity is the strongest indicator of disengagement. The larger the gap, the more likely the customer has abandoned the wallet.

### 5. `trx_count_30d`
- **Type:** Frequency (Behavioral)
- **Definition:** Number of outgoing transactions initiated by the customer in the last 30 days of the window (2024-03-02 to 2024-03-31).
- **Hypothesis:** Captures the customer's active daily/weekly transaction habits. High frequency in the last month indicates strong current adoption.

### 6. `trx_count_90d`
- **Type:** Frequency (Behavioral)
- **Definition:** Total number of outgoing transactions initiated by the customer in the entire 90-day window.
- **Hypothesis:** Captures the long-term frequency and transactional footprint of the customer.

### 7. `total_amt_90d`
- **Type:** Monetary (Behavioral)
- **Definition:** Sum of transaction amounts for all outgoing transactions initiated by the customer.
- **Hypothesis:** High monetary spend indicates high transactional value and commitment to the wallet. High spenders are more valuable and less likely to churn.

### 8. `mean_amt_30d`
- **Type:** Monetary (Behavioral)
- **Definition:** Average transaction amount for outgoing transactions initiated in the last 30 days.
- **Hypothesis:** Captures the average ticket size of recent transactions. A drop in ticket size can signal disengagement.

### 9. `total_amt_in_90d`
- **Type:** Monetary (Incoming)
- **Definition:** Sum of transaction amounts received by the customer (P2P receive, CashIn) in the 90 days.
- **Hypothesis:** Captures the incoming funds. Customers who receive regular peer transfers or perform cash-ins have funds to spend, which reduces churn risk.

### 10. `last_balance`
- **Type:** Balance (Behavioral)
- **Definition:** Available balance on `2024-03-31` (or the last available date in the observation window).
- **Hypothesis:** Customers with a non-zero balance are less likely to churn because they still have money in their wallet. A zero balance indicates a drained wallet.

### 11. `mean_balance_90d`
- **Type:** Balance (Behavioral)
- **Definition:** Average daily available balance over the 90-day window.
- **Hypothesis:** High average balance indicates the wallet is being used as a store of value, signaling high trust and lock-in.

### 12. `balance_std_90d`
- **Type:** Balance (Behavioral)
- **Definition:** Standard deviation of the daily available balance over the 90-day window.
- **Hypothesis:** High volatility (standard deviation) suggests active cash flow (frequent deposits and spending), which correlates with high activity and low churn.

### 13. `balance_trend`
- **Type:** Balance (Behavioral)
- **Definition:** Ratio of mean available balance in the last 30 days to the mean available balance in the first 30 days. Formula: `(mean_bal_last_30d + 1) / (mean_bal_first_30d + 1)`.
- **Hypothesis:** A declining trend (ratio < 1) indicates the customer is gradually draining their wallet, a clear signal of imminent churn. An increasing or stable trend is protective.

### 14. `zero_balance_days`
- **Type:** Balance (Behavioral)
- **Definition:** Number of days where the daily available balance was zero or negative.
- **Hypothesis:** Frequent zero-balance days indicate that the customer is not keeping money in the wallet, which points to transactional-only use and a higher risk of churn.

### 15. `weekend_trx_ratio`
- **Type:** Specialty / Profile
- **Definition:** Ratio of outgoing transactions on weekends (Friday & Saturday in Bangladesh) to total outgoing transactions.
- **Hypothesis:** Helps profile the customer's lifestyle. High weekend usage indicates personal/retail use rather than business/professional use, which has different churn dynamics.

---

### 16. `service_diversity`
- **Type:** Specialty / Adoption
- **Definition:** Number of unique transaction types (`TRX_TYPE`) the customer initiated (e.g. P2P, BillPay, MerchantPay, CashOut).
- **Hypothesis:** Customers who use multiple different services (high diversity) are deeper in the FictiPay ecosystem and have much higher switching costs, making them highly sticky.

### 17. `velocity_freq_30_90`
- **Type:** Frequency (Behavioral)
- **Definition:** Ratio of transaction count in the last 30 days to transaction count in the 90 days. Formula: `trx_count_30d / (trx_count_90d + 1)`.
- **Hypothesis:** Captures the velocity of activity. A ratio significantly below 0.33 indicates a recent slowdown in activity, signaling disengagement and churn.

### 18. Type Ratios (`p2p_ratio`, `merchant_pay_ratio`, `bill_pay_ratio`, `cashout_ratio`)
- **Type:** Behavioral (Service mix)
- **Definition:** Ratio of specific transaction type amount to total outgoing amount.
- **Hypothesis:** Certain transaction types are more anchoring. For example, high `bill_pay_ratio` signals that the customer uses FictiPay for regular utilities, which anchors them to the platform.
