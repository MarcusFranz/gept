# Grand Exchange Tax Rules

## Current Rules (May 2025 Update)

As of **29 May 2025**, the Grand Exchange tax was updated from 1% to 2%.

### Tax Calculation Rules

1. **Rate**: 2% of the sell price
2. **Rounding**: Always rounds DOWN to the nearest whole number
3. **Floor**: Items sold below 50gp have **0 tax**
4. **Cap**: Maximum **5,000,000gp** tax per item (not per transaction)
5. **Per-Item Calculation**: Tax is calculated per item, then multiplied by quantity

### Formula

```python
if sell_price < 50:
    tax = 0
else:
    tax_per_item = min(int(sell_price * 0.02), 5_000_000)
    total_tax = tax_per_item * quantity
```

### Examples

| Sell Price | Tax | Net Proceeds | Notes |
|------------|-----|--------------|-------|
| 49gp | 0gp | 49gp | Below floor |
| 50gp | 1gp | 49gp | Boundary case |
| 100gp | 2gp | 98gp | Standard 2% |
| 149gp | 2gp | 147gp | 2.98 rounds down to 2 |
| 150gp | 3gp | 147gp | Exact 3gp |
| 300M gp | 5M gp | 295M gp | Hits cap |

### Boundary Behavior

At the 50gp threshold, sellers receive the same net amount:
- Selling at 49gp: 49 - 0 = **49gp net**
- Selling at 50gp: 50 - 1 = **49gp net**

This creates a "dead zone" where prices between 49gp and 50gp yield identical proceeds.

## Historical Data Considerations

### Pre-May 2025 Data

Data collected before 29 May 2025 was subject to the **1% tax rate**. When training models or analyzing historical patterns:

1. **Training Data**: Models should ideally be trained on post-May 2025 data to reflect current tax environment
2. **Feature Engineering**: Historical returns and fill probabilities may differ due to tax change
3. **EV Calculations**: Pre-May data had different profitability characteristics

### Recommendations

- **Flag historical data**: Mark records pre-2025-05-29 in training pipelines
- **Separate models**: Consider separate models for pre/post tax change if using historical data
- **Feature adjustment**: If using pre-May data, adjust tax-dependent features to 2% rate

### Impact on Predictions

The tax increase from 1% to 2% affects:
- **Expected Value**: Net margins decreased by 1% across all trades
- **Optimal Offsets**: Breakeven offsets shifted higher
- **Fill Probabilities**: Trader behavior may have adapted to new tax rate

## Implementation

See `src/tax_calculator.py` for the reference implementation of all GE tax rules.

Tests are located in `tests/test_tax_calculator.py`.

## References

- [OSRS Wiki - Grand Exchange Tax](https://oldschool.runescape.wiki/w/Grand_Exchange#Tax)
- Issue #8: Comprehensive Tax Calculation Audit
