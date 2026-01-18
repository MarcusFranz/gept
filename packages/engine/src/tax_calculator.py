"""Grand Exchange tax calculation utilities.

Implements GE tax rules as of May 2025 update:
- Rate: 2% (updated from 1% on 29 May 2025)
- Rounding: Always rounds DOWN to nearest whole number
- Floor: Items sold below 50gp have 0 tax
- Cap: Maximum 5,000,000gp tax per item (not per transaction)
"""

import logging

logger = logging.getLogger(__name__)


# GE Tax Constants (May 2025 update)
TAX_RATE = 0.02  # 2%
TAX_FLOOR_GP = 50  # No tax on items sold below 50gp
TAX_CAP_GP = 5_000_000  # Maximum tax per item


def calculate_tax(sell_price: int, qty: int = 1) -> int:
    """Calculate Grand Exchange tax for a sale.

    Args:
        sell_price: Price per item in gold pieces
        qty: Quantity of items being sold

    Returns:
        Total tax in gold pieces (always rounded down)

    Tax Rules (May 2025):
        - 2% tax rate (updated from 1%)
        - Rounds DOWN to nearest whole number
        - Items below 50gp: 0 tax
        - Maximum 5,000,000gp tax per item

    Examples:
        >>> calculate_tax(49, 1)  # Below floor
        0
        >>> calculate_tax(50, 1)  # At floor
        1
        >>> calculate_tax(100, 1)  # Standard
        2
        >>> calculate_tax(149, 1)  # Rounds down: 2.98 -> 2
        2
        >>> calculate_tax(150, 1)  # Rounds down: 3.0
        3
        >>> calculate_tax(300_000_000, 1)  # Hits cap
        5_000_000
        >>> calculate_tax(1000, 100)  # Bulk calculation
        2000
    """
    if sell_price < TAX_FLOOR_GP:
        return 0

    # Calculate tax per item (rounded down)
    tax_per_item = int(sell_price * TAX_RATE)

    # Apply cap per item
    tax_per_item = min(tax_per_item, TAX_CAP_GP)

    # Total tax for quantity
    return tax_per_item * qty


def calculate_net_proceeds(sell_price: int, qty: int = 1) -> int:
    """Calculate net proceeds after GE tax.

    Args:
        sell_price: Price per item in gold pieces
        qty: Quantity of items being sold

    Returns:
        Net gold received after tax

    Examples:
        >>> calculate_net_proceeds(100, 1)
        98
        >>> calculate_net_proceeds(50, 1)
        49
        >>> calculate_net_proceeds(49, 1)  # No tax
        49
    """
    gross = sell_price * qty
    tax = calculate_tax(sell_price, qty)
    return gross - tax


def calculate_flip_profit(buy_price: int, sell_price: int, qty: int = 1) -> int:
    """Calculate profit from a flip after GE tax.

    Args:
        buy_price: Price paid per item
        sell_price: Price sold per item
        qty: Quantity flipped

    Returns:
        Net profit after tax (can be negative)

    Examples:
        >>> calculate_flip_profit(98, 102, 1)
        2
        >>> calculate_flip_profit(100, 105, 10)
        20
    """
    cost = buy_price * qty
    proceeds = calculate_net_proceeds(sell_price, qty)
    return proceeds - cost


def effective_tax_rate(sell_price: int) -> float:
    """Calculate effective tax rate for a given price.

    Due to rounding down, the effective rate is often less than 2%.

    Args:
        sell_price: Price per item in gold pieces

    Returns:
        Effective tax rate as a decimal (e.g., 0.0196 for 1.96%)

    Examples:
        >>> round(effective_tax_rate(100), 4)
        0.02
        >>> round(effective_tax_rate(149), 4)
        0.0134
        >>> effective_tax_rate(49)
        0.0
    """
    if sell_price < TAX_FLOOR_GP:
        return 0.0

    tax = calculate_tax(sell_price, 1)
    return tax / sell_price


def get_tax_info() -> dict:
    """Get current tax configuration.

    Returns:
        Dictionary with tax constants and effective date

    Examples:
        >>> info = get_tax_info()
        >>> info['rate']
        0.02
        >>> info['floor_gp']
        50
    """
    return {
        "rate": TAX_RATE,
        "floor_gp": TAX_FLOOR_GP,
        "cap_gp": TAX_CAP_GP,
        "effective_date": "2025-05-29",
        "notes": "Tax rate updated from 1% to 2% on 29 May 2025",
    }
