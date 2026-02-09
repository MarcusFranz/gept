// packages/web/src/lib/ge-tax.ts
//
// GE tax/profit helpers used by both client components and server routes.
// Keep this file free of Node/browser-specific APIs.

// GE tax rules (as of 2025-05-29 engine update):
// - 2% tax rate, rounded down
// - No tax when sell price < 50 gp
// - Cap: 5,000,000 gp tax per item (not per transaction)
const TAX_DIVISOR = 50; // 2% == 1/50
const TAX_FLOOR_GP = 50;
const TAX_CAP_GP = 5_000_000;

export function calculateGeTaxPerItem(sellPrice: number): number {
  const price = Math.trunc(sellPrice);
  if (!Number.isFinite(price) || price <= 0) return 0;
  if (price < TAX_FLOOR_GP) return 0;

  // Use integer division instead of floats (avoids rounding surprises).
  const tax = Math.floor(price / TAX_DIVISOR);
  return Math.min(tax, TAX_CAP_GP);
}

export function calculateGeTax(sellPrice: number, qty: number = 1): number {
  const q = Math.max(0, Math.trunc(qty));
  if (q === 0) return 0;
  return calculateGeTaxPerItem(sellPrice) * q;
}

export function calculateNetProceeds(sellPrice: number, qty: number = 1): number {
  const q = Math.max(0, Math.trunc(qty));
  const price = Math.max(0, Math.trunc(sellPrice));
  const gross = price * q;
  return gross - calculateGeTax(price, q);
}

export function calculateFlipProfit(buyPrice: number, sellPrice: number, qty: number = 1): number {
  const q = Math.max(0, Math.trunc(qty));
  const buy = Math.max(0, Math.trunc(buyPrice));
  const cost = buy * q;
  const proceeds = calculateNetProceeds(sellPrice, q);
  return proceeds - cost;
}

