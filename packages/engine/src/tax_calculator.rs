const tax_div: u32 = 50;
const tax_floor: u32 = 50;
const tax_cap: u32 = 5_000_000;

fn calculate_tax(sell_price: u32, qty: u32) -> u32 {
    