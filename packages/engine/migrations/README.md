# Database Migrations

This directory contains SQL migration scripts for the GePT prediction engine database.

## Applying Migrations

Migrations should be applied manually using `psql` or your preferred PostgreSQL client.

### Using psql

```bash
# Connect to the database
psql postgresql://[USER]:[PASS]@[HOST]:5432/gept

# Apply a migration
\i migrations/001_create_trade_outcomes_table.sql
```

You can also use the engine's `DB_CONNECTION_STRING` environment variable:

```bash
psql "$DB_CONNECTION_STRING" -f migrations/001_create_trade_outcomes_table.sql
```

### Using psql from command line

```bash
psql postgresql://[USER]:[PASS]@[HOST]:5432/gept -f migrations/001_create_trade_outcomes_table.sql
```

## Migration Files

- `001_create_trade_outcomes_table.sql` - Creates the `trade_outcomes` table for storing user-reported trade results
- `002_create_model_registry_table.sql` - Creates the `model_registry` table for tracking model runs/status
- `003_create_recommendation_feedback_table.sql` - Creates the `recommendation_feedback` table for structured feedback
- `004_add_offset_pct_to_outcomes_and_feedback.sql` - Adds `offset_pct` attribution to outcomes and feedback
- `005_add_model_id_to_trade_outcomes.sql` - Adds `model_id` attribution to trade outcomes for model comparisons

## Migration Naming Convention

Files are named with the pattern: `NNN_description.sql`

- `NNN` - Sequential number (001, 002, etc.)
- `description` - Brief description using snake_case

## Notes

- Migrations are designed to be idempotent (safe to run multiple times)
- Each migration includes `CREATE TABLE IF NOT EXISTS` or similar safe constructs
- Always test migrations on a development database first
- Keep migrations focused on a single logical change
