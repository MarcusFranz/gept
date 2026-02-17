# Database Migrations

This directory contains SQL migration scripts for the GePT prediction engine database.

## Applying Migrations

Migrations should be applied manually using `psql` or your preferred PostgreSQL client.

Run commands from the `packages/engine` directory so relative paths resolve correctly.

### Using psql

```bash
# Connect to the database
psql -v ON_ERROR_STOP=1 postgresql://[USER]:[PASS]@[HOST]:5432/osrs_data

# Apply a migration
\i migrations/001_create_trade_outcomes_table.sql
```

You can also use the engine's `DB_CONNECTION_STRING` environment variable:

```bash
psql -v ON_ERROR_STOP=1 "$DB_CONNECTION_STRING" -f migrations/001_create_trade_outcomes_table.sql
```

### Using psql from command line

```bash
psql -v ON_ERROR_STOP=1 postgresql://[USER]:[PASS]@[HOST]:5432/osrs_data -f migrations/001_create_trade_outcomes_table.sql
```

### Apply in Order

Apply migrations in numeric order:

```bash
ls -1 migrations/*.sql | sort
```

## Migration Files

- `001_create_trade_outcomes_table.sql` - Trade outcomes table
- `002_create_model_registry_table.sql` - Model registry table
- `003_create_recommendation_feedback_table.sql` - Recommendation feedback table
- `004_add_offset_pct_to_outcomes_and_feedback.sql` - Adds `offset_pct` column

## Migration Naming Convention

Files are named with the pattern: `NNN_description.sql`

- `NNN` - Sequential number (001, 002, etc.)
- `description` - Brief description using snake_case

## Notes

- Migrations are designed to be idempotent (safe to run multiple times)
- Each migration includes `CREATE TABLE IF NOT EXISTS` or similar safe constructs
- Always test migrations on a development database first
- Keep migrations focused on a single logical change
