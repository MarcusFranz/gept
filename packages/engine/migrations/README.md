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

### Using psql from command line

```bash
psql postgresql://[USER]:[PASS]@[HOST]:5432/gept -f migrations/001_create_trade_outcomes_table.sql
```

Tip: run these commands from `packages/engine` or adjust the migration path.

If you want to avoid typing the password in the URL, you can use `PGPASSWORD`:

```bash
PGPASSWORD=your_password psql postgresql://[USER]@[HOST]:5432/gept -f migrations/001_create_trade_outcomes_table.sql
```

## Migration Files

- `001_create_trade_outcomes_table.sql` - Creates the `trade_outcomes` table for storing user-reported trade results

## Migration Naming Convention

Files are named with the pattern: `NNN_description.sql`

- `NNN` - Sequential number (001, 002, etc.)
- `description` - Brief description using snake_case

## Notes

- Migrations are designed to be idempotent (safe to run multiple times)
- Each migration includes `CREATE TABLE IF NOT EXISTS` or similar safe constructs
- Always test migrations on a development database first
- Keep migrations focused on a single logical change
