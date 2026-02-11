# GePT Developer Setup Guide

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/MarcusFranz/gept-foundations.git
cd gept-foundations
```

### 2. Set Up Environment Variables
```bash
cp .env.example .env
```

Edit `.env` with the actual credentials (request via a secure channel):
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=osrs_data
DB_USER=osrs_user
DB_PASS=<secure_value>
GRAFANA_PASSWORD=<secure_value>
```

Keep `.env` local only (it is gitignored). If you share a machine, lock it down:
```bash
chmod 600 .env
```

### 3. SSH Tunnel to Database
The database runs on a remote server. To connect locally:
```bash
ssh -i <ssh_key_path> -L 5432:localhost:5432 <ssh_user>@<host> -N &
```

You'll need the SSH key file from the repository maintainer.
To stop the background tunnel later:
```bash
pkill -f "ssh -i .* -L 5432:localhost:5432"
```

### 4. Install Python Dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Verify Database Connection
```bash
source .env
python -c "from src.db_utils import get_simple_connection; c = get_simple_connection(); print('Connected!'); c.close()"
```

You can also verify with `psql` (avoids writing the password to shell history):
```bash
source .env
PGPASSWORD="$DB_PASS" psql -h localhost -U "$DB_USER" -d "$DB_NAME" -c "SELECT now();"
```

---

## Database Access

### Connection Details
- **Host**: `localhost` (via SSH tunnel)
- **Port**: `5432`
- **Database**: `osrs_data`
- **User**: `osrs_user`

### Key Tables
| Table | Description | Size |
|-------|-------------|------|
| `price_data_5min` | 5-minute price snapshots (2021-2026) | ~6GB compressed |
| `prices_latest_1m` | 1-minute tick prices | ~2GB |
| `player_counts` | OSRS player counts | ~22MB |
| `predictions` | Model predictions | ~500MB |
| `items` | Item metadata | ~1MB |

### Example Queries
```sql
-- Recent prices for Cannonball (item_id=2)
SELECT timestamp, avg_high_price, avg_low_price
FROM price_data_5min
WHERE item_id = 2
ORDER BY timestamp DESC
LIMIT 10;

-- Player count trend
SELECT timestamp, player_count
FROM player_counts
WHERE timestamp > NOW() - INTERVAL '1 day'
ORDER BY timestamp;
```

---

## Project Structure

```
gept/
├── packages/
│   ├── inference/
│   │   ├── src/            # Inference pipeline code
│   │   ├── scripts/        # Backups + migrations + ops helpers
│   │   ├── Dockerfile      # GHCR image build
│   │   └── .env.example    # Environment template (local)
│   ├── collectors/         # Data collection + monitoring (Docker Compose)
│   ├── engine/             # Recommendation Engine API
│   └── web/                # Vercel-deployed site
└── infra/systemd/           # Server systemd units (docker-based)
```

---

## Running Code

### Load Environment Variables
Always load your `.env` first:
```bash
source .env
# or
export $(cat .env | xargs)
```

### Run Inference
```bash
source .env
python src/pipeline/run_patchtst_inference.py --model-path /path/to/best_model.pt
```

### Run Tests
```bash
source .env
pytest tests/
```

---

## Monitoring

### Dashboards (VPN/SSH required)
- **Status Dashboard**: internal URL (redacted)
- **Grafana**: internal URL (redacted)
- **Prometheus**: internal URL (redacted)

---

## Common Issues

### "DB_PASS environment variable required"
You forgot to load your `.env`:
```bash
source .env
```

### "Connection refused on port 5432"
SSH tunnel not running:
```bash
ssh -i .secrets/oracle_key.pem -L 5432:localhost:5432 ubuntu@<server_ip> -N &
```

### Missing SSH key
Ask Marcus for the `.secrets/oracle_key.pem` file.

---

## Dependency Management

This project uses [pip-tools](https://pip-tools.readthedocs.io/) for deterministic dependency management.

### File Structure
- `requirements.in` - Source constraints (what we want)
- `requirements.txt` - Pinned lockfile (what gets installed)
- `../collectors/requirements.in` - Collector source constraints
- `../collectors/requirements.txt` - Collector pinned lockfile

### Installing Dependencies
```bash
# Install pinned versions (reproducible)
pip install -r requirements.txt
```

### Updating Dependencies
```bash
# Install pip-tools
pip install pip-tools

# Update root dependencies
pip-compile requirements.in --upgrade

# Update collectors dependencies
pip-compile ../collectors/requirements.in -o ../collectors/requirements.txt --upgrade

# Sync your environment
pip-sync requirements.txt
```

### Adding New Dependencies
1. Add the package to `requirements.in` (with minimum version constraint)
2. Run `pip-compile requirements.in`
3. Commit both `.in` and `.txt` files

---

## Security Notes

- **Never commit `.env`** - it's in `.gitignore`
- **Never commit SSH keys** - keep in `.secrets/`
- **Never hardcode credentials** - always use `os.environ['DB_PASS']`

---

## Contact

Questions? Reach out to the repository maintainer.
