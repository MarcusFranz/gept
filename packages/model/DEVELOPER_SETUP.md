# GePT Developer Setup Guide

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/MarcusFranz/gept.git
cd gept
```

### 2. Set Up Environment Variables
Create a local `.env` in `packages/model` with the values provided by the maintainer:
```bash
cd packages/model
cat <<'EOF' > .env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=osrs_data
DB_USER=osrs_user
DB_PASS=<provided_secure_value>
GRAFANA_PASSWORD=<provided_secure_value>
EOF
```

### 3. SSH Tunnel to Database
The database runs on a remote server. To connect locally:
```bash
ssh -i .secrets/your-key.pem -L 5432:localhost:5432 ubuntu@your-inference-host -N &
```

You'll need the SSH key file in `.secrets/` from the maintainer.

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
packages/model/
├── src/                    # Core ML/prediction code
│   ├── db_utils.py         # Database connections (use this!)
│   ├── feature_engine.py   # Feature computation
│   ├── predictor.py        # Prediction logic
│   └── ...
├── collectors/             # Data collection services
├── training/               # Model training scripts
├── scripts/                # Utility scripts
└── .env                    # Local env file (not committed)
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
python run_inference.py
```

### Run Tests
```bash
source .env
pytest tests/
```

---

## Monitoring

Dashboards are available on internal hosts. Request the current URLs and access details from the maintainer.

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
ssh -i .secrets/your-key.pem -L 5432:localhost:5432 ubuntu@your-inference-host -N &
```

### Missing SSH key
Request the `.secrets/` SSH key from the maintainer.

---

## Dependency Management

This project uses [pip-tools](https://pip-tools.readthedocs.io/) for deterministic dependency management.

### File Structure
- `requirements.in` - Source constraints (what we want)
- `requirements.txt` - Pinned lockfile (what gets installed)
- `collectors/requirements.in` - Collector source constraints
- `collectors/requirements.txt` - Collector pinned lockfile

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
pip-compile collectors/requirements.in -o collectors/requirements.txt --upgrade

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

Questions? Reach out to Marcus.
