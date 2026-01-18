#!/bin/bash
# Deploy Continuous Training Pipeline to Hydra Server
# Usage: ./scripts/deploy_continuous_training.sh

set -e

# Configuration
HYDRA_HOST="${HYDRA_HOST:-ubuntu@10.0.0.146}"
HYDRA_KEY="${HYDRA_KEY:-.secrets/wsl_key.pem}"
REMOTE_DIR="/home/ubuntu/gept"

echo "============================================"
echo "Deploying Continuous Training Pipeline"
echo "============================================"
echo "Host: $HYDRA_HOST"
echo ""

# Check SSH key
if [ ! -f "$HYDRA_KEY" ]; then
    echo "Error: SSH key not found at $HYDRA_KEY"
    exit 1
fi

# Sync code
echo "1. Syncing code to Hydra..."
rsync -avz --progress \
    -e "ssh -i $HYDRA_KEY" \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'models/' \
    --exclude 'data_cache/' \
    --exclude '.secrets' \
    --exclude 'node_modules' \
    ./ "$HYDRA_HOST:$REMOTE_DIR/"

# Run database migration
echo ""
echo "2. Running database migration..."
ssh -i "$HYDRA_KEY" "$HYDRA_HOST" << 'MIGRATION'
cd /home/ubuntu/gept
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gept

# Check if we can connect to database
if ! psql -h localhost -U osrs_user -d osrs_data -c "SELECT 1" > /dev/null 2>&1; then
    echo "Warning: Cannot connect to database. Ensure db-tunnel.service is running."
    echo "You may need to run the migration manually later."
else
    echo "Running migration..."
    psql -h localhost -U osrs_user -d osrs_data -f migrations/003_continuous_training.sql || {
        echo "Migration failed or already applied (this is OK if tables exist)"
    }
fi
MIGRATION

# Install systemd services
echo ""
echo "3. Installing systemd services..."
ssh -i "$HYDRA_KEY" "$HYDRA_HOST" << 'SYSTEMD'
cd /home/ubuntu/gept

# Copy service files
sudo cp config/systemd/gept-scheduler.service /etc/systemd/system/
sudo cp config/systemd/gept-data-sync.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services (but don't start yet)
sudo systemctl enable gept-data-sync.service
sudo systemctl enable gept-scheduler.service

echo "Services installed:"
echo "  - gept-data-sync.service"
echo "  - gept-scheduler.service"
SYSTEMD

# Create data cache directory
echo ""
echo "4. Creating data cache directory..."
ssh -i "$HYDRA_KEY" "$HYDRA_HOST" << 'CACHE'
mkdir -p /home/ubuntu/gept/data_cache/price_data
mkdir -p /home/ubuntu/gept/data_cache/metadata
chmod -R 755 /home/ubuntu/gept/data_cache
CACHE

echo ""
echo "============================================"
echo "Deployment Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Ensure db-tunnel.service is running:"
echo "     sudo systemctl status db-tunnel"
echo ""
echo "  2. Start data sync service:"
echo "     sudo systemctl start gept-data-sync"
echo "     sudo systemctl status gept-data-sync"
echo ""
echo "  3. Wait for initial data sync (~5-10 minutes)"
echo "     journalctl -u gept-data-sync -f"
echo ""
echo "  4. Start scheduler (optional - starts training):"
echo "     sudo systemctl start gept-scheduler"
echo "     sudo systemctl status gept-scheduler"
echo ""
echo "  5. Monitor training:"
echo "     journalctl -u gept-scheduler -f"
echo ""
