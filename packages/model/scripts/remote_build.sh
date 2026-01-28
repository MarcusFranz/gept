#!/bin/bash
# Remote Dataset Builder
# Runs on Hydra, executes build on Ampere, transfers dataset back
#
# Usage: ./remote_build.sh <recipe_name> [output_dir]

set -e

RECIPE_NAME=${1:-baseline}
OUTPUT_DIR=${2:-/workspace/datasets}
AMPERE_HOST="ubuntu@150.136.170.128"
AMPERE_PATH="/home/ubuntu/gept/packages/model"
SSH_KEY_SOURCE="/root/.ssh/oracle_key.pem"
SSH_KEY="/tmp/oracle_key.pem"

# Copy SSH key to writable location and set permissions
cp $SSH_KEY_SOURCE $SSH_KEY
chmod 600 $SSH_KEY

# Configure SSH to skip host key verification (safe for internal servers)
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Create empty known_hosts to avoid warnings
touch ~/.ssh/known_hosts
chmod 600 ~/.ssh/known_hosts

cat > ~/.ssh/config <<EOF
Host 150.136.170.128
    StrictHostKeyChecking no
    UserKnownHostsFile ~/.ssh/known_hosts
    LogLevel ERROR
    IdentityFile $SSH_KEY
EOF
chmod 600 ~/.ssh/config

# Set rsync to use our SSH configuration
export RSYNC_RSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=~/.ssh/known_hosts"

echo "================================"
echo "Remote Dataset Builder"
echo "================================"
echo "Recipe: $RECIPE_NAME"
echo "Remote: $AMPERE_HOST"
echo "Output: $OUTPUT_DIR"
echo ""

# Test SSH connection first (this also populates known_hosts)
echo "→ Testing SSH connection to Ampere..."
if ! ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=~/.ssh/known_hosts $AMPERE_HOST "echo 'Connection successful'" 2>&1 | grep -q "Connection successful"; then
    echo "✗ Error: Cannot connect to Ampere. Check SSH key and network."
    exit 1
fi
echo "✓ SSH connection verified"

# Step 1: Ensure directories exist and sync recipe to Ampere
echo "→ Step 1: Syncing recipe configuration to Ampere..."
ssh -i $SSH_KEY $AMPERE_HOST "mkdir -p ${AMPERE_PATH}/configs/dataset_recipes"
rsync -az \
    /workspace/configs/dataset_recipes/${RECIPE_NAME}.yaml \
    ${AMPERE_HOST}:${AMPERE_PATH}/configs/dataset_recipes/

# Step 2: Build dataset on Ampere
echo "→ Step 2: Building dataset on Ampere (this may take several minutes)..."
ssh -i $SSH_KEY $AMPERE_HOST bash << ENDSSH
set -e

# Check if gept repo exists
if [ ! -d "/home/ubuntu/gept" ]; then
    echo "✗ Error: /home/ubuntu/gept directory not found on Ampere"
    echo "Please clone the repository first:"
    echo "  ssh ubuntu@150.136.170.128"
    echo "  git clone <repo-url> gept"
    exit 1
fi

cd /home/ubuntu/gept/packages/model

# Ensure .env exists with localhost config
if [ ! -f .env ]; then
    echo "Creating .env for Ampere..."
    cat > .env <<'EOF'
DB_HOST=localhost
DB_PORT=5432
DB_NAME=osrs_data
DB_USER=osrs_user
DB_PASS=${DB_PASS:-your_password_here}
EOF
fi

# Check if this is a git repo and pull if so
if [ -d .git ]; then
    echo "Pulling latest code..."
    git pull origin main || git pull origin master || echo "Note: Could not pull latest code"
fi

# Check if docker compose is available (v2 vs v1)
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "✗ Error: Neither 'docker compose' nor 'docker-compose' found on Ampere"
    exit 1
fi

# Build container if needed
echo "Building dataset-builder container..."
\$DOCKER_COMPOSE build dataset-builder

# Run dataset builder
echo "Running dataset builder for recipe: ${RECIPE_NAME}..."
\$DOCKER_COMPOSE run --rm dataset-builder python scripts/build_dataset.py --recipe ${RECIPE_NAME}

ENDSSH

echo "→ Step 3: Retrieving dataset version..."
DATASET_VERSION=$(ssh -i $SSH_KEY $AMPERE_HOST \
    "ls -t ${AMPERE_PATH}/datasets | grep ${RECIPE_NAME} | head -1")

if [ -z "$DATASET_VERSION" ]; then
    echo "✗ Error: Could not find built dataset on Ampere"
    exit 1
fi

echo "Found dataset: $DATASET_VERSION"

# Step 4: Transfer dataset back to Hydra
echo "→ Step 4: Transferring dataset to Hydra..."
rsync -avz --progress \
    ${AMPERE_HOST}:${AMPERE_PATH}/datasets/${DATASET_VERSION}/ \
    ${OUTPUT_DIR}/${DATASET_VERSION}/

echo ""
echo "✅ Dataset build complete!"
echo "   Version: $DATASET_VERSION"
echo "   Location: ${OUTPUT_DIR}/${DATASET_VERSION}"
echo "   Files:"
ls -lh ${OUTPUT_DIR}/${DATASET_VERSION}/ | tail -n +2

# Step 5: Verify dataset
echo ""
echo "→ Step 5: Verifying dataset integrity..."
if [ -f "${OUTPUT_DIR}/${DATASET_VERSION}/metadata.json" ]; then
    echo "Metadata:"
    cat ${OUTPUT_DIR}/${DATASET_VERSION}/metadata.json | python3 -m json.tool | head -20
    echo "✓ Dataset verified"
else
    echo "✗ Warning: metadata.json not found"
fi
