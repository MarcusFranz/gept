#!/bin/bash
# ==============================================================================
# WSL Training Machine Setup Script
# ==============================================================================
#
# Run this script INSIDE your WSL instance to prepare it for remote GPU training.
#
# Prerequisites:
#   - WSL2 with Ubuntu (20.04 or 22.04)
#   - NVIDIA GPU with drivers installed on Windows
#   - CUDA toolkit in WSL
#
# Usage:
#   # Copy this script to WSL and run:
#   chmod +x setup_wsl_training.sh
#   ./setup_wsl_training.sh
#
# After running this script:
#   1. Run the Windows setup script (setup_windows_portforward.ps1)
#   2. Copy your SSH public key to WSL
#   3. Test connection from Ampere server
#
# ==============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${BLUE}[STEP]${NC} $1"; }

echo "=============================================================="
echo "  GePT WSL Training Machine Setup"
echo "=============================================================="
echo ""

# Check if running in WSL
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    log_error "This script must be run inside WSL!"
    exit 1
fi

GEPT_DIR="/home/$USER/gept"
VENV_DIR="$GEPT_DIR/venv"

# ==============================================================================
# Step 1: System Updates
# ==============================================================================
log_step "Step 1: Updating system packages..."

sudo apt-get update
sudo apt-get upgrade -y

# ==============================================================================
# Step 2: Install SSH Server
# ==============================================================================
log_step "Step 2: Installing and configuring SSH server..."

sudo apt-get install -y openssh-server

# Configure SSH
sudo tee /etc/ssh/sshd_config.d/gept-training.conf > /dev/null << 'SSHCONFIG'
# GePT Training SSH Configuration
Port 22
PermitRootLogin no
PubkeyAuthentication yes
PasswordAuthentication no
ChallengeResponseAuthentication no
UsePAM yes
X11Forwarding no
PrintMotd no
AcceptEnv LANG LC_*
Subsystem sftp /usr/lib/openssh/sftp-server

# Security hardening
MaxAuthTries 3
MaxSessions 5
ClientAliveInterval 300
ClientAliveCountMax 2
SSHCONFIG

# Generate host keys if they don't exist
sudo ssh-keygen -A

# Start SSH service
sudo service ssh start

log_info "SSH server installed and started"

# ==============================================================================
# Step 3: Create SSH Key Directory
# ==============================================================================
log_step "Step 3: Setting up SSH keys directory..."

mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

log_info "SSH directory created at ~/.ssh"
log_warn "You need to add your Ampere server's public key to ~/.ssh/authorized_keys"

# ==============================================================================
# Step 4: Install Python and Dependencies
# ==============================================================================
log_step "Step 4: Installing Python and build dependencies..."

sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    rsync \
    htop \
    nvtop \
    curl \
    wget

# Make python3.10 the default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 || true
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 || true

log_info "Python $(python3 --version) installed"

# ==============================================================================
# Step 5: Check NVIDIA/CUDA
# ==============================================================================
log_step "Step 5: Checking NVIDIA GPU and CUDA..."

if command -v nvidia-smi &> /dev/null; then
    log_info "NVIDIA driver detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

    # Check CUDA
    if command -v nvcc &> /dev/null; then
        log_info "CUDA version: $(nvcc --version | grep release | awk '{print $6}')"
    else
        log_warn "CUDA toolkit not found. Installing..."

        # Install CUDA toolkit (for CatBoost GPU support)
        wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update
        sudo apt-get install -y cuda-toolkit-12-2
        rm cuda-keyring_1.1-1_all.deb

        # Add to PATH
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

        log_info "CUDA toolkit installed. Please restart your shell."
    fi
else
    log_error "NVIDIA driver not detected!"
    log_error "Make sure you have:"
    log_error "  1. NVIDIA GPU drivers installed on Windows"
    log_error "  2. WSL2 (not WSL1)"
    log_error "  3. Latest Windows 11 or Windows 10 21H2+"
    exit 1
fi

# ==============================================================================
# Step 6: Create GePT Directory Structure
# ==============================================================================
log_step "Step 6: Creating GePT directory structure..."

mkdir -p "$GEPT_DIR"/{src,models,data/prepared,logs,cloud}
mkdir -p "$GEPT_DIR/.secrets"
chmod 700 "$GEPT_DIR/.secrets"

log_info "Directory structure created at $GEPT_DIR"

# ==============================================================================
# Step 7: Create Python Virtual Environment
# ==============================================================================
log_step "Step 7: Creating Python virtual environment..."

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install training dependencies
pip install \
    numpy \
    pandas \
    pyarrow \
    scikit-learn \
    catboost \
    numba \
    psycopg2-binary \
    tqdm \
    pyyaml \
    python-dateutil

log_info "Virtual environment created and dependencies installed"

# Verify CatBoost GPU support
python -c "
from catboost import CatBoostClassifier
import catboost
print(f'CatBoost version: {catboost.__version__}')
try:
    # Test GPU availability
    model = CatBoostClassifier(task_type='GPU', devices='0', iterations=1, verbose=False)
    print('GPU support: Available')
except Exception as e:
    print(f'GPU support: Not available ({e})')
"

# ==============================================================================
# Step 8: Create Startup Script
# ==============================================================================
log_step "Step 8: Creating SSH startup script..."

# Create a script to start SSH on WSL boot
cat > ~/start_ssh.sh << 'STARTSCRIPT'
#!/bin/bash
# Start SSH server in WSL
# Add this to Windows Task Scheduler to run on boot:
#   wsl -d Ubuntu -u root service ssh start

if ! pgrep -x "sshd" > /dev/null; then
    sudo service ssh start
    echo "SSH server started"
else
    echo "SSH server already running"
fi
STARTSCRIPT

chmod +x ~/start_ssh.sh

# ==============================================================================
# Step 9: Create Environment File Template
# ==============================================================================
log_step "Step 9: Creating environment file template..."

cat > "$GEPT_DIR/.secrets/training.env.template" << 'ENVTEMPLATE'
# GePT Training Environment Variables
# Copy this to training.env and fill in values

# Database connection (via SSH tunnel from Ampere)
DB_PASS=your_database_password_here

# SSH tunnel command (run this on WSL before training if needed):
# ssh -i ~/.ssh/ampere_key.pem -L 5432:localhost:5432 $AMPERE_HOST -N &
# (AMPERE_HOST is defined in config/servers.env, default: ubuntu@150.136.170.128)
ENVTEMPLATE

log_info "Environment template created at $GEPT_DIR/.secrets/training.env.template"

# ==============================================================================
# Step 10: Display Summary and Next Steps
# ==============================================================================
echo ""
echo "=============================================================="
echo "  Setup Complete!"
echo "=============================================================="
echo ""
log_info "WSL IP address: $(hostname -I | awk '{print $1}')"
log_info "GePT directory: $GEPT_DIR"
log_info "Virtual env: $VENV_DIR"
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. On your WINDOWS machine, run the port forwarding script:"
echo "   (Run PowerShell as Administrator)"
echo "   .\\setup_windows_portforward.ps1"
echo ""
echo "2. Generate SSH key on AMPERE server (if not already done):"
echo "   ssh-keygen -t ed25519 -f ~/.ssh/wsl_key -N ''"
echo ""
echo "3. Copy the Ampere public key to this WSL machine:"
echo "   # On Ampere, display the public key:"
echo "   cat ~/.ssh/wsl_key.pub"
echo ""
echo "   # On WSL, add it to authorized_keys:"
echo "   echo 'PASTE_PUBLIC_KEY_HERE' >> ~/.ssh/authorized_keys"
echo ""
echo "4. Copy the private key to the GePT repo on Ampere:"
echo "   cp ~/.ssh/wsl_key /home/ubuntu/gept/.secrets/wsl_key.pem"
echo "   chmod 600 /home/ubuntu/gept/.secrets/wsl_key.pem"
echo ""
echo "5. Test SSH connection from Ampere:"
echo "   ssh -i .secrets/wsl_key.pem -p 2222 $USER@YOUR_WINDOWS_IP"
echo ""
echo "6. Copy training scripts to WSL:"
echo "   rsync -avz cloud/ $USER@YOUR_WINDOWS_IP:$GEPT_DIR/cloud/"
echo "   rsync -avz src/ $USER@YOUR_WINDOWS_IP:$GEPT_DIR/src/"
echo ""
echo "7. Configure WSL_HOST in training_config.yaml on Ampere"
echo ""
echo "=============================================================="
