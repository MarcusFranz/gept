# ==============================================================================
# Windows Port Forwarding Setup for WSL Training
# ==============================================================================
#
# This script configures Windows to forward SSH connections to WSL.
# Run this script in PowerShell AS ADMINISTRATOR.
#
# What it does:
#   1. Gets the WSL IP address
#   2. Creates port forwarding rule (Windows port 2222 -> WSL port 22)
#   3. Creates firewall rule to allow incoming connections
#   4. Creates a scheduled task to maintain forwarding after reboot
#
# Usage:
#   1. Open PowerShell as Administrator
#   2. Run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   3. Run: .\setup_windows_portforward.ps1
#
# ==============================================================================

# Check if running as Administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  Windows Port Forwarding Setup for WSL Training" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$WINDOWS_PORT = 2222      # Port to listen on Windows
$WSL_PORT = 22            # SSH port in WSL
$RULE_NAME = "GePT-WSL-Training"

# ==============================================================================
# Step 1: Get WSL IP Address
# ==============================================================================
Write-Host "[STEP 1] Getting WSL IP address..." -ForegroundColor Blue

$WSL_IP = (wsl hostname -I).Trim().Split(" ")[0]

if ([string]::IsNullOrEmpty($WSL_IP)) {
    Write-Host "ERROR: Could not get WSL IP address. Is WSL running?" -ForegroundColor Red
    Write-Host "Try running: wsl" -ForegroundColor Yellow
    exit 1
}

Write-Host "  WSL IP: $WSL_IP" -ForegroundColor Green

# ==============================================================================
# Step 2: Remove Existing Port Forwarding (if any)
# ==============================================================================
Write-Host "[STEP 2] Removing any existing port forwarding rules..." -ForegroundColor Blue

# Remove existing port proxy rules for this port
netsh interface portproxy delete v4tov4 listenport=$WINDOWS_PORT listenaddress=0.0.0.0 2>$null
netsh interface portproxy delete v4tov4 listenport=$WINDOWS_PORT listenaddress=127.0.0.1 2>$null

Write-Host "  Existing rules removed" -ForegroundColor Green

# ==============================================================================
# Step 3: Create Port Forwarding Rule
# ==============================================================================
Write-Host "[STEP 3] Creating port forwarding rule..." -ForegroundColor Blue
Write-Host "  Windows:0.0.0.0:$WINDOWS_PORT -> WSL:$WSL_IP`:$WSL_PORT" -ForegroundColor Gray

netsh interface portproxy add v4tov4 `
    listenport=$WINDOWS_PORT `
    listenaddress=0.0.0.0 `
    connectport=$WSL_PORT `
    connectaddress=$WSL_IP

if ($LASTEXITCODE -eq 0) {
    Write-Host "  Port forwarding rule created" -ForegroundColor Green
} else {
    Write-Host "ERROR: Failed to create port forwarding rule" -ForegroundColor Red
    exit 1
}

# Verify the rule was created
Write-Host ""
Write-Host "  Current port proxy rules:" -ForegroundColor Gray
netsh interface portproxy show all

# ==============================================================================
# Step 4: Configure Firewall
# ==============================================================================
Write-Host ""
Write-Host "[STEP 4] Configuring Windows Firewall..." -ForegroundColor Blue

# Remove existing firewall rule if it exists
Remove-NetFirewallRule -DisplayName $RULE_NAME -ErrorAction SilentlyContinue

# Create new firewall rule
New-NetFirewallRule `
    -DisplayName $RULE_NAME `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort $WINDOWS_PORT `
    -Action Allow `
    -Profile Any `
    -Description "Allow SSH connections to WSL for GePT training"

Write-Host "  Firewall rule '$RULE_NAME' created" -ForegroundColor Green

# ==============================================================================
# Step 5: Create Update Script (for WSL IP changes)
# ==============================================================================
Write-Host ""
Write-Host "[STEP 5] Creating WSL IP update script..." -ForegroundColor Blue

$UPDATE_SCRIPT_PATH = "$env:USERPROFILE\update_wsl_portforward.ps1"

$updateScript = @"
# Auto-generated script to update WSL port forwarding
# Run this after WSL restarts (IP may change)

`$WSL_IP = (wsl hostname -I).Trim().Split(" ")[0]
if ([string]::IsNullOrEmpty(`$WSL_IP)) {
    Write-Host "WSL not running" -ForegroundColor Red
    exit 1
}

Write-Host "Updating port forward to WSL IP: `$WSL_IP"

netsh interface portproxy delete v4tov4 listenport=$WINDOWS_PORT listenaddress=0.0.0.0 2>`$null
netsh interface portproxy add v4tov4 listenport=$WINDOWS_PORT listenaddress=0.0.0.0 connectport=$WSL_PORT connectaddress=`$WSL_IP

Write-Host "Port forwarding updated successfully" -ForegroundColor Green
"@

$updateScript | Out-File -FilePath $UPDATE_SCRIPT_PATH -Encoding UTF8

Write-Host "  Update script saved to: $UPDATE_SCRIPT_PATH" -ForegroundColor Green

# ==============================================================================
# Step 6: Create Scheduled Task (Optional)
# ==============================================================================
Write-Host ""
Write-Host "[STEP 6] Creating scheduled task for startup..." -ForegroundColor Blue

$taskName = "GePT-WSL-PortForward"

# Remove existing task if it exists
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Create the task
$action = New-ScheduledTaskAction `
    -Execute "PowerShell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$UPDATE_SCRIPT_PATH`""

$trigger = New-ScheduledTaskTrigger -AtStartup

$principal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Updates WSL port forwarding for GePT training on startup"

Write-Host "  Scheduled task '$taskName' created" -ForegroundColor Green

# ==============================================================================
# Step 7: Start SSH in WSL
# ==============================================================================
Write-Host ""
Write-Host "[STEP 7] Starting SSH server in WSL..." -ForegroundColor Blue

wsl -u root service ssh start

Write-Host "  SSH server started in WSL" -ForegroundColor Green

# ==============================================================================
# Step 8: Get Windows IP for Remote Access
# ==============================================================================
Write-Host ""
Write-Host "[STEP 8] Getting Windows IP addresses..." -ForegroundColor Blue

$windowsIPs = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
    Select-Object -ExpandProperty IPAddress

Write-Host "  Your Windows IP address(es):" -ForegroundColor Green
foreach ($ip in $windowsIPs) {
    Write-Host "    - $ip" -ForegroundColor Cyan
}

# ==============================================================================
# Summary
# ==============================================================================
Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Port Forwarding:" -ForegroundColor Yellow
Write-Host "  Windows Port: $WINDOWS_PORT" -ForegroundColor White
Write-Host "  WSL IP: $WSL_IP`:$WSL_PORT" -ForegroundColor White
Write-Host ""
Write-Host "To connect from Ampere server:" -ForegroundColor Yellow
foreach ($ip in $windowsIPs) {
    Write-Host "  ssh -i .secrets/wsl_key.pem -p $WINDOWS_PORT ubuntu@$ip" -ForegroundColor Cyan
}
Write-Host ""
Write-Host "IMPORTANT:" -ForegroundColor Red
Write-Host "  1. Make sure WSL SSH server is configured (run setup_wsl_training.sh in WSL)" -ForegroundColor White
Write-Host "  2. Add your Ampere server's SSH public key to WSL ~/.ssh/authorized_keys" -ForegroundColor White
Write-Host "  3. If WSL restarts and IP changes, run: $UPDATE_SCRIPT_PATH" -ForegroundColor White
Write-Host ""
Write-Host "If your router/firewall blocks incoming connections:" -ForegroundColor Yellow
Write-Host "  - You may need to configure port forwarding on your router" -ForegroundColor White
Write-Host "  - Or use a reverse SSH tunnel from WSL to Ampere" -ForegroundColor White
Write-Host ""
