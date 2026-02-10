#!/bin/bash
# =============================================================================
# Quick Rollback Script for GePT Data Collectors
# =============================================================================
# Convenience wrapper to quickly rollback the collectors deployment.
#
# Usage:
#   ./rollback_collectors.sh              # Rollback to previous version
#   ./rollback_collectors.sh VERSION      # Rollback to specific version
#   ./rollback_collectors.sh --list       # List available versions
#   ./rollback_collectors.sh --dry-run    # Preview rollback
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Pass all arguments to the main deploy script with --rollback
if [[ "$1" == "--list" ]]; then
    exec "$SCRIPT_DIR/deploy_collectors.sh" --list
elif [[ "$1" == "--dry-run" ]]; then
    shift
    exec "$SCRIPT_DIR/deploy_collectors.sh" --dry-run --rollback "$@"
elif [[ -n "$1" && ! "$1" =~ ^-- ]]; then
    # Specific version provided
    exec "$SCRIPT_DIR/deploy_collectors.sh" --rollback "$1"
else
    # Default: rollback to previous
    exec "$SCRIPT_DIR/deploy_collectors.sh" --rollback
fi
