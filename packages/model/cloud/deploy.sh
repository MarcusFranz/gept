#!/bin/bash
# =============================================================================
# Deploy Cloud Run Job for daily model training
# =============================================================================
# Supports versioned image deployments with rollback capability.
#
# Prerequisites:
# - gcloud CLI authenticated
# - Project ID set: gcloud config set project PROJECT_ID
# - Container Registry access enabled
#
# Usage:
#   ./deploy.sh                    # Deploy with default settings
#   ./deploy.sh --tasks 500        # Override task count
#   ./deploy.sh --tier-mode        # Enable tiered training
#   ./deploy.sh --rollback         # Rollback to previous version
#   ./deploy.sh --rollback VERSION # Rollback to specific version
#   ./deploy.sh --list             # List available versions
#   ./deploy.sh --dry-run          # Preview deployment
#
# Image Tagging Strategy:
#   gcr.io/$PROJECT/gept-trainer:latest          # Current active version
#   gcr.io/$PROJECT/gept-trainer:20260113_150000 # Timestamped version
# =============================================================================

set -euo pipefail

# Error handler for debugging
error_handler() {
    echo "ERROR: Script failed on line $1" >&2
    exit 1
}
trap 'error_handler $LINENO' ERR

# Configuration
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${REGION:-us-central1}"
JOB_NAME="${JOB_NAME:-gept-daily-train}"
GCS_BUCKET="${GCS_BUCKET:-osrs-models-mof}"
IMAGE_BASE="gcr.io/${PROJECT_ID}/gept-trainer"
KEEP_VERSIONS=5
DRY_RUN=false

# Default resources (optimized for cost)
TASKS="${TASKS:-2380}"
MEMORY="${MEMORY:-2Gi}"
CPU="${CPU:-1}"
TIMEOUT="${TIMEOUT:-15m}"
MAX_RETRIES="${MAX_RETRIES:-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_help() {
    echo "GePT Cloud Run Job - Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --tasks N             Number of parallel tasks (default: 2380)"
    echo "  --tier-mode           Enable tiered training strategy"
    echo "  --discovery           Force discovery run"
    echo "  --memory SIZE         Memory per task (default: 2Gi)"
    echo "  --cpu N               CPU cores per task (default: 1)"
    echo "  --rollback [VERSION]  Rollback to previous or specific version"
    echo "  --list                List available versions"
    echo "  --dry-run             Preview deployment without executing"
    echo "  --keep N              Keep N versions (default: 5)"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Deploy new version"
    echo "  $0 --tasks 500        # Deploy with 500 tasks"
    echo "  $0 --tier-mode        # Deploy with tiered training"
    echo "  $0 --rollback         # Rollback to previous version"
    echo "  $0 --list             # Show available versions"
}

list_versions() {
    log_info "Available versions for $IMAGE_BASE:"
    echo ""

    # Get current version from Cloud Run job
    local current_image
    current_image=$(gcloud run jobs describe "$JOB_NAME" --region="$REGION" --format='value(template.template.containers[0].image)' 2>/dev/null || echo "")
    local current_version=""
    if [[ -n "$current_image" ]]; then
        current_version=$(echo "$current_image" | sed 's/.*://')
    fi

    # List images from Container Registry
    local images
    images=$(gcloud container images list-tags "$IMAGE_BASE" --format='value(tags)' --sort-by='~timestamp' --limit=20 2>/dev/null || echo "")

    if [[ -z "$images" ]]; then
        echo "  No versions found in registry"
        return
    fi

    echo "$images" | while IFS= read -r tag; do
        # Handle multiple tags on same image
        for t in $tag; do
            if [[ "$t" == "latest" ]]; then
                continue  # Skip latest tag in listing
            fi
            if [[ "$t" == "$current_version" ]]; then
                echo "  * $t (current)"
            else
                echo "    $t"
            fi
        done
    done
    echo ""
}

get_previous_version() {
    # Get the second most recent timestamped version (not 'latest')
    gcloud container images list-tags "$IMAGE_BASE" \
        --format='value(tags)' \
        --sort-by='~timestamp' \
        --limit=10 2>/dev/null | \
        tr ' ' '\n' | \
        grep -E '^[0-9]{8}_[0-9]{6}$' | \
        head -2 | tail -1
}

get_current_version() {
    local current_image
    current_image=$(gcloud run jobs describe "$JOB_NAME" --region="$REGION" --format='value(template.template.containers[0].image)' 2>/dev/null || echo "")
    if [[ -n "$current_image" ]]; then
        echo "$current_image" | sed 's/.*://'
    fi
}

do_rollback() {
    local target_version=$1

    if [[ -z "$target_version" ]]; then
        target_version=$(get_previous_version)
    fi

    if [[ -z "$target_version" ]]; then
        log_error "No previous version available for rollback"
        exit 1
    fi

    local current_version
    current_version=$(get_current_version)

    if [[ "$target_version" == "$current_version" ]]; then
        log_warning "Version $target_version is already current"
        exit 0
    fi

    log_info "Rolling back from $current_version to $target_version..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would update job to use image: $IMAGE_BASE:$target_version"
        return
    fi

    # Verify target version exists in registry
    if ! gcloud container images describe "$IMAGE_BASE:$target_version" &>/dev/null; then
        log_error "Version $target_version does not exist in registry"
        echo ""
        list_versions
        exit 1
    fi

    # Update Cloud Run job to use the rollback version
    gcloud run jobs update "$JOB_NAME" \
        --image "$IMAGE_BASE:$target_version" \
        --region "$REGION"

    # Update latest tag to point to rollback version
    log_info "Updating 'latest' tag to point to $target_version..."
    gcloud container images add-tag \
        "$IMAGE_BASE:$target_version" \
        "$IMAGE_BASE:latest" \
        --quiet

    log_success "Rolled back to version $target_version"

    # Verify rollback
    verify_deployment
}

verify_deployment() {
    log_info "Verifying deployment..."

    echo ""
    echo "=== Current Job Configuration ==="
    gcloud run jobs describe "$JOB_NAME" --region="$REGION" \
        --format='table(metadata.name,template.template.containers[0].image,template.taskCount,template.template.containers[0].resources.limits.memory)'

    echo ""
    echo "=== Recent Executions ==="
    gcloud run jobs executions list --job="$JOB_NAME" --region="$REGION" --limit=5 \
        --format='table(metadata.name,status.succeededCount,status.failedCount,status.completionTime)' 2>/dev/null || echo "No executions found"

    log_success "Verification complete"
}

prune_old_versions() {
    log_info "Pruning old versions (keeping last $KEEP_VERSIONS)..."

    # Get all timestamped versions sorted by date
    local versions
    versions=$(gcloud container images list-tags "$IMAGE_BASE" \
        --format='value(tags)' \
        --sort-by='~timestamp' 2>/dev/null | \
        tr ' ' '\n' | \
        grep -E '^[0-9]{8}_[0-9]{6}$' || true)

    local count=0
    while IFS= read -r version; do
        count=$((count + 1))
        if [[ $count -gt $KEEP_VERSIONS && -n "$version" ]]; then
            log_info "Removing old version: $version"
            if [[ "$DRY_RUN" != true ]]; then
                gcloud container images delete "$IMAGE_BASE:$version" --quiet --force-delete-tags 2>/dev/null || true
            fi
        fi
    done <<< "$versions"

    log_success "Old versions pruned"
}

deploy_version() {
    local tier_mode=$1
    local force_discovery=$2

    local deploy_version
    deploy_version=$(date +%Y%m%d_%H%M%S)
    local image_uri="$IMAGE_BASE:$deploy_version"
    local git_commit
    git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

    log_info "Deploying version: $deploy_version"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would build image: $image_uri"
        log_info "[DRY RUN] Would push to Container Registry"
        log_info "[DRY RUN] Would update Cloud Run job"
        return
    fi

    echo "=============================================="
    echo "Deploying Cloud Run Job: ${JOB_NAME}"
    echo "=============================================="
    echo "Project:     ${PROJECT_ID}"
    echo "Region:      ${REGION}"
    echo "Version:     ${deploy_version}"
    echo "Git Commit:  ${git_commit}"
    echo "Image:       ${image_uri}"
    echo "Tasks:       ${TASKS}"
    echo "Memory:      ${MEMORY}"
    echo "CPU:         ${CPU}"
    echo "Tier Mode:   ${tier_mode}"
    echo "=============================================="

    # Build Docker image with version tag
    echo ""
    log_info "Building Docker image..."
    docker build -t "$image_uri" -t "$IMAGE_BASE:latest" -f cloud/Dockerfile cloud/

    # Push both tags to Container Registry
    echo ""
    log_info "Pushing to Container Registry..."
    docker push "$image_uri"
    docker push "$IMAGE_BASE:latest"

    # Store deployment metadata in GCS
    log_info "Recording deployment metadata..."
    local previous_version
    previous_version=$(get_current_version)
    cat > /tmp/deploy_info.json << EOF
{
    "version": "$deploy_version",
    "deployed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "deployed_by": "cloud/deploy.sh",
    "git_commit": "$git_commit",
    "previous_version": "$previous_version",
    "tasks": $TASKS,
    "memory": "$MEMORY",
    "tier_mode": $tier_mode
}
EOF
    gsutil cp /tmp/deploy_info.json "gs://${GCS_BUCKET}/deployments/${deploy_version}/deploy_info.json" 2>/dev/null || true
    rm /tmp/deploy_info.json

    # Check if job exists
    if gcloud run jobs describe "${JOB_NAME}" --region="${REGION}" &>/dev/null; then
        echo ""
        log_info "Updating existing job..."
        gcloud run jobs update "${JOB_NAME}" \
            --image "${image_uri}" \
            --tasks "${TASKS}" \
            --task-timeout "${TIMEOUT}" \
            --max-retries "${MAX_RETRIES}" \
            --memory "${MEMORY}" \
            --cpu "${CPU}" \
            --region "${REGION}" \
            --set-env-vars "GCS_BUCKET=${GCS_BUCKET},TIER_MODE=${tier_mode},FORCE_DISCOVERY=${force_discovery}"
    else
        echo ""
        log_info "Creating new job..."
        gcloud run jobs create "${JOB_NAME}" \
            --image "${image_uri}" \
            --tasks "${TASKS}" \
            --task-timeout "${TIMEOUT}" \
            --max-retries "${MAX_RETRIES}" \
            --memory "${MEMORY}" \
            --cpu "${CPU}" \
            --region "${REGION}" \
            --set-env-vars "GCS_BUCKET=${GCS_BUCKET},TIER_MODE=${tier_mode},FORCE_DISCOVERY=${force_discovery}"
    fi

    log_success "Deployed version $deploy_version"

    # Prune old versions
    prune_old_versions
}

# Main script
main() {
    local action="deploy"
    local rollback_version=""
    local tier_mode="false"
    local force_discovery="false"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tasks)
                TASKS="$2"
                shift 2
                ;;
            --tier-mode)
                tier_mode="true"
                shift
                ;;
            --discovery)
                force_discovery="true"
                shift
                ;;
            --memory)
                MEMORY="$2"
                shift 2
                ;;
            --cpu)
                CPU="$2"
                shift 2
                ;;
            --rollback)
                action="rollback"
                if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                    rollback_version="$2"
                    shift
                fi
                shift
                ;;
            --list)
                action="list"
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --keep)
                KEEP_VERSIONS="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    case $action in
        rollback)
            do_rollback "$rollback_version"
            ;;
        list)
            list_versions
            ;;
        deploy)
            deploy_version "$tier_mode" "$force_discovery"
            verify_deployment

            echo ""
            echo "=============================================="
            echo "Deployment complete!"
            echo "=============================================="
            echo ""
            echo "To execute the job:"
            echo "  gcloud run jobs execute ${JOB_NAME} --region=${REGION} \\"
            echo "    --set-env-vars RUN_ID=\$(date +%Y%m%d_%H%M%S)"
            echo ""
            echo "To execute with tiered training:"
            echo "  gcloud run jobs execute ${JOB_NAME} --region=${REGION} \\"
            echo "    --set-env-vars RUN_ID=\$(date +%Y%m%d_%H%M%S),TIER_MODE=true"
            echo ""
            echo "To force discovery run:"
            echo "  gcloud run jobs execute ${JOB_NAME} --region=${REGION} \\"
            echo "    --set-env-vars RUN_ID=\$(date +%Y%m%d_%H%M%S),FORCE_DISCOVERY=true"
            echo ""
            echo "Rollback command:"
            echo "  ./cloud/deploy.sh --rollback"
            echo ""
            ;;
    esac
}

main "$@"
