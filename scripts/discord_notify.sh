#!/usr/bin/env bash

# Best-effort Discord notifications for Slurm jobs. Notifications are disabled
# unless DISCORD_WEBHOOK is exported in the submission environment.

notify_discord() {
    local message="$1"
    local webhook="${DISCORD_WEBHOOK:-}"

    if [[ -z "$webhook" ]]; then
        return 0
    fi

    curl -sS -H "Content-Type: application/json" \
        -d "{\"content\": \"$message\"}" \
        "$webhook" > /dev/null 2>&1 || true
}

discord_job_name="${SLURM_JOB_NAME:-$(basename "$0")}"

notify_discord_error() {
    local exit_code="$1"
    local line_no="$2"
    notify_discord "[FAILED] ${discord_job_name} failed on $(hostname) (job ${SLURM_JOB_ID:-local}) at line ${line_no} with exit code ${exit_code}."
    exit "$exit_code"
}

trap 'notify_discord_error "$?" "$LINENO"' ERR
notify_discord "[STARTED] ${discord_job_name} is running on $(hostname) (job ${SLURM_JOB_ID:-local})."
