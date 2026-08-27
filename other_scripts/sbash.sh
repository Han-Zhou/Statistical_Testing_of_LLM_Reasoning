#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 <job-id> [-o] [g<gpu-count> | s<session-index>]" >&2
    echo "  -o  start a new overlapping shell (do not sattach to an existing session)" >&2
}

if (( $# < 1 || $# > 3 )); then
    usage
    exit 2
fi

job_id=$1
shift

if [[ ! $job_id =~ ^[0-9]+$ ]]; then
    echo "Error: job ID must be a non-negative integer." >&2
    usage
    exit 2
fi

gpu_count=
session_index=
overlap=

for option in "$@"; do
    if [[ $option == -o ]]; then
        if [[ -n $overlap ]]; then
            echo "Error: -o may only be specified once." >&2
            usage
            exit 2
        fi
        overlap=1
    elif [[ $option =~ ^g([0-9]+)$ ]]; then
        if [[ -n $gpu_count ]]; then
            echo "Error: the GPU option may only be specified once." >&2
            usage
            exit 2
        fi
        gpu_count=${BASH_REMATCH[1]}
    elif [[ $option =~ ^s([0-9]+)$ ]]; then
        if [[ -n $session_index ]]; then
            echo "Error: the session option may only be specified once." >&2
            usage
            exit 2
        fi
        session_index=${BASH_REMATCH[1]}
    else
        echo "Error: invalid option '$option'; expected -o, g<gpu-count>, or s<session-index>." >&2
        usage
        exit 2
    fi
done

if [[ -n $gpu_count && -n $session_index ]]; then
    echo "Error: GPU and session options cannot be specified together." >&2
    usage
    exit 2
fi

if [[ -n $overlap && -n $session_index ]]; then
    echo "Error: -o and session options cannot be specified together." >&2
    usage
    exit 2
fi

if [[ -n $gpu_count ]]; then
    if (( 10#$gpu_count < 1 )); then
        echo "Error: GPU count must be greater than zero." >&2
        exit 2
    fi
    exec srun --jobid="$job_id" --overlap --gres="gpu:$((10#$gpu_count))" --pty bash
fi

if [[ -n $session_index ]]; then
    step_id="${job_id}.$((10#$session_index))"
    if ! scontrol show step "$step_id" >/dev/null 2>&1; then
        echo "Error: session '$step_id' does not exist or is not accessible." >&2
        exit 1
    fi
    exec sattach "$step_id"
fi

if [[ -n $overlap ]]; then
    exec srun --jobid="$job_id" --overlap --pty bash
fi

step_id="${job_id}.0"

if scontrol show step "$step_id" >/dev/null 2>&1; then
    exec sattach "$step_id"
else
    exec srun --jobid="$job_id" --overlap --pty bash
fi
