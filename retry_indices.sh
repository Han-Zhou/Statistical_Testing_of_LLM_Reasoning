#!/usr/bin/env bash
# Extract indices from *_error.json files in a trajectory directory.
# Usage: bash retry_indices.sh <relative/path/to/folder> <output_name>
# Output: <output_name>.txt with one index per line, sorted numerically.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <relative/path/to/folder> <output_name>" >&2
    exit 1
fi

target="${1%/}"
output_name="$2"

if [[ ! -d "$target" ]]; then
    echo "Error: '$target' is not a directory" >&2
    exit 1
fi

output="${output_name}.txt"

if [[ -e "$output" ]]; then
    echo "Error: '$output' already exists" >&2
    exit 1
fi

find "$target" -name '*_error.json' -printf '%f\n' \
    | sed -n 's/^traj_\([0-9]\+\)_error\.json$/\1/p' \
    | sort -n \
    > "$output"

echo "Created $output ($(wc -l < "$output") indices)"
