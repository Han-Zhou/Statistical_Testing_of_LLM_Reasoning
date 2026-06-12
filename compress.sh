#!/usr/bin/env bash
# Compress a folder into a sibling <folder>.tar.zst archive.
# Usage: bash compress.sh <relative/path/to/folder>
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <relative/path/to/folder>" >&2
    exit 1
fi

# Strip any trailing slash so basename/dirname behave.
target="${1%/}"

if [[ ! -d "$target" ]]; then
    echo "Error: '$target' is not a directory" >&2
    exit 1
fi

parent="$(dirname "$target")"
name="$(basename "$target")"
output="$parent/$name.tar.zst"

if [[ -e "$output" ]]; then
    echo "Error: '$output' already exists" >&2
    exit 1
fi

# -C parent keeps archive paths relative to the folder itself.
# zstd -19 -T0: high compression, all cores.
tar -C "$parent" -cf - "$name" | zstd -19 -T0 -o "$output"

echo "Created $output"
du -sh "$output"
