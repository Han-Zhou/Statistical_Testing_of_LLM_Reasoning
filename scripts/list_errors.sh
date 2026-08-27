#!/bin/bash
# Lists indices of error trajectories in a given experiment/method directory.
# Usage: bash scripts/list_errors.sh <trajectory_dir>
# Example: bash scripts/list_errors.sh trajectories/gpt-bm-618_gpt_bigbench_movie_sfull/vanilla

dir="${1:?Usage: $0 <trajectory_dir>}"

ls "$dir"/*_error.json 2>/dev/null | sed 's/.*traj_\([0-9]*\)_error.json/\1/' | sort -n
