import os
import re
import shutil

from tqdm import tqdm

# Full range of datapoints (INCLUSIVE)
RANGE = (0, 249)

# Values: path to an indices file, or "" for leftover datapoints
# PRIORITY: earlier entries claim first, later entries only get unclaimed indices. Handles overlapping index files.
trajectories_dict = {
    "trajectories/gpt-bm-s2-625_gpt_bigbench_movie_ssfull": "gpt-bm-s2-ri-625.txt",
    "trajectories/gpt-bm-supplement-623_gpt_bigbench_movie_ssfull": "gpt_retry_indices_623.txt",
    "trajectories/gpt-bm-618_gpt_bigbench_movie_sfull": "",
}

output_dir_name = "trajectories/MERGED_gpt-bm-626-bigbench_movie_sfull"

SUBDIRS = ("vanilla", "rejection", "lawyer", "stepbootstrap")

_TRAJ_IDX = re.compile(r"^traj_(\d+)")


def traj_index(filename):
    m = _TRAJ_IDX.match(filename)
    return int(m.group(1)) if m else None


def load_indices(filepath):
    indices = set()
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                indices.add(int(line))
    return indices


def merge_trajectories(trajectories_dict, output_dir_name):
    for subdir in SUBDIRS:
        os.makedirs(os.path.join(output_dir_name, subdir), exist_ok=True)

    # Assign indices with priority: earlier entries claim first, later entries
    # only get unclaimed indices. Handles overlapping index files.
    assigned_indices = {}  # src_dir -> set of indices
    claimed = set()
    leftover_dir = None
    for src_dir, indices_path in trajectories_dict.items():
        if indices_path == "":
            if leftover_dir is not None:
                raise ValueError("only one trajectory dir can be the leftover (empty string)")
            leftover_dir = src_dir
        else:
            raw = load_indices(indices_path)
            assigned = raw - claimed
            assigned_indices[src_dir] = assigned
            claimed |= assigned

    # Leftover dir gets everything not claimed by any explicit dir
    if leftover_dir is not None:
        all_indices = set(range(RANGE[0], RANGE[1] + 1))
        assigned_indices[leftover_dir] = all_indices - claimed

    # Build copy jobs
    jobs = []
    for src_dir, allowed_indices in assigned_indices.items():
        for subdir in SUBDIRS:
            src_sub = os.path.join(src_dir, subdir)
            dst_sub = os.path.join(output_dir_name, subdir)
            if not os.path.isdir(src_sub):
                continue
            for filename in os.listdir(src_sub):
                idx = traj_index(filename)
                if idx is None or idx not in allowed_indices:
                    continue
                dst_path = os.path.join(dst_sub, filename)
                if os.path.exists(dst_path):
                    raise FileExistsError(
                        f"name collision: {dst_path} (from {src_dir})"
                    )
                jobs.append((os.path.join(src_sub, filename), dst_path))

    for src_path, dst_path in tqdm(jobs, desc="merging", unit="file"):
        shutil.copy2(src_path, dst_path)
    return len(jobs)


def main():
    copied = merge_trajectories(trajectories_dict, output_dir_name)
    print(f"copied {copied} files into {output_dir_name}")


if __name__ == "__main__":
    main()
