import os
import re
import shutil

from tqdm import tqdm

# INCLUSIVE [start, end]
trajectories_dict = {
    "trajectories/llama-bm-610_llama_bigbench_movie_sfull": (0, 100),
    "trajectories/llama-bm-610-p2_llama_bigbench_movie_s101_150": (101, 149),
    "trajectories/llama-bm-610-p3_llama_bigbench_movie_s150_200": (150, 199),
    "trajectories/llama-bm-610-p4_llama_bigbench_movie_s200_250": (200, 249),
}

output_dir_name = "trajectories/MERGED_llama-bm-610-bigbench_movie_sfull"

SUBDIRS = ("vanilla", "rejection", "lawyer", "stepbootstrap")

# filenames look like: traj_<i>.json, traj_<i>_error.json, traj_<i>_sample_<j>.json
_TRAJ_IDX = re.compile(r"^traj_(\d+)")


def traj_index(filename):
    m = _TRAJ_IDX.match(filename)
    return int(m.group(1)) if m else None


def merge_trajectories(trajectories_dict, output_dir_name):
    for subdir in SUBDIRS:
        os.makedirs(os.path.join(output_dir_name, subdir), exist_ok=True)

    jobs = []
    for src_dir, (start, end) in trajectories_dict.items():
        for subdir in SUBDIRS:
            src_sub = os.path.join(src_dir, subdir)
            dst_sub = os.path.join(output_dir_name, subdir)
            for filename in os.listdir(src_sub):
                idx = traj_index(filename)
                if idx is None or not (start <= idx <= end):
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
