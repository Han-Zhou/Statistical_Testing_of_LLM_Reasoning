import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets.bigbench_movie import BigBenchMovieDataset


OUTPUT_DIR = Path("/storage/backup/han/cot/pickles")
OUTPUT_NAME = "bigbench_movie_250.pkl"


def main() -> None:
    dataset = BigBenchMovieDataset()
    datapoints = dataset.load_datapoints()

    rows = []
    for i, dp in enumerate(datapoints):   
        rows.append({"id": dp.id, "question": dp.question, "ground_truth": dp.ground_truth})

    output_path = OUTPUT_DIR / OUTPUT_NAME
    with open(output_path, "wb") as f:
        pickle.dump(rows, f)

    print(f"Wrote {len(rows)} datapoints to {output_path}")


if __name__ == "__main__":
    main()
