import json
import logging
import urllib.request

from pathlib import Path
from datasets.base import Dataset


logger = logging.getLogger(__name__)

class BigBenchCausalDataset(Dataset):
    def __init__(self):
        super().__init__("bigbench_causal")
    
    def _fetch_bbh(self, config: str) -> list[dict]:
        """Download a BBH (BigBench Hard) config JSON and return the examples list."""
        _BBH_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{}.json"
        url = _BBH_URL.format(config)
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read())["examples"]

    def load_datapoints(self) -> list:
        rows = self._fetch_bbh("causal_judgment")
        entries = []
        for i, row in enumerate(rows):
            entries.append(self._entry(
                id_=f"bigbench_causal_{i}",
                question=row["input"],
                answer=row["target"],
                source="bigbench/causal_judgment",
            ))
        logger.info(f"[bigbench_causal] {len(entries)} entries")
        return entries

    def build_messages(self, datapoint, prompt_request) -> list[dict[str, str]]:
        raise NotImplementedError

    def evaluate(self, trajectory):
        raise NotImplementedError

    def resolve_pregenerated(self, pickle_path):
        raise NotImplementedError


