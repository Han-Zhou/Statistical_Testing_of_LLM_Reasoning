import json
import logging
import urllib.request

from pathlib import Path
from datasets.base import Dataset


logger = logging.getLogger(__name__)

class BfclDataset(Dataset):
    def __init__(self):
        super().__init__("bfcl")
    

    def load_datapoints(self) -> list:

    def build_messages(self, datapoint, prompt_request) -> list[dict[str, str]]:
        raise NotImplementedError

    def evaluate(self, trajectory):
        raise NotImplementedError

    def resolve_pregenerated(self, pickle_path):
        raise NotImplementedError


