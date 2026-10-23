from typing import Dict, List, Any
import json
from pathlib import Path

from torch.utils.data import Dataset


class SelectionDataset(Dataset):
    def __init__(
        self,
        data_files: Dict[str, str],
        split: str,
        task: str,
    ) -> None:
        if task not in ("kv", "table"):
            raise ValueError("The selector task must be kv or table.")
        self.path = Path(data_files[split]).resolve(strict=True)
        self.task = task
        self.offsets: List[int] = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    raise ValueError(f"Blank JSONL row at byte {offset} in {self.path}")
                self.offsets.append(offset)
        if not self.offsets:
            raise ValueError("Selection datasets cannot be empty.")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            row = json.loads(handle.readline())
        if set(row) != {"input_id", "inputs", "labels"}:
            raise ValueError("Expected a frozen input_id/inputs/labels selector row.")
        if row["inputs"]["task"] != self.task:
            raise ValueError("A dataset cannot mix selector tasks.")
        return row
