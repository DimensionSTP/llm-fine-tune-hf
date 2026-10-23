from typing import Dict, Optional, Any
import hashlib
import json
from pathlib import Path
import uuid

import torch
from torch import nn

from peft import PeftModel

from safetensors.torch import load_file, save_file

from .selection_heads import BilinearScores


def save_selection_checkpoint(
    adapter: PeftModel,
    selector: nn.Module,
    output: Path,
    selection_config: Dict[str, Any],
    input_contract: Dict[str, Any],
    model_reference: Dict[str, Any],
    adapter_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    selector_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    if output.exists():
        raise FileExistsError(output)
    staging_dir = output.parent / f"staging_{output.name}_{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True)
    adapter.save_pretrained(
        save_directory=staging_dir,
        safe_serialization=True,
        state_dict=adapter_state_dict,
    )
    required_files = ("adapter_config.json", "adapter_model.safetensors")
    if any(not (staging_dir / name).is_file() for name in required_files):
        raise ValueError(
            "A checkpoint must include the PEFT adapter, not only the selector."
        )
    head_weights = (
        selector.state_dict() if selector_state_dict is None else selector_state_dict
    )
    save_file(
        {
            name: value.detach().cpu().contiguous().clone()
            for name, value in head_weights.items()
        },
        str(staging_dir / "selection_heads.safetensors"),
    )
    payloads = {
        "selection_config.json": selection_config,
        "vlm_input_contract.json": input_contract,
        "model_reference.json": model_reference,
    }
    for name, payload in payloads.items():
        with (staging_dir / name).open("x") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
            )
    manifest = {
        "version": selection_config["version"],
        "task": selector.task,
        "kind": "adapter_and_selection_weights_not_optimizer_resume",
        "files": {
            str(path.relative_to(staging_dir)): _file_hash(path=path)
            for path in staging_dir.rglob("*")
            if path.is_file()
        },
    }
    with (staging_dir / "run_manifest.json").open("x") as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
        )
    if output.exists():
        raise FileExistsError(output)
    staging_dir.rename(output)


def validate_selection_checkpoint(
    selector: nn.Module,
    checkpoint: Path,
    selection_config: Dict[str, Any],
    input_contract: Dict[str, Any],
    model_reference: Dict[str, Any],
) -> Dict[str, Any]:
    manifest = json.loads((checkpoint / "run_manifest.json").read_text())
    if manifest["task"] != selector.task:
        raise ValueError("Checkpoint task differs from the selector task.")
    for name, digest in manifest["files"].items():
        if _file_hash(path=checkpoint / name) != digest:
            raise ValueError(f"Checkpoint artifact changed: {name}")
    expected = {
        "selection_config.json": selection_config,
        "vlm_input_contract.json": input_contract,
        "model_reference.json": model_reference,
    }
    for name, value in expected.items():
        if json.loads((checkpoint / name).read_text()) != value:
            raise ValueError(f"Checkpoint contract differs: {name}")
    return manifest


def load_selection_weights(
    selector: nn.Module,
    checkpoint: Path,
    selection_config: Dict[str, Any],
    input_contract: Dict[str, Any],
    model_reference: Dict[str, Any],
) -> Dict[str, Any]:
    manifest = validate_selection_checkpoint(
        selector=selector,
        checkpoint=checkpoint,
        selection_config=selection_config,
        input_contract=input_contract,
        model_reference=model_reference,
    )
    state_dict = load_file(str(checkpoint / "selection_heads.safetensors"))
    for name, module in selector.named_modules():
        if isinstance(module, BilinearScores):
            state_dict[f"{name}.bias"] = state_dict[f"{name}.bias"].reshape(1)
    selector.load_state_dict(
        state_dict,
        strict=True,
    )
    return manifest


def _file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(
            handle,
            "sha256",
        ).hexdigest()
