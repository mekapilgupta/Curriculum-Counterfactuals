"""
Provenance, hashing, git tracking, and reproducibility utilities.
"""

import hashlib
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
import yaml
from curriculum_retrieval.schemas import RunManifest


def compute_text_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash of normalized text."""
    normalized = " ".join(text.strip().split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_raw_hash(text: str) -> str:
    """Compute SHA-256 hash of raw UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_dict_hash(d: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of a dictionary."""
    serialized = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_git_commit() -> Optional[str]:
    """Retrieve the current git commit hash if available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return None


def set_seed(seed: int = 42) -> None:
    """Set seeds across standard library, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load and parse YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_run_manifest(
    command: str,
    config: Dict[str, Any],
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    metrics: Dict[str, Any] = None,
    output_path: Optional[str | Path] = None,
) -> RunManifest:
    """Create and persist a RunManifest JSON file."""
    config_hash = compute_dict_hash(config)
    git_commit = get_git_commit()
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{config_hash[:8]}"

    manifest = RunManifest(
        run_id=run_id,
        command=command,
        config_hash=config_hash,
        git_commit=git_commit,
        timestamp=datetime.utcnow().isoformat(),
        inputs={k: str(v) for k, v in inputs.items()},
        outputs={k: str(v) for k, v in outputs.items()},
        metrics=metrics or {},
    )

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2)

    return manifest
