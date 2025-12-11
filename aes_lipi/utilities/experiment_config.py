"""Centralized experiment configuration loader and dataclass.

Supports JSON and YAML (if PyYAML is installed). Provides an
`ExperimentConfig` dataclass and helpers to load/save configs.
"""
from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, Optional


@dataclass
class ExperimentConfig:
    epochs: int = 400
    learning_rate: float = 1e-05
    visualize: str = "final"
    trials: int = 1
    checkpoint_interval: int = 10000
    rng_seed: Optional[int] = 1
    radius: int = 1
    log_level: str = "debug"
    ae_quality_measures: str = "L1"
    calculate_test_loss: bool = True
    output_dir: str = "out_experiments"
    batch_size: int = 5
    lexi_threshold: float = 0.1
    # allow arbitrary extra fields
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        base_fields = {f.name for f in cls.__dataclass_fields__.values()}
        init_kwargs = {k: v for k, v in d.items() if k in base_fields and k != "extras"}
        extras = {k: v for k, v in d.items() if k not in init_kwargs}
        inst = cls(**init_kwargs)
        inst.extras.update(extras)
        return inst

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        # flatten extras back into top-level
        extras = out.pop("extras", {})
        out.update(extras)
        return out


def _load_yaml(content: str) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as e:
        raise ImportError("PyYAML is required to load YAML files. Install with `pip install pyyaml`") from e
    # tolerate markdown-style fenced code blocks (```yaml / ```)
    txt = content
    # strip leading/trailing whitespace
    txt = txt.strip()
    # remove starting fence like ```yaml or ``` if present
    if txt.startswith("```"):
        lines = txt.splitlines()
        # if first line is a fence, drop it
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        # if last line is a fence, drop it
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        txt = "\n".join(lines)
    return yaml.safe_load(txt)


def load_config(path: str) -> Dict[str, Any]:
    """Load a JSON or YAML config file and return a dict."""
    with open(path, "r") as fd:
        text = fd.read()
    if path.endswith(".json"):
        return json.loads(text)
    if path.endswith(".yml") or path.endswith(".yaml"):
        return _load_yaml(text)
    # try JSON then YAML
    try:
        return json.loads(text)
    except Exception:
        return _load_yaml(text)


def save_config(path: str, config: Dict[str, Any]) -> None:
    """Save the config dict as JSON or YAML according to file extension."""
    if path.endswith(".json"):
        with open(path, "w") as fd:
            json.dump(config, fd, indent=2)
        return
    try:
        import yaml

        with open(path, "w") as fd:
            yaml.safe_dump(config, fd)
        return
    except Exception:
        # fallback to JSON
        with open(path, "w") as fd:
            json.dump(config, fd, indent=2)
