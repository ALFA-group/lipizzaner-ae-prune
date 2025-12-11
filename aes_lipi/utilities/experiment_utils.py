import datetime
import json
import os
from typing import Any, Dict, Optional


def timestamp_str(ts: Optional[datetime.datetime] = None) -> str:
    if ts is None:
        ts = datetime.datetime.now()
    return "{:%Y-%m-%d_%H:%M:%S.%f}".format(ts)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def build_variant_name(variant: Dict[str, Any]) -> str:
    return "_".join(["_".join([str(k)[:2], str(v)]) for k, v in variant.items()])


def build_trial_output_dir(
    base_dir: str,
    trial_idx: int,
    variant: Dict[str, Any],
    timestamp: Optional[datetime.datetime] = None,
) -> str:
    variant_name = build_variant_name(variant)
    ts = timestamp_str(timestamp)
    path = os.path.join(base_dir, f"trial_{trial_idx}_{variant_name}_{ts}")
    ensure_dir(path)
    return path


def save_params_json(out_dir: str, params: Dict[str, Any]) -> str:
    out_path = os.path.join(out_dir, "params.json")
    with open(out_path, "w") as fd:
        json.dump(params, fd, indent=1)
    return out_path


def normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    def _convert_value(v: Any) -> Any:
        # normalize booleans in string form
        if isinstance(v, str):
            if v in ("True", "False"):
                return v == "True"
            # try int
            try:
                iv = int(v)
                return iv
            except Exception:
                pass
            # try float
            try:
                fv = float(v)
                # convert floats that are whole numbers to int
                if fv.is_integer():
                    return int(fv)
                return fv
            except Exception:
                return v
        # lists/tuples: recursively convert
        if isinstance(v, list):
            return [_convert_value(x) for x in v]
        if isinstance(v, tuple):
            return tuple(_convert_value(x) for x in v)
        return v

    for k, v in list(params.items()):
        params[k] = _convert_value(v)
    return params
