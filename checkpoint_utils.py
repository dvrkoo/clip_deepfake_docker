from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

import torch


def _normalize_fc_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor] | None:
    keys = set(state_dict.keys())
    if keys == {"weight", "bias"}:
        return {"weight": state_dict["weight"], "bias": state_dict["bias"]}

    if {"fc.weight", "fc.bias"}.issubset(keys):
        return {
            "weight": state_dict["fc.weight"],
            "bias": state_dict["fc.bias"],
        }

    return None


def load_checkpoint_into_model(
    model, checkpoint_path: str, strict: bool = True
) -> Tuple[str, str]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")

    full_state_candidates = []
    if isinstance(ckpt_obj, (dict, OrderedDict)):
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], (dict, OrderedDict)):
            full_state_candidates.append(("checkpoint['model']", ckpt_obj["model"]))
        if "state_dict" in ckpt_obj and isinstance(
            ckpt_obj["state_dict"], (dict, OrderedDict)
        ):
            full_state_candidates.append(
                ("checkpoint['state_dict']", ckpt_obj["state_dict"])
            )
        full_state_candidates.append(("checkpoint", ckpt_obj))

    full_errors = []
    for source, state_dict in full_state_candidates:
        try:
            model.load_state_dict(state_dict, strict=strict)
            return "full", f"Loaded full model weights from {source}"
        except Exception as exc:
            full_errors.append(f"{source}: {exc}")

    if not hasattr(model, "fc"):
        detail = (
            " | ".join(full_errors) if full_errors else "No compatible state_dict found"
        )
        raise ValueError(
            f"Could not load checkpoint as full model and model has no fc layer. {detail}"
        )

    fc_candidates = []
    if isinstance(ckpt_obj, (dict, OrderedDict)):
        normalized = _normalize_fc_state_dict(ckpt_obj)
        if normalized is not None:
            fc_candidates.append(("checkpoint", normalized))

        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], (dict, OrderedDict)):
            normalized_model = _normalize_fc_state_dict(ckpt_obj["model"])
            if normalized_model is not None:
                fc_candidates.append(("checkpoint['model']", normalized_model))

        if "state_dict" in ckpt_obj and isinstance(
            ckpt_obj["state_dict"], (dict, OrderedDict)
        ):
            normalized_sd = _normalize_fc_state_dict(ckpt_obj["state_dict"])
            if normalized_sd is not None:
                fc_candidates.append(("checkpoint['state_dict']", normalized_sd))

    fc_errors = []
    for source, fc_state_dict in fc_candidates:
        try:
            model.fc.load_state_dict(fc_state_dict, strict=True)
            return "fc_only", f"Loaded fc-only weights from {source}"
        except Exception as exc:
            fc_errors.append(f"{source}: {exc}")

    details = []
    if full_errors:
        details.append("full load failed: " + " | ".join(full_errors))
    if fc_errors:
        details.append("fc-only load failed: " + " | ".join(fc_errors))
    detail_msg = " ; ".join(details) if details else "unknown checkpoint format"
    raise ValueError(f"Unsupported checkpoint format for this model: {detail_msg}")
