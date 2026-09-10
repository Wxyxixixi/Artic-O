"""Config system for LatentArc.

Uses OmegaConf for YAML loading and structured configs with attribute-style
access. Models are registered by name so build_model() can instantiate them
without hard-coded imports.

Typical usage:
    from src.configs import load_config, build_model

    cfg = load_config("ptscond")          # load by name from src/configs/
    cfg = load_config("path/to/my.yaml")  # or load by explicit path

    model = build_model(cfg)              # instantiates cfg.model.name(**cfg.model.params)

    # CLI-style overrides via OmegaConf dotlist:
    cfg = load_config("ptscond", overrides=["model.params.num_3d_tokens=512"])
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, List, Optional

from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Maps config name -> (module_path, class_name).
# Add new models here so build_model() can find them without manual imports.
_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "Nova3rImgCondArticulated": (
        "src.models.nova3r_img_cond_articulated",
        "Nova3rImgCondArticulated",
    ),
}

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
_DATASET_REGISTRY: dict[str, tuple[str, str]] = {
    "BaseDataset": ("src.datasets.base_dataset", "BaseDataset"),
    "ArticODataset": ("src.datasets.artic_o_dataset", "ArticODataset"),
}


_CONFIGS_DIR = Path(__file__).parent


def load_config(
    name_or_path: str,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Load a config by short name or explicit file path.

    Args:
        name_or_path: Either a short name (e.g. ``"ptscond"``) that resolves
            to ``src/configs/<name>.yaml``, or an explicit path to a YAML file.
        overrides: Optional list of OmegaConf dotlist overrides, e.g.
            ``["model.params.num_3d_tokens=512", "amp_dtype=fp32"]``.

    Returns:
        :class:`~omegaconf.DictConfig` with attribute-style access.
    """
    path = Path(name_or_path)
    if not path.exists():
        # Try resolving as a short name under src/configs/
        path = _CONFIGS_DIR / f"{name_or_path}.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {name_or_path!r}. "
            f"Looked in {_CONFIGS_DIR} and as a direct path."
        )

    cfg: DictConfig = OmegaConf.load(path)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def build_model(cfg: DictConfig, **extra_kwargs: Any):
    """Instantiate the model described in *cfg*.

    Expects ``cfg.model.name`` to be a registered model name and
    ``cfg.model.params`` to hold the constructor keyword arguments.

    Args:
        cfg: Top-level config loaded via :func:`load_config`.
        **extra_kwargs: Extra keyword arguments forwarded to the model
            constructor, overriding anything in ``cfg.model.params``.

    Returns:
        An :class:`~torch.nn.Module` instance.
    """
    model_name: str = cfg.model.name
    if model_name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {model_name!r}. "
            f"Registered models: {list(_MODEL_REGISTRY)}"
        )

    module_path, class_name = _MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)

    # OmegaConf -> plain dict so **-unpacking works with all constructors
    params: dict = OmegaConf.to_container(cfg.model.params, resolve=True)
    params.update(extra_kwargs)

    # The inner 'cfg' key must be a DictConfig (the model reads it with
    # attribute access and .get()), so convert back after to_container.
    if "cfg" in params and isinstance(params["cfg"], dict):
        params["cfg"] = OmegaConf.create(params["cfg"])

    return model_cls(**params)


def build_dataset(cfg: DictConfig, **extra_kwargs: Any):
    """Instantiate the dataset described in *cfg*.

    Expects ``cfg.name`` to be a registered dataset name and
    ``cfg.params`` to hold the constructor keyword arguments.

    Args:
        cfg: Top-level config loaded via :func:`load_config`.
        **extra_kwargs: Extra keyword arguments forwarded to the dataset
            constructor, overriding anything in ``cfg.params``.

    Returns:
        A :class:`~torch.utils.data.Dataset` instance.
    """
    dataset_name: str = cfg.name
    if dataset_name not in _DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset {dataset_name!r}. "
            f"Registered datasets: {list(_DATASET_REGISTRY)}"
        )

    module_path, class_name = _DATASET_REGISTRY[dataset_name]
    module = importlib.import_module(module_path)
    dataset_cls = getattr(module, class_name)

    params: dict = OmegaConf.to_container(cfg.params, resolve=True)
    params.update(extra_kwargs)

    return dataset_cls(**params)


def register_dataset(name: str, module_path: str, class_name: str) -> None:
    """Register a dataset so it can be built by :func:`build_dataset`."""
    _DATASET_REGISTRY[name] = (module_path, class_name)



def register_model(name: str, module_path: str, class_name: str) -> None:
    """Register a model so it can be built by :func:`build_model`.

    Args:
        name: The string used in ``cfg.model.name``.
        module_path: Dotted Python module path (e.g. ``"src.models.my_model"``).
        class_name: Class name inside that module.
    """
    _MODEL_REGISTRY[name] = (module_path, class_name)
