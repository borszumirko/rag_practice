"""Compose the same Hydra configuration for ASGI imports and the CLI."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Load server defaults without reading CLI arguments or requiring credentials."""
    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path(__file__).parent / "conf")
    ):
        return compose(config_name="server", overrides=overrides or [])
