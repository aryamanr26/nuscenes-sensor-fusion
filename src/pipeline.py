"""Scaffold for the colorized point cloud pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    dataroot: Path
    version: str
    scenes_config: Path
    output_dir: Path


class ColorizationPipeline:
    """Modular pipeline scaffold.

    Intended stages:
    1) scene/token resolution
    2) LiDAR -> camera transformations
    3) projection and pixel color sampling
    4) multi-camera fusion
    5) point cloud export
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    def run(self) -> None:
        raise NotImplementedError(
            "Pipeline execution is intentionally scaffolded; "
            "implement stage modules incrementally."
        )
