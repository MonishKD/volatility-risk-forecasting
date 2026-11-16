"""Experiment configuration loaded from a YAML file."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class Config:
    symbols: tuple[str, ...]
    data_start: str
    data_end: str
    auto_adjust: bool
    raw_dir: Path
    initial_train_end: pd.Timestamp
    dev_start: pd.Timestamp
    dev_end: pd.Timestamp
    horizons: tuple[int, ...]
    refit_frequency: str
    models: tuple[str, ...]
    floor_fraction: float
    output_dir: Path
    raw: dict
    config_id: str


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    # Hash of the parsed content: identical settings give an identical id,
    # so every forecast row can be traced back to the settings that produced it.
    config_id = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:12]
    cfg = Config(
        symbols=tuple(raw["symbols"]),
        data_start=str(raw["data"]["start"]),
        data_end=str(raw["data"]["end"]),
        auto_adjust=bool(raw["data"]["auto_adjust"]),
        raw_dir=Path(raw["data"]["raw_dir"]),
        initial_train_end=pd.Timestamp(raw["periods"]["initial_train_end"]),
        dev_start=pd.Timestamp(raw["periods"]["dev_start"]),
        dev_end=pd.Timestamp(raw["periods"]["dev_end"]),
        horizons=tuple(int(h) for h in raw["horizons"]),
        refit_frequency=raw["refit_frequency"],
        models=tuple(raw["models"]),
        floor_fraction=float(raw["floor"]["fraction_of_train_mean_variance"]),
        output_dir=Path(raw["output_dir"]),
        raw=raw,
        config_id=config_id,
    )
    if not cfg.initial_train_end < cfg.dev_start <= cfg.dev_end:
        raise ValueError("periods must satisfy initial_train_end < dev_start <= dev_end")
    if cfg.refit_frequency != "monthly":
        raise ValueError("only refit_frequency: monthly is implemented")
    if cfg.floor_fraction <= 0:
        raise ValueError("the variance floor must be strictly positive")
    return cfg
