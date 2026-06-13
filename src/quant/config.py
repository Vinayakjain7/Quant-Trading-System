"""Configuration loading.

All tunables live in config.yaml so the strategy is reproducible and
there are no magic numbers scattered through the code.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    # src/quant/config.py -> project root is two levels up from src/
    return Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: dict[str, Any]

    # convenience accessors -------------------------------------------------
    @property
    def tickers(self) -> list[str]:
        return list(self.raw["universe"]["tickers"])

    @property
    def universe_name(self) -> str:
        return self.raw["universe"].get("name", "universe")

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {})

    def get(self, path: str, default: Any = None) -> Any:
        """Dotted lookup, e.g. cfg.get('risk.capital')."""
        node: Any = self.raw
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def end_date(self) -> str:
        end = self.get("data.end")
        return end or _dt.date.today().isoformat()

    @property
    def start_date(self) -> str:
        return self.get("data.start", "2018-01-01")


def load_config(path: str | Path | None = None) -> Config:
    """Load config.yaml from an explicit path or the project root."""
    if path is None:
        path = _project_root() / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path}. Copy the template at the repo root."
        )
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    _validate(raw)
    return Config(raw=raw)


def _validate(raw: dict[str, Any]) -> None:
    required = ["universe", "data", "strategy", "risk", "backtest", "ml"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"config.yaml missing sections: {missing}")
    if not raw["universe"].get("tickers"):
        raise ValueError("config.yaml: universe.tickers is empty")
