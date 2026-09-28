"""Config models and YAML loaders.

Everything industry-specific lives in config/<industry>/{prompts,brands}.yaml;
shared settings live in config/settings.yaml and config/pricing.yaml.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


# ---------------------------------------------------------------- prompts.yaml
class VariableValue(BaseModel):
    value: str  # tag value stored for filtering
    text: str   # text inserted into the template

    @model_validator(mode="before")
    @classmethod
    def _from_str(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"value": data, "text": data}
        if isinstance(data, dict) and "text" not in data:
            return {**data, "text": data["value"]}
        return data


class Template(BaseModel):
    id: str
    text: str
    intent: str
    language: str = "en"
    sample: int | None = Field(default=None, ge=1)
    tags: dict[str, str] = Field(default_factory=dict)  # static extra tags

    @property
    def placeholders(self) -> list[str]:
        return list(dict.fromkeys(PLACEHOLDER_RE.findall(self.text)))


class PromptConfig(BaseModel):
    industry: str
    seed: int = 0
    segments: list[str]
    tag_defaults: dict[str, str] = Field(default_factory=dict)
    variables: dict[str, list[VariableValue]]
    templates: list[Template]

    @model_validator(mode="after")
    def _check(self) -> "PromptConfig":
        ids = [t.id for t in self.templates]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate template ids")
        for t in self.templates:
            missing = [p for p in t.placeholders if p not in self.variables]
            if missing:
                raise ValueError(f"template {t.id!r} uses undefined variables {missing}")
        return self


# ----------------------------------------------------------------- brands.yaml
class Brand(BaseModel):
    key: str
    name: str
    parent: str | None = None
    aliases: list[str] = Field(default_factory=list)
    properties: list[str] = Field(default_factory=list)
    case_sensitive: bool = False
    ambiguous: bool = False
    exclude_patterns: list[str] = Field(default_factory=list)
    context_words: list[str] = Field(default_factory=list)

    @field_validator("key")
    @classmethod
    def _key(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z0-9_]+", v):
            raise ValueError(f"brand key must be snake_case: {v!r}")
        return v

    @model_validator(mode="after")
    def _defaults(self) -> "Brand":
        if self.name not in self.aliases:
            self.aliases.insert(0, self.name)
        for p in self.exclude_patterns:
            re.compile(p)  # fail fast on bad regex
        return self


class BrandConfig(BaseModel):
    industry: str
    default_context_words: list[str] = Field(default_factory=list)
    brands: list[Brand]

    @model_validator(mode="after")
    def _unique(self) -> "BrandConfig":
        keys = [b.key for b in self.brands]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate brand keys")
        return self

    def by_key(self) -> dict[str, Brand]:
        return {b.key: b for b in self.brands}


# --------------------------------------------------------------- settings.yaml
class CollectionSettings(BaseModel):
    concurrency: int = 8
    max_retries: int = 6
    request_timeout_s: float = 120
    max_output_tokens: int = 1500
    reasoning_effort: str | None = "low"
    default_budget_usd: float = 5.0
    seed: int = 42


class EstimateSettings(BaseModel):
    avg_output_tokens: int = 1000
    avg_web_search_input_tokens: int = 8000


class DetectionSettings(BaseModel):
    extraction_model: str = "gpt-5-nano"
    extraction_provider: str = "openai"
    context_window_chars: int = 80


class Settings(BaseModel):
    db_path: str = "data/brand_tracking.db"
    default_industry: str = "luxury_hotels"
    collection: CollectionSettings = Field(default_factory=CollectionSettings)
    estimates: EstimateSettings = Field(default_factory=EstimateSettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)

    @property
    def db_url(self) -> str:
        path = Path(os.environ.get("BT_DB_PATH", self.db_path))
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path}"


# --------------------------------------------------------------------- loaders
def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def industry_dir(industry: str, config_dir: Path = CONFIG_DIR) -> Path:
    d = config_dir / industry
    if not d.is_dir():
        available = sorted(p.name for p in config_dir.iterdir() if p.is_dir())
        raise FileNotFoundError(f"no config for industry {industry!r}; available: {available}")
    return d


def load_prompt_config(industry: str, config_dir: Path = CONFIG_DIR) -> PromptConfig:
    return PromptConfig(**_load_yaml(industry_dir(industry, config_dir) / "prompts.yaml"))


def load_brand_config(industry: str, config_dir: Path = CONFIG_DIR) -> BrandConfig:
    return BrandConfig(**_load_yaml(industry_dir(industry, config_dir) / "brands.yaml"))


def load_settings(config_dir: Path = CONFIG_DIR) -> Settings:
    path = config_dir / "settings.yaml"
    return Settings(**_load_yaml(path)) if path.exists() else Settings()


def load_pricing(config_dir: Path = CONFIG_DIR) -> dict[str, dict[str, Any]]:
    return _load_yaml(config_dir / "pricing.yaml")
