"""Expand prompt templates × variables into a tagged, deduplicated prompt set."""
from __future__ import annotations

import hashlib
import itertools
import random
from collections import Counter
from dataclasses import dataclass, field

from brand_tracking.config import PromptConfig, Template


@dataclass(frozen=True)
class GeneratedPrompt:
    text: str
    template_id: str
    language: str
    tags: dict[str, str] = field(hash=False)

    @property
    def prompt_hash(self) -> str:
        return prompt_hash(self.text, self.language)


def prompt_hash(text: str, language: str) -> str:
    return hashlib.sha1(f"{language}\n{text.strip()}".encode("utf-8")).hexdigest()


def _expand_template(cfg: PromptConfig, t: Template) -> list[GeneratedPrompt]:
    names = t.placeholders
    combos = itertools.product(*(cfg.variables[n] for n in names)) if names else [()]
    out = []
    for combo in combos:
        chosen = dict(zip(names, combo))
        text = t.text.format(**{n: v.text for n, v in chosen.items()})
        tags = {k: cfg.tag_defaults.get(k, "unspecified") for k in cfg.segments}
        tags.update({n: v.value for n, v in chosen.items()})
        tags.update(t.tags)
        tags["intent"] = t.intent
        tags["language"] = t.language
        out.append(GeneratedPrompt(text=text, template_id=t.id, language=t.language, tags=tags))
    return out


def _balanced_sample(items: list[GeneratedPrompt], keys: list[str], n: int, seed: str) -> list[GeneratedPrompt]:
    """Seeded sample where each variable value appears as evenly as possible.

    Greedy: repeatedly take the (shuffled) combination whose values have been used least.
    """
    rng = random.Random(seed)  # string seeds are stable across runs
    pool = list(range(len(items)))
    rng.shuffle(pool)
    used: Counter = Counter()
    chosen: list[int] = []
    for _ in range(n):
        best = min(pool, key=lambda i: sum(used[(k, items[i].tags[k])] for k in keys))
        pool.remove(best)
        chosen.append(best)
        used.update((k, items[best].tags[k]) for k in keys)
    return [items[i] for i in sorted(chosen)]


def combination_count(cfg: PromptConfig, t: Template) -> int:
    n = 1
    for name in t.placeholders:
        n *= len(cfg.variables[name])
    return n


def generate(cfg: PromptConfig) -> list[GeneratedPrompt]:
    """Return the full prompt set: every template expanded, sampled if requested, deduped."""
    prompts: list[GeneratedPrompt] = []
    seen: set[str] = set()
    for t in cfg.templates:
        expanded = _expand_template(cfg, t)
        if t.sample and t.sample < len(expanded):
            expanded = _balanced_sample(expanded, t.placeholders, t.sample, f"{cfg.seed}:{t.id}")
        for p in expanded:
            if p.prompt_hash not in seen:
                seen.add(p.prompt_hash)
                prompts.append(p)
    return prompts


def summarize(cfg: PromptConfig, prompts: list[GeneratedPrompt]) -> dict:
    per_template = Counter(p.template_id for p in prompts)
    return {
        "total": len(prompts),
        "templates": [
            {
                "id": t.id,
                "intent": t.intent,
                "combinations": combination_count(cfg, t),
                "selected": per_template.get(t.id, 0),
            }
            for t in cfg.templates
        ],
        "segments": {
            seg: dict(Counter(p.tags.get(seg, "unspecified") for p in prompts).most_common())
            for seg in cfg.segments
        },
    }
