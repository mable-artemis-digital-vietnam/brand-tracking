import pytest

from brand_tracking.config import PromptConfig, load_brand_config, load_prompt_config
from brand_tracking.prompts.generator import generate, summarize


def _cfg(**over):
    base = {
        "industry": "test",
        "seed": 1,
        "segments": ["destination", "traveler_type", "intent", "language"],
        "tag_defaults": {"destination": "global"},
        "variables": {
            "destination": ["Bali", {"value": "Maldives", "text": "the Maldives"}],
            "traveler_type": [{"value": "honeymoon", "text": "a honeymoon"}, "business"],
        },
        "templates": [
            {"id": "a", "intent": "recommendation", "text": "Best hotel in {destination} for {traveler_type}?"},
            {"id": "b", "intent": "best_of_list", "text": "Top brands for {traveler_type}?"},
        ],
    }
    base.update(over)
    return PromptConfig(**base)


def test_cartesian_expansion_and_tags():
    prompts = generate(_cfg())
    assert len(prompts) == 4 + 2
    p = next(p for p in prompts if "the Maldives" in p.text and "a honeymoon" in p.text)
    assert p.tags == {"destination": "Maldives", "traveler_type": "honeymoon",
                      "intent": "recommendation", "language": "en"}
    # template without {destination} falls back to tag_defaults
    b = next(p for p in prompts if p.template_id == "b")
    assert b.tags["destination"] == "global"


def test_sampling_is_deterministic():
    cfg = _cfg(templates=[{"id": "a", "intent": "x", "sample": 2,
                           "text": "Hotel in {destination} for {traveler_type}?"}])
    first = [p.text for p in generate(cfg)]
    assert len(first) == 2
    assert first == [p.text for p in generate(cfg)]


def test_dedupe_identical_text():
    cfg = _cfg(templates=[
        {"id": "a", "intent": "x", "text": "Hotels in {destination}?"},
        {"id": "b", "intent": "y", "text": "Hotels in {destination}?"},
    ])
    assert len(generate(cfg)) == 2


def test_undefined_variable_rejected():
    with pytest.raises(ValueError, match="undefined variables"):
        _cfg(templates=[{"id": "a", "intent": "x", "text": "Hi {nope}"}])


def test_sample_config_is_about_50_prompts():
    cfg = load_prompt_config("luxury_hotels")
    prompts = generate(cfg)
    s = summarize(cfg, prompts)
    assert 40 <= s["total"] <= 60
    assert set(s["segments"]["traveler_type"]) >= {"honeymoon", "family", "business", "solo"}
    assert all(set(p.tags) == set(cfg.segments) for p in prompts)


def test_sample_brands_config_loads():
    brands = load_brand_config("luxury_hotels")
    assert len(brands.brands) == 20
    assert "Four Seasons" in brands.by_key()["four_seasons"].aliases


def test_balanced_sampling_spreads_values():
    from collections import Counter
    cfg = load_prompt_config("luxury_hotels")
    prompts = [p for p in generate(cfg) if p.template_id == "rec_by_traveler"]  # 12 of 8×4
    dest = Counter(p.tags["destination"] for p in prompts)
    trav = Counter(p.tags["traveler_type"] for p in prompts)
    assert max(dest.values()) - min(dest.values()) <= 1 and len(dest) == 8
    assert set(trav.values()) == {3}
