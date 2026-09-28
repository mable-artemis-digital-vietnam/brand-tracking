"""Brand-tracking management CLI.

    python bt.py generate [--industry luxury_hotels] [--show 10] [--csv prompts.csv]
"""
from __future__ import annotations

import csv
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from brand_tracking.config import load_prompt_config, load_settings
from brand_tracking.prompts.generator import generate as generate_prompts
from brand_tracking.prompts.generator import summarize

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.callback()
def main():
    """Brand visibility tracker: generate prompts, detect brands, compute metrics, export."""


def _industry(industry: str | None) -> str:
    return industry or load_settings().default_industry


@app.command()
def generate(
    industry: str = typer.Option(None, help="Industry config folder under config/."),
    show: int = typer.Option(5, help="Number of sample prompts to print."),
    csv_path: Path = typer.Option(None, "--csv", help="Also write the expanded prompts to CSV."),
):
    """Expand prompt templates and show the count before anything is run."""
    industry = _industry(industry)
    cfg = load_prompt_config(industry)
    prompts = generate_prompts(cfg)
    s = summarize(cfg, prompts)

    t = Table(title=f"Prompt templates — {industry}")
    for col in ("template", "intent", "combinations", "selected"):
        t.add_column(col, justify="right" if col in ("combinations", "selected") else "left")
    for row in s["templates"]:
        t.add_row(row["id"], row["intent"], str(row["combinations"]), str(row["selected"]))
    t.add_section()
    t.add_row("[bold]TOTAL", "", "", f"[bold]{s['total']}")
    console.print(t)

    for seg, counts in s["segments"].items():
        console.print(f"[cyan]{seg}[/]: " + ", ".join(f"{k} ({v})" for k, v in counts.items()))

    console.print(f"\n[bold]Sample prompts[/] ({min(show, len(prompts))} of {len(prompts)}):")
    step = max(1, len(prompts) // max(show, 1))
    for p in prompts[::step][:show]:
        tags = " ".join(f"{k}={v}" for k, v in p.tags.items())
        console.print(f"  • {p.text}\n    [dim]{p.template_id} | {tags}[/]")

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["prompt_hash", "template_id", "text", *cfg.segments])
            for p in prompts:
                w.writerow([p.prompt_hash, p.template_id, p.text, *(p.tags[k] for k in cfg.segments)])
        console.print(f"\nWrote {len(prompts)} prompts to {csv_path}")


if __name__ == "__main__":
    app()
