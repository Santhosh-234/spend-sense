#!/usr/bin/env python3
"""Spend-Sense AI — CLI entry point."""
import sys
from pathlib import Path

import click
import yaml
from loguru import logger


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _setup_logging(config: dict) -> None:
    cfg = config["logging"]
    logger.remove()
    logger.add(sys.stderr, level=cfg["level"], colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add(
        cfg["file"],
        rotation=cfg["rotation"],
        retention=cfg["retention"],
        level=cfg["level"],
        enqueue=True,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
@click.option("--config", default="config.yaml", show_default=True, help="Path to config.yaml")
@click.pass_context
def cli(ctx: click.Context, config: str) -> None:
    """Spend-Sense AI — Automated Expense Intelligence System."""
    cfg = _load_config(config)
    _setup_logging(cfg)
    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg


@cli.command()
@click.pass_context
def watch(ctx: click.Context) -> None:
    """Start the directory watcher. Receipts dropped in data/raw are auto-processed."""
    from src.pipeline import SpendSensePipeline
    from src.ingestion.watcher import ReceiptWatcher

    cfg      = ctx.obj["config"]
    pipeline = SpendSensePipeline(cfg)
    watcher  = ReceiptWatcher(cfg, pipeline.process_receipt)
    watcher.start()


@cli.command()
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def ingest(ctx: click.Context, image_path: Path) -> None:
    """Manually ingest a single receipt IMAGE_PATH."""
    from src.pipeline import SpendSensePipeline

    pipeline = SpendSensePipeline(ctx.obj["config"])
    result   = pipeline.process_receipt(image_path)

    click.echo("\n── Receipt Summary ──────────────────────────")
    click.echo(f"  Merchant  : {result.get('merchant', 'N/A')}")
    click.echo(f"  Date      : {result.get('date', 'N/A')}")
    click.echo(f"  Amount    : ${result.get('total_amount') or 0:.2f}")
    click.echo(f"  Category  : {result['category']}  ({result['category_confidence']:.1%})")
    click.echo("─────────────────────────────────────────────\n")


@cli.command()
@click.argument("question")
@click.pass_context
def ask(ctx: click.Context, question: str) -> None:
    """Ask a natural-language QUESTION about your expenses (uses RAG + Claude)."""
    from src.pipeline import SpendSensePipeline

    pipeline = SpendSensePipeline(ctx.obj["config"])
    result   = pipeline.ask(question)

    click.echo(f"\n{result['answer']}")
    if result["sources"]:
        click.echo("\n── Sources ──────────────────────────────────")
        for s in result["sources"]:
            amount = f"${s['amount']:.2f}" if s["amount"] else "N/A"
            click.echo(f"  • {s['merchant']} | {s['date']} | {amount} | [{s['category']}]")
        click.echo("─────────────────────────────────────────────\n")


@cli.command()
@click.pass_context
def train(ctx: click.Context) -> None:
    """(Re)train the expense classifier on synthetic bootstrap data."""
    from src.models.classifier import ExpenseClassifier

    clf     = ExpenseClassifier(ctx.obj["config"])
    metrics = clf.train_on_synthetic()
    click.echo(
        f"Training complete — "
        f"CV F1: {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f} "
        f"over {metrics['n_samples']} samples"
    )


if __name__ == "__main__":
    cli()
