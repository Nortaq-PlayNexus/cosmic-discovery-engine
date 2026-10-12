"""Command-line interface for Cosmic Discovery Engine."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

import click

from .config import get_db_path, load_config
from .database import Database
from .models import CandidateStatus
from .pipeline.ingest import create_demo_pipeline, create_mast_pipeline, IngestPipeline


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config YAML file.")
@click.option("--log-level", "-l", default="INFO", show_default=True, help="Logging level.")
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], log_level: str) -> None:
    """Cosmic Discovery Engine — astronomical anomaly discovery platform."""
    _setup_logging(log_level)
    cfg = load_config(config) if config else load_config()
    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg
    ctx.obj["db"] = Database(get_db_path(cfg))


# ── Demo commands ──────────────────────────────────────────────────────────


@cli.command("demo-ingest")
@click.option("--target", "-t", multiple=True, help="Target source ID(s) to process.")
@click.option("--all", "process_all", is_flag=True, help="Process all demo targets.")
@click.pass_context
def demo_ingest(ctx: click.Context, target: tuple[str, ...], process_all: bool) -> None:
    """Run the pipeline on demo (synthetic) data."""
    pipeline: IngestPipeline = create_demo_pipeline()
    config = ctx.obj["config"]

    if process_all:
        from .sources.demo import DEMO_TARGETS
        source_ids = [t[0] for t in DEMO_TARGETS]
    elif target:
        source_ids = list(target)
    else:
        click.echo("No targets specified. Use --all or --target TIC_100000001")
        return

    click.echo(f"Processing {len(source_ids)} demo targets...")
    results = pipeline.process_many(source_ids)

    n_success = sum(1 for r in results if r.success)
    n_candidates = sum(len(r.candidates) for r in results)
    click.echo(f"Results: {n_success}/{len(results)} succeeded, {n_candidates} candidates found")

    for result in results:
        if result.success:
            status = "OK"
            if result.candidates:
                c = result.candidates[0]
                status += f" -> candidate P={c.period_days:.4f}d, score={c.score:.1f}"
            click.echo(f"  {result.source_id}: {status}")
        else:
            click.echo(f"  {result.source_id}: FAILED - {result.error}")


@cli.command("demo-search")
@click.option("--ra", type=float, required=True, help="Right ascension in degrees.")
@click.option("--dec", type=float, required=True, help="Declination in degrees.")
@click.option("--radius", type=float, default=600.0, show_default=True, help="Search radius in arcsec.")
@click.pass_context
def demo_search(ctx: click.Context, ra: float, dec: float, radius: float) -> None:
    """Search demo targets near a sky position."""
    from .sources.demo import DemoSource

    source = DemoSource()
    result = source.search_targets(ra, dec, radius)
    if result.success and result.data:
        click.echo(f"Found {len(result.data)} demo target(s):")
        for t in result.data:
            click.echo(f"  {t.source_id}: RA={t.ra_deg:.4f}, Dec={t.dec_deg:.4f}, V={t.magnitude}")
    else:
        click.echo(f"Search failed: {result.error}")


# ── MAST commands ──────────────────────────────────────────────────────────


@cli.command("mast-search")
@click.option("--ra", type=float, required=True, help="Right ascension in degrees.")
@click.option("--dec", type=float, required=True, help="Declination in degrees.")
@click.option("--radius", type=float, default=600.0, show_default=True, help="Search radius in arcsec.")
@click.pass_context
def mast_search(ctx: click.Context, ra: float, dec: float, radius: float) -> None:
    """Search MAST for TESS targets near a sky position."""
    config = ctx.obj["config"]
    pipeline = create_mast_pipeline()
    result = pipeline.source.search_targets(ra, dec, radius)
    if result.success and result.data:
        click.echo(f"Found {len(result.data)} MAST target(s):")
        for t in result.data:
            click.echo(f"  {t.source_id}: RA={t.ra_deg:.4f}, Dec={t.dec_deg:.4f}")
    else:
        click.echo(f"Search failed: {result.error}")


@cli.command("mast-ingest")
@click.argument("source_ids", nargs=-1, required=True)
@click.option("--sector", type=int, default=None, help="Specific TESS sector.")
@click.pass_context
def mast_ingest(ctx: click.Context, source_ids: tuple[str, ...], sector: Optional[int]) -> None:
    """Fetch and process TESS light curves from MAST."""
    config = ctx.obj["config"]
    pipeline = create_mast_pipeline()
    kwargs = {"sector": sector} if sector else {}

    click.echo(f"Fetching {len(source_ids)} target(s) from MAST...")
    results = pipeline.process_many(list(source_ids), **kwargs)

    n_success = sum(1 for r in results if r.success)
    n_candidates = sum(len(r.candidates) for r in results)
    click.echo(f"Results: {n_success}/{len(results)} succeeded, {n_candidates} candidates found")

    for result in results:
        if result.success:
            status = "OK"
            if result.candidates:
                c = result.candidates[0]
                status += f" -> candidate P={c.period_days:.4f}d, score={c.score:.1f}"
            click.echo(f"  {result.source_id}: {status}")
        else:
            click.echo(f"  {result.source_id}: FAILED - {result.error}")


# ── Candidate listing ──────────────────────────────────────────────────────


@cli.command("list-candidates")
@click.option("--status", "-s", type=click.Choice(["new", "known", "false_positive", "unverified"]), default=None)
@click.option("--min-score", type=float, default=0.0)
@click.option("--limit", "-n", type=int, default=50)
@click.option("--format", "output_format", type=click.Choice(["table", "json", "csv"]), default="table")
@click.pass_context
def list_candidates(
    ctx: click.Context,
    status: Optional[str],
    min_score: float,
    limit: int,
    output_format: str,
) -> None:
    """List discovery candidates."""
    db: Database = ctx.obj["db"]
    status_enum = CandidateStatus(status) if status else None
    candidates = db.get_candidates(status=status_enum, min_score=min_score, limit=limit)

    if not candidates:
        click.echo("No candidates found.")
        return

    if output_format == "json":
        click.echo(json.dumps(candidates, indent=2, default=str))
    elif output_format == "csv":
        import csv
        writer = csv.DictWriter(sys.stdout, fieldnames=candidates[0].keys())
        writer.writeheader()
        writer.writerows(candidates)
    else:
        click.echo(f"{'ID':<18} {'Source':<18} {'Period(d)':<12} {'Depth(ppm)':<12} {'SNR':<8} {'Score':<8} {'Status':<14}")
        click.echo("-" * 100)
        for c in candidates:
            click.echo(
                f"{c['candidate_id']:<18} {c['source_id']:<18} "
                f"{c['period_days']:<12.4f} {c['depth_ppm']:<12.0f} "
                f"{c['snr']:<8.1f} {c['score']:<8.1f} {c['status']:<14}"
            )
        click.echo(f"\nTotal: {len(candidates)} candidate(s)")


@cli.command("stats")
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show database statistics."""
    db: Database = ctx.obj["db"]
    counts = db.get_candidate_count()
    click.echo("Candidate status counts:")
    for status, count in sorted(counts.items()):
        click.echo(f"  {status}: {count}")
    if not counts:
        click.echo("  (no candidates yet)")

    vcounts = db.get_validation_count()
    if vcounts:
        click.echo("\nValidation verdicts:")
        for verdict, count in sorted(vcounts.items()):
            click.echo(f"  {verdict}: {count}")


# ── Validation ──────────────────────────────────────────────────────────────


@cli.command("validate")
@click.argument("source_ids", nargs=-1)
@click.option("--all", "process_all", is_flag=True, help="Validate every stored candidate.")
@click.option("--trials", "-n", type=int, default=25, show_default=True,
              help="Surrogate trials per candidate. p-value floor is 1/(1+n).")
@click.option("--alpha", type=float, default=0.05, show_default=True, help="FDR level.")
@click.option("--method", type=click.Choice(["BY", "BH"]), default="BY", show_default=True,
              help="FDR procedure. BY is valid under arbitrary dependence.")
@click.option("--sensitivity", is_flag=True, help="Also run the preprocessing sweep (slow).")
@click.pass_context
def validate(
    ctx: click.Context,
    source_ids: tuple[str, ...],
    process_all: bool,
    trials: int,
    alpha: float,
    method: str,
    sensitivity: bool,
) -> None:
    """Run statistical validation on candidates and apply family FDR.

    Each candidate gets an empirical null p-value from surrogate data. The
    whole family is then corrected together, because candidates drawn from the
    same star are not independent.
    """
    from .pipeline.validation import apply_family_fdr, validate_candidate
    from .sources.demo import DemoSource

    db: Database = ctx.obj["db"]
    pipeline_cfg = ctx.obj["config"].get("pipeline", {})
    min_p = pipeline_cfg.get("min_period_days", 0.1)
    max_p = pipeline_cfg.get("max_period_days", 50.0)
    grid = pipeline_cfg.get("bls_duration_grid")

    # Collect candidates to validate.
    from .models import TransitCandidate

    if source_ids:
        rows = []
        for sid in source_ids:
            for c in db.get_candidates(limit=10000):
                if c["source_id"] == sid:
                    rows.append(c)
    elif process_all:
        rows = db.get_candidates(limit=10000)
    else:
        click.echo("Specify source IDs or --all")
        return

    if not rows:
        click.echo("No candidates to validate.")
        return

    click.echo(f"Validating {len(rows)} candidate(s) with {trials} surrogate trials each.")
    click.echo(f"  p-value floor is {1/(1+trials):.4f}; family correction is {method} at alpha={alpha}")
    if method == "BY" and trials < 29:
        click.echo("  note: with fewer than ~29 trials the BY floor may exceed its threshold")
    click.echo("")

    source = DemoSource()
    reports = []
    for row in rows:
        cand = TransitCandidate(
            candidate_id=row["candidate_id"], source_id=row["source_id"],
            period_days=row["period_days"], epoch_bjd=row["epoch_bjd"],
            duration_hours=row["duration_hours"], depth_ppm=row["depth_ppm"],
            snr=row["snr"], n_transits=row["n_transits"],
            duty_cycle=row["duty_cycle"], score=row["score"],
        )
        lc_result = source.fetch_lightcurve(row["source_id"])
        if not lc_result.success or lc_result.data is None:
            click.echo(f"  {row['candidate_id'][:12]}: no light curve available, skipped")
            continue

        report = validate_candidate(
            cand, lc_result.data,
            n_null_trials=trials, alpha=alpha, fdr_method=method,
            min_period_days=min_p, max_period_days=max_p,
            duration_grid=grid, run_sensitivity=sensitivity,
        )
        reports.append(report)
        click.echo(f"  {row['candidate_id'][:12]}: p={report.null_result.p_value:.4f} "
                   f"snr={report.null_result.observed_snr:.1f} -> {report.verdict}")

    # One correction across the whole family.
    apply_family_fdr(reports, alpha=alpha, method=method)

    click.echo("")
    for r in reports:
        db.upsert_validation(r)

    rejected = sum(1 for r in reports if r.fdr and r.candidate_id in set(r.fdr.rejected))
    click.echo(f"Family of {len(reports)}: {rejected} rejected by {method} at alpha={alpha}")
    click.echo("")
    for r in reports:
        mark = "REJECT" if (r.fdr and r.candidate_id in set(r.fdr.rejected)) else "keep"
        click.echo(f"  [{mark:6}] {r.candidate_id[:12]}  p={r.null_result.p_value:.4f}  {r.verdict}")
        for reason in r.reasons:
            click.echo(f"            - {reason}")


@cli.command("benchmark")
@click.option("--trials", "-s", type=int, default=3, show_default=True, help="Seeds per SNR.")
@click.option("--periods", type=int, default=2000, show_default=True, help="BLS period grid size.")
@click.pass_context
def benchmark(ctx: click.Context, trials: int, periods: int) -> None:
    """Positive control: inject known signals and measure recovery."""
    from .pipeline.validation import injection_benchmark

    click.echo("Running injection benchmark (this is the positive control)...")
    bench = injection_benchmark(n_seeds=trials, n_periods=periods)

    click.echo("")
    click.echo(f"{'Inj SNR':>8} {'Depth ppm':>10} {'Detected':>9} {'Rate':>6} "
               f"{'Period err %':>13} {'Depth err %':>12}")
    click.echo("-" * 64)
    for r in bench.results:
        click.echo(
            f"{r.snr_injected:>8.1f} {r.depth_injected_ppm:>10.0f} "
            f"{'yes' if r.detected else 'no':>9} {r.detection_rate:>6.0%} "
            f"{r.period_error_pct:>13.2f} {r.depth_error_pct:>12.1f}"
        )

    click.echo("")
    click.echo(f"Detection threshold in use: {bench.detect_threshold_snr:.1f}")
    limit = bench.sensitivity_limit_snr(0.8)
    if limit is not None:
        click.echo(f"Sensitivity limit (80% recovery): injected SNR >= {limit:.1f}")
    click.echo(f"Median period error at detected signals: {bench.median_period_error_pct:.2f}%")


# ── Dashboard ──────────────────────────────────────────────────────────────


@cli.command("dashboard")
@click.option("--host", default=None, help="Dashboard bind host.")
@click.option("--port", type=int, default=None, help="Dashboard bind port.")
@click.option("--debug", is_flag=True, help="Enable Flask debug mode.")
@click.pass_context
def dashboard(ctx: click.Context, host: Optional[str], port: Optional[int], debug: bool) -> None:
    """Start the local web dashboard."""
    config = ctx.obj["config"]
    host = host or config.get("dashboard", {}).get("host", "127.0.0.1")
    port = port or config.get("dashboard", {}).get("port", 5050)

    from .dashboard.app import create_app
    app = create_app(ctx.obj["db"])
    click.echo(f"Dashboard running at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


# ── Export ─────────────────────────────────────────────────────────────────


@cli.command("export")
@click.argument("output", type=click.Path())
@click.option("--format", "output_format", type=click.Choice(["json", "csv", "markdown"]), default="json")
@click.option("--status", "-s", type=click.Choice(["new", "known", "false_positive", "unverified"]), default=None)
@click.pass_context
def export(ctx: click.Context, output: str, output_format: str, status: Optional[str]) -> None:
    """Export candidates to a file."""
    db: Database = ctx.obj["db"]
    status_enum = CandidateStatus(status) if status else None
    candidates = db.get_candidates(status=status_enum, limit=10000)

    if not candidates:
        click.echo("No candidates to export.")
        return

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(candidates, f, indent=2, default=str)
    elif output_format == "csv":
        import csv
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=candidates[0].keys())
            writer.writeheader()
            writer.writerows(candidates)
    elif output_format == "markdown":
        lines = [
            "# Cosmic Discovery Engine — Candidate Export",
            "",
            f"Generated: {datetime.now(UTC).isoformat()}",
            f"Total candidates: {len(candidates)}",
            "",
            "| ID | Source | Period (d) | Depth (ppm) | SNR | Score | Status |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in candidates:
            lines.append(
                f"| {c['candidate_id']} | {c['source_id']} | "
                f"{c['period_days']:.4f} | {c['depth_ppm']:.0f} | "
                f"{c['snr']:.1f} | {c['score']:.1f} | {c['status']} |"
            )
        output_path.write_text("\n".join(lines), encoding="utf-8")

    click.echo(f"Exported {len(candidates)} candidate(s) to {output}")


def main() -> None:
    """Entry point for the cosmic-discovery CLI."""
    cli()


if __name__ == "__main__":
    main()
