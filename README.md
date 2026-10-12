# Cosmic Discovery Engine

A production-quality astronomical anomaly discovery platform for Windows 10/11. Discovers and investigates potentially overlooked astronomical phenomena using real, publicly accessible scientific datasets.

## Quick Start

```powershell
# Run the setup script (creates venv, installs deps)
.\setup.ps1

# Run on demo (synthetic) data — no network needed
cosmic-discovery demo-ingest --all

# List discovered candidates
cosmic-discovery list-candidates

# Export to JSON
cosmic-discovery export candidates.json --format json

# Start the web dashboard
cosmic-discovery dashboard
```

## Data Sources

| Source | Type | Network Required |
|---|---|---|
| **Demo** (synthetic) | Offline synthetic light curves with known transits | No |
| **MAST/TESS** | Real TESS 2-minute cadence light curves via astroquery | Yes |
| **NASA Exoplanet Archive** | TAP queries for known exoplanets (cross-matching) | Yes |
| **Gaia** | Astrometric catalogue (planned) | Yes |

## Architecture

```
src/cosmic_discovery/
├── cli.py                  # Click-based command-line interface
├── config.py               # YAML + environment configuration
├── database.py             # SQLite persistence layer (WAL mode)
├── models.py               # Typed dataclasses for all entities
├── sources/                # Data source adapters
│   ├── base.py             #   Abstract base class
│   ├── mast.py             #   MAST/TESS via astroquery
│   ├── exoplanet_archive.py #  NASA Exoplanet Archive TAP
│   └── demo.py             #   Synthetic offline data
├── pipeline/               # Discovery pipeline
│   ├── ingest.py           #   End-to-end orchestrator
│   ├── features.py         #   Lomb-Scargle + BLS transit search
│   ├── candidates.py       #   Candidate scoring and flagging
│   └── crossmatch.py       #   Known-object cross-matching
└── dashboard/
    └── app.py              # Flask web dashboard
```

## Pipeline Stages

1. **Ingest** — Fetch light curves from MAST (TESS) or demo source
2. **Clean** — Sigma-clip outliers, remove NaNs, detrend with moving median
3. **Feature extraction** — Lomb-Scargle periodogram + Box Least Squares transit search
4. **Candidate generation** — Score by SNR, depth, transit count, even/odd consistency
5. **Cross-match** — Compare against known exoplanet catalogues
6. **Report** — Export Markdown, JSON, or CSV; view in dashboard

## CLI Commands

```
cosmic-discovery demo-ingest --all              # Process all demo targets
cosmic-discovery demo-ingest -t TIC_100000001  # Process one target
cosmic-discovery mast-search --ra 15 --dec 45   # Search MAST
cosmic-discovery mast-ingest TIC 25155310       # Fetch from MAST
cosmic-discovery list-candidates                # Show candidates
cosmic-discovery list-candidates --status new   # Filter by status
cosmic-discovery export report.md --format markdown
cosmic-discovery stats                          # Database statistics
cosmic-discovery dashboard                      # Start web UI
```

## Configuration

Edit `config/default.yaml` or set environment variables:

| Variable | Default | Description |
|---|---|---|
| `CDE_DATA_DIR` | `./data` | Data directory |
| `CDE_LOG_LEVEL` | `INFO` | Logging level |
| `CDE_DASHBOARD_HOST` | `127.0.0.1` | Dashboard bind address |
| `CDE_DASHBOARD_PORT` | `5050` | Dashboard port |
| `CDE_MAST_TOKEN` | (none) | MAST API token (increases rate limits) |

## Testing

```powershell
# Run all tests
pytest

# Run with coverage
pytest --cov=cosmic_discovery --cov-report=term-missing

# Run specific test file
pytest tests/test_features.py -v
```

## Scientific Integrity

- Demo data is **clearly flagged as synthetic** in metadata
- Real observations are never replaced with fabricated data
- Candidates are scored and flagged, never confirmed without cross-matching
- Multiple hypothesis testing is addressed via SNR thresholds
- All data sources respect API rate limits and terms of service

## Requirements

- Python 3.11+
- Windows 10/11 (PowerShell)
- See `pyproject.toml` for Python package dependencies

## License

MIT
