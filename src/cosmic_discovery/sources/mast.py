"""MAST/TESS data source via astroquery."""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from astropy.io import fits

from ..models import DataSource, LightCurve, StellarSource
from .base import DataSourceBase, SourceResult

logger = logging.getLogger(__name__)


class MASTSource(DataSourceBase):
    """Access TESS light curves through the MAST archive via astroquery."""

    name = "mast"

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        token: Optional[str] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.token = token
        self._obs: Any = None

    def _get_obs(self) -> Any:
        if self._obs is None:
            from astroquery.mast import Observations

            kwargs: dict[str, Any] = {}
            if self.token:
                kwargs["token"] = self.token
            self._obs = Observations(**kwargs)
        return self._obs

    def health_check(self) -> SourceResult:
        try:
            obs = self._get_obs()
            # Simple query to verify connectivity
            obs.query_criteria(
                obs_collection="TESS",
                target_name="TIC 25155310",
                radius="0.01 deg",
            )
            return SourceResult(success=True, metadata={"source": self.name})
        except Exception as e:
            return SourceResult(success=False, error=str(e), metadata={"source": self.name})

    def search_targets(
        self, ra_deg: float, dec_deg: float, radius_arcsec: float, **kwargs: Any
    ) -> SourceResult:
        """Search MAST for TESS targets near a sky position."""
        try:
            obs = self._get_obs()
            radius_deg = radius_arcsec / 3600.0
            obs_table = obs.query_criteria(
                obs_collection="TESS",
                coordinates=f"{ra_deg} {dec_deg}",
                radius=f"{radius_deg} deg",
                dataproduct_type="timeseries",
            )
            if obs_table is None or len(obs_table) == 0:
                return SourceResult(success=True, data=[], metadata={"count": 0})

            targets = []
            seen_tic: set[int] = set()
            for row in obs_table:
                tic_id = row.get("target_name", "").replace("TIC ", "").strip()
                try:
                    tic_int = int(tic_id)
                except (ValueError, TypeError):
                    continue
                if tic_int in seen_tic:
                    continue
                seen_tic.add(tic_int)
                targets.append(
                    StellarSource(
                        source_id=f"TIC {tic_int}",
                        ra_deg=float(row.get("s_ra", ra_deg)),
                        dec_deg=float(row.get("s_dec", dec_deg)),
                        tic_id=tic_int,
                        metadata={"socat": str(row.get("socat", ""))},
                    )
                )
            return SourceResult(success=True, data=targets, metadata={"count": len(targets)})
        except Exception as e:
            logger.error("MAST search failed: %s", e)
            return SourceResult(success=False, error=str(e))

    def fetch_lightcurve(
        self, source_id: str, **kwargs: Any
    ) -> SourceResult:
        """Download and parse a TESS light curve from MAST.

        Parameters
        ----------
        source_id : str
            TIC identifier (e.g. "TIC 25155310").
        sector : int, optional
            Specific TESS sector. If None, downloads all available.
        """
        sector: Optional[int] = kwargs.get("sector")
        try:
            lc = self._download_with_retry(source_id, sector=sector)
            return SourceResult(success=True, data=lc)
        except Exception as e:
            logger.error("MAST light curve fetch failed for %s: %s", source_id, e)
            return SourceResult(success=False, error=str(e))

    def _download_with_retry(
        self, source_id: str, sector: Optional[int] = None
    ) -> LightCurve:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._download(source_id, sector=sector)
            except Exception as e:
                last_err = e
                logger.warning(
                    "MAST download attempt %d/%d failed for %s: %s",
                    attempt + 1, self.max_retries, source_id, e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        raise last_err or RuntimeError("download failed")

    def _download(self, source_id: str, sector: Optional[int] = None) -> LightCurve:
        obs = self._get_obs()

        # Query for the target
        tic_id = source_id.replace("TIC", "").strip()
        query_kwargs: dict[str, Any] = {
            "obs_collection": "TESS",
            "target_name": f"TIC {tic_id}",
            "dataproduct_type": "timeseries",
            "productSubGroupDescription": ["LC"],
        }
        if sector is not None:
            query_kwargs["sequence_number"] = sector

        obs_table = obs.query_criteria(**query_kwargs)
        if obs_table is None or len(obs_table) == 0:
            raise RuntimeError(f"No TESS light curves found for {source_id}")

        # Download the products
        products = obs.get_product_list(obs_table)
        if products is None or len(products) == 0:
            raise RuntimeError(f"No downloadable products for {source_id}")

        # Filter to light curve files (suffix _lc.fits)
        lc_mask = [
            fn.endswith("_lc.fits") or fn.endswith("_lc.fits.gz")
            for fn in products["productFilename"]
        ]
        lc_products = products[lc_mask]
        if len(lc_products) == 0:
            lc_products = products  # fallback to whatever is available

        manifest = obs.download_products(lc_products, download_dir=str(self.cache_dir) if self.cache_dir else None)

        # Parse the downloaded FITS files
        all_time: list[np.ndarray] = []
        all_flux: list[np.ndarray] = []
        all_flux_err: list[np.ndarray] = []
        all_quality: list[np.ndarray] = []
        sectors: list[int] = []

        for local_path in manifest["Local Path"]:
            path = Path(local_path)
            if not path.exists():
                continue
            with fits.open(path, memmap=False) as hdul:
                data = hdul[1].data
                time_arr = np.array(data["TIME"], dtype=np.float64)
                flux_arr = np.array(data["PDCSAP_FLUX"], dtype=np.float64)
                flux_err_arr = np.array(data["PDCSAP_FLUX_ERR"], dtype=np.float64)
                quality_arr = np.array(data["QUALITY"], dtype=np.int32) if "QUALITY" in data.columns.names else np.zeros(len(time_arr), dtype=np.int32)

                # Only keep finite values
                mask = np.isfinite(time_arr) & np.isfinite(flux_arr) & (flux_arr > 0)
                mask &= (quality_arr & 0x7FFF) == 0  # remove flagged cadences

                if mask.sum() == 0:
                    continue

                all_time.append(time_arr[mask])
                all_flux.append(flux_arr[mask])
                all_flux_err.append(flux_err_arr[mask])
                all_quality.append(quality_arr[mask])

                hdr = hdul[1].header
                seq = hdr.get("SECTOR", None)
                if seq is not None:
                    sectors.append(int(seq))

        if not all_time:
            raise RuntimeError(f"No valid cadences in downloaded files for {source_id}")

        combined_time = np.concatenate(all_time)
        combined_flux = np.concatenate(all_flux)
        combined_err = np.concatenate(all_flux_err)
        combined_quality = np.concatenate(all_quality)

        # Sort by time
        order = np.argsort(combined_time)
        combined_time = combined_time[order]
        combined_flux = combined_flux[order]
        combined_err = combined_err[order]
        combined_quality = combined_quality[order]

        checksum = hashlib.sha256(combined_time.tobytes() + combined_flux.tobytes()).hexdigest()[:16]
        used_sector = sectors[0] if len(sectors) == 1 else None

        return LightCurve(
            source_id=source_id,
            time=combined_time,
            flux=combined_flux,
            flux_err=combined_err,
            source=DataSource.MAST,
            sector=used_sector,
            quality_flags=combined_quality,
            checksum=checksum,
            retrieved_at=datetime.utcnow(),
        )
