"""
US Southern/Central Plains — ERA5 single-level download + NOAA Storm Events join.

Why this domain. The England ablation showed the convective feature family contributing *nothing*
to occurrence skill (removing it cost 0.0 PR-AUC), because English extreme wind is overwhelmingly
synoptic — Atlantic depressions in the cold season. That makes the CIEW regime head untestable on
England: there is no convective mode for it to separate.

The Plains is the opposite regime. Severe convective wind (derechos, bow echoes, downbursts) is the
dominant warm-season hazard, and it coexists with cold-season synoptic gradient wind. Transferring
the same framework here is therefore a genuine generalisation test rather than a repeated run, and
it is the only place the convective/synoptic regime head can actually be falsified.

NOAA Storm Events additionally supplies *impact* labels — damage in dollars, injuries, deaths — which
feed the hazard-management chain (early warning -> resource allocation) with observed consequences
instead of reanalysis proxies alone.

Usage
-----
    python download_plains_era5.py --era5          # ERA5 fields (slow: CDS queue, hours to days)
    python download_plains_era5.py --storm         # NOAA Storm Events (fast: minutes)
    python download_plains_era5.py --era5 --storm

Both stages are resumable: anything already on disk is skipped.
"""
import argparse
import datetime as dt
import gzip
import io
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent

# --- domain -----------------------------------------------------------------
# N, W, S, E. Covers OK, KS, NE, north TX, MO, AR — the severe-wind corridor.
AREA = [42.0, -102.0, 32.0, -92.0]
GRID = [0.25, 0.25]
YEARS = range(2015, 2027)

DATA = Path(__file__).resolve().parents[1] / "data"
ERA5_DIR = DATA / "era5_plains_raw"     # sits beside era5_england_raw_full
STORM_DIR = DATA / "noaa_storm_events"
# Stop where the England archive stops, so both domains cover the identical period.
END_MONTH = dt.date(2026, 6, 1)

# States intersecting the AREA box, used to place zone-located events that have no coordinates.
DOMAIN_STATES = {
    "OKLAHOMA", "KANSAS", "NEBRASKA", "MISSOURI", "ARKANSAS", "IOWA",
    "TEXAS", "COLORADO", "NEW MEXICO", "LOUISIANA", "SOUTH DAKOTA",
}

# Identical variable set to download_england_full.py so the two domains are directly comparable.
GROUPS = {
    "a": [  # kinematic + gust
        "10m_u_component_of_wind", "10m_v_component_of_wind",
        "100m_u_component_of_wind", "100m_v_component_of_wind",
        "instantaneous_10m_wind_gust", "10m_wind_gust_since_previous_post_processing",
        "boundary_layer_height",
    ],
    "b": [  # convective + thermodynamic + pressure
        "convective_available_potential_energy", "convective_inhibition",
        "k_index", "total_column_water_vapour",
        "mean_sea_level_pressure", "surface_pressure",
        "2m_temperature", "2m_dewpoint_temperature",
    ],
}


# ----------------------------------------------------------------------------- ERA5
def finalize(fp):
    """CDS returns a zip when a request spans several data streams. Detect it, extract the inner
    NetCDFs and merge them into one clean file. (Same helper as the England script.)"""
    import xarray as xr
    fp = Path(fp)
    with open(fp, "rb") as fh:
        if fh.read(2) != b"PK":
            return
    with zipfile.ZipFile(fp) as z:
        members = [n for n in z.namelist() if n.endswith(".nc")]
        td = tempfile.mkdtemp()
        z.extractall(td)
    dss = [xr.open_dataset(os.path.join(td, n)) for n in members]
    merged = xr.merge(dss, compat="override", combine_attrs="override")
    tmp = str(fp) + ".tmp.nc"
    merged.to_netcdf(tmp)
    for d in dss:
        d.close()
    merged.close()
    shutil.rmtree(td, ignore_errors=True)
    os.replace(tmp, fp)


def download_era5():
    import cdsapi
    ERA5_DIR.mkdir(parents=True, exist_ok=True)

    now = dt.date.today()
    last = min(END_MONTH, dt.date(now.year, now.month, 1))
    todo = []
    for y in YEARS:
        for m in range(1, 13):
            if dt.date(y, m, 1) > last:
                continue
            for g in GROUPS:
                todo.append((y, m, g))

    print(f"ERA5 target requests: {len(todo)}  ->  {ERA5_DIR}", flush=True)
    print(f"Domain N{AREA[0]} W{AREA[1]} S{AREA[2]} E{AREA[3]} at {GRID[0]} deg", flush=True)

    for existing in ERA5_DIR.glob("*.nc"):
        try:
            finalize(existing)
        except Exception as e:
            print(f"[repair] {existing.name} failed: {e}", flush=True)

    c = cdsapi.Client()
    done = 0
    for y, m, g in todo:
        fp = ERA5_DIR / f"era5_plains_{y}_{m:02d}_{g}.nc"
        if fp.exists() and fp.stat().st_size > 500_000:
            done += 1
            continue
        print(f"[{y}-{m:02d}/{g}] requesting {len(GROUPS[g])} vars ...", flush=True)
        t0 = time.time()
        try:
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": GROUPS[g],
                    "year": [str(y)],
                    "month": [f"{m:02d}"],
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": AREA,
                    "grid": GRID,
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
                str(fp),
            )
            finalize(fp)
            done += 1
            print(f"[{y}-{m:02d}/{g}] saved {fp.stat().st_size/1e6:.1f} MB in "
                  f"{time.time()-t0:.0f}s ({done}/{len(todo)})", flush=True)
        except Exception as e:
            print(f"[{y}-{m:02d}/{g}] FAILED: {e}", flush=True)

    print(f"\nERA5 done. {done}/{len(todo)} requests present in {ERA5_DIR}")


# ----------------------------------------------------------------------------- NOAA
NCEI_ROOT = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"


def download_storm_events():
    """Fetch StormEvents 'details' CSVs for YEARS and filter to wind events in the domain.

    Filenames carry a creation-date suffix that NCEI revises, so the directory index is listed and
    matched by year rather than guessed.
    """
    import urllib.request
    import numpy as np
    import pandas as pd

    STORM_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Listing {NCEI_ROOT} ...", flush=True)
    with urllib.request.urlopen(NCEI_ROOT, timeout=120) as r:
        index = r.read().decode("utf-8", errors="ignore")

    frames = []
    for y in YEARS:
        pat = rf'(StormEvents_details-ftp_v1\.0_d{y}_c\d+\.csv\.gz)'
        matches = sorted(set(re.findall(pat, index)))
        if not matches:
            print(f"  [{y}] no details file listed (year may not be published yet)", flush=True)
            continue
        fname = matches[-1]                       # newest revision
        local = STORM_DIR / fname
        if not local.exists():
            print(f"  [{y}] downloading {fname} ...", flush=True)
            urllib.request.urlretrieve(NCEI_ROOT + fname, local)
        else:
            print(f"  [{y}] cached {fname}", flush=True)

        with gzip.open(local, "rt", encoding="latin-1") as fh:
            df = pd.read_csv(fh, low_memory=False)
        frames.append(df)

    if not frames:
        print("No Storm Events files retrieved.")
        return

    ev = pd.concat(frames, ignore_index=True)
    print(f"\nAll events {len(ev):,}")

    # --- wind hazards only ---
    WIND_TYPES = ["Thunderstorm Wind", "High Wind", "Strong Wind", "Marine Thunderstorm Wind",
                  "Marine High Wind", "Marine Strong Wind", "Tornado", "Dust Storm"]
    ev = ev[ev["EVENT_TYPE"].isin(WIND_TYPES)].copy()
    print(f"Wind-type events {len(ev):,}")

    # --- geographic filter ------------------------------------------------
    # Point-located events (Thunderstorm Wind, Tornado, Marine) carry BEGIN_LAT/LON.
    # Zone-located events (High Wind, Strong Wind, Dust Storm) do NOT -- NCEI records them against
    # a forecast zone, so every one of them has a null coordinate. Filtering on lat/lon alone
    # therefore silently discards 100% of the synoptic wind events, which are exactly the mode that
    # makes this domain comparable to England. Those fall back to a state filter.
    lat = pd.to_numeric(ev.get("BEGIN_LAT"), errors="coerce")
    lon = pd.to_numeric(ev.get("BEGIN_LON"), errors="coerce")
    N, W, S, E = AREA
    has_point = lat.notna() & lon.notna()
    inbox = (lat.between(S, N) & lon.between(W, E)).fillna(False)

    ev["geo_quality"] = np.where(has_point, "point", "zone")
    in_states = ev["STATE"].str.upper().isin(DOMAIN_STATES)
    keep_mask = (has_point & inbox) | (~has_point & in_states)
    ev = ev[keep_mask].copy()

    n_pt = int((ev.geo_quality == "point").sum())
    n_zn = int((ev.geo_quality == "zone").sum())
    print(f"Inside domain {len(ev):,}  (point-located {n_pt:,} | zone-located {n_zn:,})")
    print("  NOTE: zone-located rows are state-filtered, not point-filtered. Join them to the ERA5"
          "\n        grid by zone centroid, or restrict to geo_quality=='point' for strict work.")

    # --- normalise the impact columns ---
    def money(x):
        """'2.5K' / '10.00M' / '1B' -> float dollars."""
        if not isinstance(x, str) or not x.strip():
            return 0.0
        s = x.strip().upper()
        mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(s[-1], 1.0)
        try:
            return float(s[:-1] if s[-1] in "KMBT" else s) * mult
        except ValueError:
            return 0.0

    ev["damage_property_usd"] = ev.get("DAMAGE_PROPERTY", "").map(money)
    ev["damage_crops_usd"] = ev.get("DAMAGE_CROPS", "").map(money)
    for c in ("INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT"):
        ev[c.lower()] = pd.to_numeric(ev.get(c), errors="coerce").fillna(0)
    ev["injuries"] = ev["injuries_direct"] + ev["injuries_indirect"]
    ev["deaths"] = ev["deaths_direct"] + ev["deaths_indirect"]

    ev["begin_time"] = pd.to_datetime(
        ev["BEGIN_DATE_TIME"], format="%d-%b-%y %H:%M:%S", errors="coerce")
    ev["magnitude_kt"] = pd.to_numeric(ev.get("MAGNITUDE"), errors="coerce")

    keep = ["begin_time", "EVENT_TYPE", "geo_quality", "STATE", "CZ_NAME",
            "BEGIN_LAT", "BEGIN_LON", "magnitude_kt", "MAGNITUDE_TYPE",
            "damage_property_usd", "damage_crops_usd",
            "injuries", "deaths", "TOR_F_SCALE", "EPISODE_ID", "EVENT_ID"]
    ev = ev[[c for c in keep if c in ev.columns]].sort_values("begin_time")

    out = STORM_DIR / "plains_wind_events.csv"
    ev.to_csv(out, index=False)
    print(f"\nWritten -> {out}  ({len(ev):,} rows)")
    print(f"Period: {ev.begin_time.min()} -> {ev.begin_time.max()}")
    print(f"Total property damage: ${ev.damage_property_usd.sum()/1e9:.2f} B")
    print(f"Injuries {int(ev.injuries.sum()):,} | deaths {int(ev.deaths.sum()):,}")
    print("\nBy event type:")
    print(ev.groupby("EVENT_TYPE").agg(
        n=("EVENT_ID", "size"),
        damage_musd=("damage_property_usd", lambda s: round(s.sum() / 1e6, 1)),
        deaths=("deaths", "sum"),
    ).sort_values("n", ascending=False).to_string())


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--era5", action="store_true", help="download ERA5 fields (slow)")
    ap.add_argument("--storm", action="store_true", help="download NOAA Storm Events (fast)")
    args = ap.parse_args()

    if not (args.era5 or args.storm):
        ap.print_help()
        sys.exit(0)
    if args.storm:
        download_storm_events()
    if args.era5:
        download_era5()
