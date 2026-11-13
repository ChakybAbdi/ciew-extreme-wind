"""
ERA5 England Extreme Wind Events â€” Download + Process Pipeline
=============================================================
Downloads hourly ERA5 data for England (50-56Â°N, 6Â°W-2Â°E), 2015-2026.
Applies POT (P95 threshold, 72h declustering) to extract extreme storm events.
Outputs: england_extreme_events_2015_2026.csv

Run: python download_england_era5.py
"""

import os, warnings
import numpy as np
import pandas as pd
import xarray as xr
import cdsapi
from pathlib import Path

warnings.filterwarnings("ignore")

# â”€â”€ CONFIG â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
AREA         = [56.0, -6.0, 50.0, 2.0]   # [N, W, S, E] â€” all of England + coastal
YEARS        = list(range(2015, 2027))     # 2015â€“2026 inclusive
POT_Q        = 0.95                        # per-grid-point P95 threshold
DECLUST_HRS  = 72                          # independence window (hours)
TMP_DIR      = Path(__file__).resolve().parents[1] / "data" / "era5_england_raw"
OUT_CSV      = Path(__file__).resolve().parents[1] / "data" / "england_extreme_events_2015_2026.csv"

VARIABLES = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '100m_u_component_of_wind',
    '100m_v_component_of_wind',
    'boundary_layer_height',
    'surface_pressure',
    '2m_temperature',
]
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

TMP_DIR.mkdir(parents=True, exist_ok=True)


# â”€â”€ STEP 1: DOWNLOAD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def download_year(client, year, month):
    """Download one calendar month for all variables."""
    import datetime

    # Skip future months (CDS won't have them)
    now = datetime.datetime.utcnow()
    if datetime.date(year, month, 1) > datetime.date(now.year, now.month, 1):
        print(f"  [{year}-{month:02d}] in the future â€” skipping")
        return None

    fpath = TMP_DIR / f"era5_england_{year}_{month:02d}.nc"
    if fpath.exists():
        print(f"  [{year}-{month:02d}] already on disk â€” skipping")
        return fpath

    print(f"  [{year}-{month:02d}] queuing CDS request â€¦")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable":     VARIABLES,
            "year":         str(year),
            "month":        f"{month:02d}",
            "day":          [f"{d:02d}" for d in range(1, 32)],
            "time":         [f"{h:02d}:00" for h in range(24)],
            "area":         AREA,
            "grid":         [0.25, 0.25],
            "format":       "netcdf",
        },
        str(fpath),
    )
    sz = fpath.stat().st_size / 1e6
    print(f"  [{year}-{month:02d}] saved -> {fpath.name}  ({sz:.0f} MB)")
    return fpath


print("=" * 60)
print("STEP 1 â€” Downloading ERA5 (monthly chunks to stay within CDS limits)")
print("=" * 60)
c = cdsapi.Client()
nc_files = []
for yr in YEARS:
    for mo in range(1, 13):
        try:
            fp = download_year(c, yr, mo)
            if fp:
                nc_files.append(fp)
        except Exception as e:
            print(f"  [{yr}-{mo:02d}] WARNING: {e} â€” skipping")

print(f"\nDownloaded {len(nc_files)} yearly files.")


# â”€â”€ STEP 2: LOAD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 60)
print("STEP 2 â€” Loading NetCDF files")
print("=" * 60)

def _normalise(ds):
    """Normalise dimension names across old and new ERA5 CDS formats."""
    rename = {}
    if "lat" in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims:
        rename["lon"] = "longitude"
    if "valid_time" in ds.dims and "time" not in ds.dims:
        rename["valid_time"] = "time"
    if rename:
        ds = ds.rename(rename)
    # Drop expver coord if present (causes combine conflicts)
    if "expver" in ds.coords:
        ds = ds.drop_vars("expver")
    return ds

ds = xr.open_mfdataset(
    sorted(nc_files),
    combine="by_coords",
    engine="netcdf4",
    chunks={"time": 744},
    preprocess=_normalise,
)

lats  = ds.latitude.values
lons  = ds.longitude.values
times = pd.DatetimeIndex(ds.time.values)

print(f"  Grid:  {len(lats)} lats Ã— {len(lons)} lons = {len(lats)*len(lons)} points")
print(f"  Time:  {times[0]}  â†’  {times[-1]}  ({len(times):,} hours)")

# Compute wind speed fields (still lazy)
ws100 = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2)
ws10  = np.sqrt(ds["u10"]  ** 2 + ds["v10"]  ** 2)

print("  Loading ws100 into RAM â€¦")
ws100_np = ws100.values.astype(np.float32)          # (T, lat, lon)
ws10_np  = ws10.values.astype(np.float32)

print("  Loading auxiliary variables â€¦")
blh_np = ds["blh"].values.astype(np.float32)
sp_np  = (ds["sp"].values / 100.0).astype(np.float32)   # Pa â†’ hPa
t2m_np = (ds["t2m"].values - 273.15).astype(np.float32) # K â†’ Â°C


# â”€â”€ STEP 3: P95 THRESHOLD PER GRID POINT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 60)
print("STEP 3 â€” Computing P95 threshold per grid point")
print("=" * 60)

threshold_np = np.nanquantile(ws100_np, POT_Q, axis=0)   # (lat, lon)
print(f"  Mean threshold: {threshold_np.mean():.2f} m/s  "
      f"| min: {threshold_np.min():.2f}  | max: {threshold_np.max():.2f}")


# â”€â”€ STEP 4: POT + RUNS DECLUSTERING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 60)
print("STEP 4 â€” Extracting extreme events (POT + 72h declustering)")
print("=" * 60)


def decluster_peaks(ws_series, thr, declust_hrs):
    """
    Runs declustering on a 1-D hourly wind speed array.
    Returns list of integer indices (hours) of declustered storm peaks.
    """
    exceed = ws_series > thr
    if not exceed.any():
        return []

    # Identify contiguous exceedance runs
    run_peaks = []
    in_run = False
    run_start = 0

    for t in range(len(ws_series)):
        if exceed[t] and not in_run:
            in_run, run_start = True, t
        elif not exceed[t] and in_run:
            in_run = False
            pk = run_start + int(np.argmax(ws_series[run_start:t]))
            run_peaks.append(pk)
    if in_run:  # run reaches end of series
        pk = run_start + int(np.argmax(ws_series[run_start:]))
        run_peaks.append(pk)

    # Enforce minimum inter-event gap (declust_hrs)
    selected = [run_peaks[0]]
    for pk in run_peaks[1:]:
        gap = pk - selected[-1]          # both are hour-indices â†’ gap in hours
        if gap >= declust_hrs:
            selected.append(pk)
        else:
            # Merge: keep the stronger peak
            if ws_series[pk] > ws_series[selected[-1]]:
                selected[-1] = pk

    return selected


records = []
n_gp = len(lats) * len(lons)
done = 0

for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        thr = float(threshold_np[i, j])

        ws_gp  = ws100_np[:, i, j]
        peaks  = decluster_peaks(ws_gp, thr, DECLUST_HRS)
        if not peaks:
            done += 1
            continue

        prev_dt = None
        for pk in peaks:
            dt      = times[pk]
            ws_val  = float(ws_gp[pk])
            ws10_v  = float(ws10_np[pk, i, j])
            excess  = ws_val - thr

            ia_hrs       = float((dt - prev_dt).total_seconds() / 3600) \
                           if prev_dt is not None else np.nan
            log_ia       = float(np.log1p(ia_hrs)) if not np.isnan(ia_hrs) else np.nan
            log_exc      = float(np.log(excess))   if excess > 0 else np.nan

            records.append({
                "peak_datetime":      dt,
                "latitude":           float(lat),
                "longitude":          float(lon),
                "ws100":              ws_val,
                "ws10":               ws10_v,
                "shear":              ws_val - ws10_v,
                "blh":                float(blh_np[pk, i, j]),
                "sp_hpa":             float(sp_np[pk, i, j]),
                "t2m_c":              float(t2m_np[pk, i, j]),
                "threshold":          thr,
                "excess":             excess,
                "log_excess_c":       log_exc,
                "inter_arrival_hrs":  ia_hrs,
                "log_inter_arrival":  log_ia,
            })
            prev_dt = dt

        done += 1
        if done % 100 == 0:
            print(f"  Processed {done}/{n_gp} grid points  "
                  f"({len(records):,} events so far)", end="\r")

print(f"\n  Total extreme events extracted: {len(records):,}")


# â”€â”€ STEP 5: FEATURES + SAVE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 60)
print("STEP 5 â€” Adding features & saving CSV")
print("=" * 60)

df = pd.DataFrame(records)
df["peak_datetime"] = pd.to_datetime(df["peak_datetime"])
df = df.sort_values(["latitude", "longitude", "peak_datetime"]).reset_index(drop=True)

month = df["peak_datetime"].dt.month
hour  = df["peak_datetime"].dt.hour

df["sin_month"] = np.sin(2 * np.pi * month / 12)
df["cos_month"] = np.cos(2 * np.pi * month / 12)
df["sin_hour"]  = np.sin(2 * np.pi * hour  / 24)
df["cos_hour"]  = np.cos(2 * np.pi * hour  / 24)

df.to_csv(OUT_CSV, index=False)

print(f"  Saved {len(df):,} rows â†’ {OUT_CSV}")
print(f"\n  Column summary:")
print(df.dtypes.to_string())
print(f"\n  Descriptive stats:")
print(df[["ws100","excess","inter_arrival_hrs","shear","blh","sp_hpa","t2m_c"]].describe().round(3).to_string())
print("\nDone.")

