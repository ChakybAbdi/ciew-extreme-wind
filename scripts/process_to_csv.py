# -*- coding: utf-8 -*-
"""
Process all ERA5 monthly NetCDF files into a single extreme-events CSV.
Run after download_england_era5.py has finished all downloads.
"""

import warnings
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

warnings.filterwarnings("ignore")

TMP_DIR     = Path(__file__).resolve().parents[1] / "data" / "era5_england_raw"
OUT_CSV     = Path(__file__).resolve().parents[1] / "data" / "england_extreme_events_2015_2026.csv"
POT_Q       = 0.95
DECLUST_HRS = 72

# ── STEP 1: FIND FILES ───────────────────────────────────────────────────────
nc_files = sorted(TMP_DIR.glob("era5_england_*.nc"))
print(f"Found {len(nc_files)} NetCDF files")


# ── STEP 2: LOAD ─────────────────────────────────────────────────────────────
print("\nSTEP 2 -- Loading NetCDF files")

def normalise(ds):
    rename = {}
    if "lat" in ds.dims:       rename["lat"] = "latitude"
    if "lon" in ds.dims:       rename["lon"] = "longitude"
    if "valid_time" in ds.dims and "time" not in ds.dims:
        rename["valid_time"] = "time"
    if rename:
        ds = ds.rename(rename)
    if "expver" in ds.coords:
        ds = ds.drop_vars("expver")
    return ds

ds = xr.open_mfdataset(
    nc_files,
    combine="by_coords",
    engine="netcdf4",
    chunks={"time": 744},
    preprocess=normalise,
)

lats  = ds.latitude.values
lons  = ds.longitude.values
times = pd.DatetimeIndex(ds.time.values)

print(f"  Grid : {len(lats)} lats x {len(lons)} lons = {len(lats)*len(lons)} points")
print(f"  Time : {times[0]}  ->  {times[-1]}  ({len(times):,} hours)")

# Wind speed
ws100 = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2)
ws10  = np.sqrt(ds["u10"]  ** 2 + ds["v10"]  ** 2)

print("  Loading ws100 into RAM ...")
ws100_np = ws100.values.astype(np.float32)
ws10_np  = ws10.values.astype(np.float32)

print("  Loading auxiliary variables ...")
blh_np = ds["blh"].values.astype(np.float32)
sp_np  = (ds["sp"].values  / 100.0).astype(np.float32)    # Pa -> hPa
t2m_np = (ds["t2m"].values - 273.15).astype(np.float32)   # K  -> degC


# ── STEP 3: P95 THRESHOLD PER GRID POINT ────────────────────────────────────
print("\nSTEP 3 -- Computing P95 threshold per grid point")

threshold_np = np.nanquantile(ws100_np, POT_Q, axis=0)
print(f"  Mean P95 threshold: {threshold_np.mean():.2f} m/s"
      f"  |  min: {threshold_np.min():.2f}  |  max: {threshold_np.max():.2f}")


# ── STEP 4: POT + 72h RUNS DECLUSTERING ─────────────────────────────────────
print("\nSTEP 4 -- Extracting extreme events (POT + 72h declustering)")

def decluster_peaks(ws_series, thr, gap_hrs):
    exceed = ws_series > thr
    if not exceed.any():
        return []
    run_peaks = []
    in_run = False
    run_start = 0
    for t in range(len(ws_series)):
        if exceed[t] and not in_run:
            in_run, run_start = True, t
        elif not exceed[t] and in_run:
            in_run = False
            run_peaks.append(run_start + int(np.argmax(ws_series[run_start:t])))
    if in_run:
        run_peaks.append(run_start + int(np.argmax(ws_series[run_start:])))

    selected = [run_peaks[0]]
    for pk in run_peaks[1:]:
        if pk - selected[-1] >= gap_hrs:
            selected.append(pk)
        elif ws_series[pk] > ws_series[selected[-1]]:
            selected[-1] = pk
    return selected


records = []
n_gp = len(lats) * len(lons)
done  = 0

for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        thr   = float(threshold_np[i, j])
        ws_gp = ws100_np[:, i, j]
        peaks = decluster_peaks(ws_gp, thr, DECLUST_HRS)
        if not peaks:
            done += 1
            continue

        prev_dt = None
        for pk in peaks:
            dt     = times[pk]
            ws_val = float(ws_gp[pk])
            ws10_v = float(ws10_np[pk, i, j])
            excess = ws_val - thr

            ia_hrs  = float((dt - prev_dt).total_seconds() / 3600) if prev_dt is not None else np.nan
            log_ia  = float(np.log1p(ia_hrs)) if not np.isnan(ia_hrs) else np.nan
            log_exc = float(np.log(excess))   if excess > 0          else np.nan

            records.append({
                "peak_datetime":     dt,
                "latitude":          float(lat),
                "longitude":         float(lon),
                "ws100":             ws_val,
                "ws10":              ws10_v,
                "shear":             ws_val - ws10_v,
                "blh":               float(blh_np[pk, i, j]),
                "sp_hpa":            float(sp_np[pk, i, j]),
                "t2m_c":             float(t2m_np[pk, i, j]),
                "threshold":         thr,
                "excess":            excess,
                "log_excess_c":      log_exc,
                "inter_arrival_hrs": ia_hrs,
                "log_inter_arrival": log_ia,
            })
            prev_dt = dt

        done += 1
        if done % 50 == 0:
            print(f"  {done}/{n_gp} grid points  ({len(records):,} events)", end="\r")

print(f"\n  Total extreme events: {len(records):,}")


# ── STEP 5: FEATURES + SAVE ──────────────────────────────────────────────────
print("\nSTEP 5 -- Adding cyclic features and saving CSV")

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

print(f"  Saved {len(df):,} rows -> {OUT_CSV}")
print("\nColumn summary:")
print(df.dtypes.to_string())
print("\nDescriptive stats:")
print(df[["ws100","excess","inter_arrival_hrs","shear","blh","sp_hpa","t2m_c"]].describe().round(3).to_string())
print("\nDone.")
