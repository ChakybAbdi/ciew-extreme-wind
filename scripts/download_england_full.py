"""
ERA5 England, FULL variable set (gusts + convective) monthly download.
Resumable: skips any month already present. Writes into era5_england_raw_full
so the existing 7-variable files are left untouched.

Run:  python download_england_full.py
"""
import sys, time, os, zipfile, tempfile, shutil, datetime as dt
from pathlib import Path
import cdsapi
import xarray as xr


def finalize(fp):
    """The new CDS returns a zip when a request spans several data streams.
    Detect it, extract the inner NetCDFs and merge them into one clean file."""
    fp = Path(fp)
    with open(fp, "rb") as fh:
        if fh.read(2) != b"PK":
            return                         # already a plain NetCDF
    with zipfile.ZipFile(fp) as z:
        members = [n for n in z.namelist() if n.endswith(".nc")]
        td = tempfile.mkdtemp(); z.extractall(td)
    dss = [xr.open_dataset(os.path.join(td, n)) for n in members]
    merged = xr.merge(dss, compat="override", combine_attrs="override")
    tmp = str(fp) + ".tmp.nc"
    merged.to_netcdf(tmp)
    for d in dss: d.close()
    merged.close()
    shutil.rmtree(td, ignore_errors=True)
    os.replace(tmp, fp)

AREA = [56.0, -6.0, 50.0, 2.0]      # N, W, S, E
GRID = [0.25, 0.25]
OUT = Path(__file__).resolve().parents[1] / "data" / "era5_england_raw_full"
OUT.mkdir(parents=True, exist_ok=True)

# The new CDS caps fields per request, so we split each month into two smaller
# requests (group a and group b). The notebook loader merges them by coordinates.
GROUPS = {
    "a": [   # kinematic + gust
        "10m_u_component_of_wind", "10m_v_component_of_wind",
        "100m_u_component_of_wind", "100m_v_component_of_wind",
        "instantaneous_10m_wind_gust", "10m_wind_gust_since_previous_post_processing",
        "boundary_layer_height",
    ],
    "b": [   # convective + thermodynamic + pressure
        "convective_available_potential_energy", "convective_inhibition",
        "k_index", "total_column_water_vapour",
        "mean_sea_level_pressure", "surface_pressure",
        "2m_temperature", "2m_dewpoint_temperature",
    ],
}

YEARS = range(2015, 2027)
NOW = dt.date.today()

c = cdsapi.Client()
todo = []
for y in YEARS:
    for m in range(1, 13):
        # skip months in the future (ERA5T has a short lag; those requests fail)
        if dt.date(y, m, 1) > dt.date(NOW.year, NOW.month, 1):
            continue
        for g in GROUPS:
            todo.append((y, m, g))

print(f"Target requests: {len(todo)}  ->  {OUT}", flush=True)

# startup: repair any files already on disk that are still zipped
for existing in OUT.glob("*.nc"):
    try:
        finalize(existing)
    except Exception as e:
        print(f"[repair] {existing.name} failed: {e}", flush=True)

done = 0
for y, m, g in todo:
    fp = OUT / f"era5_england_{y}_{m:02d}_{g}.nc"
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
        print(f"[{y}-{m:02d}/{g}] saved {fp.stat().st_size/1e6:.1f} MB in {time.time()-t0:.0f}s "
              f"({done}/{len(todo)})", flush=True)
    except Exception as e:
        print(f"[{y}-{m:02d}/{g}] FAILED: {e}", flush=True)

print(f"\nDone. {done}/{len(todo)} requests present in {OUT}")
