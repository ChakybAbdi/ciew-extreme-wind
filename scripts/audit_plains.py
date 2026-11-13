"""
Read-only audit of an ERA5 month-file archive.

Checks every file the loader will read, because "the download finished" and "the archive is sound"
are different claims. This run has already produced two silent defects that a file count would not
have caught: months missing k_index entirely, and months merged onto an 82-column longitude grid
with half the values NaN.

Per file: readable, correct grid, full hour count, the group's full variable set, the -180..180
longitude convention, matching coordinate vectors, and no all-NaN variable that should carry data.

Run:  python audit_plains.py [--dir <path>] [--prefix era5_plains]
"""
import argparse
import calendar
import collections
import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr

GROUPS = {
    "a": ["u10", "v10", "u100", "v100", "i10fg", "fg10", "blh"],
    "b": ["cape", "cin", "kx", "tcwv", "msl", "sp", "t2m", "d2m"],
}
# ERA5 masks CIN wherever it is undefined -- 99.5% NaN in a winter month is correct, not damage.
MASKED_OK = {"cin"}


def audit(directory, prefix, end_month):
    files = sorted(directory.glob(f"{prefix}_*.nc"))
    if not files:
        print(f"no files matching {prefix}_*.nc in {directory}")
        return 1

    ref_lat = ref_lon = None
    problems = collections.defaultdict(list)
    months = set()

    for fp in files:
        try:
            y, m, g = int(fp.stem.split("_")[2]), int(fp.stem.split("_")[3]), fp.stem.split("_")[4]
        except (IndexError, ValueError):
            problems["unparseable filename"].append(fp.name)
            continue
        months.add((y, m, g))
        try:
            with xr.open_dataset(fp) as d:
                full = calendar.monthrange(y, m)[1] * 24
                floor = full - 24 if (y, m) == (end_month.year, end_month.month) else full
                if d.sizes.get("valid_time", 0) < floor:
                    problems["short hour count"].append(
                        f"{fp.name}: {d.sizes.get('valid_time')} < {floor}")

                have = set(d.data_vars)
                missing = [v for v in GROUPS[g] if v not in have]
                if missing:
                    problems["missing variables"].append(f"{fp.name}: {missing}")
                extra = [v for v in have if v not in GROUPS[g]]
                if extra:
                    problems["unexpected variables"].append(f"{fp.name}: {extra}")

                lat, lon = d.sizes.get("latitude"), d.sizes.get("longitude")
                if ref_lat is None:
                    ref_lat, ref_lon = d.latitude.values, d.longitude.values
                    print(f"reference grid from {fp.name}: {lat}x{lon}  "
                          f"lat {ref_lat.max():.2f}..{ref_lat.min():.2f}  "
                          f"lon {ref_lon.min():.2f}..{ref_lon.max():.2f}")
                else:
                    if (lat, lon) != (len(ref_lat), len(ref_lon)):
                        problems["grid shape"].append(f"{fp.name}: {lat}x{lon}")
                    elif not (np.array_equal(d.latitude.values, ref_lat)
                              and np.array_equal(d.longitude.values, ref_lon)):
                        problems["coordinate mismatch"].append(fp.name)

                if float(d.longitude.max()) > 180:
                    problems["0-360 longitude"].append(fp.name)

                for v in have:
                    if v in MASKED_OK:
                        continue
                    if not np.isfinite(d[v].values).any():
                        problems["all-NaN variable"].append(f"{fp.name}: {v}")
        except Exception as e:
            problems["unreadable"].append(f"{fp.name}: {str(e)[:80]}")

    expected = {(y, m, g)
                for g in GROUPS for y in range(2015, end_month.year + 1)
                for m in range(1, 13)
                if dt.date(y, m, 1) <= end_month}
    for y, m, g in sorted(expected - months):
        problems["absent month file"].append(f"{y}-{m:02d}/{g}")

    print(f"\n{len(files)} file(s) audited, {len(expected)} expected\n")
    if not problems:
        print("PASS -- every file has the right grid, hours, variables and longitude convention.")
        return 0
    for kind, items in sorted(problems.items(), key=lambda kv: -len(kv[1])):
        print(f"{kind}: {len(items)}")
        for it in items[:8]:
            print(f"    {it}")
        if len(items) > 8:
            print(f"    ... and {len(items)-8} more")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parents[1] / "data" / "era5_plains_raw"))
    ap.add_argument("--prefix", default="era5_plains")
    ap.add_argument("--end", default="2026-06")
    args = ap.parse_args()
    y, m = (int(x) for x in args.end.split("-"))
    print(f"auditing {args.dir}  ({args.prefix}_*.nc, through {args.end})\n")
    raise SystemExit(audit(Path(args.dir), args.prefix, dt.date(y, m, 1)))


if __name__ == "__main__":
    main()
