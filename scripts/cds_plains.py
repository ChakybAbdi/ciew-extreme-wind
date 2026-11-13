"""
Queue-aware CDS runner for the Plains ERA5 download.

Three problems with the thread-pool runners this replaces:

1. Latency, not bandwidth, is the cost: a request spends 25 min to 3 h in the CADS queue to deliver
   ~19 MB. Queueing more of them at once does NOT help -- CADS caps the number of queued requests
   per user per dataset and rejects the excess ("Number queued requests for this dataset is
   temporarily limited"), six seconds after accepting them. Measured cap: 4. So the target in-flight
   count starts at 4, halves toward 2 on rejection, and climbs back after a clean stretch.

2. Killing the process orphaned its jobs. They stayed queued under the account, eating fair share,
   with nobody left to download them. Every job ID is written to a registry file as soon as it is
   submitted, and on startup any job already on the account is adopted -- finished ones are
   downloaded without re-queueing, unfinished ones are waited on.

3. Requests were smaller than they had to be. CADS costs a request at 12 * n_vars * n_hours with a
   ceiling of 121,000, i.e. 10,083 variable-hours. One month of group a spends only 62% of that, so
   two short months travel for the price of one request.

Run:  python cds_plains.py [--max-flight 4] [--dry-run]
"""
import argparse
import calendar
import datetime as dt
import importlib.util
import json
import os
import time
from pathlib import Path

import requests
import xarray as xr

_spec = importlib.util.spec_from_file_location(
    "dl", Path(__file__).with_name("download_plains_era5.py"))
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)

DATASET = "reanalysis-era5-single-levels"
COST_LIMIT = 121_000          # CADS "size" cost ceiling for this dataset
COST_PER_VAR_HOUR = 12.0      # measured against the /costing endpoint
MAX_VAR_HOURS = int(COST_LIMIT / COST_PER_VAR_HOUR)
DL_RETRIES = 5                # download attempts before a finished job is finally given up

TMP_DIR = dl.ERA5_DIR / "_tmp"
PARTS_DIR = dl.ERA5_DIR / "_parts"      # per-variable slices awaiting their siblings
GRID_SHAPE = (int(round((dl.AREA[0] - dl.AREA[2]) / dl.GRID[0])) + 1,
              int(round((dl.AREA[3] - dl.AREA[1]) / dl.GRID[1])) + 1)   # (lat, lon) = 41x41
REGISTRY = dl.ERA5_DIR / "_jobs.json"

# CDS request names -> the short names it writes into the NetCDF, so a partly-filled month can be
# checked variable by variable instead of being re-fetched whole
SHORT_NAME = {
    "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
    "100m_u_component_of_wind": "u100", "100m_v_component_of_wind": "v100",
    "instantaneous_10m_wind_gust": "i10fg",
    "10m_wind_gust_since_previous_post_processing": "fg10",
    "boundary_layer_height": "blh",
    "convective_available_potential_energy": "cape", "convective_inhibition": "cin",
    "k_index": "kx", "total_column_water_vapour": "tcwv",
    "mean_sea_level_pressure": "msl", "surface_pressure": "sp",
    "2m_temperature": "t2m", "2m_dewpoint_temperature": "d2m",
}
assert all(v in SHORT_NAME for vs in dl.GROUPS.values() for v in vs)

_rc = dict(l.split(": ", 1) for l in
           Path.home().joinpath(".cdsapirc").read_text().strip().splitlines())
API = _rc["url"].strip()
HDR = {"PRIVATE-TOKEN": _rc["key"].strip()}


def log(msg):
    print(f"{dt.datetime.now():%H:%M:%S} {msg}", flush=True)


# --------------------------------------------------------------------------- archive state
def month_path(y, m, g):
    return dl.ERA5_DIR / f"era5_plains_{y}_{m:02d}_{g}.nc"


def expected_hours(y, m, last_month):
    """Full month, except the archive's final month, which ERA5T truncates a few hours short."""
    full = calendar.monthrange(y, m)[1] * 24
    return full - 24 if (y, m) == (last_month.year, last_month.month) else full


def month_ok(y, m, g, last_month):
    """Complete means the right hours, the right grid AND the full variable group.

    Hour count alone is not enough: the 2015-2019 group-b files are full-length but carry 7 of the
    8 variables, k_index having been lost in the old zip-merge path.
    """
    fp = month_path(y, m, g)
    if not fp.exists() or fp.stat().st_size < 500_000:
        return False
    try:
        with xr.open_dataset(fp) as d:
            return (d.sizes.get("valid_time", 0) >= expected_hours(y, m, last_month)
                    and len(d.data_vars) >= len(dl.GROUPS[g])
                    and (d.sizes.get("latitude"), d.sizes.get("longitude")) == GRID_SHAPE)
    except Exception:
        return False


def missing_months(last_month):
    out = {}
    for g in dl.GROUPS:
        out[g] = [(y, m) for y in dl.YEARS for m in range(1, 13)
                  if dt.date(y, m, 1) <= last_month and not month_ok(y, m, g, last_month)]
    return out


def missing_by_variable(last_month, no_backfill=False):
    """-> {(group, variable): [(year, month), ...]} still to fetch.

    Per variable rather than per month, because a request now carries a variable subset: after an
    interrupted run some variables of a month are already sitting in _parts, and re-requesting the
    whole group would pay for them twice.

    no_backfill skips months that already have a file on disk, i.e. it gives up on completing a
    partly-filled month and spends the queue on months with no data at all. It does not touch
    months still being built out of parts, which have no file yet.
    """
    done, held = set(), {}
    for g in dl.GROUPS:
        for y in dl.YEARS:
            for m in range(1, 13):
                if dt.date(y, m, 1) > last_month:
                    continue
                if month_ok(y, m, g, last_month):
                    done.add((y, m, g))
                    continue
                fp = month_path(y, m, g)
                if fp.exists():         # incomplete file: keep what it already holds
                    try:
                        with xr.open_dataset(fp) as d:
                            held[(y, m, g)] = set(d.data_vars)
                    except Exception:
                        pass
    out = {}
    for g, gvars in dl.GROUPS.items():
        for v in gvars:
            ms = [(y, m) for y in dl.YEARS for m in range(1, 13)
                  if dt.date(y, m, 1) <= last_month
                  and (y, m, g) not in done
                  and not part_path(y, m, g, v).exists()
                  and SHORT_NAME[v] not in held.get((y, m, g), ())
                  and not (no_backfill and (y, m, g) in held)]
            if ms:
                out[(g, v)] = ms
    return out, done


def build_requests(mbv):
    """Fill the cost ceiling as fully as possible.

    The ceiling is on variable-hours, so the a/b variable split is what wastes it: a month of group
    b spends 5,952 of the 10,083 available and two months never fit. Trading variables for months
    is free -- one variable for a whole year is 8,760, 87% of the ceiling -- and a request costs the
    same ~50 min in the queue whatever it carries. So: take every missing month of a year at once,
    then carry as many variables as still fit.
    """
    by_key = {}
    for (g, v), ms in mbv.items():
        by_key.setdefault((g, tuple(ms)), []).append(v)

    reqs = []
    for (g, ms), vs in by_key.items():
        ordered = [v for v in dl.GROUPS[g] if v in vs]      # stable, group-ordered
        for y in sorted({yy for yy, _ in ms}):              # one year per request: the API takes
            ys = [(yy, m) for yy, m in ms if yy == y]       # year and month as independent lists,
            hours = sum(calendar.monthrange(yy, m)[1] * 24 for yy, m in ys)   # so mixing years
            per_req = max(1, MAX_VAR_HOURS // hours)        # would over-select every combination
            for i in range(0, len(ordered), per_req):
                reqs.append({"months": ys, "group": g, "vars": ordered[i:i + per_req]})
    return reqs


# --------------------------------------------------------------------------- CDS REST
def body_for(req):
    return {"inputs": {
        "product_type": ["reanalysis"],
        "variable": list(req["vars"]),
        "year": [str(req["months"][0][0])],
        "month": [f"{m:02d}" for _, m in req["months"]],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": dl.AREA, "grid": dl.GRID,
        "data_format": "netcdf", "download_format": "unarchived"}}


def submit(req):
    r = requests.post(f"{API}/retrieve/v1/processes/{DATASET}/execution",
                      headers=HDR, json=body_for(req), timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} {r.text[:200]}")
    return r.json()["jobID"]


def job_status(job_id):
    try:
        r = requests.get(f"{API}/retrieve/v1/jobs/{job_id}", headers=HDR, timeout=60)
    except Exception:
        return "unknown"                      # transient network blip: leave the job alone
    if r.status_code == 404:
        return "gone"
    if r.status_code >= 400:
        return "unknown"
    return r.json().get("status", "unknown")


def job_request(job_id):
    """The month/group selection a job was submitted with -- used to adopt orphans."""
    r = requests.get(f"{API}/retrieve/v1/jobs/{job_id}?request=True", headers=HDR, timeout=60)
    if r.status_code >= 400:
        return None
    req = (r.json().get("metadata") or {}).get("request", {})
    # the account's job record nests the selection under "ids"; a submitted body uses "inputs"
    inp = req.get("ids") or req.get("inputs") or req
    try:
        year = inp["year"]
        y = int(year[0] if isinstance(year, list) else year)
        ms = inp["month"] if isinstance(inp["month"], list) else [inp["month"]]
        vs = inp["variable"] if isinstance(inp["variable"], list) else [inp["variable"]]
        # a request may carry only part of a group now, so match by subset, not equality
        g = next((k for k, v in dl.GROUPS.items() if set(vs) <= set(v)), None)
        return {"months": [(y, int(m)) for m in ms], "group": g, "vars": list(vs)} if g else None
    except Exception:
        return None


def download_result(job_id, fp):
    r = requests.get(f"{API}/retrieve/v1/jobs/{job_id}/results", headers=HDR, timeout=120)
    r.raise_for_status()
    href = r.json()["asset"]["value"]["href"]
    tmp = str(fp) + ".part"
    with requests.get(href, stream=True, timeout=1800) as s:
        s.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in s.iter_content(1 << 20):
                fh.write(chunk)
    os.replace(tmp, fp)


def delete_job(job_id):
    try:
        requests.delete(f"{API}/retrieve/v1/jobs/{job_id}", headers=HDR, timeout=60)
    except Exception:
        pass


# --------------------------------------------------------------------------- result handling
def canonical_lon(ds):
    """Force longitude to the -180..180 convention.

    CDS does not answer in one convention: for the same area, k_index and total_column_water_vapour
    came back on 258..268 while t2m, msl and the rest came back on -102..-92. They describe the same
    grid (east - 360 == west), but merging them as-is outer-joins into an 82-column half-NaN grid.
    """
    if float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude")
    # ERA5 carries number/expver sometimes as coordinates, sometimes as data variables; xr.merge
    # refuses to combine sources that disagree, so pin them as coordinates everywhere
    for name in ("number", "expver"):
        if name in ds.data_vars:
            ds = ds.set_coords(name)
    return ds


def write_slice(sel, src, out):
    """Write one month, carrying the CDS encoding (float32, zlib level 1) across."""
    enc = {}
    for v in sel.data_vars:
        e = {k: val for k, val in src[v].encoding.items()
             if k in ("dtype", "zlib", "complevel", "_FillValue")}
        e.setdefault("zlib", True)
        e.setdefault("complevel", 1)
        enc[v] = e
    tmp = str(out) + ".tmp.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    sel.load().to_netcdf(tmp, encoding=enc)
    os.replace(tmp, out)


def part_path(y, m, g, var):
    return PARTS_DIR / f"era5_plains_{y}_{m:02d}_{g}__{var}.nc"


def split_to_months(tmp_fp, req, last_month):
    """Cut a multi-month download into per-month files.

    A request carrying the whole variable group lands straight on the final month file. A partial
    request lands in _parts and waits for its siblings, so a month file only ever exists complete.
    """
    g = req["group"]
    whole = set(req["vars"]) == set(dl.GROUPS[g])
    kept, short = [], []
    with xr.open_dataset(tmp_fp) as raw:
        ds = canonical_lon(raw)
        yr = ds.valid_time.dt.year.values
        mo = ds.valid_time.dt.month.values
        for y, m in sorted({(int(a), int(b)) for a, b in zip(yr, mo)}):
            sel = ds.sel(valid_time=(ds.valid_time.dt.year == y) & (ds.valid_time.dt.month == m))
            n = sel.sizes["valid_time"]
            if n < expected_hours(y, m, last_month):
                short.append((y, m, n))
                continue
            if whole:
                write_slice(sel, ds, month_path(y, m, g))
            else:
                # one part per variable, whatever the request carried: the planner asks "is this
                # variable already on disk?" by filename, so a combined name like
                # "__2m_temperature-surface_pressure" would read as both being absent
                for v in req["vars"]:
                    sn = SHORT_NAME[v]
                    if sn in sel.data_vars:
                        write_slice(sel[[sn]], ds, part_path(y, m, g, v))
            kept.append((y, m))
    return kept, short


def assemble(y, m, g, last_month):
    """Build the final month file once every variable of the group is available.

    Sources are the per-variable parts plus, when it exists, the month file already on disk: the
    2015-2019 group-b files came out of the old zip-merge path missing k_index, so backfilling one
    variable has to merge into what is already there rather than rewrite the month from scratch.
    """
    parts = sorted(PARTS_DIR.glob(f"era5_plains_{y}_{m:02d}_{g}__*.nc"))
    existing = month_path(y, m, g)
    if not parts:
        return False
    def read(fp):
        """Read fully and close before returning.

        canonical_lon returns a derived Dataset, so closing *that* leaves the underlying NetCDF
        handle on the source file open -- and Windows then refuses to delete the part afterwards.
        """
        with xr.open_dataset(fp) as src:
            return canonical_lon(src.load())

    dss = []
    try:
        for p in parts:
            dss.append(read(p))
        if existing.exists():
            dss.append(read(existing))
    except Exception:
        for d in dss:
            d.close()
        return False
    try:
        # count rather than name-match: CDS returns short names (u10, i10fg) that do not map
        # mechanically to the long request names, and each part carries distinct variables
        have = set().union(*(set(d.data_vars) for d in dss))
        if len(have) < len(dl.GROUPS[g]):
            return False
        merged = xr.merge(dss, compat="override", combine_attrs="override", join="exact")
        if merged.sizes.get("valid_time", 0) < expected_hours(y, m, last_month):
            return False
        if (merged.sizes.get("latitude"), merged.sizes.get("longitude")) != GRID_SHAPE:
            log(f"  REFUSING {y}-{m:02d}/{g}: grid is "
                f"{merged.sizes.get('latitude')}x{merged.sizes.get('longitude')}, expected "
                f"{GRID_SHAPE[0]}x{GRID_SHAPE[1]}")
            return False
        merged = merged.load()      # read fully before the sources close: the target may BE one
    finally:
        for d in dss:
            d.close()
    try:
        write_slice(merged, merged, existing)
    finally:
        merged.close()
    for p in parts:
        p.unlink(missing_ok=True)
    return True


def sweep_parts(last_month):
    """Merge any month whose parts are all present -- e.g. the run died between the last part
    landing and the merge."""
    seen, made = set(), 0
    for p in PARTS_DIR.glob("era5_plains_*__*.nc"):
        try:
            _, _, y, m, g = p.stem.split("__")[0].split("_")
            key = (int(y), int(m), g)
        except ValueError:
            continue
        if key in seen:
            continue
        seen.add(key)
        if not month_ok(*key, last_month) and assemble(*key, last_month):
            made += 1
    return made


def handle_finished(job_id, req, last_month, stats):
    tmp_fp = TMP_DIR / f"job_{job_id}.nc"
    download_result(job_id, tmp_fp)
    dl.finalize(tmp_fp)
    kept, short = split_to_months(tmp_fp, req, last_month)
    tmp_fp.unlink(missing_ok=True)
    g = req["group"]
    whole = set(req["vars"]) == set(dl.GROUPS[g])
    done = [(y, m) for y, m in kept if whole or assemble(y, m, g, last_month)]
    stats["months"] += len(done)
    nv = len(req["vars"])
    log(f"  <- {g} {req['months'][0][0]} {[m for _, m in req['months']]} x{nv}var -> "
        f"{len(kept)} slice(s), {len(done)} month file(s) complete"
        f"{'  SHORT ' + str(short) if short else ''}   [{stats['months']} months this run]")
    delete_job(job_id)


# --------------------------------------------------------------------------- registry
def load_registry():
    if REGISTRY.exists():
        try:
            return json.loads(REGISTRY.read_text())
        except Exception:
            pass
    return {}


def save_registry(reg):
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=1))


def adopt_account_jobs(reg, last_month):
    """Pick up jobs already on the account (from killed runs or failed downloads) rather than waste
    them. A finished job whose payload we already hold is left alone."""
    page, found, skipped = f"{API}/retrieve/v1/jobs?limit=100", 0, 0
    while page:
        r = requests.get(page, headers=HDR, timeout=60).json()
        for j in r.get("jobs", []):
            # Only unfinished jobs. Our own finished-but-undownloaded jobs stay in the registry
            # until the download succeeds, so they need no rescuing from the account listing --
            # and sweeping up every old successful job would fill the in-flight budget with
            # results we already hold, blocking new submissions. harvest_old_jobs.py does that
            # deliberately, on demand.
            if j["status"] in ("accepted", "running") and j["jobID"] not in reg:
                got = job_request(j["jobID"])
                if got:
                    if j["status"] == "successful" and not still_needed(got, last_month):
                        skipped += 1          # already on disk: an old job from a finished batch
                        continue
                    reg[j["jobID"]] = got
                    found += 1
        page = next((l["href"] for l in r.get("links", []) if l.get("rel") == "next"), None)
    if found or skipped:
        log(f"adopted {found} job(s) left over from a previous run "
            f"({skipped} finished job(s) already on disk, ignored)")
    return reg


def still_needed(req, last_month):
    """Would this request's payload actually add anything we do not already hold?"""
    g, vs = req["group"], req["vars"]
    for y, m in req["months"]:
        if month_ok(y, m, g, last_month):
            continue
        if any(not part_path(y, m, g, v).exists() for v in vs):
            return True
    return False


def normalise(info):
    """JSON round-trips tuples to lists, and entries written by an older run carry no var list."""
    return {"months": [tuple(x) for x in info["months"]],
            "group": info["group"],
            "vars": list(info.get("vars") or dl.GROUPS[info["group"]])}


def already_covered(req, reg, last_month):
    """Skip work an adopted job is already going to deliver."""
    months, g, vs = {tuple(x) for x in req["months"]}, req["group"], set(req["vars"])
    for info in reg.values():
        if (info["group"] == g and {tuple(x) for x in info["months"]} >= months
                and set(info.get("vars", dl.GROUPS[g])) >= vs):
            return True
    return all(month_ok(y, m, g, last_month) for y, m in months)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-flight", type=int, default=4)   # per-dataset queue cap
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backfill", action="store_true",
                    help="do not spend queue slots completing months that already have a file")
    args = ap.parse_args()

    dl.ERA5_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    now = dt.date.today()
    last_month = min(dl.END_MONTH, dt.date(now.year, now.month, 1))

    swept = sweep_parts(last_month)      # finish anything a previous run left assembled-but-not-merged
    if swept:
        log(f"assembled {swept} month file(s) from parts left by a previous run")

    mbv, done = missing_by_variable(last_month, args.no_backfill)
    if args.no_backfill:
        skipped = sum(1 for g in dl.GROUPS for y in dl.YEARS for m in range(1, 13)
                      if dt.date(y, m, 1) <= last_month and (y, m, g) not in done
                      and month_path(y, m, g).exists())
        log(f"--no-backfill: leaving {skipped} partly-filled month file(s) as they are")
    n_missing = sum(1 for g in dl.GROUPS for y in dl.YEARS for m in range(1, 13)
                    if dt.date(y, m, 1) <= last_month and (y, m, g) not in done)
    # years first: a month file is only complete once every variable of its group has landed, so
    # finishing one year before starting the next keeps the parts directory small
    todo = sorted(build_requests(mbv), key=lambda r: (r["months"][0][0], r["group"]))

    log(f"domain N{dl.AREA[0]} W{dl.AREA[1]} S{dl.AREA[2]} E{dl.AREA[3]} @ {dl.GRID[0]} deg "
        f"| through {last_month:%Y-%m}")
    log(f"{n_missing} month-files missing -> {len(todo)} requests "
        f"(cap {MAX_VAR_HOURS} var-hours each, target {args.max_flight} in flight)")
    if args.dry_run:
        for r in todo:
            hrs = sum(calendar.monthrange(y, m)[1] * 24 for y, m in r["months"])
            log(f"    {r['group']}: {r['months'][0][0]} months {[m for _, m in r['months']]} "
                f"x {len(r['vars'])} var = {len(r['vars'])*hrs} var-hours "
                f"({100*len(r['vars'])*hrs/MAX_VAR_HOURS:.0f}% of cap)")
        return

    reg = adopt_account_jobs(load_registry(), last_month)
    save_registry(reg)
    pending = [r for r in todo if not already_covered(r, reg, last_month)]

    stats = {"months": 0}
    flight = [args.max_flight]      # adaptive: shrinks on rejection, grows back on success
    rejects = [0]
    dl_fails = {}                   # jobID -> consecutive download failures
    t0 = time.time()
    seen = None
    try:
        while pending or reg:
            while pending and len(reg) < flight[0]:
                req = pending.pop(0)
                try:
                    jid = submit(req)
                    reg[jid] = req
                    save_registry(reg)
                    log(f"  -> submitted {req['group']} {req['months'][0][0]} "
                        f"{[m for _, m in req['months']]} x{len(req['vars'])}var  "
                        f"({len(reg)} in flight, {len(pending)} left to submit)")
                    time.sleep(3)
                except Exception as e:
                    log(f"  submit refused ({str(e)[:140]}); backing off 60 s")
                    pending.insert(0, req)
                    time.sleep(60)
                    break

            time.sleep(args.poll)

            running = 0
            for jid in list(reg):
                st = job_status(jid)
                if st == "successful":
                    info = normalise(reg[jid])
                    try:
                        handle_finished(jid, info, last_month, stats)
                        reg.pop(jid, None)
                        save_registry(reg)
                        dl_fails.pop(jid, None)
                        if stats["months"] and stats["months"] % 6 == 0 and flight[0] < args.max_flight:
                            flight[0] += 1
                            log(f"  clean stretch; target in-flight -> {flight[0]}")
                    except Exception as e:
                        # the job is finished on the server -- the result is still there. Discarding
                        # it would cost another full trip through the queue for a transient blip.
                        dl_fails[jid] = dl_fails.get(jid, 0) + 1
                        if dl_fails[jid] >= DL_RETRIES:
                            reg.pop(jid, None)
                            save_registry(reg)
                            pending.append(info)
                            log(f"  download failed {dl_fails[jid]}x for {jid[:8]} "
                                f"({str(e)[:110]}); giving up, requeueing")
                        else:
                            log(f"  download failed for {jid[:8]} ({str(e)[:110]}); "
                                f"keeping job, retry {dl_fails[jid]}/{DL_RETRIES} next cycle")
                elif st in ("failed", "dismissed", "gone", "rejected"):
                    info = normalise(reg.pop(jid))
                    save_registry(reg)
                    pending.insert(0, info)
                    if st == "rejected":
                        # "Number queued requests for this dataset is temporarily limited" --
                        # we were over the per-dataset queue cap. Back the target off and retry.
                        rejects[0] += 1
                        flight[0] = max(2, min(flight[0], len(reg)) - 1)
                        log(f"  job {jid[:8]} rejected (queue cap); target in-flight -> {flight[0]}")
                    else:
                        log(f"  job {jid[:8]} {st}; requeueing {info['group']} {info['months']}")
                elif st == "running":
                    running += 1
            state = (running, len(reg))
            if state != seen:
                log(f"  [queue] {running} running / {len(reg)} in flight / {len(pending)} to submit")
                seen = state
    finally:
        save_registry(reg)

    have = sorted(dl.ERA5_DIR.glob("era5_plains_*.nc"))
    size = sum(f.stat().st_size for f in have) / 1e9
    log(f"finished in {(time.time()-t0)/3600:.2f} h | {len(have)} month-files, {size:.2f} GB")


if __name__ == "__main__":
    main()
