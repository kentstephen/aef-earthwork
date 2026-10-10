"""Satellite laser truth for the earthwork model: where ICESat-2 or GEDI measured the same ground twice.

Repeat airborne lidar (3DEP and the national programs) is the model's truth where it exists: the United States
and parts of Europe. Over most of Africa and Asia nothing flew twice. Two space lasers did measure the ground
there, again and again, along narrow lines:

  ICESat-2 (2018 on), two ways:
  ATL03 bins. Photons kept at high confidence, binned every 10 m along each beam: a bin's median height is the
  ground where its photons are tightly stacked (90th minus 10th percentile under 1 m, "bare"); under canopy
  the stack spreads and the bin is left out. Single photons range to about 0.1 m; the geolocation is good to
  about 4 m. Read without a login through OpenAltimetry (the passes from NASA's CMR, which is public).
  ATL08 20 m terrain. The land product's ground fit every 20 m (h_te_best_fit_20m), from photons it
  classed as ground at any confidence, under canopy too, with the canopy height over it (h_canopy_20m) and,
  per 100 m segment, the ground photon count (n_te_photons), the along-track slope (terrain_slope) and the
  product's reference DEM height (dem_h). Read from the granules at NSIDC (Earthdata login, earthaccess), only
  the datasets and rows over the box. h_te_uncertainty is kept but not used: it runs 5 to 100 m at these
  sites, far from the pairs' own spread, so it is no guide to a 20 m height.
  GEDI (2019 to 2023, again from 2024). L2A elev_lowestmode, the ground under each 25 m footprint, kept where
  quality_flag (l2a_quality_flag_rel3 in version 3) is 1, degrade_flag 0 and sensitivity at least 0.95. About
  0.5 to 1 m on open ground, 1 to 3 m under canopy; footprints placed to within about 10 m. Needs a NASA
  Earthdata login (~/.netrc, machine urs.earthdata.nasa.gov), read with earthaccess.

A PAIR is two measurements of the same ground at two dates: ATL03 bins within 10 m of each other, ATL08 20 m
heights within 15 m, GEDI footprints within 12 m, or a GEDI footprint and an ATL03 bin or ATL08 height within
12 m (their constant offset, the median difference over the site, removed; GEDI's rules, and the ICESat-2
side as its own pairs ask: a bare ATL03 bin, an ATL08 height under low canopy with enough ground photons). Each measurement is paired with the nearest
one of every other date, so ground crossed by n passes gives a pair for every two of their dates (repeat
tracks of one reference ground track and crossings of others alike). On a slope s two measurements r m apart
differ by r * s with no ground moved, so the Copernicus GLO-30 DEM's height difference between the two places
is taken off dz, and half of what was taken off is added to the moved threshold (the DEM is 30 m, a 2011 to
2015 surface, over trees). Each pair is one AlphaEarth 10 m pixel, at the AlphaEarth years
earthwork_model.aef_years gives the two dates; only pairs whose dates fall in different AlphaEarth years are
kept. The labels are the lidar sites' with thresholds wide enough for the noise:

  ATL03 bins  moved |dz| > 2.0 m (both bins bare)    unchanged |dz| < 0.5 m (both bare, slope under 15%),
              or the ground clusters within 0.7 m under canopy (unchanged only)
  ATL08       moved |dz| > 3.0 m (both canopy under   unchanged |dz| < 0.75 m (along-track and DEM slope
              10 m, DEM slope under 30%)               under 15%, any canopy)
              both: at least 20 ground photons in the 100 m segment, within 75 m of the product's DEM
  GEDI        moved |dz| > 3.0 m (both low canopy)   unchanged |dz| < 1.0 m (low canopy, rh98 under 5 m),
              or < 1.5 m under canopy at sensitivity 0.98 and up. Moved only against ATL08: GEDI's own
              pairs and its pairs with ATL03 bins called moved as often on ground AlphaEarth saw
              unchanged as on ground it saw change, so they teach unchanged ground only

The ATL08 thresholds come from the pairs themselves. On ground nobody dug (Serengeti savanna, Unki's
woodland around the mine) the slope-corrected dz of ATL08 pairs spreads 0.25 to 0.4 m (1.48 times the median
absolute deviation), strong and weak beams alike, under woodland canopy as on open ground (the correction
brings Unki's 15 m pairs from 0.61 to 0.41 m): 0.75 m is about two and a half of those. Of 989 Serengeti
pairs, 8 pass 3 m and 1 passes the moved rule below. Against AlphaEarth's own change at the pair's
pixel and years, ATL08 dz past 3 m marks changed ground at the mines (Sierra Rutile) as often as ATL03's bare
2 m does, except where the 20 m segment holds tall canopy (h_canopy_20m 10 m and up), few ground photons
(under 20 in the 100 m segment) or a steep hillside: at Shurugwi those pairs, mostly from one rainy-season
pass, swing 3 to 10 m with no change on the ground. A canopy taller than 10 m is no bar to unchanged ground:
the ground fit under it spreads no wider.

Anything between is left out. Where pairs at one pixel and year pair disagree (one moved, one unchanged), the
pixel is left out of that year pair. Each site is cached as data/earthwork/space-<name>.npz in the format of
earthwork_model.site_data (X, y, plain, dz, patch, wet, meta), so the training script can add them as sites;
meta also holds the box and the moved and unchanged counts by source.

Run: uv run --with icechunk --with earthaccess --with h5py python earthwork_points_space.py [icesat2|atl08|gedi|both|all] [site ...]
     (icesat2 is the ATL03 bins and needs no login; the rest need ~/.netrc for urs.earthdata.nasa.gov; all,
     the default, is ATL03 bins, ATL08 and GEDI)
"""

import datetime
import gzip
import json
import os
import re
import sys
import time

import numpy as np
import requests
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score

import earthwork_model as em

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(em.DATA, "space-raw")
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
OA = "https://openaltimetry.earthdatacloud.nasa.gov/data/api/icesat2"
GEDI_EPOCH = datetime.datetime(2018, 1, 1)

# (name, kind, a point inside the site, half the box in degrees: one number for a square, or (east-west,
# north-south)). Kinds as earthwork_model: mining and building teach both ways, the rest only as not digging
# (their unchanged ground). The mining boxes sit on the strongest AlphaEarth change near each place (plain
# change 2023 to 2025 and 2019 to 2025 over a 40 to 50 km window, checked against mapped mines):
#   Shurugwi: the chrome workings run north and south in a belt about 3 km wide along 30.00 to 30.04 E, from
#   19.52 to 19.74 S (the largest active patch, 35 ha, at 30.017 E, 19.729 S); Unki's platinum mine and its
#   tailings 7 km east. The larger 2019 to 2025 change at 29.87 E, 19.74 S is a reservoir on the Runde (its
#   water level), not digging.
#   Sierra Rutile: the Gangama and Lanti workings west of the first box (to 12.39 W), and the Gbeni and Sembehun
#   dry mining 8 km south.
#   Kolwezi: the Sicomines pits west of the first box (25.32 to 25.35 E, 10.74 S) and KOV / Kamoto.
SITES = [
    ("shurugwi-zw", "mining", (30.015, -19.63), (0.025, 0.12)),  # chrome and gold workings along the Great Dyke
    ("unki-zw", "mining", (30.085, -19.625), 0.02),  # Unki platinum mine, its plant and tailings
    ("mobimbi-sl", "mining", (-12.365, 7.745), 0.045),  # Sierra Rutile mineral sands: Gangama, Lanti, Nitti
    ("sembehun-sl", "mining", (-12.305, 7.675), 0.03),  # Sierra Rutile: Gbeni and Sembehun dry mining
    ("kolwezi-cd", "mining", (25.36, -10.74), (0.06, 0.04)),  # copper and cobalt pits: Sicomines, KOV
    ("morowali-id", "mining", (122.15, -2.83), 0.03),  # nickel mines and the industrial park
    ("weda-bay-id", "mining", (127.90, 0.45), 0.03),  # nickel mines
    ("new-capital-eg", "building", (31.75, 30.02), 0.03),  # Egypt's New Administrative Capital
    ("eko-atlantic-ng", "building", (3.41, 6.41), 0.03),  # land reclaimed from the sea at Lagos
    ("xiongan-cn", "building", (115.95, 39.00), 0.03),  # Xiong'an new area
    ("amaravati-in", "building", (80.52, 16.52), 0.03),  # Amaravati capital region
    ("serengeti-tz", "savanna", (34.80, -2.30), 0.03),
    ("salonga-cd", "rainforest", (21.00, -2.00), 0.03),
    ("banaue-ph", "terraces", (121.06, 16.92), 0.03),  # rice terraces
]
TEACH_KINDS = ("building", "mining")

IS2_PAIR_M, IS2_BIN_M = 10.0, 10.0
IS2_MOVED_M, IS2_STILL_M, IS2_BARE_M, IS2_SLOPE, IS2_STILL_CANOPY_M = 2.0, 0.5, 1.0, 0.15, 0.7
GEDI_PAIR_M = 12.0
GEDI_MOVED_M, GEDI_STILL_M, GEDI_STILL_CANOPY_M, GEDI_LOW_RH98, GEDI_SLOPE = 3.0, 1.0, 1.5, 5.0, 0.15
# ATL08 20 m terrain heights (the reasoning in the module docstring)
A08_PAIR_M = 15.0
A08_MOVED_M, A08_STILL_M, A08_CANOPY_M, A08_STEEP, A08_SLOPE = 3.0, 0.75, 10.0, 0.30, 0.15
A08_MIN_PHOTONS, A08_MAX_FROM_REF = 20, 75.0
SOURCES = ("is2", "atl08", "gedi")
GEDI_BEAMS = ("BEAM0000", "BEAM0001", "BEAM0010", "BEAM0011", "BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011")


def _get(url, params, tries=4):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=180)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(3 + 5 * k)
    return None


def _box(point, half):
    lon, lat = point
    hx, hy = half if isinstance(half, (tuple, list)) else (half, half)
    return lon - hx, lat - hy, lon + hx, lat + hy


def _fresh(name, box):
    """Raw files are kept per site; when a site's box moves, its old raw files are dropped (they hold another
    box's ground). The box is noted in <name>-box.json."""
    os.makedirs(RAW, exist_ok=True)
    note = f"{RAW}/{name}-box.json"
    want = [round(v, 6) for v in box]
    if os.path.exists(note):
        with open(note) as f:
            if json.load(f) == want:
                return
        gone = [g for g in os.listdir(RAW) if g.startswith(name + "-") and not g.endswith("-box.json")]
        for g in gone:
            os.remove(os.path.join(RAW, g))
        print(f"  {name}: the box moved, {len(gone)} old raw files dropped")
    with open(note, "w") as f:
        json.dump(want, f)


def _local(lon, lat, lon0, lat0):
    """Meters east and north of (lon0, lat0)."""
    return (lon - lon0) * 111320 * np.cos(np.radians(lat0)), (lat - lat0) * 110574


_FS = {}


def _granule(job, tries=3):
    """One granule read in a worker process: (url, reader name, box, what) -> the reader's list of dicts. The
    file is opened over HTTPS in 2 MB blocks: h5py asks for many small pieces, and fsspec's default read-ahead
    moved about ten times the bytes for the same datasets."""
    import aiohttp
    import earthaccess
    import fsspec
    import h5py

    url, reader, box, what = job
    if "token" not in _FS:
        _FS["token"] = earthaccess.login(strategy="netrc").token["access_token"]
    for k in range(tries):
        try:
            # earthaccess's own session, with timeouts (without them a stalled read waits forever), new for each
            # granule: a session kept across granules stalled after a few
            fs = fsspec.filesystem("https", skip_instance_cache=True, client_kwargs={
                "headers": {"Authorization": f"Bearer {_FS['token']}"}, "trust_env": False,
                "timeout": aiohttp.ClientTimeout(total=300, sock_connect=30, sock_read=90)})
            with fs.open(url, "rb", block_size=2 ** 21, cache_type="blockcache") as fh:
                with h5py.File(fh, "r") as f:
                    return globals()[reader](f, box)
        except Exception as e:
            err = e
            time.sleep(5 + 10 * k)
    print(f"    a {what} granule skipped: {type(err).__name__}: {str(err)[:80]}", flush=True)
    return []


def _granules(found, reader, box, what, workers=5):
    """The reader (a module function of an open h5py file and the box) over each granule found, in worker
    processes: h5py holds one lock for every thread, so threads read one granule at a time, and each small
    read waits a second or more on the server. More than about five at once and the GEDI server (LP DAAC)
    stalls every read."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(g.data_links()[0], reader, box, what) for g in found]
    with ProcessPoolExecutor(workers) as pool:
        for k, got in enumerate(pool.map(_granule, jobs)):
            if k % 20 == 19:
                print(f"    {what}: {k + 1} of {len(jobs)} granules read", flush=True)
            yield got


# ---- ICESat-2 through OpenAltimetry (no login) ------------------------------------------------------------


def is2_passes(box):
    """The ICESat-2 passes over the box: (date, reference ground track), from the ATL08 granules in CMR."""
    out, page = set(), 1
    while True:
        got = _get(CMR, {"short_name": "ATL08", "version": "007", "bounding_box": ",".join(map(str, box)),
                         "page_size": 500, "page_num": page})
        e = (got or {}).get("feed", {}).get("entry", [])
        for g in e:
            m = re.search(r"ATL08_(\d{8})\d{6}_(\d{4})", g["title"])
            if m:
                d = m.group(1)
                out.add((f"{d[:4]}-{d[4:6]}-{d[6:]}", int(m.group(2))))
        if len(e) < 500:
            return sorted(out)
        page += 1


IS2_BAND_DEG = 0.015  # the box asked in latitude bands this tall: the API returns at most about 50,000 photons a request, noise included


def is2_photons(name, box, date, rgt):
    """High-confidence ATL03 photons of one pass over the box: {beam: (n, 3) lat, lon, height}, cached per band."""
    os.makedirs(RAW, exist_ok=True)
    out = {}
    bands = int(np.ceil((box[3] - box[1]) / IS2_BAND_DEG - 1e-6))
    edges = np.linspace(box[1], box[3], bands + 1)

    def band(k):
        path = f"{RAW}/{name}-atl03-{date}-{rgt}-b{k}.json.gz"
        if os.path.exists(path):
            try:
                with gzip.open(path, "rt") as f:
                    return json.load(f)
            except Exception:
                os.remove(path)  # cut short by an interrupted run
        got = _get(f"{OA}/atl03", {"date": date, "trackId": rgt, "minx": box[0], "miny": float(edges[k]), "maxx": box[2],
                                   "maxy": float(edges[k + 1]), "outputFormat": "json"})
        if got is not None:
            with gzip.open(path, "wt") as f:
                json.dump(got, f)
        return got

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(8) as pool:
        for got in pool.map(band, range(bands)):
            for s in got if isinstance(got, list) else []:
                hi = [x["data"] for x in s.get("series", []) if x.get("name") == "High" and x.get("data")]
                if hi:
                    out.setdefault(s["beam_name"], []).extend(hi[0])
    return {b: np.asarray(v, np.float64) for b, v in out.items() if len(v) >= 20}


def is2_bins(ph, lon0, lat0):
    """One beam's photons in 10 m bins along its line: (n, 7) x, y, median height, 90th minus 10th percentile,
    photon count, along-track slope of the ground, and the ground under canopy: the mean of the photons within
    0.5 m above the bin's 5th percentile when at least 3 lie there (NaN otherwise)."""
    x, y = _local(ph[:, 1], ph[:, 0], lon0, lat0)
    xy = np.c_[x, y]
    c = xy.mean(0)
    u = np.linalg.svd(xy - c, full_matrices=False)[2][0]  # the line's direction
    s = (xy - c) @ u
    k = np.floor((s - s.min()) / IS2_BIN_M).astype(int)
    rows = []
    for b in np.unique(k):
        m = k == b
        if m.sum() < 5:
            continue
        h = ph[m, 2]
        p5, p10, p50, p90 = np.percentile(h, [5, 10, 50, 90])
        low = h[h < p5 + 0.5]
        rows.append([x[m].mean(), y[m].mean(), p50, p90 - p10, m.sum(), float(b), low.mean() if len(low) >= 3 else np.nan])
    if len(rows) < 2:
        return np.zeros((0, 7))
    r = np.array(rows)
    # slope from the neighboring bins' ground (their median where no ground cluster), rise over the run between them
    g = np.where(np.isfinite(r[:, 6]), r[:, 6], r[:, 2])
    slope = np.full(len(r), np.nan)
    for i in range(len(r)):
        j = [t for t in (i - 1, i + 1) if 0 <= t < len(r) and abs(r[t, 5] - r[i, 5]) == 1]
        if j:
            dx = [np.hypot(r[t, 0] - r[i, 0], r[t, 1] - r[i, 1]) for t in j]
            slope[i] = max(abs(g[t] - g[i]) / max(d, 1.0) for t, d in zip(j, dx))
    r[:, 5] = slope
    return r


def is2_measurements(name, point, half):
    """Every 10 m bin of every pass over the site: dict of arrays x, y, h, spread, slope, date (datetime64[D])."""
    box = _box(point, half)
    cols = {k: [] for k in ("x", "y", "h", "spread", "slope", "ground", "date")}
    passes = is2_passes(box)
    for date, rgt in passes:
        for beam, ph in is2_photons(name, box, date, rgt).items():
            r = is2_bins(ph, *point)
            if not len(r):
                continue
            for k, j in (("x", 0), ("y", 1), ("h", 2), ("spread", 3), ("slope", 5), ("ground", 6)):
                cols[k].append(r[:, j])
            cols["date"].append(np.full(len(r), np.datetime64(date, "D")))
    print(f"  {name}: {len(passes)} ICESat-2 passes, {sum(len(v) for v in cols['h']):,} bins")
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}


# ---- GEDI through earthaccess (Earthdata login) -----------------------------------------------------------


def _read_gedi(f, box):
    """One granule's good footprints in the box. Only each beam's coordinates are read whole (byte ranges,
    a few MB); the rest only over the run of footprints that falls in the box."""
    got = []
    for beam in GEDI_BEAMS:
        if beam not in f:
            continue
        g = f[beam]
        lat, lon = g["lat_lowestmode"][:], g["lon_lowestmode"][:]
        m = (lon >= box[0]) & (lon <= box[2]) & (lat >= box[1]) & (lat <= box[3])
        if not m.any():
            continue
        i0, i1 = np.flatnonzero(m)[[0, -1]]
        sl, mm = slice(i0, i1 + 1), m[i0:i1 + 1]
        q = g["quality_flag"] if "quality_flag" in g else g["l2a_quality_flag_rel3"]  # version 3 renames it
        ok = (q[sl][mm] == 1) & (g["degrade_flag"][sl][mm] == 0) & (g["sensitivity"][sl][mm] >= 0.95)
        if not ok.any():
            continue
        t = g["delta_time"][sl][mm][ok]
        got.append({"lon": lon[sl][mm][ok], "lat": lat[sl][mm][ok], "h": g["elev_lowestmode"][sl][mm][ok],
                    "rh98": g["rh"][sl, 98][mm][ok], "sens": g["sensitivity"][sl][mm][ok],
                    "date": np.array([np.datetime64(GEDI_EPOCH + datetime.timedelta(seconds=float(v)), "D") for v in t])})
    return got


def gedi_measurements(name, point, half):
    """Every good GEDI L2A footprint over the site: dict of arrays x, y, h, rh98, sens, date. Cached per site."""
    os.makedirs(RAW, exist_ok=True)
    path = f"{RAW}/{name}-gedi.npz"
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    import earthaccess

    earthaccess.login(strategy="netrc")
    box = _box(point, half)
    found = earthaccess.search_data(short_name="GEDI02_A", version="003", bounding_box=box)
    cols = {k: [] for k in ("lon", "lat", "h", "rh98", "sens", "date")}

    for got in _granules(found, "_read_gedi", box, "GEDI"):
        for d in got:
            for k in cols:
                cols[k].append(d[k])
    out = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    out["x"], out["y"] = _local(out["lon"], out["lat"], *point) if len(out["lon"]) else (np.zeros(0), np.zeros(0))
    np.savez_compressed(path, **out)
    print(f"  {name}: {len(found)} GEDI granules, {len(out['h']):,} good footprints")
    return out


# ---- ICESat-2 ATL08 through earthaccess (Earthdata login) -------------------------------------------------


ATL08_BEAMS = ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r")
ATL08_FILL = 1e30  # float fill values are 3.4e38
ATLAS_EPOCH = datetime.datetime(2018, 1, 1)


def _read_atl08(f, box):
    """One ATL08 granule's 20 m terrain heights in the box (rows as atl08_measurements)."""
    got = []
    orient = int(f["orbit_info/sc_orient"][0])  # 0 backward (left beams strong), 1 forward (right strong)
    if orient not in (0, 1):
        return got  # turning: beam strength unknown
    for beam in ATL08_BEAMS:
        if f"{beam}/land_segments" not in f:
            continue
        g = f[f"{beam}/land_segments"]
        lat, lon = g["latitude"][:], g["longitude"][:]
        pad = 0.002  # a 100 m segment's 20 m parts reach 40 m past its center
        m = (lon >= box[0] - pad) & (lon <= box[2] + pad) & (lat >= box[1] - pad) & (lat <= box[3] + pad)
        if not m.any():
            continue
        i0, i1 = np.flatnonzero(m)[[0, -1]]
        sl = slice(i0, i1 + 1)
        lat20, lon20 = g["latitude_20m"][sl], g["longitude_20m"][sl]
        h20 = g["terrain/h_te_best_fit_20m"][sl].astype(np.float64)
        c20 = g["canopy/h_canopy_20m"][sl].astype(np.float64)
        n = len(h20)
        per = {"slope": g["terrain/terrain_slope"][sl], "unc": g["terrain/h_te_uncertainty"][sl],
               "nte": g["terrain/n_te_photons"][sl], "ref": g["dem_h"][sl], "cloud": g["cloud_flag_atm"][sl]}
        t = g["delta_time"][sl]
        day = np.array([np.datetime64(ATLAS_EPOCH + datetime.timedelta(seconds=float(v)), "D") for v in t])
        rep = lambda v: np.repeat(np.asarray(v), 5)
        d = {"lon": lon20.ravel(), "lat": lat20.ravel(), "h": h20.ravel(), "canopy": c20.ravel(),
             **{k: rep(v).astype(np.float32) for k, v in per.items()},
             "strong": np.full(n * 5, beam.endswith("l") == (orient == 0)), "date": rep(day)}
        ok = (np.abs(d["h"]) < ATL08_FILL) & (np.abs(d["lat"]) < 90) & (d["lon"] >= box[0]) & (d["lon"] <= box[2]) \
            & (d["lat"] >= box[1]) & (d["lat"] <= box[3])
        if ok.any():
            d["canopy"] = np.where(np.abs(d["canopy"]) < ATL08_FILL, d["canopy"], np.nan)
            for k in ("slope", "unc", "ref"):
                d[k] = np.where(np.abs(d[k]) < ATL08_FILL, d[k], np.nan)
            got.append({k: v[ok] for k, v in d.items()})
    return got


def atl08_measurements(name, point, half):
    """Every ATL08 20 m terrain height over the site: dict of arrays x, y, h (h_te_best_fit_20m), canopy
    (h_canopy_20m, NaN where none), and the 100 m segment's slope (terrain_slope, along track), unc
    (h_te_uncertainty), nte (n_te_photons), ref (dem_h, the product's reference DEM), cloud (cloud_flag_atm),
    strong (the beam), date. Read from the granules over HTTPS: each beam's 100 m latitudes and longitudes whole
    (two chunks of 10,000), the rest only over the run of segments in the box. Cached per site."""
    os.makedirs(RAW, exist_ok=True)
    path = f"{RAW}/{name}-atl08.npz"
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    import earthaccess

    earthaccess.login(strategy="netrc")
    box = _box(point, half)
    found = earthaccess.search_data(short_name="ATL08", version="007", bounding_box=box)
    keys = ("lon", "lat", "h", "canopy", "slope", "unc", "nte", "ref", "cloud", "strong", "date")

    cols = {k: [] for k in keys}
    for got in _granules(found, "_read_atl08", box, "ATL08"):
        for d in got:
            for k in cols:
                cols[k].append(d[k])
    out = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    out["x"], out["y"] = _local(out["lon"], out["lat"], *point) if len(out["lon"]) else (np.zeros(0), np.zeros(0))
    np.savez_compressed(path, **out)
    print(f"  {name}: {len(found)} ATL08 granules, {len(out['h']):,} 20 m terrain heights")
    return out


# ---- the ground's slope, for pairs whose two measurements are a few meters apart -------------------------


def copdem(name, point, half):
    """Copernicus GLO-30 heights over the box (public COGs on AWS, no login), cached per site: a function
    giving the height at local x, y (bilinear). The DEM is a 2011 to 2015 surface (TanDEM-X), so it only
    stands for the slope between two nearby points, not for any date's ground."""
    from scipy.ndimage import map_coordinates

    os.makedirs(RAW, exist_ok=True)
    path = f"{RAW}/{name}-copdem.npz"
    box = _box(point, half)
    pad = 0.002
    if not os.path.exists(path):
        import rasterio
        from rasterio.merge import merge

        tiles = []
        for la in range(int(np.floor(box[1] - pad)), int(np.floor(box[3] + pad)) + 1):
            for lo in range(int(np.floor(box[0] - pad)), int(np.floor(box[2] + pad)) + 1):
                t = f"Copernicus_DSM_COG_10_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM"
                tiles.append(f"/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{t}/{t}.tif")
        srcs = []
        for t in tiles:
            try:
                srcs.append(rasterio.open(t))
            except Exception:
                pass  # open sea has no tile
        if not srcs:
            return lambda x, y: np.zeros(np.shape(x))
        a, tr = merge(srcs, bounds=(box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad))
        for s_ in srcs:
            s_.close()
        # above 50 degrees GLO-30 columns are wider than rows, so both spacings are kept
        np.savez_compressed(path, z=a[0].astype(np.float32), W=tr.c, N=tr.f, rx=tr.a, ry=abs(tr.e))
    g = np.load(path)
    z, W, N, rx, ry = g["z"], float(g["W"]), float(g["N"]), float(g["rx"]), float(g["ry"])

    def at(x, y):
        lon = point[0] + np.asarray(x) / (111320 * np.cos(np.radians(point[1])))
        lat = point[1] + np.asarray(y) / 110574
        return map_coordinates(z, [(N - lat) / ry - 0.5, (lon - W) / rx - 0.5], order=1, mode="nearest")

    return at


# ---- pairs and labels -----------------------------------------------------------------------------------


def _pairs(a, b, radius, same=False):
    """Index pairs (i in a, j in b) within radius m and on different dates, the earlier date first when a is b
    (same): for each measurement, the nearest one of every other date (repeat passes over the same ground give
    a pair for every two of their dates)."""
    if not len(a["x"]) or not len(b["x"]):
        return np.zeros((0, 2), int)
    ta = cKDTree(np.c_[a["x"], a["y"]])
    if same:
        P = ta.query_pairs(radius, output_type="ndarray")
    else:
        sp = ta.sparse_distance_matrix(cKDTree(np.c_[b["x"], b["y"]]), radius, output_type="ndarray")
        P = np.c_[sp["i"], sp["j"]].astype(int)
    if not len(P):
        return np.zeros((0, 2), int)
    i, j = P[:, 0], P[:, 1]
    keep = a["date"][i] != b["date"][j]
    i, j = i[keep], j[keep]
    if same:  # earlier first
        swap = a["date"][i] > a["date"][j]
        i, j = np.where(swap, j, i), np.where(swap, i, j)
    d = np.hypot(a["x"][i] - b["x"][j], a["y"][i] - b["y"][j])
    o = np.argsort(d, kind="stable")
    _, first = np.unique(np.c_[i[o], b["date"][j[o]].astype(np.int64)], axis=0, return_index=True)
    return np.c_[i[o][first], j[o][first]]


def _slope(dem, x, y, step=15.0):
    """The DEM's slope (rise over run) at x, y over +-step m."""
    gx = (dem(x + step, y) - dem(x - step, y)) / (2 * step)
    gy = (dem(x, y + step) - dem(x, y - step)) / (2 * step)
    return np.hypot(gx, gy)


def label_pairs(src, kind, m1, m2, idx, bias=0.0, dem=None):
    """The pairs as rows: x, y, dz (later minus earlier, less the DEM's height difference between the two
    places), first and second date, label (1 moved, 0 unchanged, -1 left out), src. src: 'is2' (ATL03 bins),
    'atl08', 'gedi', 'cross' (m1 GEDI, m2 ATL03 bins) or 'cross08' (m1 GEDI, m2 ATL08).

    Two measurements of a pair sit up to the pairing radius apart, so on a slope s they differ by up to
    radius * s with no ground moved (10 m on a 20% slope: 2 m, the moved threshold). The Copernicus DEM's
    height difference between the two places is taken off dz, and as the DEM is 30 m, older than the pairs
    and a surface over trees, half of what it took off is added to the moved threshold."""
    if not len(idx):
        return None
    i, j = idx[:, 0], idx[:, 1]
    d1, d2 = m1["date"][i], m2["date"][j]
    later = d2 > d1
    x1, y1, x2, y2 = m1["x"][i], m1["y"][i], m2["x"][j], m2["y"][j]
    corr = np.zeros(len(i)) if dem is None else np.where(later, dem(x2, y2) - dem(x1, y1), dem(x1, y1) - dem(x2, y2))
    dz = np.where(later, m2["h"][j] - m1["h"][i], m1["h"][i] - m2["h"][j]) - np.where(later, bias, -bias) - corr
    big = np.abs(dz) - 0.5 * np.abs(corr)  # dz past the correction's own doubt
    steep = np.zeros(len(i)) if dem is None else _slope(dem, (x1 + x2) / 2, (y1 + y2) / 2)
    lab = np.full(len(i), -1, np.int8)
    if src == "is2":
        bare = (m1["spread"][i] < IS2_BARE_M) & (m2["spread"][j] < IS2_BARE_M)
        flat = (np.nan_to_num(np.maximum(m1["slope"][i], m2["slope"][j]), nan=1.0) < IS2_SLOPE) & (steep < IS2_SLOPE)
        lab[bare & (big > IS2_MOVED_M)] = 1
        lab[bare & flat & (np.abs(dz) < IS2_STILL_M)] = 0
        # under canopy: the ground clusters, unchanged only (a canopy's own change is not digging)
        dg = np.where(later, m2["ground"][j] - m1["ground"][i], m1["ground"][i] - m2["ground"][j]) - corr
        lab[~bare & flat & (np.abs(dg) < IS2_STILL_CANOPY_M)] = 0
    elif src == "atl08":
        good = np.ones(len(i), bool)
        for m, k in ((m1, i), (m2, j)):
            good &= (m["nte"][k] >= A08_MIN_PHOTONS) & (np.abs(m["h"][k] - m["ref"][k]) < A08_MAX_FROM_REF)
        low = (np.nan_to_num(m1["canopy"][i], nan=0) < A08_CANOPY_M) & (np.nan_to_num(m2["canopy"][j], nan=0) < A08_CANOPY_M)
        flat = (np.nan_to_num(np.maximum(np.abs(m1["slope"][i]), np.abs(m2["slope"][j])), nan=1.0) < A08_SLOPE) & (steep < A08_SLOPE)
        lab[good & low & (steep < A08_STEEP) & (big > A08_MOVED_M)] = 1
        lab[good & flat & (np.abs(dz) < A08_STILL_M)] = 0
    else:
        g = m1  # GEDI is m1 in 'gedi', 'cross' (m2 ATL03 bins) and 'cross08' (m2 ATL08)
        low = g["rh98"][i] < GEDI_LOW_RH98
        if src == "gedi":
            low &= m2["rh98"][j] < GEDI_LOW_RH98
        if src == "cross":  # the ATL03 side as its own pairs ask: a bare bin (a spread stack is often canopy)
            low &= m2["spread"][j] < IS2_BARE_M
        if src == "cross08":  # the ATL08 side as its own pairs ask
            low &= (np.nan_to_num(m2["canopy"][j], nan=0) < A08_CANOPY_M) & (m2["nte"][j] >= A08_MIN_PHOTONS) \
                & (np.abs(m2["h"][j] - m2["ref"][j]) < A08_MAX_FROM_REF)
        if src == "cross08":  # GEDI's own pairs and GEDI against ATL03 bins called moved about as often where
            lab[low & (big > GEDI_MOVED_M)] = 1  # AlphaEarth saw no change as where it did: unchanged only
        lab[low & (steep < GEDI_SLOPE) & (np.abs(dz) < GEDI_STILL_M)] = 0
        if src == "gedi":
            deep = ~low & (g["sens"][i] >= 0.98) & (m2["sens"][j] >= 0.98) & (steep < GEDI_SLOPE)
            lab[deep & (np.abs(dz) < GEDI_STILL_CANOPY_M)] = 0
    return {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2, "dz": dz, "corr": corr, "d1": np.minimum(d1, d2),
            "d2": np.maximum(d1, d2), "y_": lab, "src": np.full(len(i), src)}


def _cat(parts):
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def site_pairs(name, kind, point, half, sources=SOURCES):
    """Every labeled pair over the site from the sources asked (rows as label_pairs, -1 included)."""
    _fresh(name, _box(point, half))
    dem = copdem(name, point, half)
    is2 = is2_measurements(name, point, half) if "is2" in sources else None
    a08 = atl08_measurements(name, point, half) if "atl08" in sources else None
    gedi = gedi_measurements(name, point, half) if "gedi" in sources else None
    parts = []
    if is2 is not None:
        parts.append(label_pairs("is2", kind, is2, is2, _pairs(is2, is2, IS2_PAIR_M, same=True), dem=dem))
    if a08 is not None:
        parts.append(label_pairs("atl08", kind, a08, a08, _pairs(a08, a08, A08_PAIR_M, same=True), dem=dem))
    if gedi is not None:
        parts.append(label_pairs("gedi", kind, gedi, gedi, _pairs(gedi, gedi, GEDI_PAIR_M, same=True), dem=dem))
    for other, src in ((is2, "cross"), (a08, "cross08")):
        if other is None or gedi is None:
            continue
        idx = _pairs(gedi, other, GEDI_PAIR_M)
        if len(idx):
            raw = other["h"][idx[:, 1]] - gedi["h"][idx[:, 0]]
            bias = float(np.median(raw[np.abs(raw - np.median(raw)) < 5]))  # the sensors' constant offset
            parts.append(label_pairs(src, kind, gedi, other, idx, bias, dem=dem))
    return _cat(parts)


def site_points(name, kind, point, half, sources=SOURCES, path=None, pairs=None, only=None):
    """The site's labeled pairs at AlphaEarth pixels, cached as data/earthwork/space-<name>.npz in the format of
    earthwork_model.site_data (one row per pixel and year pair). Where two pairs at one pixel and year pair
    disagree (one moved, one unchanged) the pixel is left out of that year pair; where they agree, the pair
    with the largest |dz| speaks for it. pairs: site_pairs' result if already at hand; only: the pair sources
    kept (all by default)."""
    path = path or f"{em.DATA}/space-{name}.npz"
    p = pairs if pairs is not None else site_pairs(name, kind, point, half, sources)
    if p is not None and only is not None:
        p = {k: v[np.isin(p["src"], only)] for k, v in p.items()}
    if p is None or not (p["y_"] >= 0).any():
        print(f"  {name}: no labeled pairs")
        return None
    keep = p["y_"] >= 0
    if kind not in TEACH_KINDS:
        keep &= p["y_"] == 0  # quiet kinds: only their unchanged ground
    p = {k: v[keep] for k, v in p.items()}
    # each pair's AlphaEarth pixel and years
    lon = point[0] + p["x"] / (111320 * np.cos(np.radians(point[1])))
    lat = point[1] + p["y"] / 110574
    col = np.floor((lon - em.AEF_X0) / em.AEF_RES).astype(int)
    row = np.floor((em.AEF_Y0 - lat) / em.AEF_RES).astype(int)
    day = lambda d: datetime.date.fromisoformat(str(d))
    yrs = np.array([em.aef_years((day(a), day(a)), (day(b), day(b))) for a, b in zip(p["d1"], p["d2"])])
    ok = yrs[:, 1] > yrs[:, 0]
    rc = (row.min(), row.max() + 1, col.min(), col.max() + 1)
    X, plain, rows_ = [], [], []
    emb, torn = {}, 0
    for yb, ya in sorted({tuple(v) for v in yrs[ok]}):
        sel = np.flatnonzero(ok & (yrs[:, 0] == yb) & (yrs[:, 1] == ya))
        key = np.c_[row[sel], col[sel]]
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        lo, hi = np.zeros(inv.max() + 1), np.zeros(inv.max() + 1)
        np.maximum.at(hi, inv, p["y_"][sel])
        lo[:] = 1
        np.minimum.at(lo, inv, p["y_"][sel])
        mixed = (hi != lo)[inv]
        torn += int((hi != lo).sum())
        sel = sel[~mixed]
        if not len(sel):
            continue
        order = sel[np.argsort(-np.abs(p["dz"][sel]))]
        _, first = np.unique(np.c_[row[order], col[order]], axis=0, return_index=True)
        sel = order[first]
        for y_ in (yb, ya):
            if y_ not in emb:
                emb[y_] = em.aef_unit(y_, rc)
        (b, okb), (a, oka) = emb[yb], emb[ya]
        r_, c_ = row[sel] - rc[0], col[sel] - rc[2]
        good = okb[r_, c_] & oka[r_, c_]
        sel, r_, c_ = sel[good], r_[good], c_[good]
        bv, av = b[:, r_, c_], a[:, r_, c_]
        X.append(em.features(bv, av).astype(np.float16))
        plain.append((1 - (av * bv).sum(0)).astype(np.float32))
        rows_.append(sel)
    if not rows_:
        print(f"  {name}: no pairs inside AlphaEarth's years")
        return None
    sel = np.concatenate(rows_)
    y = p["y_"][sel].astype(np.int8)
    by = {str(s_): [int(((p["src"][sel] == s_) & (y == 1)).sum()), int(((p["src"][sel] == s_) & (y == 0)).sum())]
          for s_ in sorted(set(p["src"][sel]))}
    out = {"X": np.concatenate(X), "y": y, "plain": np.concatenate(plain),
           "dz": p["dz"][sel].astype(np.float32), "patch": np.zeros(len(sel), np.float32), "wet": np.zeros(len(sel), bool),
           "meta": json.dumps({"name": name, "kind": kind, "source": "+".join(only or sources), "point": list(point),
                               "box": [round(v, 4) for v in _box(point, half)], "by_source": by,
                               "aef": sorted({f"{a}-{b}" for a, b in yrs[sel]}),
                               "dates": [str(p["d1"][sel].min()), str(p["d2"][sel].max())]})}
    np.savez_compressed(path, **out)
    print(f"  {name}: moved {int((y == 1).sum()):,}, unchanged {int((y == 0).sum()):,} pixels, {torn} left out where "
          f"pairs disagreed; by source (moved, unchanged) {by}")
    return out


# ---- scoring with the saved models ------------------------------------------------------------------------


def score(d, model="earthwork-lr.npz"):
    """The map's score on the site's pixels (gated as on the map): AP where both labels exist, and the share of
    unchanged ground over .5."""
    m = np.load(os.path.join(HERE, "models", model))
    z = d["X"].astype(np.float32) @ m["w"].astype(np.float32) + float(m["b"])
    s = em.gate(1 / (1 + np.exp(-z)), d["plain"])
    y = d["y"]
    ap = average_precision_score(y, s) if (y == 1).any() and (y == 0).any() else float("nan")
    still = float((s[y == 0] > 0.5).mean()) if (y == 0).any() else float("nan")
    moved = float((s[y == 1] > 0.5).mean()) if (y == 1).any() else float("nan")
    return ap, still, moved


def main():
    args = sys.argv[1:]
    choice = {"icesat2": ("is2",), "atl08": ("atl08",), "gedi": ("gedi",), "both": ("is2", "gedi"), "all": SOURCES}
    sources = choice.get(args[0], SOURCES) if args else SOURCES
    if args and args[0] in choice:
        args = args[1:]
    want = set(args)
    models = ["earthwork-lr.npz", "earthwork-lr-building-pivots.npz"]
    rows = []
    for name, kind, point, half in SITES:
        if want and name not in want:
            continue
        d = site_points(name, kind, point, half, sources)
        if d is None:
            continue
        rows.append((name, kind, int((d["y"] == 1).sum()), int((d["y"] == 0).sum()), [score(d, m) for m in models]))
    print("\n  site               kind        moved  unchanged   " + "   ".join(f"{m[13:-4] or 'current':>16s} AP / still>.5 / moved>.5" for m in models))
    for name, kind, nm, nu, sc in rows:
        print(f"  {name:18s} {kind:10s} {nm:6d} {nu:10d}   " + "   ".join(f"{ap:16.2f} / {st:9.1%} / {mv:8.1%}" for ap, st, mv in sc))


if __name__ == "__main__":
    main()
