"""Repeat DEM truth from Asia for the earthwork model: Shizuoka (Japan) and Hong Kong.

The sites here are built exactly as earthwork_model.site_data builds the 3DEP ones (two bare earth DEMs
differenced on one 2 m grid, the median removed, each AlphaEarth 10 m pixel labeled MOVED or UNCHANGED) and
cached in the same format (data/earthwork/<name>.npz), so the training script can read them beside the US sites.

Shizuoka: VIRTUAL SHIZUOKA, the prefecture's airborne laser surveys on AWS (s3://virtual-shizuoka, CC BY 4.0 and
ODbL). Each fiscal year flew different areas, a few of them twice; the Grid product is a 0.5 m ground DEM as XYZ
text per 1:500 sheet (400 x 300 m, EPSG:6676, heights on JGD2011). Flight dates come from the ground points'
GPS times where they are stored as adjusted standard time, else the fiscal year.

Hong Kong: CEDD's territory-wide airborne lidar DTMs, 0.5 m GeoTIFF tiles (EPSG:2326), flown Dec 2010 to Jan 2011
and Dec 2019 to Feb 2020 (bulkdata.csdi.gov.hk, tile indexes on the CSDI portal). The first flight predates
AlphaEarth (2017), so ground that moved between the flights may have moved before AlphaEarth's first year: these
sites teach only their UNCHANGED ground (their moved pixels are left out, label -1).

Run: uv run --with laspy --with icechunk python earthwork_sites_asia.py   (downloads only the site windows)
"""

import datetime
import io
import json
import os
import zipfile

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from scipy import ndimage

import earthwork_model as em

RAW = os.path.join(em.DATA, "asia-raw")
SHIZUOKA = "https://virtual-shizuoka.s3.ap-northeast-1.amazonaws.com"
HK_INDEX = "https://portal.csdi.gov.hk/csdi-webpage/file-api?dataset_id=cedd_rcd_1629267205233_87895&format=geojson&layer_name={}"
HK_FLIGHTS = ((datetime.date(2010, 12, 1), datetime.date(2011, 1, 8)), (datetime.date(2019, 12, 20), datetime.date(2020, 2, 2)))

# Shizuoka: the pairs of fiscal years flown over the same sheets, longest span first (a sheet goes to the first
# pair that has it), with each flight's dates (start, end): from the ground points' GPS times (adjusted standard
# time) where the files keep them, else the fiscal year's winter. Each 3 km block of 8 or more sheets is a site
SHIZUOKA_PAIRS = [
    (2021, 2025, ((2021, 7, 18), (2021, 11, 13)), ((2025, 4, 14), (2025, 6, 5))),
    (2019, 2021, ((2019, 12, 1), (2020, 3, 31)), ((2021, 7, 18), (2021, 11, 13))),
    (2022, 2025, ((2022, 10, 1), (2023, 3, 31)), ((2025, 4, 14), (2025, 6, 5))),
    (2019, 2020, ((2019, 12, 1), (2019, 12, 31)), ((2020, 10, 1), (2021, 3, 31))),
    (2021, 2022, ((2021, 7, 18), (2021, 11, 13)), ((2022, 10, 1), (2023, 3, 31))),
]
# what each site turned out to hold (looked at in the imagery once built): the sites not listed only test
SHIZUOKA_KINDS = {}

# Hong Kong: (name, kind, (lon, lat) of the window's center), each a 3 km window, quiet only
HK_SITES = [
    ("hk-tai-lam-forest", "forest", (114.035, 22.395)),
    ("hk-tai-mo-shan-forest", "forest", (114.110, 22.415)),
    ("hk-yuen-long-ponds", "ponds", (114.040, 22.480)),
    ("hk-kowloon-urban", "urban", (114.170, 22.320)),
    ("hk-lantau-south", "forest", (113.930, 22.240)),
    ("hk-fanling-farms", "farm", (114.105, 22.510)),
]

_L = "ABCDEFGHIJKLMNOPQRST"


def sheet_bounds(code):
    """EPSG:6676 bounds (west, south, east, north) of a 1:500 sheet code such as 08ME2849: a 1:50,000 sheet (two
    letters, 30 x 40 km from (300 km N, 160 km W)), its 1:5,000 sheet (3 x 4 km), its 1:500 sheet (300 x 400 m)."""
    y = 300000 - 30000 * _L.index(code[2]) - 3000 * int(code[4]) - 300 * int(code[6])
    x = -160000 + 40000 * _L.index(code[3]) + 4000 * int(code[5]) + 400 * int(code[7])
    return x, y - 300, x + 400, y


def _fetch(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get(url, timeout=300)
        if r.status_code != 200:
            return None
        with open(path + ".part", "wb") as f:
            f.write(r.content)
        os.replace(path + ".part", path)
    return path


def shizuoka_sheets(year):
    """Every 1:500 sheet with a Grid file in the fiscal year's LP survey."""
    path = os.path.join(RAW, "shizuoka", f"grid-{year}.json")
    if not os.path.exists(path):
        keys, token = [], None
        while True:
            params = {"list-type": 2, "prefix": f"{year}/LP/Grid/"} | ({"continuation-token": token} if token else {})
            r = requests.get(SHIZUOKA, params=params, timeout=120).text
            keys += [k.split("<")[0] for k in r.split("<Key>")[1:]]
            if "<NextContinuationToken>" not in r:
                break
            token = r.split("<NextContinuationToken>")[1].split("<")[0]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(sorted(keys), f)
    with open(path) as f:
        return {k.rsplit("/", 1)[-1].split(".")[0]: k for k in json.load(f)}


def shizuoka_sheet_dem(key):
    """One sheet's 0.5 m ground DEM averaged to 2 m (150 x 200, north up), NaN where it has no data."""
    path = _fetch(f"{SHIZUOKA}/{key}", os.path.join(RAW, "shizuoka", key))
    if path is None:
        return None
    z = zipfile.ZipFile(path)
    name = next(n for n in z.namelist() if n.lower().endswith((".txt", ".csv")))
    raw = z.read(name)
    # x y z (2019 to 2022) or index,x,y,z,flag (2025)
    cols = len(raw[:raw.index(b"\n")].replace(b",", b" ").split())
    v = np.array(raw.replace(b",", b" ").split(), dtype=np.float64)
    xyz = v[: len(v) // cols * cols].reshape(-1, cols)[:, (1, 2, 3) if cols == 5 else (0, 1, 2)]
    W, S, E, N = sheet_bounds(key.rsplit("/", 1)[-1].split(".")[0])
    a = np.full((600, 800), np.nan, np.float32)
    c = np.floor((xyz[:, 0] - W) / 0.5).astype(int)
    r = np.floor((N - xyz[:, 1]) / 0.5).astype(int)
    ok = (c >= 0) & (c < 800) & (r >= 0) & (r < 600) & (xyz[:, 2] > -100)
    a[r[ok], c[ok]] = xyz[ok, 2]
    with np.errstate(invalid="ignore"):
        return np.nanmean(a.reshape(150, 4, 200, 4), axis=(1, 3))


def sheet_blocks(sheets, size=3000, least=8):
    """The sheets binned into size x size m blocks, each block of least sheets or more: the years mostly overlap
    in strips along the edges of their survey areas, cut here into sites of about the US sites' size."""
    blocks = {}
    for sh in sheets:
        x0, y0, _, _ = sheet_bounds(sh)
        blocks.setdefault((int(x0 // size), int(y0 // size)), []).append(sh)
    return [sorted(g) for _, g in sorted(blocks.items()) if len(g) >= least]


def shizuoka_mosaic(sheets, year_keys):
    """Both years' 2 m DEMs over the sheets, and where the sheets are: (a, b, transform, cover)."""
    bs = np.array([sheet_bounds(s) for s in sheets])
    W, S, E, N = bs[:, 0].min(), bs[:, 1].min(), bs[:, 2].max(), bs[:, 3].max()
    w, h = int((E - W) / 2), int((N - S) / 2)
    out, cover = [], np.zeros((h, w), bool)
    for s in sheets:
        x0, _, _, y1 = sheet_bounds(s)
        r0, c0 = int((N - y1) / 2), int((x0 - W) / 2)
        cover[r0:r0 + 150, c0:c0 + 200] = True
    for keys in year_keys:
        m = np.full((h, w), np.nan, np.float32)
        for s in sheets:
            d = shizuoka_sheet_dem(keys[s])
            if d is None:
                continue
            x0, _, _, y1 = sheet_bounds(s)
            r0, c0 = int((N - y1) / 2), int((x0 - W) / 2)
            m[r0:r0 + 150, c0:c0 + 200] = d
        out.append(m)
    return out[0], out[1], from_origin(W, N, 2, 2), cover


def hk_tiles(year):
    path = os.path.join(RAW, "hk", f"index-{year}.json")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(requests.get(HK_INDEX.format(f"DTM_{year}"), timeout=120).text)
    with open(path) as f:
        return json.load(f)["features"]


def hk_mosaic(point, half=1500):
    """Both flights' 2 m DTMs over a window around point: (a, b, transform)."""
    from pyproj import Transformer
    x, y = Transformer.from_crs(4326, 2326, always_xy=True).transform(*point)
    W, S, E, N = (np.floor((x - half) / 2) * 2, np.floor((y - half) / 2) * 2, np.ceil((x + half) / 2) * 2, np.ceil((y + half) / 2) * 2)
    lo = transform_bounds(2326, 4326, W, S, E, N)
    out = []
    for year in (2010, 2020):
        srcs = []
        for f in hk_tiles(year):
            ring = np.array(f["geometry"]["coordinates"][0])[:, :2]
            if ring[:, 0].max() < lo[0] or ring[:, 0].min() > lo[2] or ring[:, 1].max() < lo[1] or ring[:, 1].min() > lo[3]:
                continue
            url = f["properties"]["URL"]
            p = _fetch(url, os.path.join(RAW, "hk", str(year), url.rsplit("/", 1)[-1]))
            if p:
                srcs.append(rasterio.open(f"zip://{p}!/" + zipfile.ZipFile(p).namelist()[0]))
        a, _ = merge(srcs, bounds=(W, S, E, N), res=2, resampling=Resampling.average, nodata=-9999)
        for s in srcs:
            s.close()
        a = a[0].astype(np.float32)
        a[a < -1000] = np.nan
        out.append(a)
    return out[0], out[1], from_origin(W, N, 2, 2)


def change(a, b, cover=None):
    """Second minus first with the median removed, and the water in both flights, as em.dem_change does it. The
    overlap share is over cover (the sheets flown, where the grid is wider than them), else the whole grid."""
    dz = b - a
    share = float(np.isfinite(dz)[cover].mean() if cover is not None else np.isfinite(dz).mean())
    dz -= np.nanmedian(dz)

    def flat(z):
        ok = np.isfinite(z)
        f = np.where(ok, z.astype(np.float64) - np.nanmedian(z), 0.0)
        m = ndimage.uniform_filter(f, 5)
        full = ndimage.uniform_filter(ok.astype(np.float64), 5) > 0.999
        return ok & full & (np.sqrt(np.maximum(ndimage.uniform_filter(f * f, 5) - m * m, 0)) < 0.001)

    fa, fb = flat(a), flat(b)
    water = fa & fb & np.isfinite(dz)
    with np.errstate(invalid="ignore"):
        dz[water | (fa & ~fb & (dz < 0)) | (fb & ~fa & (dz > 0))] = np.nan
    return dz, share, water


def build(name, kind, a, b, tr, crs, d1, d2, quiet_only, source, lat, cover=None):
    """A site's cache in site_data's format, from its two 2 m DEMs."""
    dz, share, water = change(a, b, cover)
    if share < em.MIN_OVERLAP:
        print(f"  {name}: skipped (the flights overlap on {share:.0%} of the window)")
        return None
    mean, moved, wet, rc = em.to_aef_grid(dz, tr, crs, water)
    yb, ya = em.aef_years(d1, d2)
    if ya <= yb:
        print(f"  {name}: skipped (no AlphaEarth year between the flights)")
        return None
    bv, okb = em.aef_unit(yb, rc)
    av, oka = em.aef_unit(ya, rc)
    ok = okb & oka & np.isfinite(mean)
    lab = np.full(mean.shape, -1, np.int8)
    lab[ok & (np.abs(mean) > em.MOVED_M)] = 1
    lab[ok & (np.abs(mean) < em.STILL_M) & (moved < 0.05)] = 0
    lab[okb & oka & (wet > 0.5) & (lab != 1)] = 0
    n_moved = int((lab == 1).sum())
    if quiet_only:
        lab[lab == 1] = -1
    px_ha = (em.AEF_RES * 111320 * np.cos(np.radians(lat))) * (em.AEF_RES * 110574) / 1e4
    cc, _ = ndimage.label(lab == 1, structure=np.ones((3, 3)))
    patch = (np.bincount(cc.ravel())[cc] * px_ha).astype(np.float32)
    out = {"X": em.features(bv, av).astype(np.float16), "y": lab.ravel(), "plain": (1 - (av * bv).sum(0)).ravel().astype(np.float32),
           "dz": mean.ravel(), "patch": patch.ravel(), "wet": (wet > 0.5).ravel(),
           "meta": json.dumps({"name": name, "kind": kind, "flights": [f"{d1[0]}..{d1[1]}", f"{d2[0]}..{d2[1]}"], "aef": [int(yb), int(ya)],
                               "overlap": round(share, 3), "tiles": [], "rc": [int(v) for v in rc], "source": source,
                               "quiet_only": quiet_only, "moved_seen": n_moved, "shape": list(mean.shape)})}
    np.savez_compressed(os.path.join(em.DATA, f"{name}.npz"), **out)
    print(f"  {name}: AlphaEarth {yb} vs {ya}, overlap {share:.0%}, moved {n_moved:,} px{' (left out: quiet only)' if quiet_only else ''}, "
          f"unchanged {int((lab == 0).sum()):,} (water {int((wet > 0.5).sum()):,})")
    return out


def main():
    os.makedirs(em.DATA, exist_ok=True)
    used = set()
    to_ll = __import__("pyproj").Transformer.from_crs(6676, 4326, always_xy=True)
    for y1, y2, f1, f2 in SHIZUOKA_PAIRS:
        ka, kb = shizuoka_sheets(y1), shizuoka_sheets(y2)
        for g in sheet_blocks(sorted(set(ka) & set(kb) - used)):
            used |= set(g)
            bs = np.array([sheet_bounds(s) for s in g])
            lon, lat = to_ll.transform((bs[:, 0].min() + bs[:, 2].max()) / 2, (bs[:, 1].min() + bs[:, 3].max()) / 2)
            name = f"shizuoka-{y1}-{y2}-{lat:.3f}n{lon:.3f}e".replace(".", "")
            if os.path.exists(os.path.join(em.DATA, f"{name}.npz")):
                print(f"  {name}: cached")
                continue
            print(f"  {name}: {len(g)} sheets", flush=True)
            a, b, tr, cover = shizuoka_mosaic(g, (ka, kb))
            d1 = tuple(datetime.date(*v) for v in f1)
            d2 = tuple(datetime.date(*v) for v in f2)
            build(name, SHIZUOKA_KINDS.get(name, "survey"), a, b, tr, rasterio.crs.CRS.from_epsg(6676), d1, d2, False,
                  f"VIRTUAL SHIZUOKA {y1} / {y2}, {len(g)} sheets", lat, cover)
    for name, kind, point in HK_SITES:
        if os.path.exists(os.path.join(em.DATA, f"{name}.npz")):
            print(f"  {name}: cached")
            continue
        a, b, tr = hk_mosaic(point)
        build(name, kind, a, b, tr, rasterio.crs.CRS.from_epsg(2326), *HK_FLIGHTS, True, "CEDD LiDAR 2010 / 2020", point[1])


if __name__ == "__main__":
    main()
