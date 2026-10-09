"""The earthwork model: 3DEP repeat lidar teaches AlphaEarth what moved ground looks like.

Where 3DEP has flown the same ground twice since AlphaEarth began (2017), the difference between the two
1 m DEMs is the truth of where earth moved: dug, filled, graded, mined. Per AlphaEarth 10 m pixel, the
mean elevation change between the flights labels it MOVED (more than 0.5 m) or UNCHANGED (under 0.1 m, and
almost none of its 2 m cells moved 0.5 m); anything between is left out. The features are AlphaEarth's
year before the first flight and its year of the second, as unit vectors b and a: [b, a, a * b, (a - b)^2],
256 numbers. A logistic regression learns MOVED from them. Plain AlphaEarth change (1 - cos(b, a)) is the
baseline it has to beat.

Each digging site is scored by a model taught on all the OTHER digging sites (leave one site out), so every
number is on ground the model never saw. Then detection is broken down by how deep the change was and how big the patch
of moved ground was. The final model, taught on every digging site, is saved to models/earthwork-lr.npz.

Sites are found by name in the TNM 1 m tile list, with the tile's flight dates from ScienceBase (or the 3DEP
index); the two tiles are read on one 2 m grid over where they meet, and a site whose two flights cover less
than 15% of it is skipped. The model is for digging by people: building sites (site grading) and mines
teach it. Burn sites teach it only what is NOT digging (their unchanged ground; a burn scar looks like
digging to a model shown only building sites). Storm sites only test it. (Burns and storms once taught both
ways, which made the model busier on bare ground and a little worse at digging.) DEM differences and AlphaEarth features are cached
in data/earthwork/.

Run: uv run python earthwork_model.py   (a few minutes the first time)
"""

import datetime
import json
import os
import time

import numpy as np
import rasterio
import requests
import zarr
from obstore.store import S3Store
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from zarr.storage import ObjectStore

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "earthwork")
MODEL = os.path.join(HERE, "models", "earthwork-lr.npz")
AEF_RES, AEF_Y0, AEF_X0, AEF_NODATA = 8.983111749910169e-05, 83.68570533713473, -180.0, -128
INDEX = "https://index.nationalmap.gov/arcgis/rest/services/3DEPElevationIndex/MapServer/8/query"
TNM = "https://tnmaccess.nationalmap.gov/api/v1/products"
MOVED_M, STILL_M = 0.5, 0.1
MIN_OVERLAP = 0.15  # share of the shared grid both flights cover
# what the model is for: digging by people (site grading, pits, quarries). Only these sites teach it; the
# rest (burns, storm coasts) test whether it stays quiet where the surface changed but nobody dug
TEACH_KINDS = ("building", "mining")
# ground nobody dug that looks a lot like digging to AlphaEarth (a burn scar): taught only as NOT digging, its
# unchanged pixels and none of its moved ones, so the model learns that look is not earthwork
QUIET_KINDS = ("burn",)

# (name, kind, a point inside the site, first project, second project); the 1 m tile holding the point is
# used. Kinds: building (site grading) and mining (pits, quarries) teach; storm (coastal) and burn only test
# (surface change over ground nobody dug: false alarms to watch). Tried and left out, the two flights
# sharing too little of the tile: Bingham Canyon UT (2018 has the pit, Salt Lake County 2023 only the valley),
# Sierrita AZ, Morenci AZ, Las Vegas NV, Paradise CA, SW Washington clear-cuts, and second tiles at Columbus
# NE and Aurora. Repeat 1 m lidar over working mines is rare: a 3 km window at 33 US mines and quarries found
# two flights both with data only at Mountain Pass and Four Corners (and part of a Hibbing tile)
SITES = [
    ("new-albany-oh", "building", (-82.80, 40.05), "OH_Columbus_2019", "OH_Statewide_Phase3_2021"),
    ("katy-tx", "building", (-95.80, 29.79), "TX_CoastalRegion_2018", "TX_Houston_B24"),
    ("columbus-ne-oh", "building", (-82.70, 40.05), "OH_Columbus_2019", "OH_Statewide_Phase2_2020"),
    ("aurora-co", "building", (-104.70, 39.79), "CO_EasternColorado_2018", "CO_DRCOG_2020"),
    ("brighton-co", "building", (-104.707, 40.065), "CO_EasternColorado_2018", "CO_DRCOG_2020"),
    ("lakewood-ranch-fl", "building", (-82.40, 27.42), "FL_Peninsular_FDEM_2018", "FL_ManateeCounty_B25"),
    ("huntsville-al", "building", (-86.80, 34.69), "AL_NorthAL_2019", "AL_11County_B23"),
    ("fishers-in", "building", (-85.95, 39.98), "IN_Central_Hamilton_2017", "IN_HamiltonCounty_A25"),
    ("mountain-pass-ca", "mining", (-115.53, 35.48), "CA_MountainPass_2019", "CA_FEMAR9Southeast_D24"),
    ("four-corners-fl", "mining", (-82.10, 27.65), "FL_Peninsular_FDEM_2018", "FL_ManateeCounty_B25"),
    ("hibbing-mn", "mining", (-93.066, 47.448), "MN_LakeSuperior_2021", "MN_UpperMissRiver_B22"),
    ("mexico-beach-fl", "storm", (-85.394, 29.956), "FL_Lower_Choctawhatchee_2017", "FL_HurricaneMichael_2020"),
    ("cameron-peak-co", "burn", (-105.650, 40.604), "CO_DRCOG_2020", "CO_ArapahoRooseveltPikeNF_D23"),
    ("grizzly-flats-ca", "burn", (-120.416, 38.595), "CA_UpperSouthAmerican_Eldorado_2019", "CA_SierraNevada_B22"),
]


def _get(url, params, tries=4):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=90)
            return r.json()
        except Exception:
            time.sleep(2 + 3 * k)
    return {}


_norm = lambda t: "".join(ch for ch in t.lower() if ch.isalnum())


def tile_for(lon, lat, project):
    """The 1 m DEM tile holding the point from the project (matched loosely on its title or path): (url, bbox,
    ScienceBase item)."""
    for _ in range(3):
        items = _get(TNM, {"datasets": "Digital Elevation Model (DEM) 1 meter", "bbox": f"{lon},{lat},{lon},{lat}",
                           "outputFormat": "JSON"}).get("items")
        if items is not None:
            break
        time.sleep(5)
    key = _norm(project)
    for i in items or []:
        if key in _norm(i["downloadURL"]) or key in _norm(i["title"]):
            b = i["boundingBox"]
            return i["downloadURL"], (b["minX"], b["minY"], b["maxX"], b["maxY"]), i.get("metaUrl")
    return None


def tile_dates(meta_url):
    """(start, end) dates the tile was flown, from its ScienceBase item."""
    if not meta_url:
        return None
    d = {x.get("type"): x.get("dateString") for x in _get(meta_url, {"format": "json"}).get("dates", [])}
    if not (d.get("Start") and d.get("End")):
        return None
    return datetime.date.fromisoformat(d["Start"][:10]), datetime.date.fromisoformat(d["End"][:10])


def flight_dates(bbox, project):
    """(start, end) dates of the 3DEP lidar project (matched loosely on its name) over the tile, from the 3DEP
    index; for when the tile's own dates are missing."""
    d = _get(INDEX, dict(geometry=",".join(map(str, bbox)), geometryType="esriGeometryEnvelope", inSR=4326,
                         spatialRel="esriSpatialRelIntersects", outFields="project,collect_start,collect_end",
                         returnGeometry="false", f="json"))
    key = _norm(project)
    for f in d.get("features", []):
        a = f["attributes"]
        if _norm(a["project"]).startswith(key) or key.startswith(_norm(a["project"])):
            day = lambda ms: datetime.date.fromtimestamp(ms / 1000)
            return day(a["collect_start"]), day(a["collect_end"])
    return None


def aef_years(first, second):
    """AlphaEarth's year before the first flight (the flight's own year if it began in the second half) and
    its year at the second (the year before, if the flight ended in the first half), inside 2017-2025."""
    b = first[0].year if first[0].month >= 7 else first[0].year - 1
    a = second[1].year if second[1].month >= 7 else second[1].year - 1
    return max(2017, b), min(2025, a)


def dem_change(url_a, url_b):
    """Second DEM minus first, read at 2 m on one grid over where the two tiles meet (projects cut their tiles
    a little differently), the median removed (a datum or geoid offset between flights): (dz, transform, crs,
    overlap share)."""
    with rasterio.open("/vsicurl/" + url_a) as sa, rasterio.open("/vsicurl/" + url_b) as sb:
        if sa.crs != sb.crs:
            return None
        crs = sa.crs
        W, S = (np.ceil(max(u, v) / 2) * 2 for u, v in ((sa.bounds.left, sb.bounds.left), (sa.bounds.bottom, sb.bounds.bottom)))
        E, N = (np.floor(min(u, v) / 2) * 2 for u, v in ((sa.bounds.right, sb.bounds.right), (sa.bounds.top, sb.bounds.top)))
        w, h = int((E - W) / 2), int((N - S) / 2)
        if w < 500 or h < 500:
            return None
        ta = from_origin(W, N, 2, 2)
        def read(s):
            a = s.read(1, window=from_bounds(W, S, E, N, s.transform), out_shape=(h, w), resampling=Resampling.average).astype(np.float32)
            a[(a == s.nodata) | (a < -1000)] = np.nan
            return a
        a, b = read(sa), read(sb)
    dz = b - a
    share = float(np.isfinite(dz).mean())
    dz -= np.nanmedian(dz)
    # standing water: lidar DEMs flatten it to its level on the day, so a pond higher or lower between flights
    # reads as fill or a cut. Ground flat in BOTH DEMs (5 x 5 spread under 2 cm) is water both times: left out.
    # A pond dug between the flights is flat only in the second, and stays in (it is digging)
    def flat(z):
        f = np.nan_to_num(z, nan=-9999.0)
        m = ndimage.uniform_filter(f, 5)
        return np.sqrt(np.maximum(ndimage.uniform_filter(f * f, 5) - m * m, 0)) < 0.02
    dz[flat(a) & flat(b)] = np.nan
    return dz, ta, crs, share


def to_aef_grid(dz, tr, crs):
    """Each AlphaEarth 10 m pixel over the tile: mean dz and the share of its 2 m cells that moved > 0.5 m
    (5 x 5 samples a pixel), with the mosaic's row and column range."""
    h, w = dz.shape
    inv = Transformer.from_crs(crs, 4326, always_xy=True)
    c = [inv.transform(*(tr * (x, y))) for x, y in ((0, 0), (w, 0), (0, h), (w, h))]
    W, E = max(c[0][0], c[2][0]), min(c[1][0], c[3][0])
    N, S = min(c[0][1], c[1][1]), max(c[2][1], c[3][1])
    x0, x1 = int(np.ceil((W - AEF_X0) / AEF_RES)), int((E - AEF_X0) / AEF_RES)
    y0, y1 = int(np.ceil((AEF_Y0 - N) / AEF_RES)), int((AEF_Y0 - S) / AEF_RES)
    sub = (np.arange(5) + 0.5) / 5
    cols = x0 + (np.arange(x1 - x0)[:, None] + sub[None]).ravel()
    rows = y0 + (np.arange(y1 - y0)[:, None] + sub[None]).ravel()
    LON, LAT = np.meshgrid(AEF_X0 + cols * AEF_RES, AEF_Y0 - rows * AEF_RES)
    X, Y = Transformer.from_crs(4326, crs, always_xy=True).transform(LON, LAT)
    cc, rr = (~tr) * (X, Y)
    v = ndimage.map_coordinates(dz, [rr - 0.5, cc - 0.5], order=0, mode="constant", cval=np.nan)
    v = v.reshape(y1 - y0, 5, x1 - x0, 5)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(v, axis=(1, 3))
        moved = np.nanmean(np.abs(v) > MOVED_M, axis=(1, 3))
    return mean.astype(np.float32), moved.astype(np.float32), (y0, y1, x0, x1)


_emb = {}


def aef_unit(year, rc):
    """AlphaEarth unit embeddings (64, h, w) for the year over the mosaic rows and columns, and the valid mask."""
    if "arr" not in _emb:
        store = S3Store("tge-labs", endpoint="https://data.source.coop", region="us-west-2",
                        virtual_hosted_style_request=False, skip_signature=True, prefix="aef-mosaic")
        g = zarr.open_group(ObjectStore(store, read_only=True), mode="r")
        _emb["times"], _emb["arr"] = g["time"][:], g["embeddings"]
    y0, y1, x0, x1 = rc
    e = np.asarray(_emb["arr"][int(np.where(_emb["times"] == year)[0][0]), :, y0:y1, x0:x1])
    v = e.astype(np.float32)
    v = np.sign(v) * (v / 127.5) ** 2
    return v / np.maximum(np.linalg.norm(v, axis=0), 1e-9), e[0] != AEF_NODATA


def features(b, a):
    """[b, a, a * b, (a - b)^2] per pixel: (n, 256)."""
    return np.concatenate([b, a, a * b, (a - b) ** 2], 0).reshape(256, -1).T


def site_data(name, kind, point, p1, p2):
    """Features, labels, plain change, mean dz and patch size (ha) of moved ground for one site, cached."""
    path = f"{DATA}/{name}.npz"
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    lon, lat = point
    t1, t2 = tile_for(lon, lat, p1), tile_for(lon, lat, p2)
    if not (t1 and t2):
        print(f"  {name}: skipped (no 1 m tile at the point for {'both' if not (t1 or t2) else p1 if not t1 else p2})")
        return None
    (u1, bb, m1), (u2, _, m2) = t1, t2
    d1, d2 = tile_dates(m1) or flight_dates(bb, p1), tile_dates(m2) or flight_dates(bb, p2)
    if not (d1 and d2):
        print(f"  {name}: skipped (no flight dates for {p1 if not d1 else p2})")
        return None
    got = dem_change(u1, u2)
    if got is None or got[3] < MIN_OVERLAP:
        print(f"  {name}: skipped (the flights overlap on {0 if got is None else got[3]:.0%} of the tile)")
        return None
    dz, tr, crs, share = got
    mean, moved, rc = to_aef_grid(dz, tr, crs)
    yb, ya = aef_years(d1, d2)
    if ya <= yb:
        print(f"  {name}: skipped (no AlphaEarth year between the flights {d1[0]}..{d1[1]} and {d2[0]}..{d2[1]})")
        return None
    b, okb = aef_unit(yb, rc)
    a, oka = aef_unit(ya, rc)
    ok = okb & oka & np.isfinite(mean)
    lab = np.full(mean.shape, -1, np.int8)
    lab[ok & (np.abs(mean) > MOVED_M)] = 1
    lab[ok & (np.abs(mean) < STILL_M) & (moved < 0.05)] = 0
    # each moved pixel's patch: connected moved pixels, in hectares
    px_ha = (AEF_RES * 111320 * np.cos(np.radians(lat))) * (AEF_RES * 110574) / 1e4
    cc, _ = ndimage.label(lab == 1, structure=np.ones((3, 3)))
    patch = (np.bincount(cc.ravel())[cc] * px_ha).astype(np.float32)
    out = {"X": features(b, a).astype(np.float16), "y": lab.ravel(), "plain": (1 - (a * b).sum(0)).ravel().astype(np.float32),
           "dz": mean.ravel(), "patch": patch.ravel(), "meta": json.dumps({"name": name, "kind": kind, "flights": [f"{d1[0]}..{d1[1]}", f"{d2[0]}..{d2[1]}"],
           "aef": [int(yb), int(ya)], "overlap": round(share, 3), "tiles": [u1, u2], "shape": list(mean.shape)})}
    np.savez_compressed(path, **out)
    return out


def sample(d, rng, n_pos=20000, n_neg=200000):
    """A site's share of a training set: up to n_pos moved and n_neg unchanged pixels, so no one site (Katy,
    a quarter of it moved) outweighs the rest."""
    pos, neg = np.flatnonzero(d["y"] == 1), np.flatnonzero(d["y"] == 0)
    idx = np.concatenate([rng.choice(pos, min(len(pos), n_pos), replace=False), rng.choice(neg, min(len(neg), n_neg), replace=False)])
    return d["X"][idx].astype(np.float32), d["y"][idx]


def fit(parts):
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    return LogisticRegression(C=0.3, max_iter=4000, class_weight="balanced").fit(X, y)


def main():
    t = time.time()
    os.makedirs(DATA, exist_ok=True)
    sites = {}
    for name, kind, point, p1, p2 in SITES:
        d = site_data(name, kind, point, p1, p2)
        if d is not None:
            m = json.loads(str(d["meta"]))
            lab = d["y"]
            print(f"  {name}: flights {m['flights'][0]} / {m['flights'][1]}, AlphaEarth {m['aef'][0]} vs {m['aef'][1]}, "
                  f"overlap {m['overlap']:.0%}, moved {np.sum(lab == 1):,} px, unchanged {np.sum(lab == 0):,}  ({time.time() - t:.0f} s)")
            d["kind"] = kind
            sites[name] = d
    rng = np.random.default_rng(0)
    teach = [n for n, d in sites.items() if d["kind"] in TEACH_KINDS]
    quiet = [n for n, d in sites.items() if d["kind"] in QUIET_KINDS]
    samples = {n: sample(sites[n], rng) for n in teach} | {n: sample(sites[n], rng, n_pos=0) for n in quiet}
    taught = teach + quiet
    final = fit([samples[n] for n in taught])
    print(f"\nTaught on digging ({', '.join(TEACH_KINDS)}), and on {', '.join(QUIET_KINDS)} as not digging: each of those sites")
    print("scored by a model taught on the others; every other site (test only) by the model taught on all of them.")
    print("plain = AlphaEarth change, 1 - cos.")
    print("Quiet ground: the share of unchanged pixels scoring over 0.5 and over 0.8 (lower is quieter).")
    print(f"  {'site':20s} {'kind':9s} {'moved':>6s}   AP taught/plain   top 5% moved taught/plain   "
          f"moved > .5 / > .8   unchanged > .5 / > .8")
    held = {}
    for name, d in sites.items():
        model = fit([samples[n] for n in taught if n != name]) if name in taught else final
        m = d["y"] >= 0
        yt = d["y"][m]
        p = model.predict_proba(d["X"][m].astype(np.float32))[:, 1]
        q = d["plain"][m]
        k = max(1, int(0.05 * len(yt)))
        top = lambda s: yt[np.argsort(-s)[:k]].mean()
        ap = lambda s: average_precision_score(yt, s) if 0 < yt.sum() < len(yt) else np.nan
        over = lambda c, v: (p[yt == c] > v).mean() if (yt == c).any() else np.nan
        print(f"  {name:20s} {d['kind']:9s} {yt.mean():6.1%}   {ap(p):.2f}/{ap(q):.2f}          {top(p):5.1%}/{top(q):5.1%}"
              f"               {over(1, .5):5.1%} / {over(1, .8):5.1%}     {over(0, .5):5.1%} / {over(0, .8):5.1%}"
              + ("" if name in teach else "   (not digging)" if name in quiet else "   (test only)"))
        if name in teach:
            held[name] = (p, q, yt, d["dz"][m], d["patch"][m])
    # is it only big digs? recall of moved pixels by depth and by patch size, at a cutoff that flags as many
    # pixels as truly moved at each site (so taught and plain flag the same number); every site, then mines
    for group, names in (("every digging site", list(held)), ("mining sites", [n for n in held if sites[n]["kind"] == "mining"])):
        if not names:
            continue
        print(f"\nShare of moved pixels found, by how deep and how big (held out, {group} pooled)")
        rows = {"taught": [], "plain": []}
        for n_ in names:
            p, q, yt, dz, patch = held[n_]
            n = int(yt.sum())
            for key, s in (("taught", p), ("plain", q)):
                cut = np.sort(s)[-n] if n else np.inf
                rows[key].append((s >= cut)[yt == 1])
        dzs = np.concatenate([held[n_][3][held[n_][2] == 1] for n_ in names])
        pas = np.concatenate([held[n_][4][held[n_][2] == 1] for n_ in names])
        hit = {k: np.concatenate(v) for k, v in rows.items()}
        for label, vals, bins in (("depth |dz| m", np.abs(dzs), [0.5, 1, 2, 5, 20, 1e9]), ("patch ha", pas, [0, 0.1, 0.5, 2, 10, 1e9])):
            print(f"  {label}")
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (vals >= lo) & (vals < hi)
                if m.sum() < 50:
                    continue
                span = f"{lo:g} to {hi:g}" if hi < 1e8 else f"over {lo:g}"
                print(f"    {span:12s} {m.sum():9,d} px   taught {hit['taught'][m].mean():5.1%}   plain {hit['plain'][m].mean():5.1%}")
    # the final model, taught on every digging site
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    np.savez(MODEL, w=final.coef_[0].astype(np.float32), b=np.float32(final.intercept_[0]),
             features=np.array("[b, a, a*b, (a-b)^2] of AlphaEarth unit embeddings: b the year before the first flight, a the year after"),
             sites=np.array(teach), not_digging=np.array(quiet), tested=np.array([n for n in sites if n not in taught]), moved_m=MOVED_M, still_m=STILL_M)
    print(f"\nsaved {os.path.relpath(MODEL, HERE)} (taught on {len(teach)} digging sites and {len(quiet)} not), {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
