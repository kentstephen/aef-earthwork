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
teach it. Burn sites and a coastal marsh teach it only what is NOT digging (their unchanged ground; a burn
scar or tidal water looks like digging to a model shown only building sites), as does water lying in both
flights at every site. Storm sites and the other marshes only test it. (Burns and storms once taught both
ways, which made the model busier on bare ground and a little worse at digging.) DEM differences and AlphaEarth features are cached
in data/earthwork/.

Crop changes are not digging either: a field green one year and bare the next is a large AlphaEarth change,
and the model, shown no farmland, lit fields up wherever crops rotate. Two sources teach it they are quiet:
farmland flown twice by 3DEP (kind "farm", taught like the burns, only its unchanged ground), and USDA's
Cropland Data Layer, whose pixels that grew a field crop every year of a pair but a different one at each end
are taught as not digging in several US farm regions (CDL_TEACH) and tested in others (CDL_TEST). With
EARTHWORK_CROP=0 the script leaves both out (the farm sites then only test) and saves the model as before;
otherwise it saves models/earthwork-lr-crop.npz.

Variants, each saved to models/earthwork-lr-<EARTHWORK_VARIANT>.npz when that is set:
  EARTHWORK_PIVOTS=1    more center-pivot country: CDL_PIVOTS taught too, and PIVOT_SITES (farm lidar) added
  EARTHWORK_MIDDLE=1    the year between: three more features, from AlphaEarth's middle year m of the pair,
                        (1 - cos(b, m)), (1 - cos(m, a)) and the out-and-back (1 - cos(b, m)) + (1 - cos(m, a))
                        - (1 - cos(b, a)): crops swing and come back, digging moves once (the map then reads a
                        third year per frame)
  EARTHWORK_TEACH=building    only building sites teach digging (mining then only tests)
  EARTHWORK_CDL=0       crop quiet from the farm lidar alone: no CDL switches taught (still tested)

On the map (earthwork.py) the score is also gated by plain change: multiplied by clip((1 - cos(b, a) - 0.05)
/ 0.10, 0, 1). The model reads each year's look as well as the change, and ground bare in both years (a
graded pad here) reads as dug: open desert abroad, which barely changes, lit up wholesale without it. Held
out, the gate costs no digging and quiets the desert (Cairo 50% to 4%, Sahara 51% to 2%).

Run: uv run --with icechunk python earthwork_model.py   (a few minutes the first time; the CDL is an Icechunk store)
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
CROP = os.environ.get("EARTHWORK_CROP", "1") != "0"
CROP_MODEL = os.path.join(HERE, "models", "earthwork-lr-crop.npz")
VARIANT = os.environ.get("EARTHWORK_VARIANT", "")
PIVOTS = os.environ.get("EARTHWORK_PIVOTS", "0") == "1"
MIDDLE = os.environ.get("EARTHWORK_MIDDLE", "0") == "1"
CDL_TAUGHT = CROP and os.environ.get("EARTHWORK_CDL", "1") != "0"
AEF_RES, AEF_Y0, AEF_X0, AEF_NODATA = 8.983111749910169e-05, 83.68570533713473, -180.0, -128
INDEX = "https://index.nationalmap.gov/arcgis/rest/services/3DEPElevationIndex/MapServer/8/query"
TNM = "https://tnmaccess.nationalmap.gov/api/v1/products"
MOVED_M, STILL_M = 0.5, 0.1
MIN_OVERLAP = 0.15  # share of the shared grid both flights cover
# what the model is for: digging by people (site grading, pits, quarries). Only these sites teach it; the
# rest (burns, storm coasts) test whether it stays quiet where the surface changed but nobody dug
TEACH_KINDS = tuple(os.environ.get("EARTHWORK_TEACH", "building,mining").split(","))
# ground nobody dug that looks a lot like digging to AlphaEarth (a burn scar, bare desert sand): taught only as
# NOT digging (dunes do move, so only their unchanged ground), its
# unchanged pixels and none of its moved ones, so the model learns that look is not earthwork
QUIET_KINDS = ("burn", "dunes") + (("farm",) if CROP else ())
# and one coastal marsh, mostly open brackish water (inland ponds did not teach it tidal water); the other
# marshes only test whether that carries
QUIET_SITES = ("brazoria-tx",)

# (name, kind, a point inside the site, first project, second project); the 1 m tile holding the point is
# used. Kinds: building (site grading) and mining (pits, quarries) teach; burn (and one marsh, QUIET_SITES)
# teach only as not digging; storm (coastal) and the other marshes only test (surface change over ground nobody dug: false alarms to watch; marsh
# lidar is the least sure, the ground under reeds and shallow water partly the vegetation). Tried and left out, the two flights
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
    ("las-vegas-se-nv", "building", (-115.05, 36.00), "NV_Las_Vegas_Region_2016", "NV_ClarkCounty_B22"),
    ("las-vegas-nw-nv", "building", (-115.33, 36.28), "NV_Las_Vegas_Region_2016", "NV_ClarkCounty_B22"),
    ("lancaster-ca", "building", (-118.20, 34.75), "CA_LosAngeles_2016", "CA_LosAngeles_B23"),
    ("mountain-pass-ca", "mining", (-115.53, 35.48), "CA_MountainPass_2019", "CA_FEMAR9Southeast_D24"),
    ("four-corners-fl", "mining", (-82.10, 27.65), "FL_Peninsular_FDEM_2018", "FL_ManateeCounty_B25"),
    ("hibbing-mn", "mining", (-93.066, 47.448), "MN_LakeSuperior_2021", "MN_UpperMissRiver_B22"),
    ("mexico-beach-fl", "storm", (-85.394, 29.956), "FL_Lower_Choctawhatchee_2017", "FL_HurricaneMichael_2020"),
    ("white-sands-nm", "dunes", (-106.28, 32.81), "NM_SouthEast_2018", "NM_WhiteSandsNM_2020"),
    ("brazoria-tx", "marsh", (-95.25, 29.07), "TX_CoastalRegion_2018", "TX_Houston_B24"),
    ("anahuac-tx", "marsh", (-94.45, 29.62), "TX_CoastalRegion_2018", "TX_Houston_B24"),
    ("myakka-fl", "marsh", (-82.25, 27.30), "FL_Peninsular_FDEM_2018", "FL_ManateeCounty_B25"),
    ("cameron-peak-co", "burn", (-105.650, 40.604), "CO_DRCOG_2020", "CO_ArapahoRooseveltPikeNF_D23"),
    ("grizzly-flats-ca", "burn", (-120.416, 38.595), "CA_UpperSouthAmerican_Eldorado_2019", "CA_SierraNevada_B22"),
    # farmland (CDL: 70 to 89% field crops around the point), the crops changing between the flights. Tried and
    # left out, the flights sharing too little of the tile: Brighton CO farmland (2%), other Othello WA tiles
    ("arkansas-delta-ar", "farm", (-91.40, 34.60), "AR_NRCS_A3_2016", "AR_Eastern_D23"),
    ("fresno-farm-ca", "farm", (-120.20, 36.60), "CA_FEMAR9Fresno_2019", "CA_SanJoaquin_2021"),
    ("hamilton-farm-in", "farm", (-86.05, 40.15), "IN_Central_Hamilton_2017", "IN_HamiltonCounty_A25"),
    ("othello-pivots-wa", "farm", (-119.00, 46.85), "WA_ColumbiaValley_2018", "WA_NorthCentral_2021"),
]

# the Cropland Data Layer (30 m, 2008 to 2025, EPSG:5070) on Source Coop
CDL = dict(bucket="chill", prefix="usda-cropland-data-layer/v0.1.0.icechunk", endpoint_url="https://data.source.coop", region="us-east-1")
# field crops, hay and fallow; orchards, vines, berries and asparagus left out (pulling an orchard can be real
# ground work), as are pasture, forest, water and developed land
FIELD_CROPS = np.array(sorted((set(range(1, 62)) - {55, 56})
                              | {205, 206, 208, 209, 213, 214, 216, 219, 221, 222, 224, *range(225, 242), *range(243, 250), 254}))
# (name, a point, the AlphaEarth year pairs): a 6 km box around each point. The taught regions use two pairs
# (more crops and seasons), the tested ones the map's default window
CDL_TEACH = [
    ("tulare-ca", (-119.35, 36.20), ((2019, 2021), (2023, 2025))),
    ("story-ia", (-93.45, 42.15), ((2019, 2021), (2023, 2025))),
    ("haskell-ks-pivots", (-100.90, 37.60), ((2019, 2021), (2023, 2025))),
    ("bolivar-ms", (-90.75, 33.45), ((2019, 2021), (2023, 2025))),
    ("lubbock-tx", (-101.90, 34.00), ((2019, 2021), (2023, 2025))),
    ("minidoka-id", (-113.70, 42.60), ((2019, 2021), (2023, 2025))),
    ("red-river-mn", (-96.80, 47.50), ((2019, 2021), (2023, 2025))),
]
# center-pivot country, taught with EARTHWORK_PIVOTS=1 (well away from the pivots tested: Platte NE, Quincy WA,
# Tift GA)
CDL_PIVOTS = [
    ("antelope-ne-pivots", (-98.05, 42.20), ((2019, 2021), (2023, 2025))),
    ("box-butte-ne-pivots", (-102.95, 42.10), ((2019, 2021), (2023, 2025))),
    ("dallam-tx-pivots", (-102.55, 36.20), ((2019, 2021), (2023, 2025))),
    ("san-luis-co-pivots", (-105.95, 37.75), ((2019, 2021), (2023, 2025))),
]
# and farmland under pivots flown twice (the Texas Panhandle's two 2017 projects, Chase and Dundy NE, San Luis
# Valley and Finney and Kearny KS have one flight or two in one year)
PIVOT_SITES = [
    ("antelope-farm-ne", "farm", (-98.05, 42.20), "NE_Hat_White_Holt_2016", "NE_Northeast_Phase2_2020"),
    ("box-butte-farm-ne", "farm", (-102.95, 42.10), "NE_Hat_White_Sioux_2016", "NE_Statewide_D23"),
]
if PIVOTS:
    CDL_TEACH = CDL_TEACH + CDL_PIVOTS
    SITES = SITES + PIVOT_SITES
CDL_TEST = [
    ("imperial-ca", (-115.50, 32.90), ((2023, 2025),)),
    ("platte-ne-pivots", (-98.35, 40.85), ((2023, 2025),)),
    ("quincy-wa-pivots", (-119.60, 47.10), ((2023, 2025),)),
    ("darke-oh", (-84.60, 40.10), ((2023, 2025),)),
    ("tift-ga-pivots", (-83.60, 31.40), ((2023, 2025),)),
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


def shared_grid(sa, sb):
    """The 2 m grid over where two open tiles meet: west, south, east, north, width and height."""
    W, S = (np.ceil(max(u, v) / 2) * 2 for u, v in ((sa.bounds.left, sb.bounds.left), (sa.bounds.bottom, sb.bounds.bottom)))
    E, N = (np.floor(min(u, v) / 2) * 2 for u, v in ((sa.bounds.right, sb.bounds.right), (sa.bounds.top, sb.bounds.top)))
    return W, S, E, N, int((E - W) / 2), int((N - S) / 2)


def dem_change(url_a, url_b):
    """Second DEM minus first, read at 2 m on one grid over where the two tiles meet (projects cut their tiles
    a little differently), the median removed (a datum or geoid offset between flights): (dz, transform, crs,
    overlap share)."""
    with rasterio.open("/vsicurl/" + url_a) as sa, rasterio.open("/vsicurl/" + url_b) as sb:
        if sa.crs != sb.crs:
            return None
        crs = sa.crs
        W, S, E, N, w, h = shared_grid(sa, sb)
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
    # standing water: lidar DEMs flatten it to one exact level on the day (hydro-flattening: a 5 x 5 spread
    # under 1 mm; paved or graded ground, flat as it is, spreads 5 to 20 mm, so a looser test drops graded pads).
    # Water in BOTH flights has no dz (its level moved, not the ground) and is returned to be taught as NOT
    # digging: water's AlphaEarth numbers shift from year to year, and a model never shown steady water reads
    # it as a fresh pond. Water in ONE flight is digging only one way round: ground then lower water is a dug
    # pond, water then higher ground a filled one; water then lower ground (a drawdown baring the bed) and
    # ground then higher water (a flood) are left out
    # (in float64 about the tile's median: squared elevations in float32 lose the millimeters above ~1000 m,
    # and every 5 x 5 must be all data)
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
    return dz, ta, crs, share, water


def aef_rc(tr, crs, h, w):
    """The AlphaEarth mosaic's row and column range inside a grid of h x w cells (transform tr, crs)."""
    inv = Transformer.from_crs(crs, 4326, always_xy=True)
    c = [inv.transform(*(tr * (x, y))) for x, y in ((0, 0), (w, 0), (0, h), (w, h))]
    W, E = max(c[0][0], c[2][0]), min(c[1][0], c[3][0])
    N, S = min(c[0][1], c[1][1]), max(c[2][1], c[3][1])
    x0, x1 = int(np.ceil((W - AEF_X0) / AEF_RES)), int((E - AEF_X0) / AEF_RES)
    y0, y1 = int(np.ceil((AEF_Y0 - N) / AEF_RES)), int((AEF_Y0 - S) / AEF_RES)
    return y0, y1, x0, x1


def site_rc(meta):
    """A cached site's mosaic rows and columns, from its two tiles' headers (the grid site_data used)."""
    u1, u2 = meta["tiles"]
    with rasterio.open("/vsicurl/" + u1) as sa, rasterio.open("/vsicurl/" + u2) as sb:
        W, S, E, N, w, h = shared_grid(sa, sb)
        return aef_rc(from_origin(W, N, 2, 2), sa.crs, h, w)


def to_aef_grid(dz, tr, crs, water):
    """Each AlphaEarth 10 m pixel over the tile: mean dz, the share of its 2 m cells that moved > 0.5 m and
    the share that were water in both flights (5 x 5 samples a pixel), with the mosaic's row and column range."""
    h, w = dz.shape
    y0, y1, x0, x1 = aef_rc(tr, crs, h, w)
    sub = (np.arange(5) + 0.5) / 5
    cols = x0 + (np.arange(x1 - x0)[:, None] + sub[None]).ravel()
    rows = y0 + (np.arange(y1 - y0)[:, None] + sub[None]).ravel()
    LON, LAT = np.meshgrid(AEF_X0 + cols * AEF_RES, AEF_Y0 - rows * AEF_RES)
    X, Y = Transformer.from_crs(4326, crs, always_xy=True).transform(LON, LAT)
    cc, rr = (~tr) * (X, Y)
    v = ndimage.map_coordinates(dz, [rr - 0.5, cc - 0.5], order=0, mode="constant", cval=np.nan)
    v = v.reshape(y1 - y0, 5, x1 - x0, 5)
    wv = ndimage.map_coordinates(water.astype(np.float32), [rr - 0.5, cc - 0.5], order=0, mode="constant", cval=0)
    wet = wv.reshape(y1 - y0, 5, x1 - x0, 5).mean(axis=(1, 3))
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(v, axis=(1, 3))
        moved = np.nanmean(np.abs(v) > MOVED_M, axis=(1, 3))
    return mean.astype(np.float32), moved.astype(np.float32), wet.astype(np.float32), (y0, y1, x0, x1)


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


def middle(yb, ya, rc, idx=None):
    """The year between (EARTHWORK_MIDDLE): per pixel (n, 3) of (1 - cos(b, m)), (1 - cos(m, a)) and the
    out-and-back (1 - cos(b, m)) + (1 - cos(m, a)) - (1 - cos(b, a)), m the middle year of the pair (all zero
    when the years are adjacent). Every pixel of rc in order, or those at idx."""
    y0, y1, x0, x1 = rc
    n = (y1 - y0) * (x1 - x0) if idx is None else len(idx)
    if ya - yb < 2:
        return np.zeros((n, 3), np.float32)
    vs = [aef_unit(y, rc)[0].reshape(64, -1) for y in (yb, (yb + ya) // 2, ya)]
    b, m, a = (v if idx is None else v[:, idx] for v in vs)
    d = lambda u, v: 1 - (u * v).sum(0)
    bm, ma = d(b, m), d(m, a)
    return np.stack([bm, ma, bm + ma - d(b, a)], 1).astype(np.float32)


def site_middle(name, d):
    """middle() for a cached site, cached beside it."""
    path = f"{DATA}/mid-{name}.npz"
    if os.path.exists(path):
        return np.load(path)["M"]
    meta = json.loads(str(d["meta"]))
    rc = site_rc(meta)
    if [rc[1] - rc[0], rc[3] - rc[2]] != meta["shape"]:
        raise ValueError(f"{name}: the grid from the tiles ({rc}) is not the cached one ({meta['shape']})")
    M = middle(*meta["aef"], rc).astype(np.float16)
    np.savez_compressed(path, M=M)
    return M


_cdl = {}


def cdl_classes(rc, years):
    """The CDL class under each AlphaEarth pixel over the mosaic rows and columns (its nearest 30 m cell) for each
    year, and whether that cell's 3 x 3 all agree (inside a field, not on its edge)."""
    if "ct" not in _cdl:
        import icechunk
        st = icechunk.s3_storage(**CDL, anonymous=True, force_path_style=True)
        g = zarr.open_group(icechunk.Repository.open(st).readonly_session("main").store, mode="r", path="30m")
        _cdl.update(ct=g["crop_type"], years=g["year"][:], x0=float(g["x"][0]), y0=float(g["y"][0]))
    y0, y1, x0, x1 = rc
    LON, LAT = np.meshgrid(AEF_X0 + (np.arange(x0, x1) + 0.5) * AEF_RES, AEF_Y0 - (np.arange(y0, y1) + 0.5) * AEF_RES)
    X, Y = Transformer.from_crs(4326, 5070, always_xy=True).transform(LON, LAT)
    c, r = np.rint((X - _cdl["x0"]) / 30).astype(int), np.rint((_cdl["y0"] - Y) / 30).astype(int)
    r0, c0 = r.min() - 1, c.min() - 1
    out, inside = [], []
    for yr in years:
        a = np.asarray(_cdl["ct"][int(np.where(_cdl["years"] == yr)[0][0]), r0:r.max() + 2, c0:c.max() + 2])
        same = ndimage.maximum_filter(a, 3) == ndimage.minimum_filter(a, 3)
        out.append(a[r - r0, c - c0])
        inside.append(same[r - r0, c - c0])
    return out, inside


def cdl_switches(name, point, yb, ya, n=25000):
    """Up to n AlphaEarth pixels in a 6 km box that grew a field crop every year from yb to ya, a different one
    in ya than in yb, inside a field both years: features, plain change and the two classes, cached."""
    path = f"{DATA}/cdl-{name}-{yb}-{ya}.npz"
    lon, lat = point
    dy, dx = 0.03, 0.03 / np.cos(np.radians(lat))
    rc = (int((AEF_Y0 - lat - dy) / AEF_RES), int((AEF_Y0 - lat + dy) / AEF_RES),
          int((lon - dx - AEF_X0) / AEF_RES), int((lon + dx - AEF_X0) / AEF_RES))
    if os.path.exists(path):
        z = np.load(path)
        out = {k: z[k] for k in z.files}
        if not MIDDLE:
            return out
        mpath = f"{DATA}/mid-cdl-{name}-{yb}-{ya}.npz"
        if os.path.exists(mpath):
            return out | {"M": np.load(mpath)["M"]}
        if "idx" in out:
            M = middle(yb, ya, rc, out["idx"]).astype(np.float16)
            np.savez_compressed(mpath, M=M)
            return out | {"M": M}
        # cached before the pixels were kept: taken again (the same pixels, the draw is seeded)
    cls, inside = cdl_classes(rc, range(yb, ya + 1))
    crop = np.all([np.isin(c, FIELD_CROPS) for c in cls], axis=0)
    b, okb = aef_unit(yb, rc)
    a, oka = aef_unit(ya, rc)
    idx = np.flatnonzero((crop & (cls[0] != cls[-1]) & inside[0] & inside[-1] & okb & oka).ravel())
    if len(idx) > n:
        idx = np.sort(np.random.default_rng(0).choice(idx, n, replace=False))
    b, a = b.reshape(64, -1)[:, idx], a.reshape(64, -1)[:, idx]
    out = {"X": features(b, a).astype(np.float16), "plain": (1 - (a * b).sum(0)).astype(np.float32),
           "from": cls[0].ravel()[idx], "to": cls[-1].ravel()[idx], "fields": np.int64(crop.sum()), "idx": idx}
    np.savez_compressed(path, **out)
    if MIDDLE:
        M = middle(yb, ya, rc, idx).astype(np.float16)
        np.savez_compressed(f"{DATA}/mid-cdl-{name}-{yb}-{ya}.npz", M=M)
        out["M"] = M
    return out


def gate(p, plain):
    """The map's score: the model's, quieted where AlphaEarth barely changed."""
    return p * np.clip((plain - 0.05) / 0.10, 0, 1)


def features(b, a):
    """[b, a, a * b, (a - b)^2] per pixel: (n, 256)."""
    return np.concatenate([b, a, a * b, (a - b) ** 2], 0).reshape(256, -1).T


def site_data(name, kind, point, p1, p2):
    """Features, labels, plain change, mean dz and patch size (ha) of moved ground for one site, cached."""
    path = f"{DATA}/{name}.npz"
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        if "wet" in z.files:  # a site cached before water was kept (no "wet") is built again
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
    dz, tr, crs, share, water = got
    mean, moved, wet, rc = to_aef_grid(dz, tr, crs, water)
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
    # water in both flights (most of the pixel): unchanged, not digging
    lab[okb & oka & (wet > 0.5) & (lab != 1)] = 0
    # each moved pixel's patch: connected moved pixels, in hectares
    px_ha = (AEF_RES * 111320 * np.cos(np.radians(lat))) * (AEF_RES * 110574) / 1e4
    cc, _ = ndimage.label(lab == 1, structure=np.ones((3, 3)))
    patch = (np.bincount(cc.ravel())[cc] * px_ha).astype(np.float32)
    out = {"X": features(b, a).astype(np.float16), "y": lab.ravel(), "plain": (1 - (a * b).sum(0)).ravel().astype(np.float32),
           "dz": mean.ravel(), "patch": patch.ravel(), "wet": (wet > 0.5).ravel(), "meta": json.dumps({"name": name, "kind": kind, "flights": [f"{d1[0]}..{d1[1]}", f"{d2[0]}..{d2[1]}"],
           "aef": [int(yb), int(ya)], "overlap": round(share, 3), "tiles": [u1, u2], "shape": list(mean.shape)})}
    np.savez_compressed(path, **out)
    return out


def feats(d, sel):
    """The features of a site's (or CDL region's) selected pixels, with the middle year's when it has them."""
    X = d["X"][sel].astype(np.float32)
    return np.concatenate([X, d["M"][sel].astype(np.float32)], 1) if "M" in d else X


def sample(d, rng, n_pos=20000, n_neg=200000):
    """A site's share of a training set: up to n_pos moved and n_neg unchanged pixels, so no one site (Katy,
    a quarter of it moved) outweighs the rest."""
    pos, neg = np.flatnonzero(d["y"] == 1), np.flatnonzero(d["y"] == 0)
    idx = np.concatenate([rng.choice(pos, min(len(pos), n_pos), replace=False), rng.choice(neg, min(len(neg), n_neg), replace=False)])
    return feats(d, idx), d["y"][idx]


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
                  f"overlap {m['overlap']:.0%}, moved {np.sum(lab == 1):,} px, unchanged {np.sum(lab == 0):,} (water {int(d['wet'].sum()):,})  ({time.time() - t:.0f} s)")
            d["kind"] = kind
            if MIDDLE:
                d["M"] = site_middle(name, d)
            sites[name] = d
    rng = np.random.default_rng(0)
    teach = [n for n, d in sites.items() if d["kind"] in TEACH_KINDS]
    quiet = [n for n, d in sites.items() if d["kind"] in QUIET_KINDS or n in QUIET_SITES]
    samples = {n: sample(sites[n], rng) for n in teach} | {n: sample(sites[n], rng, n_pos=0) for n in quiet}
    taught = teach + quiet
    # crop switches from the CDL, taught as not digging in every fit
    crops = []
    for name, point, pairs in CDL_TEACH if CDL_TAUGHT else ():
        for yb, ya in pairs:
            c = cdl_switches(name, point, yb, ya)
            print(f"  CDL {name} {yb} to {ya}: {len(c['plain']):,} crop switch px of {int(c['fields']):,} field px  ({time.time() - t:.0f} s)")
            crops.append((feats(c, slice(None)), np.zeros(len(c["plain"]), np.int8)))
    final = fit([samples[n] for n in taught] + crops)
    print(f"\nTaught on digging ({', '.join(TEACH_KINDS)}), and on {', '.join(quiet)} as not digging: each of those sites")
    print("scored by a model taught on the others; every other site (test only) by the model taught on all of them.")
    print("plain = AlphaEarth change, 1 - cos.")
    print("Quiet ground: the share of unchanged pixels scoring over 0.5 and over 0.8 (lower is quieter).")
    print(f"  {'site':20s} {'kind':9s} {'moved':>6s}   AP taught/plain   top 5% moved taught/plain   "
          f"moved > .5 / > .8   unchanged > .5 / > .8")
    held, water, summ = {}, {}, {}
    for name, d in sites.items():
        model = fit([samples[n] for n in taught if n != name] + crops) if name in taught else final
        m = d["y"] >= 0
        yt = d["y"][m]
        p = model.predict_proba(feats(d, m))[:, 1]
        q = d["plain"][m]
        k = max(1, int(0.05 * len(yt)))
        top = lambda s: yt[np.argsort(-s)[:k]].mean()
        ap = lambda s: average_precision_score(yt, s) if 0 < yt.sum() < len(yt) else np.nan
        over = lambda c, v: (p[yt == c] > v).mean() if (yt == c).any() else np.nan
        print(f"  {name:20s} {d['kind']:9s} {yt.mean():6.1%}   {ap(p):.2f}/{ap(q):.2f}          {top(p):5.1%}/{top(q):5.1%}"
              f"               {over(1, .5):5.1%} / {over(1, .8):5.1%}     {over(0, .5):5.1%} / {over(0, .8):5.1%}"
              + ("" if name in teach else "   (not digging)" if name in quiet else "   (test only)"))
        summ[name] = {"kind": d["kind"], "ap": ap(p), "still5": over(0, .5), "still8": over(0, .8)}
        if name in teach:
            held[name] = (p, q, yt, d["dz"][m], d["patch"][m])
        wet = d["wet"][m] & (yt == 0)
        if wet.sum() >= 100:
            water[name] = (int(wet.sum()), (p[wet] > .5).mean(), (p[wet] > .8).mean())
    print("\nWater in both flights (unchanged, taught as not digging), held out: share scoring over 0.5 / over 0.8")
    for name, (n, a, b) in water.items():
        print(f"  {name:20s} {n:9,d} px   {a:5.1%} / {b:5.1%}")
    # crop changes where the model was never taught: the map's score (gated by plain change) on fields that
    # switched crops, per region and pooled
    print(f"\nCrop switches (CDL) in regions never taught, the map's score: share over 0.5 / over 0.8")
    pool = []
    for name, point, pairs in CDL_TEST:
        for yb, ya in pairs:
            c = cdl_switches(name, point, yb, ya)
            if len(c["plain"]) < 100:
                print(f"  {name:20s} {yb}-{ya}: {len(c['plain'])} px, too few")
                continue
            g = gate(final.predict_proba(feats(c, slice(None)))[:, 1], c["plain"])
            pool.append(g)
            summ[f"cdl:{name}"] = {"over5": (g > .5).mean(), "over8": (g > .8).mean()}
            print(f"  {name:20s} {yb}-{ya} {len(g):7,d} px   {(g > .5).mean():5.1%} / {(g > .8).mean():5.1%}")
    if pool:
        g = np.concatenate(pool)
        print(f"  {'all':20s}           {len(g):7,d} px   {(g > .5).mean():5.1%} / {(g > .8).mean():5.1%}")
        summ["cdl:all"] = {"over5": (g > .5).mean(), "over8": (g > .8).mean()}
    # the numbers to compare variants by: digging AP (each site held out, or test only), quiet ground at the
    # building sites, the farm lidar's unchanged ground and the untaught crop switches
    kind_mean = lambda k, f: float(np.nanmean([v[f] for v in summ.values() if v.get("kind") == k]))
    print(f"\nSummary: AP building {kind_mean('building', 'ap'):.3f}, mining {kind_mean('mining', 'ap'):.3f}; building unchanged"
          f" > .5 / > .8 {kind_mean('building', 'still5'):.1%} / {kind_mean('building', 'still8'):.1%}; farm unchanged > .5 "
          + ", ".join(f"{n} {v['still5']:.1%}" for n, v in summ.items() if v.get("kind") == "farm"))
    print("SUMMARY " + json.dumps({n: {k: (round(float(x), 4) if not isinstance(x, str) else x) for k, x in v.items()} for n, v in summ.items()}))
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
    out = os.path.join(HERE, "models", f"earthwork-lr-{VARIANT}.npz") if VARIANT else CROP_MODEL if CROP else MODEL
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, w=final.coef_[0].astype(np.float32), b=np.float32(final.intercept_[0]),
             features=np.array("[b, a, a*b, (a-b)^2] of AlphaEarth unit embeddings: b the year before the first flight, a the year after"
                               + (", then (1 - cos(b, m)), (1 - cos(m, a)), out-and-back, m the middle year (b + a) // 2" if MIDDLE else "")),
             middle=np.bool_(MIDDLE),
             sites=np.array(teach), not_digging=np.array(quiet + ([f"cdl:{n}" for n, _, _ in CDL_TEACH] if CDL_TAUGHT else [])), tested=np.array([n for n in sites if n not in taught]), moved_m=MOVED_M, still_m=STILL_M)
    print(f"\nsaved {os.path.relpath(out, HERE)} (taught on {len(teach)} digging sites and {len(quiet)} not), {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
