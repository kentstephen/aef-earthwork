"""Repeat terrain models outside the US, as earthwork_model.py's site caches: more ground for the model to learn from.

3DEP only flies the US, so the model learned what digging and quiet ground look like on US ground alone. Other
producers publish a second bare-earth terrain model of the same ground, and the difference of the two is the
same kind of truth: where earth moved between them. This builds those sites in exactly the format of
earthwork_model.site_data() (data/earthwork/<name>.npz with X, y, plain, dz, patch, wet, meta), so the
training script picks them up with a line of its own.

Sources reached here:
  IGN France, Geoplateforme WMS-R (raw float32 heights, any window): each territory's own RGE ALTI 1 m layer
  and the LiDAR HD terrain model, read on a 2 m grid in the territory's UTM zone (the world-wide RGE ALTI layer,
  ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES, is resampled to about 5 m there: three quarters of its 1 m
  neighbors repeat), with the LiDAR HD flight dates per tile from its metadata (WFS
  IGNF_LIDAR-HD_METADONNEE:metadata). Licence Ouverte / Etalab 2.0.
  Overseas, LiDAR HD has flown Reunion (2023) and Guadeloupe (2024) so far; Martinique, Mayotte, French Guiana
  and Saint Martin have none yet. Their RGE ALTI there is Litto3D lidar from about 2008 to 2013, before
  AlphaEarth's first year (2017): its moved ground may have moved before 2017, so these sites teach only as
  NOT digging (kind "tropic", their unchanged ground; ground unchanged from about 2010 to 2023 was unchanged
  through 2017 to 2023 too).
Spain's PNOA (CNIG, IDEE) has three national coverages but did not answer from here (its hosts time out).

Run: uv run --with icechunk python earthwork_sites_intl.py   (a few minutes the first time; the windows are cached)
"""

import datetime
import json
import os

import numpy as np
import requests
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from scipy import ndimage

import earthwork_model as em

RAW = os.path.join(em.DATA, "intl-raw")
WMSR = "https://data.geopf.fr/wms-r"
WFS = "https://data.geopf.fr/wfs/ows"
# island -> (its UTM zone, its RGE ALTI layer, the LiDAR HD layer read there)
ISLANDS = {"reunion": ("EPSG:2975", "RGEALTI-MNT_PYR-ZIP_REU_RGR92UTM40S_WMS", "IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.RGR92UTM40S"),
           "guadeloupe": ("EPSG:32620", "RGEALTI-MNT_PYR-ZIP_GLP_WGS84UTM20_WMS", "IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.WGS84G")}
# RGE ALTI overseas is Litto3D lidar (topo and topo/bathy): only its span is known here, not each tile's day
LITTO3D = {"reunion": (datetime.date(2008, 1, 1), datetime.date(2011, 12, 31)),
           "guadeloupe": (datetime.date(2010, 1, 1), datetime.date(2013, 12, 31))}
STEP_M = 2.0  # the 2 m grid the US sites are read on
HALF_KM = 2.0  # each site a 4 km square about its point
TROPIC_KINDS = ("tropic",)

# (name, kind, a point inside the site, island): quiet tropical ground of many looks, the moved ground left to
# the labels (and not taught)
SITES_INTL = [
    ("reunion-sud-cane", "tropic", (55.46, -21.29), "reunion"),        # sugar cane above Saint-Pierre
    ("reunion-est-forest", "tropic", (55.62, -21.12), "reunion"),      # rainforest and cane, Saint-Benoit
    ("reunion-ouest-port", "tropic", (55.30, -20.96), "reunion"),      # Le Port and the Galets river bed
    ("reunion-nord-city", "tropic", (55.47, -20.90), "reunion"),       # Saint-Denis and its slopes
    ("guadeloupe-moule-cane", "tropic", (-61.39, 16.33), "guadeloupe"),  # Grande-Terre cane and pasture
    ("guadeloupe-capesterre", "tropic", (-61.58, 16.05), "guadeloupe"),  # bananas under the volcano
    ("guadeloupe-pointe", "tropic", (-61.52, 16.25), "guadeloupe"),     # Pointe-a-Pitre, mangrove and port
]


def _box(lon, lat, crs, half_km=HALF_KM):
    """West, south, east, north (meters in crs) of a square half_km about the point, on the 2 m grid, and its
    size in cells."""
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    W, S = np.floor((x - half_km * 1000) / STEP_M) * STEP_M, np.floor((y - half_km * 1000) / STEP_M) * STEP_M
    n = int(round(2 * half_km * 1000 / STEP_M))
    return W, S, W + n * STEP_M, S + n * STEP_M, n, n


def wmsr_read(layer, crs, W, S, E, N, nx, ny, tile=1000):
    """A layer's heights (float32, NaN where empty) on the nx by ny grid over the box (crs), from WMS-R in
    pieces of at most tile x tile; cached under data/earthwork/intl-raw/."""
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, f"{layer}_{crs.replace(':', '')}_{W:.0f}_{S:.0f}_{nx}x{ny}.npy")
    if os.path.exists(path):
        return np.load(path)
    out = np.full((ny, nx), np.nan, np.float32)
    dx, dy = (E - W) / nx, (N - S) / ny
    for r0 in range(0, ny, tile):
        for c0 in range(0, nx, tile):
            h, w = min(tile, ny - r0), min(tile, nx - c0)
            n_, s_ = N - r0 * dy, N - (r0 + h) * dy
            w_, e_ = W + c0 * dx, W + (c0 + w) * dx
            params = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": layer, "STYLES": "",
                      "CRS": crs, "BBOX": f"{w_},{s_},{e_},{n_}", "WIDTH": w, "HEIGHT": h, "FORMAT": "image/geotiff"}
            for k in range(4):
                try:
                    r = requests.get(WMSR, params=params, timeout=180)
                    if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
                        break
                except requests.RequestException:
                    pass
            else:
                raise RuntimeError(f"WMS-R {layer} failed at {params['BBOX']}")
            with MemoryFile(r.content) as mf, mf.open() as d:
                a = d.read(1).astype(np.float32)
                bad = (a < -1000) | (a == (d.nodata if d.nodata is not None else -99999))
            a[bad] = np.nan
            out[r0:r0 + h, c0:c0 + w] = a
    np.save(path, out)
    return out


def lidarhd_dates(W, S, E, N):
    """(start, end) of the LiDAR HD flights over the box, from the tiles' metadata."""
    d = requests.get(WFS, params={"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature", "TYPENAMES": "IGNF_LIDAR-HD_METADONNEE:metadata",
                                  "OUTPUTFORMAT": "application/json", "COUNT": 1000, "BBOX": f"{S},{W},{N},{E},urn:ogc:def:crs:EPSG::4326"},
                     timeout=120).json()
    day = lambda s: datetime.date.fromisoformat(s[:10])
    p = [f["properties"] for f in d.get("features", []) if f["properties"].get("date_debut_acquisition")]
    if not p:
        return None
    return min(day(x["date_debut_acquisition"]) for x in p), max(day(x["date_fin_acquisition"]) for x in p)


def change(a, b):
    """Second minus first with the median removed, and water in both: earthwork_model.dem_change's rules on two
    arrays already on one grid (hydro-flattened water has no dz; water in one only counts as digging one way)."""
    dz = b - a
    share = float(np.isfinite(dz).mean())
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


def site_data_intl(name, kind, point, island):
    """One site's cache in earthwork_model.site_data()'s format, built from RGE ALTI then LiDAR HD."""
    path = f"{em.DATA}/{name}.npz"
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    lon, lat = point
    crs, l_old, l_new = ISLANDS[island]
    W, S, E, N, nx, ny = _box(lon, lat, crs)
    d2 = lidarhd_dates(lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02)
    if d2 is None:
        print(f"  {name}: skipped (no LiDAR HD over it)")
        return None
    d1 = LITTO3D[island]
    a = wmsr_read(l_old, crs, W, S, E, N, nx, ny)
    b = wmsr_read(l_new, crs, W, S, E, N, nx, ny)
    dz, share, water = change(a, b)
    if share < em.MIN_OVERLAP:
        print(f"  {name}: skipped (both models cover {share:.0%} of it)")
        return None
    tr = from_origin(W, N, STEP_M, STEP_M)
    mean, moved, wet, rc = em.to_aef_grid(dz, tr, crs, water)
    yb, ya = em.aef_years(d1, d2)
    bv, okb = em.aef_unit(yb, rc)
    av, oka = em.aef_unit(ya, rc)
    ok = okb & oka & np.isfinite(mean)
    lab = np.full(mean.shape, -1, np.int8)
    lab[ok & (np.abs(mean) > em.MOVED_M)] = 1
    lab[ok & (np.abs(mean) < em.STILL_M) & (moved < 0.05)] = 0
    lab[okb & oka & (wet > 0.5) & (lab != 1)] = 0
    px_ha = (em.AEF_RES * 111320 * np.cos(np.radians(lat))) * (em.AEF_RES * 110574) / 1e4
    cc, _ = ndimage.label(lab == 1, structure=np.ones((3, 3)))
    patch = (np.bincount(cc.ravel())[cc] * px_ha).astype(np.float32)
    out = {"X": em.features(bv, av).astype(np.float16), "y": lab.ravel(), "plain": (1 - (av * bv).sum(0)).ravel().astype(np.float32),
           "dz": mean.ravel(), "patch": patch.ravel(), "wet": (wet > 0.5).ravel(),
           "meta": json.dumps({"name": name, "kind": kind, "flights": [f"{d1[0]}..{d1[1]}", f"{d2[0]}..{d2[1]}"], "aef": [int(yb), int(ya)],
                               "overlap": round(share, 3), "tiles": [f"wms-r:{l_old}", f"wms-r:{l_new}"], "crs": crs, "shape": list(mean.shape),
                               "source": "IGN France (RGE ALTI, LiDAR HD), Licence Ouverte / Etalab 2.0", "quiet_only": True})}
    np.savez_compressed(path, **out)
    return out


def main():
    os.makedirs(em.DATA, exist_ok=True)
    for name, kind, point, island in SITES_INTL:
        d = site_data_intl(name, kind, point, island)
        if d is None:
            continue
        m = json.loads(str(d["meta"]))
        y = d["y"]
        print(f"  {name}: {m['flights'][0]} / {m['flights'][1]}, AlphaEarth {m['aef'][0]} vs {m['aef'][1]}, overlap {m['overlap']:.0%}, "
              f"moved {np.sum(y == 1):,} px (not taught), unchanged {np.sum(y == 0):,} (water {int(d['wet'].sum()):,})")


if __name__ == "__main__":
    main()
