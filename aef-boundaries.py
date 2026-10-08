# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "anywidget>=0.9",
#     "traitlets",
#     "numpy",
#     "scipy",
#     "zarr>=3.1",
#     "obstore>=0.9.2",
#     "pillow",
#     "requests",
#     "pyarrow",
#     "duckdb",
#     "async-geotiff>=0.4",
#     "pyproj",
#     "h3ronpy>=0.22.0",
# ]
# ///
"""AEF boundaries: elevation in H3 hexagons, carved by where AlphaEarth says the ground changes.

The idea of the Denali flow map, with AlphaEarth in Whitebox's place. There, elevation and
flow accumulation were each folded to H3 and combined in one expression, and the flow
lines showed as impressions in the mountain. Here the second quantity is BOUNDARY
STRENGTH: for every 10 m pixel, the mean of 1 - cosine between its AlphaEarth embedding
and its four neighbors' (an edge detector over the 64 bands, no model). Inside uniform
ground (unbroken forest, a lake, a field) it is near zero; where two kinds of ground
meet (treeline, riverbank, the rim of a clearcut, a road cut, forest against rock) it
is high. Each hexagon's mean elevation is lowered by a small amount at the outlines:
GROOVE meters (the slider, 5 m to start) where the boundary strength reaches the view's
98th percentile, proportionally less on weaker ones; then colored with one palette. The seams of the land show as grooves pressed into it. It
is one moment (the years slider's average, 2023 to 2025 by default), not change.

A shows the plain hexagons (no grooves). The images are drawn at screen
resolution, each pixel taking its hexagon's value, so the hexagons stay crisp.

Everything the viewport sees is read ONCE, at one resolution: the 10 m mosaic when the
whole view fits AEF_MAX_PX pixels a year (about 600 km2), else the one AlphaEarth
overview level that fits it (20 m at the finest, coarser for wide or 3D views). The
boundaries are computed for each year and averaged, one year in memory at a time. The
image is drawn at the screen's full pixel density. The hexagons' res comes
from that read (10 m gives res 12, 20 m res 11, 40 m res 10), so one hexagon size
covers the screen. The range is fitted to the whole view when the map opens and on S,
and held while you pan.

Memory: both reads are capped in pixels at any zoom; the kernel keeps only each part's
hexagons and the image-pixel index to recolor; caches as in the other notebooks.

Keys: A grooves or plain, S fit, R reverse, T 3D, arrows pan, Shift + arrows turn and
tilt, L place names, / search, F full screen, \\ hides the panel.

Run: uv run marimo run aef-boundaries.py --sandbox --watch
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import anywidget
    import traitlets
    import asyncio
    import io
    import math
    import os
    import tempfile
    import time
    from collections import OrderedDict
    from concurrent.futures import ThreadPoolExecutor

    import duckdb
    import numpy as np
    import pyarrow.parquet as pq
    import requests
    import zarr
    from async_geotiff import GeoTIFF, Window
    from h3ronpy.vector import coordinates_to_cells
    from obstore.store import HTTPStore, S3Store
    from PIL import Image
    from pyproj import Transformer
    from scipy import ndimage
    from zarr.storage import ObjectStore

    return (
        GeoTIFF,
        HTTPStore,
        Image,
        ObjectStore,
        OrderedDict,
        S3Store,
        ThreadPoolExecutor,
        Transformer,
        Window,
        anywidget,
        coordinates_to_cells,
        asyncio,
        duckdb,
        io,
        math,
        mo,
        ndimage,
        np,
        os,
        pq,
        requests,
        tempfile,
        time,
        traitlets,
        zarr,
    )


@app.cell
def _(
    Image,
    ObjectStore,
    OrderedDict,
    S3Store,
    ThreadPoolExecutor,
    asyncio,
    io,
    np,
    requests,
    zarr,
):
    # AlphaEarth's global mosaic on Source Cooperative: EPSG:4326, about 10 m, (time, 64, y, x)
    # int8, 256 px chunks. Read through the data.source.coop proxy, never the bucket behind it
    AEF_RES, AEF_Y0, AEF_X0, AEF_NODATA = 8.983111749910169e-05, 83.68570533713473, -180.0, -128
    AEF_YEARS = (2022, 2023, 2024, 2025)  # the years the window picks from (as on_the_fly); it opens on 2023-2025
    AEF_MAX_PX = 6_000_000  # pixels a year for the view (about 600 km2 at 10 m); years are read one at a time, about 390 MB each
    AEF_MIN_ZOOM = 11.5
    MOSAIC_CACHE_BYTES = 384 * 1024**2  # 4 MB chunks, least recently used dropped first
    DEM_Z = 13  # Mapterhorn 512 px tiles, about 6.6 m at 46 N
    DEM_CACHE_TILES = 96  # 1 MB each as float32
    DEM_URL = "https://tiles.mapterhorn.com/{z}/{x}/{y}.webp"

    _store = S3Store("tge-labs", endpoint="https://data.source.coop", region="us-west-2",
                     virtual_hosted_style_request=False, skip_signature=True, prefix="aef-mosaic",
                     client_options={"timeout": "8s", "connect_timeout": "3s"})
    _grp = zarr.open_group(ObjectStore(_store, read_only=True), mode="r")
    _times = _grp["time"][:]
    _emb = _grp["embeddings"]
    _cb = int(_emb.chunks[-1])
    _arr = _emb._async_array
    _chunks = OrderedDict()  # (ti, r, c) -> int8 (64, cb, cb)
    _held = [0]

    async def aef_window(box, year):
        """The mosaic's int8 (64, h, w) window under box = (W, S, E, N), with the window's
        own edges (W, N, E, S) in degrees, or None."""
        W, S, E, N = box
        x0, x1 = int((W - AEF_X0) / AEF_RES), int((E - AEF_X0) / AEF_RES)
        y0, y1 = int((AEF_Y0 - N) / AEF_RES), int((AEF_Y0 - S) / AEF_RES)
        if x1 <= x0 or y1 <= y0:
            return None
        ti = int(np.where(_times == year)[0][0])
        out = np.empty((64, y1 - y0, x1 - x0), np.int8)
        sem = asyncio.Semaphore(32)

        async def one(r, c):
            k = (ti, r, c)
            b = _chunks.pop(k, None)
            if b is None:
                async with sem:
                    b = await _arr.getitem((ti, slice(None), slice(r, r + _cb), slice(c, c + _cb)))
                _held[0] += b.nbytes
            _chunks[k] = b
            while _held[0] > MOSAIC_CACHE_BYTES and len(_chunks) > 1:
                _held[0] -= _chunks.popitem(last=False)[1].nbytes
            r0, r1, c0, c1 = max(r, y0), min(r + _cb, y1), max(c, x0), min(c + _cb, x1)
            out[:, r0 - y0:r1 - y0, c0 - x0:c1 - x0] = b[:, r0 - r:r1 - r, c0 - c:c1 - c]

        await asyncio.gather(*(one(r, c) for r in range(y0 // _cb * _cb, y1, _cb)
                               for c in range(x0 // _cb * _cb, x1, _cb)))
        edges = (AEF_X0 + x0 * AEF_RES, AEF_Y0 - y0 * AEF_RES, AEF_X0 + x1 * AEF_RES, AEF_Y0 - y1 * AEF_RES)
        return out, edges

    _dem = OrderedDict()  # (z, x, y) -> float32 (512, 512) meters, or None for no tile (sea)
    _pool = ThreadPoolExecutor(16)

    def _dem_tile(t):
        if t in _dem:
            _dem.move_to_end(t)
            return _dem[t]
        z, x, y = t
        v = None
        try:
            r = requests.get(DEM_URL.format(z=z, x=x, y=y), timeout=15)
            if r.status_code == 200:
                p = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB")).astype(np.float32)
                v = p[..., 0] * 256 + p[..., 1] + p[..., 2] / 256 - 32768
        except requests.RequestException:
            return None
        _dem[t] = v
        while len(_dem) > DEM_CACHE_TILES:
            _dem.popitem(last=False)
        return v

    def merc(lon, lat, z=DEM_Z):
        """Web Mercator pixel coordinates of 512 px tiles at zoom z."""
        n = 2**z * 512
        s = np.sin(np.radians(lat))
        return (np.asarray(lon) + 180) / 360 * n, (0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * n

    def dem_at(lon, lat, z=DEM_Z):
        """Elevation in meters at the grid lon (w,) x lat (h,), bilinear from Mapterhorn's
        512 px tiles at zoom z."""
        from scipy import ndimage
        LON, LAT = np.meshgrid(lon, lat)
        px, py = merc(LON, LAT, z)
        tx0, tx1 = int(px.min() // 512), int(px.max() // 512)
        ty0, ty1 = int(py.min() // 512), int(py.max() // 512)
        tiles = [(x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)]
        got = dict(zip(tiles, _pool.map(_dem_tile, [(z, x, y) for x, y in tiles])))
        big = np.full(((ty1 - ty0 + 1) * 512, (tx1 - tx0 + 1) * 512), np.nan, np.float32)
        for (x, y), v in got.items():
            if v is not None:
                big[(y - ty0) * 512:(y - ty0 + 1) * 512, (x - tx0) * 512:(x - tx0 + 1) * 512] = v
        return ndimage.map_coordinates(big, [(py - ty0 * 512 - 0.5).ravel(), (px - tx0 * 512 - 0.5).ravel()],
                                       order=1, mode="nearest").reshape(LAT.shape)

    def cache_bytes():
        return {"mosaic_mb": round(_held[0] / 1e6), "dem_tiles": len(_dem)}

    return (
        AEF_MAX_PX,
        AEF_MIN_ZOOM,
        AEF_NODATA,
        AEF_RES,
        AEF_YEARS,
        aef_window,
        cache_bytes,
        dem_at,
        merc,
    )


@app.cell
def _(
    AEF_NODATA,
    AEF_YEARS,
    GeoTIFF,
    HTTPStore,
    OrderedDict,
    Transformer,
    Window,
    asyncio,
    duckdb,
    math,
    ndimage,
    np,
    os,
    pq,
    tempfile,
):
    # THE WHOLE VIEWPORT, COARSE: AlphaEarth's per-tile COGs (UTM, 8192 px at 10 m, overviews
    # halving down to 20, 40, 80 m and on) read at the overview whose pixel keeps the view under
    # COARSE_PX pixels, then laid on an even lon/lat grid. The bytes stay about the same at any
    # zoom or pitch; the 10 m mosaic covers the near ground on top. Never finer than 40 m: the
    # 20 m overview reads slowly (14 s for a 24 km view) and the near ground is 10 m anyway
    AEF_INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
    COARSE_PX = 700_000
    COARSE_MAX_FILES = 60
    COG_KEEP = 96  # opened COG headers kept
    _idx_path = os.path.join(tempfile.gettempdir(), "aef-elevation", f"aef_index_{min(AEF_YEARS)}_{max(AEF_YEARS)}.parquet")
    if not os.path.exists(_idx_path):
        os.makedirs(os.path.dirname(_idx_path), exist_ok=True)
        _c = duckdb.connect()
        _c.execute("INSTALL httpfs; LOAD httpfs")
        _t = _c.execute(f"""SELECT year, path, crs, wgs84_west, wgs84_south, wgs84_east, wgs84_north
                            FROM read_parquet('{AEF_INDEX_URL}')
                            WHERE year IN ({", ".join(map(str, AEF_YEARS))})""").arrow().read_all()
        pq.write_table(_t, _idx_path)
        _c.close()
    _tab = pq.read_table(_idx_path)
    _yr = _tab["year"].to_numpy()
    _ix = {k: _tab[k].to_numpy() for k in ("wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north")}
    _paths = _tab["path"].to_pylist()
    _crs = _tab["crs"].to_pylist()
    _cog_store = HTTPStore.from_url("https://data.source.coop", client_options={"timeout": "15s", "connect_timeout": "3s"})
    _opened = OrderedDict()
    _tfs = {}

    async def _cog(i):
        rel = _paths[i].split("source.coop/")[1]
        g = _opened.pop(rel, None)
        if g is None:
            g = await GeoTIFF.open(rel, store=_cog_store)
        _opened[rel] = g
        while len(_opened) > COG_KEEP:
            _opened.popitem(last=False)
        return g

    def _tf(crs):
        if crs not in _tfs:
            _tfs[crs] = (Transformer.from_crs("EPSG:4326", crs, always_xy=True),
                         Transformer.from_crs(crs, "EPSG:4326", always_xy=True))
        return _tfs[crs]

    async def _read_file(i, li, box, sem):
        """One COG's overview li under the box: (int8 (64, n), lon (n,), lat (n,)) or None."""
        async with sem:
            g = await _cog(i)
            ov = g.overviews[li]
            H, W = ov.shape
            t = g.transform
            sx, sy = t.a * (g.width / W), t.e * (g.height / H)
            fwd, inv = _tf(_crs[i])
            W_, S_, E_, N_ = box
            lons = np.concatenate([np.linspace(W_, E_, 9), np.full(9, E_), np.linspace(E_, W_, 9), np.full(9, W_)])
            lats = np.concatenate([np.full(9, N_), np.linspace(N_, S_, 9), np.full(9, S_), np.linspace(S_, N_, 9)])
            ux, uy = fwd.transform(lons, lats)
            cc = (np.asarray(ux) - t.c) / sx
            rr = (np.asarray(uy) - t.f) / sy
            c0, c1 = max(0, int(np.floor(np.nanmin(cc)))), min(W, int(np.ceil(np.nanmax(cc))))
            r0, r1 = max(0, int(np.floor(np.nanmin(rr)))), min(H, int(np.ceil(np.nanmax(rr))))
            if c1 <= c0 or r1 <= r0:
                return None
            ra = await ov.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))

        def place():
            a = np.asarray(np.ma.filled(ra.as_masked(), AEF_NODATA)).reshape(64, -1)
            X, Y = np.meshgrid(t.c + (np.arange(c0, c1) + 0.5) * sx, t.f + (np.arange(r0, r1) + 0.5) * sy)
            lon, lat = inv.transform(X.ravel(), Y.ravel())
            return a, np.asarray(lon), np.asarray(lat)

        return await asyncio.get_running_loop().run_in_executor(None, place)

    async def aef_coarse(box, year, max_px=COARSE_PX, min_m=40.0):
        """AlphaEarth over the whole box at the coarsest overview that keeps it under max_px:
        (int8 (64, H, W), grid) on an even lon/lat grid, or (None, reason)."""
        W, S, E, N = box
        lat_mid = (N + S) / 2
        cos = math.cos(math.radians(lat_mid))
        area = (E - W) * 111320 * cos * (N - S) * 110574
        res_m = max(min_m, math.sqrt(area / max_px))
        li = min(8, max(0, math.ceil(math.log2(res_m / 10)) - 1))
        res_m = 10.0 * 2 ** (li + 1)
        hits = np.where((_yr == year) & (_ix["wgs84_east"] > W) & (_ix["wgs84_west"] < E) & (_ix["wgs84_north"] > S) & (_ix["wgs84_south"] < N))[0]
        if len(hits) == 0:
            return None, "No AlphaEarth tiles under this view"
        if len(hits) > COARSE_MAX_FILES:
            return None, f"{len(hits)} AlphaEarth tiles under this view; zoom in"
        dlat, dlon = res_m / 110574, res_m / (111320 * cos)
        H, Wd = max(1, math.ceil((N - S) / dlat)), max(1, math.ceil((E - W) / dlon))
        sem = asyncio.Semaphore(16)
        parts = await asyncio.gather(*(_read_file(int(i), li, box, sem) for i in hits))

        def lay():
            out = np.full((64, H, Wd), AEF_NODATA, np.int8)
            got = np.zeros((H, Wd), bool)
            for p in parts:
                if p is None:
                    continue
                a, lon, lat = p
                r = ((N - lat) / dlat).astype(np.int64)
                c = ((lon - W) / dlon).astype(np.int64)
                k = (r >= 0) & (r < H) & (c >= 0) & (c < Wd) & (a[0] != AEF_NODATA)
                out[:, r[k], c[k]] = a[:, k]
                got[r[k], c[k]] = True
            if got.any() and not got.all():  # the seams between UTM pixels: nearest, two pixels at most
                dist, (ir, ic) = ndimage.distance_transform_edt(~got, return_indices=True)
                fill = (~got) & (dist <= 2)
                out[:, fill] = out[:, ir[fill], ic[fill]]
            return out

        out = await asyncio.get_running_loop().run_in_executor(None, lay)
        grid = {"edges": (W, N, W + Wd * dlon, N - H * dlat), "dlon": dlon, "dlat": dlat, "lat_mid": lat_mid, "res_m": res_m}
        return out, grid

    return (aef_coarse,)


@app.cell
def _(AEF_NODATA, Image, coordinates_to_cells, io, math, np):
    # mean H3 edge length per res, meters: a part's hexagons are no smaller than about 1.5 of its pixels
    H3_EDGE_M = {7: 1406, 8: 531, 9: 201, 10: 76, 11: 29, 12: 11, 13: 4}
    MAX_IMG = 4096  # longest side of the view's image, pixels

    def res_for(pixel_m):
        """The finest res whose hexagon edge is at least the read's pixel: 10 m reads give res 12, 20 m res 11,
        40 m res 10. The read sets the size, so one size covers the whole view."""
        r = 13
        while r > 7 and H3_EDGE_M[r] < pixel_m:
            r -= 1
        return r

    def _unit(e):
        v = e.reshape(64, -1).T.astype(np.float32)
        v = np.sign(v) * (v / 127.5) ** 2
        return v / np.maximum(np.linalg.norm(v, axis=1), 1e-9)[:, None]

    def boundaries(emb):
        """For every pixel, the mean of 1 - cos to its four neighbors (left, right, above, below), NaN
        where AlphaEarth has no data. Row blocks with a one-row halo: the unit vectors never sit whole."""
        _, h, w = emb.shape
        ok = emb[0] != AEF_NODATA
        out = np.full((h, w), np.nan, np.float32)
        B = 128
        for r0 in range(0, h, B):
            r1 = min(h, r0 + B)
            a0, a1 = max(0, r0 - 1), min(h, r1 + 1)
            U = _unit(emb[:, a0:a1]).reshape(a1 - a0, w, 64)
            o = ok[a0:a1]
            s = np.zeros((a1 - a0, w), np.float32)
            c = np.zeros((a1 - a0, w), np.float32)
            dr = 1 - (U[:, :-1] * U[:, 1:]).sum(-1)
            vr = o[:, :-1] & o[:, 1:]
            s[:, :-1] += np.where(vr, dr, 0); c[:, :-1] += vr
            s[:, 1:] += np.where(vr, dr, 0); c[:, 1:] += vr
            dd = 1 - (U[:-1] * U[1:]).sum(-1)
            vd = o[:-1] & o[1:]
            s[:-1] += np.where(vd, dd, 0); c[:-1] += vd
            s[1:] += np.where(vd, dd, 0); c[1:] += vd
            e = np.where((c > 0) & o, s / np.maximum(c, 1), np.nan)
            out[r0:r1] = e[r0 - a0:r0 - a0 + (r1 - r0)]
        return out

    def hexagons(edge, elev, grid, res):
        """The part's H3 cells (sorted) with their mean elevation and mean boundary strength."""
        h, w = edge.shape
        W, N, _, _ = grid["edges"]
        LON, LAT = np.meshgrid(W + (np.arange(w) + 0.5) * grid["dlon"], N - (np.arange(h) + 0.5) * grid["dlat"])
        ok = (np.isfinite(edge) & np.isfinite(elev)).ravel()
        cells = np.asarray(coordinates_to_cells(LAT.ravel()[ok], LON.ravel()[ok], res)).astype(np.uint64)
        u, inv = np.unique(cells, return_inverse=True)
        n = np.bincount(inv, minlength=len(u)).astype(np.float64)
        e = np.bincount(inv, weights=edge.ravel()[ok], minlength=len(u)) / n
        return {"cells": u, "res": res, "n": n,
                "elev": np.bincount(inv, weights=elev.ravel()[ok], minlength=len(u)) / n,
                "edge": e, "edge98": float(np.percentile(e, 98)) or 1e-3}

    def pixel_index(hx, grid, scale):
        """For an image over the part's extent at `scale` image pixels per degree of longitude, rows even in
        Web Mercator: each image pixel's row in hx (-1 where no hexagon of the part)."""
        W, N, E, S = grid["edges"]
        w = int(min(MAX_IMG, max(2, round((E - W) * scale))))
        merc_y = lambda lat: math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
        h = int(min(MAX_IMG, max(2, round(w * (merc_y(N) - merc_y(S)) / math.radians(E - W)))))
        yy = merc_y(N) + (np.arange(h) + 0.5) / h * (merc_y(S) - merc_y(N))
        lat = np.degrees(2 * np.arctan(np.exp(yy)) - math.pi / 2)
        lon = W + (np.arange(w) + 0.5) / w * (E - W)
        LON, LAT = np.meshgrid(lon, lat)
        c = np.asarray(coordinates_to_cells(LAT.ravel(), LON.ravel(), hx["res"])).astype(np.uint64)
        i = np.clip(np.searchsorted(hx["cells"], c), 0, len(hx["cells"]) - 1)
        return np.where(hx["cells"][i] == c, i, -1).astype(np.int32).reshape(h, w)

    def carved(hx, fit, depth, carve):
        """Elevation minus a small amount at the outlines: depth meters at the view's 98th percentile
        boundary, proportionally less on weaker ones, at most 1.5 x depth."""
        if not carve or depth <= 0:
            return hx["elev"]
        return hx["elev"] - depth * np.clip(hx["edge"] / hx["edge98"], 0, 1.5)

    def render(hx, idx, fit, stops, stretch, depth, carve):
        """RGBA PNG: each image pixel takes its hexagon's carved elevation on the palette (stops low to high)."""
        v = carved(hx, fit, depth, carve)
        if stretch == "equalized":
            u = np.interp(v, fit["quant"], np.linspace(0, 1, len(fit["quant"])))
        else:
            u = np.clip((v - fit["lo"]) / (fit["hi"] - fit["lo"]), 0, 1)
        pal = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in stops], np.float32)
        x = u * (len(pal) - 1)
        k = np.minimum(x.astype(int), len(pal) - 2)
        f = (x - k)[:, None]
        rgb = (pal[k] * (1 - f) + pal[k + 1] * f).astype(np.uint8)
        out = np.zeros(idx.shape + (4,), np.uint8)
        m = idx >= 0
        out[m, :3] = rgb[idx[m]]
        out[m, 3] = 255
        buf = io.BytesIO()
        Image.fromarray(out, "RGBA").save(buf, "PNG", compress_level=1)
        return buf.getvalue()

    def fit_from(hx):
        """The held ramp: the hexagons' elevation range and quantiles, and the boundary strength's 98th percentile."""
        e = hx["elev"][hx["n"] >= 2] if (hx["n"] >= 2).sum() > 100 else hx["elev"]
        lo, hi = (float(x) for x in np.percentile(e, [2, 98]))
        quant = np.maximum.accumulate(np.percentile(e, np.linspace(2, 98, 33)) + np.arange(33) * 1e-3)
        return {"lo": lo, "hi": max(hi, lo + 1), "quant": quant}

    return boundaries, fit_from, hexagons, pixel_index, render, res_for


@app.cell
def _(
    AEF_MAX_PX,
    AEF_RES,
    aef_coarse,
    aef_window,
    asyncio,
    boundaries,
    dem_at,
    fit_from,
    hexagons,
    math,
    np,
    pixel_index,
    render,
    res_for,
    time,
):
    # the last view's parts ("coarse" over the whole view, "fine" at 10 m near the center): hexagons and
    # each image pixel's hexagon, kept to recolor (palette, depth, stretch) without a read. FIT is held
    LAST = {}
    FIT = {}
    # the latest palette, stretch, depth and grooves: a read renders with these when it finishes, not with
    # what was set when it started (pressing R mid-read left two colorings on the map)
    STYLE = {}

    def _coords(grid):
        W, N, E, S = grid["edges"]
        return [[W, N], [E, N], [E, S], [W, S]]

    def _dem_z(res_m, lat):
        return int(min(13, max(1, round(math.log2(40075016 * math.cos(math.radians(lat)) / (512 * res_m))))))

    def _fit_meta():
        f = FIT["ramp"]
        return {"lo": f["lo"], "hi": f["hi"], "quant": f["quant"].tolist()}

    async def view(box, near, fine_ok, zoom, scale, years, style, emit):
        """Read the whole view at ONE resolution and emit(meta, [PNG]): the 10 m mosaic when the view fits
        AEF_MAX_PX pixels a year, else the single overview level that fits it (20 m at the finest). One
        read, one hexagon size for everything on screen; the ramp is fitted to it when none is held."""
        t0 = time.time()
        loop = asyncio.get_running_loop()
        STYLE["cur"] = style
        if FIT.get("years") != tuple(years):
            FIT.clear()
            FIT["years"] = tuple(years)
        LAST.clear()
        W, S, E, N = box
        fine = ((E - W) / AEF_RES) * ((N - S) / AEF_RES) <= AEF_MAX_PX
        # the boundaries of each year, read one year at a time and averaged (a year's embeddings are freed
        # before the next is read); a pixel counts where every year has data
        acc, grid = None, None
        for y in years:
            if fine:
                got = await aef_window((W, S, E, N), y)
                if got is None:
                    emit({"part": "view", "note": "No AlphaEarth embeddings under this view"}, [])
                    return
                emb, (gW, gN, gE, gS) = got
                g = {"edges": (gW, gN, gE, gS), "dlon": AEF_RES, "dlat": AEF_RES, "lat_mid": (gN + gS) / 2}
            else:
                emb, g = await aef_coarse(box, y, AEF_MAX_PX, 20.0)
                if emb is None:
                    emit({"part": "view", "note": g}, [])
                    return
            e = await loop.run_in_executor(None, boundaries, emb)
            del emb
            if acc is None:
                acc, grid = e, g
            elif e.shape == acc.shape:
                acc = acc + e  # NaN where any year has none
            else:
                emit({"part": "view", "note": "The years' grids did not line up"}, [])
                return
        edge = acc / len(years)
        h, w = edge.shape
        gW, gN, _, _ = grid["edges"]
        res_m = grid["dlat"] * 110574
        lon = gW + (np.arange(w) + 0.5) * grid["dlon"]
        lat = gN - (np.arange(h) + 0.5) * grid["dlat"]
        elev = (await loop.run_in_executor(None, dem_at, lon, lat, _dem_z(res_m, grid["lat_mid"]))).astype(np.float32)
        res = res_for(res_m)
        hx = await loop.run_in_executor(None, hexagons, edge, elev, grid, res)
        idx = await loop.run_in_executor(None, pixel_index, hx, grid, scale)
        LAST["view"] = {"hx": hx, "idx": idx, "grid": grid}
        if "ramp" not in FIT:
            FIT["ramp"] = fit_from(hx)
        png = await loop.run_in_executor(None, render, hx, idx, FIT["ramp"], *STYLE["cur"])
        emit({"part": "view", "coords": _coords(grid), "res_m": round(res_m), "res": res, "hexes": int(len(hx["cells"])),
              "secs": round(time.time() - t0, 1), "years": list(years), "fit": _fit_meta()}, [png])

    def recolor(style, refit):
        """The last view in a new palette, stretch or depth (refit first when asked): (fit, [(part, PNG)]).
        The style is kept for the read in progress, if any."""
        STYLE["cur"] = style
        if refit:
            part = LAST.get("view")
            if part is None:
                return None, []
            FIT["ramp"] = fit_from(part["hx"])
        if "ramp" not in FIT:
            return None, []
        return _fit_meta(), [(k, render(v["hx"], v["idx"], FIT["ramp"], *style)) for k, v in list(LAST.items())]

    return recolor, view


@app.cell
def _(AEF_MIN_ZOOM, AEF_YEARS, anywidget, asyncio, traitlets):
    # the plain ramp's choices, as in relief.py: every ramp low to high (dark low, light high); Reverse flips.
    # CartoColor 5.0.2 (7 steps) and matplotlib (9 steps), all luminance-monotonic, none with a red-green pair
    PALETTES = {
        "Emrld": ["#074050", "#105965", "#217a79", "#4c9b82", "#6cc08b", "#97e196", "#d3f2a3"],
        "BluYl": ["#045275", "#00718b", "#089099", "#46aea0", "#7ccba2", "#b7e6a5", "#f7feae"],
        "DarkMint": ["#123f5a", "#235d72", "#3a7c89", "#559c9e", "#7bbcb0", "#a5dbc2", "#d2fbd4"],
        "TealGrn": ["#257d98", "#2c98a0", "#38b2a3", "#4cc8a3", "#67dba5", "#89e8ac", "#b0f2bc"],
        "Teal": ["#2a5674", "#3b738f", "#4f90a6", "#68abb8", "#85c4c9", "#a8dbd9", "#d1eeea"],
        "Mint": ["#0d585f", "#287274", "#448c8a", "#63a6a0", "#89c0b6", "#b4d9cc", "#e4f1e1"],
        "Purp": ["#63589f", "#826dba", "#9f82ce", "#b998dd", "#d1afe8", "#e4c7f1", "#f3e0f7"],
        "viridis": ["#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c", "#28ae80", "#5ec962", "#addc30", "#fde725"],
        "cividis": ["#00224e", "#1a386f", "#434e6c", "#61656f", "#7d7c78", "#9b9476", "#bcae6c", "#dec958", "#fee838"],
        "magma": ["#000004", "#1d1147", "#51127c", "#832681", "#b73779", "#e75263", "#fc8961", "#fec488", "#fcfdbf"],
        "inferno": ["#000004", "#210c4a", "#57106e", "#8a226a", "#bc3754", "#e45a31", "#f98e09", "#f9cb35", "#fcffa4"],
        "gist_heat": ["#000000", "#300000", "#600000", "#900000", "#c00100", "#f04100", "#ff8103", "#ffc183", "#ffffff"],
        "YlGnBu": ["#081d58", "#243392", "#225da8", "#1d90c0", "#40b5c4", "#7ecdbb", "#c6e9b4", "#edf8b1", "#ffffd9"],
        "Greens": ["#00441b", "#006c2c", "#228a44", "#40aa5d", "#73c476", "#a0d99b", "#c7e9c0", "#e5f5e0", "#f7fcf5"],
        "Blues": ["#08306b", "#08509b", "#2070b4", "#4191c6", "#6aaed6", "#9dcae1", "#c6dbef", "#deebf7", "#f7fbff"],
        "Greys": ["#000000", "#252525", "#525252", "#737373", "#969696", "#bdbdbd", "#d9d9d9", "#f0f0f0", "#ffffff"],
    }

    class CarveMap(anywidget.AnyWidget):
        _esm = r"""
        import * as maplibregl from "https://cdn.jsdelivr.net/npm/maplibre-gl@6.13.0/dist/maplibre-gl.mjs";

        const DEM_URL = "https://tiles.mapterhorn.com/{z}/{x}/{y}.webp";
        const OFM_STYLE = "https://tiles.openfreemap.org/styles/dark";
        const GLYPHS = "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf";
        const PHOTON = "https://photon.komoot.io/api/";

        function render({model, el}) {
          const css = document.createElement("link");
          css.rel = "stylesheet"; css.href = "https://cdn.jsdelivr.net/npm/maplibre-gl@6.13.0/dist/maplibre-gl.css";
          el.appendChild(css);
          el.classList.add("ar");
          // always fills the window: the notebook's other cells and marimo's page menu are hidden
          if (!document.getElementById("ar-fit")) {
            const m = document.createElement("style"); m.id = "ar-fit";
            m.textContent = "body.ar-on .marimo-cell:not(.ar-host){visibility:hidden} body.ar-on div.fixed.top-0.right-0.z-50.m-4{display:none !important}";
            document.head.appendChild(m);
          }
          const fit = () => {
            for (let e = el; e; e = e.parentElement || (e.getRootNode && e.getRootNode().host)) if (e.classList && e.classList.contains("marimo-cell")) { e.classList.add("ar-host"); break; }
            document.body.classList.add("ar-on");
          };
          fit(); requestAnimationFrame(fit);
          const mapEl = document.createElement("div"); mapEl.className = "ar-map"; el.appendChild(mapEl);
          const set = (k, v) => { model.set(k, v); model.save_changes(); };

          // ---- the panel: fixed size, every line a fixed height
          const panel = document.createElement("div");
          panel.className = "ar-panel";
          panel.innerHTML = `
            <div class="ar-search"><input class="ar-find" type="search" placeholder="Find a place" autocomplete="off" spellcheck="false"><kbd>/</kbd><div class="ar-hits"></div></div>
            <div class="ar-top"><span class="ar-head"></span><button class="ar-fold" title="Fold the panel (\\ hides it)">&#8722;</button></div>
            <div class="ar-sub"></div>
            <div class="ar-bar"></div>
            <div class="ar-ends"><span class="ar-lo"></span><span class="ar-hi"></span></div>
            <div class="ar-body">
              <div class="ar-grid">
                <span>Years</span><span class="at-win"></span>
                <span>Grooves</span><div class="ar-seg"><button data-c="1">Carved</button><button data-c="0">Plain</button><kbd>A</kbd></div>
                <span>Groove</span><div class="ar-two"><input type="range" class="ar-depth" min="0" max="50" step="1"><span class="ar-dv"></span></div>
                <span></span><div class="ar-two"><label class="ar-chk"><input type="checkbox" class="ar-rev"> Reverse <kbd>R</kbd></label><button class="ar-btn ar-fitb">Fit <kbd>S</kbd></button></div>
                <span>Colors</span><select class="ar-pal"></select>
                <span>Stretch</span><select class="ar-stretch"><option value="linear">Linear</option><option value="equalized">Equalized</option></select>
                <span>Shading</span><input type="range" class="ar-sh" min="0" max="1" step="0.05">
                <span></span><label class="ar-chk"><input type="checkbox" class="ar-3d"> 3D <kbd>T</kbd></label>
                <span>Height</span><input type="range" class="ar-ex" min="1" max="4" step="0.25">
              </div>
            </div>
            <div class="ar-stat"></div>
            <div class="ar-keys">Arrows pan, Shift + arrows turn and tilt. L place names, F full screen, \\ hides this panel</div>`;
          el.appendChild(panel);
          // the busy pill, bottom center over the map, while a read or a recolor runs (as the built-up notebooks' status)
          const busyEl = document.createElement("div");
          busyEl.className = "ar-busy";
          busyEl.innerHTML = `<span class="t"></span><span class="bar"><i></i></span>`;
          el.appendChild(busyEl);
          const busy = (t) => { busyEl.querySelector(".t").textContent = t || ""; busyEl.classList.toggle("on", !!t); };
          const $ = (s) => panel.querySelector(s);
          // a clicked button gives up focus, so a later key never draws a focus ring on it
          panel.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) b.blur(); });

          // ---- the map: Mapterhorn elevation; the plain ramp under the carved hexagons, hillshade over
          const demSrc = (z) => ({type: "raster-dem", tiles: [DEM_URL], tileSize: 512, encoding: "terrarium", maxzoom: z});
          const v0 = model.get("view");
          const map = new maplibregl.Map({
            container: mapEl,
            style: {
              version: 8, glyphs: GLYPHS,
              sources: {omt: {type: "vector", url: "https://tiles.openfreemap.org/planet"}, dem: demSrc(14), "dem-terrain": demSrc(13)},
              layers: [
                {id: "bg", type: "background", paint: {"background-color": "#0b0f14"}},
                {id: "relief", type: "color-relief", source: "dem", paint: {"color-relief-color": ["interpolate", ["linear"], ["elevation"], 0, "#1b2530", 4000, "#9fb0c0"]}},
                {id: "fine-slot", type: "background", layout: {visibility: "none"}, paint: {"background-opacity": 0}},
                {id: "shade", type: "hillshade", source: "dem", paint: {}},
              ],
              sky: {"sky-color": "#0b0f14", "horizon-color": "#1b2530", "fog-color": "#0b0f14"},
            },
            center: [v0.lon, v0.lat], zoom: v0.zoom, maxPitch: 80, attributionControl: {compact: true},
          });
          map.addControl(new maplibregl.NavigationControl({visualizePitch: true}), "top-left");
          // ready once loaded, for good (isStyleLoaded() is false whenever tiles are loading); the map takes
          // no keys of its own and shows no focus ring (the notebook's keys below do the moving)
          let ready = false;
          map.once("load", () => { ready = true; });
          map.keyboard.disable();

          // ---- the ramp: the palette over the fitted range (held until S), shared by the legend, the
          // smooth fallback under the hexagons and the kernel's images
          let ramp = null;  // {lo, hi, quant}
          const paletteStops = () => { const c = model.get("palettes")[model.get("palette")].slice(); return model.get("reverse") ? c.reverse() : c; };
          const plainStops = () => {
            if (!ramp) return null;
            const cs = paletteStops(), n = cs.length;
            const q = ramp.quant || [ramp.lo, ramp.hi];
            const at = (u) => { const x = u * (q.length - 1), i = Math.min(q.length - 2, Math.floor(x)); return q[i] + (q[i + 1] - q[i]) * (x - i); };
            return cs.map((col, i) => [model.get("stretch") === "equalized" ? at(i / (n - 1)) : ramp.lo + (ramp.hi - ramp.lo) * i / (n - 1), col]);
          };
          function paintPanel() {
            const carve = model.get("carve");
            $(".ar-head").textContent = carve ? "Elevation, carved by AlphaEarth" : "Elevation, plain hexagons";
            $(".ar-sub").textContent = carve ? "Grooves: where the ground changes, from AlphaEarth" : `${model.get("palette")}, no grooves`;
            panel.querySelectorAll(".ar-seg button").forEach((b) => b.classList.toggle("on", (b.dataset.c === "1") === carve));
            const stops = plainStops();
            if (!stops) return;
            const lo = stops[0][0], hi = stops[stops.length - 1][0];
            $(".ar-bar").style.background = `linear-gradient(90deg, ${stops.map(([v, c]) => `${c} ${(100 * (v - lo) / Math.max(1, hi - lo)).toFixed(1)}%`).join(",")})`;
            $(".ar-lo").textContent = `${Math.round(ramp.lo).toLocaleString()} m`;
            $(".ar-hi").textContent = `${Math.round(ramp.hi).toLocaleString()} m`;
          }
          function paintMap() {
            if (!ready) return;
            const ps = plainStops();
            if (ps) {
              const e = ["interpolate", ["linear"], ["elevation"]];
              for (const [v, c] of ps) e.push(v, c);
              map.setPaintProperty("relief", "color-relief-color", e);
            }
            for (const part of ["view"]) if (map.getLayer("aef-" + part)) map.setLayoutProperty("aef-" + part, "visibility", metas[part] ? "visible" : "none");
            // the hillshade's shadows take the palette's dark end
            const pal = model.get("palettes")[model.get("palette")], dark = model.get("reverse") ? pal[pal.length - 1] : pal[0];
            const d = [1, 3, 5].map((i) => parseInt(dark.slice(i, i + 2), 16)).join(","), s = model.get("shade");
            const sp = {
              "hillshade-method": "multidirectional",
              "hillshade-illumination-direction": [270, 315, 0, 45], "hillshade-illumination-altitude": [35, 35, 35, 35],
              "hillshade-exaggeration": 0.35 + 0.5 * s,
              "hillshade-shadow-color": `rgba(${d},${(0.85 * s).toFixed(3)})`,
              "hillshade-highlight-color": `rgba(255,255,255,${(0.3 * s).toFixed(3)})`,
              "hillshade-accent-color": `rgba(${d},${(0.4 * s).toFixed(3)})`,
            };
            for (const [k, v] of Object.entries(sp)) map.setPaintProperty("shade", k, v);
          }

          // ---- reads: everything in view, the 10 m near ground over the overviews
          const AEF_MIN_ZOOM = model.get("aef_min_zoom");
          let seq = 0, readT = null, held = {};
          const metas = {}, urls = {}, placed = {};  // metas: the images on screen; placed: where each part's last image sits
          const stat = (t) => { $(".ar-stat").textContent = t; };
          const style = () => ({stops: paletteStops(), stretch: model.get("stretch"), depth: model.get("depth"), carve: model.get("carve")});
          // the images are drawn at about screen resolution: image pixels per degree of longitude
          // (in 3D the view reaches four screens out: the image keeps about the near ground's density, capped in the kernel)
          const scale = () => { const n = nearBox(); return Math.min(2, window.devicePixelRatio || 1) * mapEl.clientWidth / Math.max(1e-6, n[2] - n[0]); };
          function nearBox() {
            const c = map.getCenter(), hw = 360 / 2 ** map.getZoom() * (mapEl.clientWidth / 512) / 2;
            const hh = hw * Math.cos(c.lat * Math.PI / 180) * mapEl.clientHeight / mapEl.clientWidth;
            return [c.lng - hw, c.lat - hh, c.lng + hw, c.lat + hh];
          }
          function viewBox() {  // what the map sees; in 3D at most four screens out from the center
            const b = map.getBounds(), n = nearBox();
            let box = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
            if (map.getPitch() > 20) {
              const cx = (n[0] + n[2]) / 2, cy = (n[1] + n[3]) / 2, rx = 4 * (n[2] - n[0]), ry = 4 * (n[3] - n[1]);
              box = [Math.max(box[0], cx - rx), Math.max(box[1], cy - ry), Math.min(box[2], cx + rx), Math.min(box[3], cy + ry)];
            }
            return [Math.max(-180, box[0]), Math.max(-84, box[1]), Math.min(180, box[2]), Math.min(84, box[3])];
          }
          let readBusy = false;
          function askRead() {
            readBusy = true;
            seq += 1;
            held = {};
            const fineOk = map.getZoom() >= AEF_MIN_ZOOM;
            stat("Reading AlphaEarth for everything in view");
            busy("Reading AlphaEarth for this view");
            model.send({kind: "read", seq, box: viewBox(), near: viewBox(), fine_ok: fineOk, window: model.get("window"),
                        zoom: map.getZoom(), scale: scale(), style: style()});
          }
          const readSoon = () => { clearTimeout(readT); readT = setTimeout(askRead, 400); };
          let recolorT = null;
          function askRecolor(refit) {
            if (!Object.keys(placed).length) return;
            if (!held || !Object.keys(held).length) busy(refit ? "Fitting the colors to this view" : "Redrawing");
            clearTimeout(recolorT); recolorT = setTimeout(() => busy(""), 20000);  // never stuck on
            model.send({kind: "recolor", seq, refit: !!refit, style: style()});
          }
          // out with the old: when a setting changes, the images in the old setting come off at once (the
          // plain ramp underneath shows meanwhile), and the new ones go on when they arrive, never a mix
          function clearImages() {
            for (const part of Object.keys(metas)) { if (map.getLayer("aef-" + part)) map.setLayoutProperty("aef-" + part, "visibility", "none"); delete metas[part]; }
          }
          function putImage(part, buf, coords) {
            const b = buf.buffer ? new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength) : new Uint8Array(buf);
            const url = URL.createObjectURL(new Blob([b], {type: "image/png"}));
            const id = "aef-" + part, old = urls[id];
            urls[id] = url;
            const src = map.getSource(id);
            if (src) src.updateImage({url, coordinates: coords});
            else {
              map.addSource(id, {type: "image", url, coordinates: coords});
              map.addLayer({id, type: "raster", source: id, paint: {"raster-fade-duration": 0, "raster-resampling": "nearest"}}, part === "coarse" ? "fine-slot" : "shade");
            }
            if (old) setTimeout(() => URL.revokeObjectURL(old), 4000);
          }
          function say() {
            const v = metas.view;
            if (v) stat(`AlphaEarth ${v.years[0]} to ${v.years[v.years.length - 1]} averaged, the whole view at ${v.res_m} m in res ${v.res} hexagons, ${v.secs} s`);
          }
          model.on("msg:custom", (msg, buffers) => {
            if (!msg) return;
            if (msg.kind === "recolor") {
              if (msg.fit) { ramp = msg.fit; paintPanel(); paintMap(); }
              if (msg.part && placed[msg.part] && buffers && buffers.length) {
                putImage(msg.part, buffers[0], placed[msg.part].coords);
                metas[msg.part] = placed[msg.part];
                paintMap();
                if (!readBusy) busy("");
              }
              return;
            }
            // a view's parts are held until all of them are in, then swapped in together: the near
            // ground landing seconds before the overview showed as a rectangle over the plain ramp
            if (msg.kind === "read-done") {
              if (msg.seq !== seq) return;
              readBusy = false; busy("");
              for (const [part, m] of Object.entries(held)) { putImage(part, m.buf, m.meta.coords); metas[part] = m.meta; placed[part] = m.meta; }
              for (const part of Object.keys(metas)) if (!held[part]) { if (map.getLayer("aef-" + part)) map.setLayoutProperty("aef-" + part, "visibility", "none"); delete metas[part]; }
              held = {};
              paintMap(); say();
              return;
            }
            if (msg.kind !== "read" || msg.seq !== seq) return;
            if (msg.err) { readBusy = false; busy(""); stat(`The read failed: ${msg.err}`); return; }
            if (msg.note) { readBusy = false; busy(""); stat(msg.note); return; }
            if (msg.fit) { ramp = msg.fit; paintPanel(); paintMap(); }
            if (buffers && buffers.length) held[msg.part] = {meta: msg, buf: buffers[0]};
          });

          // ---- place names (L): towns, water and peaks from OpenFreeMap's dark style
          let labelIds = [];
          async function addLabels() {
            let layers = [];
            try {
              const st = await (await fetch(OFM_STYLE)).json();
              layers = st.layers.filter((l) => l.type === "symbol" && ["place", "water_name", "waterway"].includes(l["source-layer"])).map((l) => ({...l, id: "label-" + l.id, source: "omt"}));
            } catch (e) {}
            layers.push({
              id: "label-peaks", type: "symbol", source: "omt", "source-layer": "mountain_peak", minzoom: 9, filter: ["has", "name"],
              layout: {"text-field": ["case", ["has", "ele"], ["concat", ["get", "name"], "\n", ["to-string", ["get", "ele"]], " m"], ["get", "name"]], "text-font": ["Noto Sans Italic"], "text-size": 11, "text-offset": [0, 0.6], "text-anchor": "top"},
              paint: {"text-color": "#e6edf3", "text-halo-color": "rgba(11,15,20,0.85)", "text-halo-width": 1.2},
            });
            labelIds = layers.map((l) => l.id);
            for (const l of layers) if (!map.getLayer(l.id)) { try { map.addLayer(l); } catch (e) {} }
            showLabels();
          }
          const showLabels = () => { for (const id of labelIds) if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", model.get("labels") ? "visible" : "none"); };

          // ---- search (/): Photon, as in the pair notebook
          const gc = $(".ar-find"), gcList = $(".ar-hits");
          let gcHits = [], gcSel = -1, gcTimer = null, gcSeq = 0;
          const hitName = (f) => { const q = f.properties || {}; return [q.name, q.street && !q.name ? q.street : null, q.city && q.city !== q.name ? q.city : null, q.county && q.county !== q.city && q.county !== q.name ? q.county : null, q.state, q.country].filter((x) => x).join(", "); };
          const hitKind = (f) => { const q = f.properties || {}; return [q.osm_value, q.type].filter((x) => x && x !== "yes").join(", "); };
          const gcHide = () => { gcList.style.display = "none"; gcList.replaceChildren(); gcSel = -1; };
          const gcShow = () => {
            gcList.replaceChildren();
            if (!gcHits.length) { gcHide(); return; }
            gcHits.forEach((f, i) => {
              const row = document.createElement("div"); row.className = "ar-hit" + (i === gcSel ? " on" : "");
              const nm = document.createElement("div"); nm.textContent = hitName(f);
              const kd = document.createElement("div"); kd.className = "k"; kd.textContent = hitKind(f);
              row.append(nm, kd);
              row.onmousedown = (e) => { e.preventDefault(); gcFly(f); };
              row.onmouseenter = () => { gcSel = i; gcShow(); };
              gcList.appendChild(row);
            });
            gcList.style.display = "block";
          };
          const gcAsk = async () => {
            const q = gc.value.trim();
            if (q.length < 2) { gcHits = []; gcHide(); return; }
            const s = ++gcSeq, c = map.getCenter();
            try {
              const data = await (await fetch(PHOTON + "?" + new URLSearchParams({q, limit: "6", lang: "en", lon: c.lng.toFixed(4), lat: c.lat.toFixed(4)}))).json();
              if (s !== gcSeq) return;
              gcHits = (data.features || []).filter((f) => f.geometry && f.geometry.coordinates);
              gcSel = gcHits.length ? 0 : -1; gcShow();
            } catch (e) { if (s === gcSeq) stat("Search failed: " + e.message); }
          };
          const gcFly = (f) => {
            const [lon, lat] = f.geometry.coordinates, ext = (f.properties || {}).extent;
            let zoom = 12.5;
            if (ext && ext.length === 4) { const span = Math.max(Math.abs(ext[2] - ext[0]), Math.abs(ext[1] - ext[3]) * 2, 0.01); zoom = Math.log2(360 * ((mapEl.clientWidth || 700) / 512) / span) - 0.3; }
            gc.value = hitName(f); gcHits = []; gcHide(); gc.blur();
            map.flyTo({center: [lon, lat], zoom: Math.max(4, Math.min(14, zoom)), duration: 2000});
          };
          gc.addEventListener("input", () => { clearTimeout(gcTimer); gcTimer = setTimeout(gcAsk, 250); });
          gc.addEventListener("focus", () => { if (gcHits.length) gcShow(); });
          gc.addEventListener("blur", () => setTimeout(gcHide, 120));
          gc.addEventListener("keydown", (e) => {
            e.stopPropagation();
            if (e.key === "ArrowDown" && gcHits.length) { gcSel = (gcSel + 1) % gcHits.length; gcShow(); e.preventDefault(); }
            else if (e.key === "ArrowUp" && gcHits.length) { gcSel = (gcSel - 1 + gcHits.length) % gcHits.length; gcShow(); e.preventDefault(); }
            else if (e.key === "Enter") { e.preventDefault(); if (gcHits.length) gcFly(gcHits[Math.max(0, gcSel)]); else { clearTimeout(gcTimer); gcAsk().then(() => { if (gcHits.length) gcFly(gcHits[0]); else stat("No place matches " + gc.value.trim()); }); } }
            else if (e.key === "Escape") { gcHide(); gc.blur(); }
          });

          // ---- fold and hide the panel (kept in this browser)
          const store = {get: (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} }};
          const setFold = (on) => { panel.classList.toggle("folded", on); store.set("ar-fold", on ? "1" : "0"); };
          const setGone = (on) => { panel.classList.toggle("gone", on); store.set("ar-gone", on ? "1" : "0"); };
          setFold(store.get("ar-fold") === "1"); setGone(store.get("ar-gone") === "1");
          $(".ar-fold").addEventListener("click", () => setFold(!panel.classList.contains("folded")));

          // ---- the years read and averaged: on_the_fly's range, two thumbs on one track; it rereads on release
          const AEF_YEARS = model.get("aef_years");
          const el_ = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html) e.innerHTML = html; return e; };
          const win = $(".at-win");
          const wTrk = el_("span", "trk"), wSpn = el_("span", "spn"), wTks = el_("span", "tks");
          for (const y of AEF_YEARS) wTks.appendChild(el_("span", "", `<i>’${String(y).slice(-2)}</i>`));
          const mkR = () => { const r = el_("input"); r.type = "range"; r.min = 0; r.max = AEF_YEARS.length - 1; r.step = 1; r.title = "the years read and averaged: drag either end"; return r; };
          const rFrom = mkR(), rTo = mkR();
          win.append(wTrk, wSpn, wTks, rFrom, rTo);
          let [wy0, wy1] = model.get("window").split("-").map(Number);
          function styleWin() {
            const i0 = Math.max(0, AEF_YEARS.indexOf(wy0)), i1 = Math.max(0, AEF_YEARS.indexOf(wy1)), n = Math.max(1, AEF_YEARS.length - 1);
            rFrom.value = i0; rTo.value = i1;
            rFrom.style.zIndex = i0 === n ? 3 : 2; rTo.style.zIndex = i1 === 0 ? 3 : 2;
            const usable = (win.clientWidth || 170) - 16;
            wSpn.style.left = (8 + usable * i0 / n) + "px"; wSpn.style.width = (usable * (i1 - i0) / n) + "px";
          }
          const onDrag = (which) => {
            let a = Number(rFrom.value), b = Number(rTo.value);
            if (a > b) { if (which === "from") a = b; else b = a; }
            wy0 = AEF_YEARS[a]; wy1 = AEF_YEARS[b]; styleWin();
          };
          const winRelease = () => { set("window", `${wy0}-${wy1}`); rFrom.blur(); rTo.blur(); };
          rFrom.addEventListener("input", () => onDrag("from")); rTo.addEventListener("input", () => onDrag("to"));
          rFrom.addEventListener("change", winRelease); rTo.addEventListener("change", winRelease);
          try { new ResizeObserver(styleWin).observe(win); } catch (e) {}
          for (const n of Object.keys(model.get("palettes"))) { const o = document.createElement("option"); o.value = o.textContent = n; $(".ar-pal").appendChild(o); }

          // ---- controls
          const sync = () => {
            [wy0, wy1] = model.get("window").split("-").map(Number); styleWin();
            $(".ar-pal").value = model.get("palette");
            $(".ar-stretch").value = model.get("stretch");
            $(".ar-rev").checked = model.get("reverse");
            $(".ar-depth").value = model.get("depth");
            $(".ar-dv").textContent = `${Math.round(model.get("depth"))} m`;
            $(".ar-sh").value = model.get("shade");
            $(".ar-3d").checked = model.get("terrain");
            $(".ar-ex").value = model.get("exaggeration");
          };
          sync(); paintPanel();
          panel.querySelectorAll(".ar-seg button").forEach((b) => b.addEventListener("click", () => set("carve", b.dataset.c === "1")));
          $(".ar-rev").addEventListener("change", (e) => set("reverse", e.target.checked));
          $(".ar-pal").addEventListener("change", (e) => set("palette", e.target.value));
          $(".ar-stretch").addEventListener("change", (e) => set("stretch", e.target.value));
          $(".ar-depth").addEventListener("input", (e) => { $(".ar-dv").textContent = `${e.target.value} m`; });
          $(".ar-depth").addEventListener("change", (e) => set("depth", parseFloat(e.target.value)));
          $(".ar-sh").addEventListener("input", (e) => set("shade", parseFloat(e.target.value)));
          $(".ar-3d").addEventListener("change", (e) => set("terrain", e.target.checked));
          $(".ar-ex").addEventListener("input", (e) => set("exaggeration", parseFloat(e.target.value)));
          $(".ar-fitb").addEventListener("click", () => { clearImages(); paintMap(); askRecolor(true); });
          const applyTerrain = (ease) => {
            if (!ready) return;
            map.setTerrain(model.get("terrain") ? {source: "dem-terrain", exaggeration: model.get("exaggeration")} : null);
            if (!ease) return;
            if (model.get("terrain")) { if (map.getPitch() < 10) map.easeTo({pitch: 60, duration: 700}); }
            else if (map.getPitch() > 0) map.easeTo({pitch: 0, bearing: 0, duration: 500});
          };
          model.on("change:carve", () => { clearImages(); paintPanel(); paintMap(); askRecolor(false); });
          model.on("change:palette", () => { clearImages(); sync(); paintPanel(); paintMap(); askRecolor(false); });
          model.on("change:stretch", () => { clearImages(); sync(); paintPanel(); paintMap(); askRecolor(false); });
          model.on("change:window", () => { clearImages(); sync(); paintMap(); askRead(); });
          model.on("change:reverse", () => { clearImages(); sync(); paintPanel(); paintMap(); askRecolor(false); });
          model.on("change:depth", () => { clearImages(); sync(); paintMap(); askRecolor(false); });
          model.on("change:shade", () => { sync(); paintMap(); });
          model.on("change:terrain", () => { sync(); applyTerrain(true); });
          model.on("change:exaggeration", () => { sync(); if (model.get("terrain")) applyTerrain(false); });
          model.on("change:labels", showLabels);

          // ---- keys. Arrows pan; Shift + arrows turn (left, right) and tilt (up, down), MapLibre's usual keys
          const typing = (t) => { const tag = t && t.tagName; return tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || (t && t.isContentEditable); };
          const onKey = (e) => {
            if (e.metaKey || e.ctrlKey || e.altKey || typing(e.target)) return;
            if (!el.matches(":hover")) return;
            const k = e.key.toLowerCase(), arrow = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]}[e.key];
            if (arrow && e.shiftKey) {
              if (arrow[0]) map.easeTo({bearing: map.getBearing() + 15 * arrow[0], duration: 250});
              else map.easeTo({pitch: Math.max(0, Math.min(80, map.getPitch() - 10 * arrow[1])), duration: 250});
            }
            else if (arrow) map.panBy([arrow[0] * 120, arrow[1] * 120], {duration: 250});
            else if (k === "s") { clearImages(); paintMap(); askRecolor(true); }
            else if (k === "r") set("reverse", !model.get("reverse"));
            else if (k === "a") set("carve", !model.get("carve"));
            else if (k === "t") set("terrain", !model.get("terrain"));
            else if (k === "l") set("labels", !model.get("labels"));
            else if (k === "f") { if (document.fullscreenElement) document.exitFullscreen(); else if (el.requestFullscreen) el.requestFullscreen(); }
            else if (e.key === "\\") setGone(!panel.classList.contains("gone"));
            else if (e.key === "/") { if (panel.classList.contains("gone")) setGone(false); if (panel.classList.contains("folded")) setFold(false); gc.focus(); gc.select(); }
            else return;
            e.preventDefault();
          };
          document.addEventListener("keydown", onKey);

          map.on("load", () => { applyTerrain(true); paintMap(); addLabels(); askRead(); });
          map.on("moveend", () => {
            const c = map.getCenter();
            set("view", {lon: +c.lng.toFixed(5), lat: +c.lat.toFixed(5), zoom: +map.getZoom().toFixed(2)});
            readSoon();
          });
          const ro = new ResizeObserver(() => map.resize());
          ro.observe(el);

          window.__ab = {map, ramp: () => ramp, parts: () => ({...metas})};
          return () => {
            document.removeEventListener("keydown", onKey);
            document.body.classList.remove("ar-on");
            ro.disconnect();
            clearTimeout(readT);
            for (const u of Object.values(urls)) URL.revokeObjectURL(u);
            map.remove();
          };
        }
        export default {render};
        """
        _css = r"""
        .ar { position: fixed; inset: 0; z-index: 9999; overflow: hidden; background: #0b0f14; color: #e6edf3;
              font: 13px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif; }
        .ar:fullscreen { width: 100%; height: 100%; }
        .ar-map { position: absolute; inset: 0; }
        .ar .maplibregl-canvas:focus, .ar .maplibregl-canvas:focus-visible, .ar .maplibregl-canvas-container:focus { outline: none; }
        .ar-panel { position: absolute; top: 12px; right: 12px; width: 280px; padding: 12px 14px; border-radius: 10px;
                    background: rgba(13, 19, 26, 0.84); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
                    border: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35); }
        .ar-panel.gone { display: none; }
        .ar-panel.folded .ar-body, .ar-panel.folded .ar-keys, .ar-panel.folded .ar-search, .ar-panel.folded .ar-sub { display: none; }
        .ar-search { position: relative; display: flex; align-items: center; gap: 6px; margin-bottom: 10px; }
        .ar-find { flex: 1; min-width: 0; background: #16202a; color: #e6edf3; border: 1px solid #2a3846; border-radius: 6px; padding: 5px 8px; font: inherit; }
        .ar-find::placeholder { color: #7d8d9c; }
        .ar-hits { position: absolute; left: 0; right: 0; top: calc(100% + 4px); z-index: 2; display: none; background: #16202a;
                   border: 1px solid #2a3846; border-radius: 6px; box-shadow: 0 6px 18px rgba(0, 0, 0, 0.4); overflow: hidden; }
        .ar-hit { padding: 5px 9px; cursor: pointer; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; line-height: 1.3; }
        .ar-hit .k { font-size: 11px; color: #7d8d9c; }
        .ar-hit.on { background: rgba(70, 174, 160, 0.35); }
        .ar-top { display: flex; align-items: baseline; gap: 8px; height: 20px; }
        .ar-head { font-weight: 600; }
        .ar-fold { all: unset; cursor: pointer; margin-left: auto; color: #9fb0c0; font-size: 16px; line-height: 1; padding: 0 4px; }
        .ar-sub { color: #9fb0c0; font-size: 12px; height: 18px; margin-top: 2px; }
        .ar-bar { height: 12px; border-radius: 6px; margin-top: 6px; }
        .ar-ends { display: flex; justify-content: space-between; height: 18px; margin: 3px 0 8px; color: #9fb0c0; font-size: 12px; font-variant-numeric: tabular-nums; }
        .ar-grid { display: grid; grid-template-columns: 64px 1fr; align-items: center; gap: 7px 8px; color: #c9d4de; }
        .ar-grid > span { color: #9fb0c0; }
        .ar-grid input[type=range] { width: 100%; }
        .ar input[type=range], .ar input[type=checkbox] { accent-color: #46aea0; }
        .ar-seg { display: flex; align-items: center; gap: 2px; padding: 2px; border-radius: 7px; background: rgba(255, 255, 255, 0.06); }
        .ar-seg button { all: unset; cursor: pointer; flex: 1; text-align: center; padding: 4px 6px; border-radius: 5px; color: #9fb0c0; font-weight: 500; }
        .ar-seg button.on { background: rgba(255, 255, 255, 0.16); color: #e6edf3; }
        .ar-seg kbd { margin: 0 4px; }
        .at-win { position: relative; display: block; width: 100%; height: 30px; }
        .at-win input { position: absolute; left: 0; top: 0; width: 100%; height: 22px; margin: 0; background: none; pointer-events: none; -webkit-appearance: none; appearance: none; }
        .at-win input:focus { outline: none; }
        .at-win input::-webkit-slider-runnable-track { background: none; height: 22px; }
        .at-win input::-moz-range-track { background: none; height: 22px; }
        .at-win input::-webkit-slider-thumb { pointer-events: auto; -webkit-appearance: none; appearance: none; width: 14px; height: 14px; margin-top: 4px; border-radius: 50%; background: #e6edf3; border: 2px solid #15181b; box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.3); cursor: grab; }
        .at-win input::-moz-range-thumb { pointer-events: auto; width: 14px; height: 14px; border-radius: 50%; background: #e6edf3; border: 2px solid #15181b; cursor: grab; }
        .at-win .trk { position: absolute; left: 8px; right: 8px; top: 9px; height: 4px; background: rgba(255, 255, 255, 0.18); border-radius: 2px; }
        .at-win .spn { position: absolute; top: 9px; height: 4px; background: #e6edf3; border-radius: 2px; }
        .at-win .tks { position: absolute; left: 8px; right: 8px; top: 19px; display: flex; justify-content: space-between; font-size: 10px; color: #9fb0c0; line-height: 1; }
        .at-win .tks span { width: 0; display: flex; justify-content: center; }
        .at-win .tks i { font-style: normal; }
        .ar select { background: #16202a; color: #e6edf3; border: 1px solid #2a3846; border-radius: 6px; padding: 3px 6px; font: inherit; width: 100%; }
        .ar-chk { display: flex; align-items: center; gap: 6px; white-space: nowrap; }
        .ar-dv { min-width: 40px; text-align: right; color: #9fb0c0; font-variant-numeric: tabular-nums; }
        .ar-two input[type=range] { flex: 1; min-width: 0; }
        .ar-two { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
        .ar-btn { all: unset; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; padding: 3px 9px; border-radius: 6px; background: rgba(255, 255, 255, 0.08); font-weight: 500; }
        .ar-btn:hover { background: rgba(255, 255, 255, 0.14); }
        .ar-seg button:focus-visible, .ar-btn:focus-visible, .ar-fold:focus-visible, .ar input:focus-visible { outline: 2px solid #46aea0; outline-offset: 1px; }
        .ar-busy { position: absolute; left: 50%; bottom: 30px; transform: translateX(-50%); display: none; align-items: center; gap: 10px;
                   padding: 6px 12px; border-radius: 8px; background: rgba(13, 19, 26, 0.86); border: 1px solid rgba(255, 255, 255, 0.1);
                   box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35); font-size: 12.5px; color: #e6edf3; pointer-events: none; }
        .ar-busy.on { display: flex; }
        .ar-busy .bar { position: relative; width: 70px; height: 3px; border-radius: 2px; background: rgba(255, 255, 255, 0.15); overflow: hidden; }
        .ar-busy .bar i { position: absolute; top: 0; bottom: 0; width: 30%; left: -30%; background: #46aea0; border-radius: 2px; animation: ar-slide 1.1s ease-in-out infinite; }
        @keyframes ar-slide { to { left: 100%; } }
        @media (prefers-reduced-motion: reduce) { .ar-busy .bar i { animation: none; left: 0; width: 100%; } }
        .ar-stat { height: 32px; margin-top: 10px; color: #9fb0c0; font-size: 11.5px; line-height: 16px; overflow: hidden;
                   display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
        .ar-keys { margin-top: 4px; color: #7d8d9c; font-size: 11.5px; }
        .ar kbd { font: 11px ui-monospace, monospace; padding: 0 4px; border: 1px solid #3a4856; border-radius: 4px; color: #9fb0c0; }
        @media (max-width: 600px) { .ar-panel { left: 12px; right: 12px; width: auto; max-height: 55vh; overflow: auto; } }
        """
        palettes = traitlets.Dict(PALETTES).tag(sync=True)
        palette = traitlets.Unicode("Emrld").tag(sync=True)  # the plain ramp
        stretch = traitlets.Unicode("linear").tag(sync=True)  # "linear" or "equalized"
        aef_years = traitlets.List(list(AEF_YEARS)).tag(sync=True)
        window = traitlets.Unicode("2023-2025").tag(sync=True)  # the years read and averaged, "from-to"
        carve = traitlets.Bool(True).tag(sync=True)  # the grooves on, or plain hexagons (A)
        depth = traitlets.Float(5.0).tag(sync=True)  # meters taken off at the view's 98th percentile boundary (the Groove slider)
        reverse = traitlets.Bool(False).tag(sync=True)
        shade = traitlets.Float(0.15).tag(sync=True)  # low: the grooves are the relief here, as in the Denali map
        terrain = traitlets.Bool(False).tag(sync=True)
        exaggeration = traitlets.Float(1.5).tag(sync=True)
        labels = traitlets.Bool(True).tag(sync=True)
        aef_min_zoom = traitlets.Float(AEF_MIN_ZOOM).tag(sync=True)
        # Toutle River valley below Mount St. Helens
        view = traitlets.Dict({"lon": -122.43, "lat": 46.27, "zoom": 12.3}).tag(sync=True)

        def __init__(self, **kw):
            super().__init__(**kw)
            self.read_fn = None  # async (box, near, fine_ok, zoom, scale, years, style, emit)
            self.recolor_fn = None  # (style, refit) -> (fit or None, [(part, png)])
            self._task = None
            self.on_msg(self._on_custom)

        def _on_custom(self, widget, content, buffers):
            if not isinstance(content, dict):
                return
            st = content.get("style", {})
            style = (list(st.get("stops") or ["#000000", "#ffffff"]), str(st.get("stretch", "linear")), float(st.get("depth", 5.0)), bool(st.get("carve", True)))
            if content.get("kind") == "recolor":
                try:
                    fit, pngs = self.recolor_fn(style, bool(content.get("refit")))
                except Exception as e:
                    self.send({"kind": "read", "seq": content.get("seq"), "err": f"{type(e).__name__}: {e}"})
                    return
                if fit:
                    self.send({"kind": "recolor", "fit": fit})
                for part, png in pngs:
                    self.send({"kind": "recolor", "part": part}, buffers=[png])
            elif content.get("kind") == "read":
                if self._task is not None and not self._task.done():  # a newer view replaces the one being read
                    self._task.cancel()
                try:
                    self._task = asyncio.get_running_loop().create_task(self._read(content, style))
                except RuntimeError:
                    pass

        async def _read(self, c, style):
            seq = c.get("seq")

            def emit(meta, pngs):
                self.send({"kind": "read", "seq": seq, **meta}, buffers=pngs)

            try:
                y0, y1 = (int(v) for v in c["window"].split("-"))
                years = [y for y in self.aef_years if y0 <= y <= y1]
                await self.read_fn(tuple(c["box"]), tuple(c["near"]), bool(c["fine_ok"]), float(c["zoom"]), float(c["scale"]), years, style, emit)
            except asyncio.CancelledError:
                return
            except Exception as e:
                self.send({"kind": "read", "seq": seq, "err": f"{type(e).__name__}: {e}"})
                return
            self.send({"kind": "read-done", "seq": seq})

    return (CarveMap,)


@app.cell
def _(CarveMap, mo, recolor, view):
    carve_map = CarveMap()
    carve_map.read_fn = view
    carve_map.recolor_fn = recolor
    aef_boundaries = mo.ui.anywidget(carve_map)
    aef_boundaries
    return (aef_boundaries,)


if __name__ == "__main__":
    app.run()
