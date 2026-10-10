# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "datafusion>=54.0.0",
#     "xarray-sql>=0.3.3",
#     "xarray",
#     "zarr>=3.1",
#     "h3ronpy>=0.22.0",
#     "pyarrow>=25.0.0",
#     "obstore>=0.9.2",
#     "async-geotiff>=0.4",
#     "anywidget>=0.9",
#     "numpy",
#     "duckdb>=1.5.5",
#     "pyproj",
#     "pillow",
#     "pmtiles",
#     "mapbox-vector-tile",
#     "scipy",
#     "scikit-learn",
#     "shapely>=2",
#     "rasterio",
#     "requests",
#     "traitlets==5.16.1",
# ]
# ///


"""Earthwork on the fly: where the ground itself moved, from AlphaEarth, taught by 3DEP.

Earthwork is the chance that the ground was physically dug, filled or graded between the first and
last year of the window (2023 and 2025 to start, close to the Sentinel-2 imagery's 2022 to 2025; the
Years read slider reaches back to 2017), not that its surface looked different. The model is
earthwork_model.py's: a logistic regression on AlphaEarth's two years, taught where 3DEP lidar flew
the same ground twice since 2017 (eight building sites and three mines), the difference of the two
1 m DEMs being the truth of where earth moved more than half a meter, and taught what is NOT digging
by two wildfire burns, a coastal marsh and the water lying in both flights at every site (burn scars
and water otherwise read as earthwork). Scored on sites it never saw, it finds site grading far
better than plain AlphaEarth change where much else changed (Huntsville AP .55, plain change .09); at
mines it does no better than plain change. It finds digs from about half a hectare well and most
house-pad-sized ones not at all (10 m pixels). The DEM teaches; nothing on the map reads it.

Only the window's first and last year of AlphaEarth are read, folded to H3 one level finer than the
hexagons drawn. Every finer cell is scored and each hexagon takes its highest-scoring finer cell
(carry the peak), so one dig is not averaged away by the quiet ground around it. Viridis by the
score, on all ground. Hold space for Earth Genome's Sentinel-2, or P to pair it with the map, to see
what each hot spot is. Click a hexagon for its score; the place comes from Overture's divisions.

Copied from embeddings-on-the-fly's on_the_fly.py (the map, the H3 folds, the Sentinel-2 imagery,
search and place names) with everything else taken out: AEF Change, Overture and WSF layers, the land
cover readers and teachers, the shared built-up models.

Run: uv run marimo run earthwork.py --sandbox --watch (it fills the window; X or Esc gives the
notebook back)

Attribution: "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google
DeepMind" (CC BY 4.0). 3DEP 1 m DEMs by the U.S. Geological Survey (teaching only). Sentinel-2
yearly mosaics by Earth Genome (CC BY 4.0). Photon (komoot) over OpenStreetMap data (ODbL). Place
names from Overture Maps divisions: (c) OpenStreetMap contributors, Overture Maps Foundation (ODbL),
with geoBoundaries, Esri Community Maps contributors and LINZ (CC BY 4.0). Basemap by Carto.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="Earthwork", sql_output="native")


@app.cell
def _():
    import asyncio
    import itertools
    import json
    import math
    import os
    import re
    import tempfile
    import time
    import traceback
    import urllib.parse
    import urllib.request
    import zlib

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import xarray as xr
    import zarr
    import duckdb
    import marimo as mo
    import anywidget
    import traitlets

    from obstore.store import HTTPStore, S3Store
    from zarr.storage import ObjectStore
    from async_geotiff import GeoTIFF, Window
    from datafusion import udf
    from xarray_sql import XarrayContext
    from h3ronpy import change_resolution
    from h3ronpy.vector import coordinates_to_cells
    from pyproj import Transformer

    import io
    from PIL import Image

    # CPU work (the DataFusion folds, tile compositing and PNG encoding, the
    # footprint rasterizing) leaves the event loop for ONE small pool, so the
    # network reads (asyncio, US East to the us-west-2 buckets) keep flowing
    # while it runs and a small machine (molab) is not flooded: at most
    # CPU_WORKERS such jobs at once, the rest queue
    from concurrent.futures import ThreadPoolExecutor

    CPU_WORKERS = max(2, min(4, (os.cpu_count() or 2) - 1))
    _cpu_pool = ThreadPoolExecutor(CPU_WORKERS, thread_name_prefix="cpu")

    async def cpu(fn, *args):
        """fn(*args) on the CPU pool, awaited."""
        return await asyncio.get_running_loop().run_in_executor(_cpu_pool, lambda: fn(*args))

    # Source Cooperative through its own proxy, data.source.coop (served by Cloudflare, so the
    # people who host the data there pay no egress), never the S3 bucket behind it. The proxy
    # speaks the S3 API, listing included: path-style, each account a bucket
    SOURCE_COOP = "https://data.source.coop"

    def source_coop(path="", **client_options):
        """A store on data.source.coop: the whole proxy (paths "account/key") when path is
        empty, else one account under a prefix ("mindearth/wsf/World_WSF_...zarr")."""
        if not path:
            return HTTPStore.from_url(SOURCE_COOP, client_options=client_options or None)
        account, _, prefix = path.partition("/")
        return S3Store(account, endpoint=SOURCE_COOP, region="us-west-2", virtual_hosted_style_request=False,
                       skip_signature=True, prefix=prefix or None, client_options=client_options or None)

    return (
        GeoTIFF,
        Image,
        ObjectStore,
        S3Store,
        Transformer,
        Window,
        XarrayContext,
        anywidget,
        asyncio,
        change_resolution,
        coordinates_to_cells,
        cpu,
        duckdb,
        io,
        itertools,
        json,
        math,
        mo,
        np,
        os,
        pa,
        pq,
        re,
        source_coop,
        tempfile,
        time,
        traceback,
        traitlets,
        udf,
        urllib,
        xr,
        zarr,
        zlib,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Earthwork

    **What you are looking at.** The chance the ground itself was dug,
    filled or graded between the first and last year of the window, from
    AlphaEarth alone, by a small model taught on 3DEP repeat lidar. Viridis:
    dark unlikely, yellow likely. Hexagons from zoom 9; each shows its
    highest-scoring patch.

    | Key | Does |
    | --- | --- |
    | `space` (hold) | the Sentinel-2 imagery instead of the hexagons; the map still drags |
    | `P` | the pair: Sentinel-2 on the left, the map on the right, one camera |
    | scroll, space held | the imagery year |
    | `[` `]` | the imagery year, back and forward |
    | `B` | the imagery's first year (2022) or its latest (2025), back and forth |
    | `;` `'` | the imagery darker, brighter |
    | `-` `=` | the window's first year, earlier, later |
    | `_` `+` | the window's last year, earlier, later |
    | `L` | place names on the map, off and on |
    | `/` | search a place, or paste an H3 string |
    | `X` | fill the window, and back |
    | `Esc` | clear the searched outline, then close the about box, the menu, the card, then leave the full window |

    <small>Locally: `uv run marimo run earthwork.py --sandbox --watch`</small>
    """)
    return


@app.cell
def _(os, tempfile):
    # ---- constants ----------------------------------------------------------
    # The imagery is Earth Genome's yearly mosaic, 2022 to 2025; AlphaEarth
    # runs 2017 to 2025. Not all nine years by default. The window opened at 2021, the year before the imagery starts,
    # so every imagery year could be a change year ("the scroll should
    # include all years including 22"); it now opens at 2023 for a lighter
    # demo: 3 years read, not 5, so change years
    # 2024 and 2025. Widen it with the window control.
    S2_YEARS = (2022, 2023, 2024, 2025)
    AEF_YEARS_ALL = tuple(range(2017, 2026))
    AEF_FROM0, AEF_TO0 = 2023, 2025  # the window's ends are what Earthwork compares; the slider reaches back to 2017
    # the first hold opens on the first imagery year, the scroll goes forward
    # from there,
    # and every later hold opens where the last one left off ("it should
    # persist where i leave off")
    S2_YEAR0 = 2022
    S2_SCALE0 = 1.0

    # the zoom -> H3 ladder: res 8 at zoom 9, 9 at
    # 10.4, 10 at 11.8, 11 at 13.2, 12 at 14.6, 13 from 16 (res 13 is where
    # structures start to show; res 12 only roughly traces them). The budget
    # lets res 13 through for a zoom 16 view's box
    ZOOM0, PER_RES, BASE_RES = 6.2, 1.4, 6
    MIN_RES, MAX_RES = 5, 13
    CELL_BUDGET = 800_000
    MOSAIC_MIN_RES = 11
    AEF_LEVEL_FOR_RES = {5: 7, 6: 7, 7: 5, 8: 4, 9: 3, 10: 1}
    AEF_MAX_FILES = 2500

    S2_STAC = "https://stac.earthgenome.org/search"
    S2_COLLECTION = "sentinel2-yearly-mosaics"
    # where the yearly mosaic is nodata, the same year's temporal mosaic fills
    # the hole pixel by pixel (2022 and 2023 only on this STAC)
    S2_FILL_COLLECTION = "sentinel2-temporal-mosaics"
    # the mosaic pyramid ends at z9 (L5, 306 m); z7-8 are rendered from L5 by
    # decimation (slow, so no lower)
    S2_TILE_MIN_Z, S2_PYRAMID_Z, S2_TCI_MAX_Z = 7, 9, 14

    # every S3 read to us-west-2: a short timeout so a stalled request is
    # retried instead of waited on (US East Coast, 2026-09-24)
    S3_OPTS = {"timeout": "6s", "connect_timeout": "3s"}

    AEF_PREFIX = "tge-labs/aef-mosaic"
    AEF_RES, AEF_Y0, AEF_X0 = 8.983111749910169e-05, 83.68570533713473, -180.0
    AEF_NODATA = -128
    AEF_INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
    CACHE_DIR = os.path.join(tempfile.gettempdir(), "x-sql-marimo", "aef-lcms")

    # the place under a click: the Overture divisions PMTiles answer at once in
    # the browser (locality, county, region); then the whole ladder, locality
    # up to country with each country's own word for the level (local_type),
    # from Overture's divisions GeoParquet as Fused partitions it on Source
    # Cooperative (7 s cold, 1 to 3 s after)
    OV_DIV_PM = "https://overturemaps-extras-us-west-2.s3.us-west-2.amazonaws.com/tiles/2026-08-19.0/divisions.pmtiles"
    ADMIN_PQ = "s3://fused/overture/2026-05-20-0/theme=divisions"

    VIEW_W, VIEW_H = 700, 780
    # the box read around the view, x its width and height: 2 leaves half a
    # screen each side, so a pan of up to half a screen needs no new read
    #
    # only what is in view is read (memory): the box is the view itself; a pan reads again
    PAD = 1.0
    SETTLE = 0.35
    # hexagons from zoom 9, the plain basemap below
    HEX_ZOOM = 9.0
    LABELS_SLOT = "watername_ocean"
    RASTER_TILE = 256
    HOME = {"longitude": -97.455, "latitude": 30.537, "zoom": 13}  # Samsung's Taylor, Texas fab, graded from 2022

    # how long a still press takes to become a hold, and how far the pointer
    # may drift before it counts as a pan instead
    HOLD_MS, HOLD_SLOP_PX = 200, 5

    # the hexagons reach the browser as cell ids and are drawn there as
    # geometry. The zoom ladder below picks the READ res (which AlphaEarth
    # overview is read); two knobs, same download:
    # HEX_UP: the hexagons drawn are this many levels finer than the read res
    #   (1 is about one hexagon per pixel of the read; past that, empty cells)
    # CARRY_RES: the fold runs this many levels finer than the hexagons drawn,
    #   and each hexagon shows its most-changed finer cell ("carry the peak"),
    #   so a small change is not averaged away zoomed out
    # (0, 1): hexagons at the read res, each its brightest patch
    # (1, 0): hexagons about a pixel of the read each, nothing carried
    HEX_UP = 0
    CARRY_RES = 1

    # the fill fades with how much the cell changed: quiet ground faint
    ALPHA_FILL = 235
    ALPHA_QUIET = 45
    VIRIDIS = "440154470d6048186a482374472e7c4538824241863e4a893a548c365d8d32658e2e6d8e2b758e287d8e25848e228c8d1f948c1e9c8920a38625ab822eb37c3aba7648c16e58c7656ccd5a7fd34e93d741a8db34c0df25d5e21aeae51afde725"
    return (
        ADMIN_PQ,
        AEF_FROM0,
        AEF_INDEX_URL,
        AEF_LEVEL_FOR_RES,
        AEF_MAX_FILES,
        AEF_NODATA,
        AEF_PREFIX,
        AEF_RES,
        AEF_TO0,
        AEF_X0,
        AEF_Y0,
        AEF_YEARS_ALL,
        ALPHA_FILL,
        ALPHA_QUIET,
        BASE_RES,
        CACHE_DIR,
        CARRY_RES,
        CELL_BUDGET,
        HEX_UP,
        HEX_ZOOM,
        HOLD_MS,
        HOLD_SLOP_PX,
        HOME,
        LABELS_SLOT,
        MAX_RES,
        MIN_RES,
        MOSAIC_MIN_RES,
        OV_DIV_PM,
        PAD,
        PER_RES,
        RASTER_TILE,
        S2_COLLECTION,
        S2_FILL_COLLECTION,
        S2_PYRAMID_Z,
        S2_SCALE0,
        S2_STAC,
        S2_TCI_MAX_Z,
        S2_TILE_MIN_Z,
        S2_YEAR0,
        S2_YEARS,
        S3_OPTS,
        SETTLE,
        VIEW_H,
        VIEW_W,
        VIRIDIS,
        ZOOM0,
    )


@app.cell
def _(
    BASE_RES,
    CELL_BUDGET,
    MAX_RES,
    MIN_RES,
    PAD,
    PER_RES,
    VIEW_H,
    VIEW_W,
    ZOOM0,
    math,
):
    # ---- the camera -> box and res --------------------------------------------
    CELL_KM2 = {5: 252.9, 6: 36.13, 7: 5.161, 8: 0.7373, 9: 0.1053, 10: 0.01505, 11: 0.00215, 12: 0.000307, 13: 0.0000439}

    def _lat_to_y(lat):
        r = math.radians(lat)
        return (1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2

    def _y_to_lat(y):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y))))

    def view_to_bbox(vs):
        """The flat camera footprint (W, S, E, N) of ONE pane; the widget reports
        the pane's canvas size (`w`, `h`) with every move."""
        world = 512 * (2 ** vs["zoom"])
        w, h = vs.get("w") or VIEW_W, vs.get("h") or VIEW_H
        half_lon = 360.0 * w / world / 2
        yc, half_y = _lat_to_y(vs["latitude"]), h / world / 2
        return (
            vs["longitude"] - half_lon,
            _y_to_lat(yc + half_y),
            vs["longitude"] + half_lon,
            _y_to_lat(yc - half_y),
        )

    def pad_box(b, f=PAD):
        dx, dy = (b[2] - b[0]) * (f - 1) / 2, (b[3] - b[1]) * (f - 1) / 2
        return (max(-179.9, b[0] - dx), max(-85.0, b[1] - dy), min(179.9, b[2] + dx), min(85.0, b[3] + dy))

    def box_km2(b):
        w = (b[2] - b[0]) * 111.32 * math.cos(math.radians((b[1] + b[3]) / 2))
        return abs(w * (b[3] - b[1]) * 110.57)

    def res_for_view(vs, box):
        r = max(MIN_RES, min(MAX_RES, BASE_RES + math.floor((vs["zoom"] - ZOOM0) / PER_RES)))
        while r > MIN_RES and box_km2(box) / CELL_KM2[r] > CELL_BUDGET:
            r -= 1
        return r

    def contains(outer, inner):
        return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]

    return CELL_KM2, box_km2, contains, pad_box, res_for_view, view_to_bbox


@app.cell
def _(XarrayContext, coordinates_to_cells, pa, udf):
    # THE FOLD IS THE H3 UDF INSIDE DATAFUSION (repo rule). One context, every fold.
    ctx = XarrayContext()
    ctx.register_udf(
        udf(
            lambda la, lo, r: pa.array(coordinates_to_cells(la.to_numpy(), lo.to_numpy(), r[0].as_py())),
            [pa.float64(), pa.float64(), pa.int32()],
            pa.uint64(),
            "stable",
            name="h3_latlng_to_cell",
        )
    )
    return (ctx,)



@app.cell
def _(
    AEF_INDEX_URL,
    AEF_LEVEL_FOR_RES,
    AEF_MAX_FILES,
    AEF_NODATA,
    AEF_PREFIX,
    AEF_RES,
    AEF_X0,
    AEF_Y0,
    AEF_YEARS_ALL,
    CACHE_DIR,
    GeoTIFF,
    ObjectStore,
    S3Store,
    S3_OPTS,
    source_coop,
    Transformer,
    Window,
    asyncio,
    cpu,
    ctx,
    duckdb,
    itertools,
    np,
    os,
    pq,
    time,
    xr,
    zarr,
):
    # ---- AlphaEarth: the COG overviews (mosaic past res 10), one fold per year --
    # `aef_fold(box, res, year)` for any year in AEF_YEARS_ALL (2017..2025, the
    # whole run; the window control picks from them); each year has its own COG index
    # slice (cached as parquet under tmp) and its own mosaic time index.
    class _Kept:
        """THE COG READS, KEPT FOR THE SESSION: the compressed byte ranges async-geotiff
        asks for (one per band per tile, ~0.38 MB; a 1024 tile of 64 bands
        ~24 MB, 41 km of ground at 40 m) are kept in memory, oldest dropped
        past the cap, gone when the kernel stops. Zooming, panning nearby
        and changing the years read reuse the tiles instead of downloading
        them again."""

        def __init__(self, inner, cap):
            self._in, self._cap, self._kept, self.held = inner, cap, {}, 0
            self.reused = self.fetched = 0
            self._sem = asyncio.Semaphore(48)
            self._fly = {}  # range -> its download in flight, shared

        def _take(self, k):
            b = self._kept.pop(k, None)
            if b is not None:
                self._kept[k] = b  # to the newest end
                self.reused += len(b)
            return b

        def _put(self, k, b):
            self._kept[k] = b
            self.held += len(b)
            self.fetched += len(b)
            while self.held > self._cap and self._kept:
                self.held -= len(self._kept.pop(next(iter(self._kept))))

        async def get_range_async(self, path, *, start, end=None, length=None):
            end = start + length if end is None else end
            b = self._take((path, start, end))
            if b is None:
                b = await self._in.get_range_async(path, start=start, end=end)
                self._put((path, start, end), b)
            return b

        async def _one(self, path, a, e):
            # one band's tile, on its own: S3 merges nearby ranges into one
            # request, and a file's 64 bands merged are ~48 MB, past the 6 s
            # timeout on a home connection.
            # Small requests side by side, each retried twice.
            for k in range(3):
                try:
                    async with self._sem:
                        return await self._in.get_range_async(path, start=a, end=e)
                except Exception:
                    if k == 2:
                        raise
                    await asyncio.sleep(0.5 * (k + 1))

        async def _fetch(self, k):
            try:
                b = await self._one(*k)
                self._put(k, b)
                return b
            finally:
                self._fly.pop(k, None)

        async def get_ranges_async(self, path, *, starts, ends=None, lengths=None):
            ends = [a + n for a, n in zip(starts, lengths)] if ends is None else list(ends)
            out = [self._take((path, a, e)) for a, e in zip(starts, ends)]
            miss = [i for i, b in enumerate(out) if b is None]
            if miss:
                # a range already downloading (the read ahead, or another
                # year's fold) is waited on, not asked for again; shielded, so
                # a cancelled read still leaves its bytes kept
                futs = []
                for i in miss:
                    k = (path, starts[i], ends[i])
                    if k not in self._fly:
                        self._fly[k] = asyncio.ensure_future(self._fetch(k))
                    futs.append(asyncio.shield(self._fly[k]))
                for i, b in zip(miss, await asyncio.gather(*futs)):
                    out[i] = b
            return out

    # 6 GB: one view in a 2x box downloads 1.2 to 1.8 GB, so 2 GB held about
    # one view and a pan pushed out what was just read (molab: 32 GB)
    _store = _Kept(source_coop(**S3_OPTS), 1 * 1024 ** 3)
    _mstore = source_coop(AEF_PREFIX, **S3_OPTS)
    # only the years are needed from the mosaic's metadata: its time array, read alone (opening
    # the whole dataset lists the store, 13 s through the proxy)
    _mt = zarr.open_group(ObjectStore(_mstore, read_only=True), mode="r")["time"][:]
    _ti = {y: int(np.where(_mt == y)[0][0]) for y in AEF_YEARS_ALL}
    # The mosaic is sharded (4096 px shards of 256 px chunks, 64 bands, int8):
    # one read of a window fetches its chunks ONE AFTER ANOTHER, so a 9 km
    # view took 31 s a year from the US East Coast (1.2 MB/s, 38 MB; zarr's
    # async.concurrency made no difference). The window is read instead as
    # its chunk-aligned blocks, all at once, through zarr's async API on this
    # loop: 2.8 s for the same year (measured 2026-09-24, Wuhan, 16 blocks)
    _memb = zarr.open_group(ObjectStore(_mstore, read_only=True), mode="r")["embeddings"]
    _mb = int(_memb.chunks[-1])
    _memb = _memb._async_array
    _msem = asyncio.Semaphore(48)
    _mkept, _mfly, _mheld = {}, {}, [0]

    async def _mosaic(ti, y0, y1, x0, x1):
        """The mosaic's (64, y1 - y0, x1 - x0) int8 window for time index ti."""
        out = np.empty((64, y1 - y0, x1 - x0), np.int8)

        async def one(r, c):
            k = (ti, r, c)
            b = _mkept.pop(k, None)
            if b is None:
                if k not in _mfly:
                    async def get():
                        try:
                            async with _msem:
                                return await _memb.getitem((ti, slice(None), slice(r, r + _mb), slice(c, c + _mb)))
                        finally:
                            _mfly.pop(k, None)
                    _mfly[k] = asyncio.ensure_future(get())
                b = await asyncio.shield(_mfly[k])
            _mkept[k] = b
            _mheld[0] = sum(v.nbytes for v in _mkept.values())
            while _mheld[0] > 512 * 1024 ** 2 and len(_mkept) > 1:
                _mheld[0] -= _mkept.pop(next(iter(_mkept))).nbytes
            r0, r1, c0, c1 = max(r, y0), min(r + _mb, y1), max(c, x0), min(c + _mb, x1)
            out[:, r0 - y0:r1 - y0, c0 - x0:c1 - x0] = b[:, r0 - r:r1 - r, c0 - c:c1 - c]

        await asyncio.gather(*(one(r, c) for r in range(y0 // _mb * _mb, y1, _mb) for c in range(x0 // _mb * _mb, x1, _mb)))
        return out

    os.makedirs(CACHE_DIR, exist_ok=True)
    _IDX, _PATHS, _CRS = {}, {}, {}
    for _y in AEF_YEARS_ALL:
        _idx_path = os.path.join(CACHE_DIR, f"aef_index_{_y}_world.parquet")
        if not os.path.exists(_idx_path):
            _c = duckdb.connect()
            _c.execute("INSTALL httpfs; LOAD httpfs")
            _t = _c.execute(f"""
                SELECT path, crs, utm_west, utm_south, utm_east, utm_north,
                       wgs84_west, wgs84_south, wgs84_east, wgs84_north
                FROM read_parquet('{AEF_INDEX_URL}')
                WHERE year = {_y}
            """).arrow().read_all()
            pq.write_table(_t, _idx_path)
            _c.close()
        _tab = pq.read_table(_idx_path)
        _IDX[_y] = {k: _tab[k].to_numpy() for k in _tab.column_names if k not in ("path", "crs")}
        _PATHS[_y] = _tab["path"].to_pylist()
        _CRS[_y] = _tab["crs"].to_pylist()

    _open = {}
    _sem = asyncio.Semaphore(64)
    _tf_fwd, _tf_inv = {}, {}

    def _tf(crs):
        if crs not in _tf_fwd:
            _tf_fwd[crs] = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            _tf_inv[crs] = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        return _tf_fwd[crs], _tf_inv[crs]

    async def _get(path):
        rel = path.split("source.coop/")[1]
        if rel not in _open:
            async with _sem:
                _open[rel] = await GeoTIFF.open(rel, store=_store)
        return _open[rel]

    async def _read_cog(year, i, li, box):
        """One file's overview window over the box: (int8 (64, h, w), lon, lat)
        or None. Through the file's affine (these COGs are stored south-up)."""
        g = await _get(_PATHS[year][i])
        ov = g.overviews[li]
        H, W = ov.shape
        t = g.transform
        sx, sy = t.a * (g.width / W), t.e * (g.height / H)
        fwd, inv = _tf(_CRS[year][i])
        W_, S_, E_, N_ = box
        lons = np.concatenate([np.linspace(W_, E_, 5), np.full(5, E_), np.linspace(E_, W_, 5), np.full(5, W_)])
        lats = np.concatenate([np.full(5, N_), np.linspace(N_, S_, 5), np.full(5, S_), np.linspace(S_, N_, 5)])
        ux, uy = fwd.transform(lons, lats)
        cc = (np.asarray(ux) - t.c) / sx
        rr = (np.asarray(uy) - t.f) / sy
        c0 = max(0, int(np.floor(np.nanmin(cc))))
        c1 = min(W, int(np.ceil(np.nanmax(cc))))
        r0 = max(0, int(np.floor(np.nanmin(rr))))
        r1 = min(H, int(np.ceil(np.nanmax(rr))))
        if c1 <= c0 or r1 <= r0:
            return None
        async with _sem:
            ra = await ov.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))

        def _place():
            a = np.asarray(np.ma.filled(ra.as_masked(), AEF_NODATA)).reshape(64, r1 - r0, c1 - c0)
            xs = t.c + (np.arange(c0, c1) + 0.5) * sx
            ys = t.f + (np.arange(r0, r1) + 0.5) * sy
            X, Y = np.meshgrid(xs, ys)
            lon, lat = inv.transform(X, Y)
            return a, lon, lat

        return await cpu(_place)

    _DEQ = ", ".join(f"avg(signum(e{i:02d}) * power(e{i:02d} / 127.5, 2)) AS e{i:02d}" for i in range(64))
    _seq = itertools.count()  # a table name per fold: the years fold side by side on the CPU pool

    def _compact(t):
        """A year's fold as {"cell": sorted uint64, "V": float32 (n, 64), each
        row unit length, NaN where it has none}: made once per read, so every
        frame built from it lines the years up by cell with no join and no
        restack of 64 columns (1 s of a 3.4 s frame at a 2x box), at half the
        memory of the float64 table."""
        cell = t["cell"].to_numpy().astype(np.uint64)
        o = np.argsort(cell)
        V = np.empty((len(cell), 64), np.float32)
        for i in range(64):
            V[:, i] = t[f"e{i:02d}"].to_numpy(zero_copy_only=False)[o]
        nrm = np.linalg.norm(V, axis=1)
        V /= np.maximum(nrm, 1e-9)[:, None]
        V[~np.isfinite(nrm) | (nrm == 0)] = np.nan
        return {"cell": cell[o], "V": V}

    def _fold_rows_sync(res, box, cols, lat, lon):
        W_, S_, E_, N_ = box
        name = f"aef_{next(_seq)}"
        ds1 = xr.Dataset(
            {f"e{i:02d}": (("i",), cols[i]) for i in range(64)} | {"lat": (("i",), lat), "lon": (("i",), lon)},
            coords={"i": np.arange(lat.size)},
        )
        ctx.from_dataset(name, ds1, chunks={"i": 262_144})
        try:
            return ctx.sql(f"""
                SELECT h3_latlng_to_cell(lat, lon, CAST({res} AS INT)) AS cell, count(*) AS naef, {_DEQ}
                FROM {name}
                WHERE e00 != {AEF_NODATA}
                  AND lon >= {W_} AND lon < {E_} AND lat >= {S_} AND lat < {N_}
                GROUP BY cell
            """).to_arrow_table()
        finally:
            ctx.deregister_table(name)

    async def aef_window(box, year):
        """The mosaic's native 10 m window under the box for one year:
        (int8 (64, h, w), lon0, lat0 of the north-west corner, pixel) or None."""
        W_, S_, E_, N_ = box
        x0, x1 = int((W_ - AEF_X0) / AEF_RES), int((E_ - AEF_X0) / AEF_RES)
        y0, y1 = int((AEF_Y0 - N_) / AEF_RES), int((AEF_Y0 - S_) / AEF_RES)
        if x1 <= x0 or y1 <= y0:
            return None
        emb = await _mosaic(_ti[year], y0, y1, x0, x1)
        return emb, AEF_X0 + x0 * AEF_RES, AEF_Y0 - y0 * AEF_RES, AEF_RES

    async def aef_fold(box, res, year, read_res=None):
        """Mean AlphaEarth vector per res cell over the box for one year, from
        the source that suits read_res (default res): a finer res than the
        read's own gives cells of about a pixel each (the carried peak).
        Returns (arrow table or None, stats)."""
        t0 = time.time()
        W_, S_, E_, N_ = box
        rr = res if read_res is None else read_res
        if rr >= MOSAIC_MIN_RES:
            x0, x1 = int((W_ - AEF_X0) / AEF_RES), int((E_ - AEF_X0) / AEF_RES)
            y0, y1 = int((AEF_Y0 - N_) / AEF_RES), int((AEF_Y0 - S_) / AEF_RES)
            emb = await _mosaic(_ti[year], y0, y1, x0, x1)
            lat = AEF_Y0 - (np.arange(y0, y1) + 0.5) * AEF_RES
            lon = AEF_X0 + (np.arange(x0, x1) + 0.5) * AEF_RES
            t1 = time.time()

            def _fold_mosaic():
                LON, LAT = np.meshgrid(lon, lat)
                return _compact(_fold_rows_sync(res, box, emb.reshape(64, -1), LAT.ravel(), LON.ravel()))

            # RES 13 THE OTHER WAY ROUND: a 10 m pixel holds about 2.3 res 13 cells, so folding
            # pixel centers leaves most res 13 cells empty (dots with gaps). Every res 13 cell in
            # the box takes the pixel under its center instead (as the model's store does)
            def _fold_centers():
                import shapely
                from h3ronpy.vector import cells_to_coordinates, wkb_to_cells
                cl = wkb_to_cells(pa.array([shapely.to_wkb(shapely.box(W_, S_, E_, N_))], pa.binary()), res, flatten=True)
                cells = np.sort(np.asarray(pa.array(cl)).astype(np.uint64))
                xy = cells_to_coordinates(pa.array(cells))
                r = np.floor((AEF_Y0 - np.asarray(pa.array(xy.column("lat")))) / AEF_RES).astype(np.int64) - y0
                c = np.floor((np.asarray(pa.array(xy.column("lng"))) - AEF_X0) / AEF_RES).astype(np.int64) - x0
                ok = (r >= 0) & (r < emb.shape[1]) & (c >= 0) & (c < emb.shape[2])
                q = emb[:, r[ok], c[ok]].T
                V = q.astype(np.float32)
                V = np.sign(V) * (V / 127.5) ** 2
                bad = q[:, 0] == AEF_NODATA
                nrm = np.linalg.norm(V, axis=1)
                V /= np.maximum(nrm, 1e-9)[:, None]
                V[bad | (nrm == 0)] = np.nan
                return {"cell": cells[ok], "V": V}

            if res >= 13:
                _fold_mosaic = _fold_centers  # noqa: F811

            out = await cpu(_fold_mosaic)
            return out, f"AEF {year} mosaic {t1 - t0:.1f} s · fold {len(out['cell']):,} {time.time() - t1:.1f} s"
        li = AEF_LEVEL_FOR_RES[rr]
        ix = _IDX[year]
        hit = np.where(
            (ix["wgs84_east"] > W_) & (ix["wgs84_west"] < E_) & (ix["wgs84_north"] > S_) & (ix["wgs84_south"] < N_)
        )[0]
        if len(hit) == 0:
            return None, f"AEF {year}: no COG tiles under the view"
        if len(hit) > AEF_MAX_FILES:
            return None, f"AEF {year}: {len(hit):,} tiles under the view; zoom in"
        parts = await asyncio.gather(*(_read_cog(year, int(i), li, box) for i in hit), return_exceptions=True)
        bad = [p for p in parts if isinstance(p, BaseException)]
        if any(isinstance(p, asyncio.CancelledError) for p in bad):
            raise asyncio.CancelledError()
        parts = [p for p in parts if p is not None and not isinstance(p, BaseException)]
        if not parts:
            return None, f"AEF {year}: nothing read" + (f" ({type(bad[0]).__name__}: {str(bad[0])[:80]})" if bad else "")
        t1 = time.time()

        def _cogs():
            cols = np.concatenate([p[0].reshape(64, -1) for p in parts], axis=1)
            lon = np.concatenate([p[1].ravel() for p in parts])
            lat = np.concatenate([p[2].ravel() for p in parts])
            return _compact(_fold_rows_sync(res, box, cols, lat, lon)), cols.shape[1]

        out, npx = await cpu(_cogs)
        return out, (
            f"AEF {year} ov{li} ({10 * 2 ** (li + 1)} m) {len(parts)} files {npx / 1e6:.2f} Mpx "
            f"{t1 - t0:.1f} s · fold {len(out['cell']):,} {time.time() - t1:.1f} s"
            f" · kept {_store.held / 1e6:,.0f} MB (fetched {_store.fetched / 1e6:,.0f}, reused {_store.reused / 1e6:,.0f})"
            + (f" · {len(bad)} files failed ({type(bad[0]).__name__})" if bad else "")
        )

    return aef_fold, aef_window




@app.cell
def _(
    GeoTIFF,
    Image,
    RASTER_TILE,
    S2_COLLECTION,
    S2_FILL_COLLECTION,
    S2_PYRAMID_Z,
    S2_SCALE0,
    S2_STAC,
    S2_TCI_MAX_Z,
    S2_TILE_MIN_Z,
    S3Store,
    S3_OPTS,
    source_coop,
    Window,
    asyncio,
    cpu,
    io,
    json,
    math,
    np,
    time,
    urllib,
):
    # ---- Sentinel-2 TCI tiles, by YEAR: the left pane ---------------------------
    # STAC once per (year, z9 ancestor tile), every footprint under the tile
    # composited in numpy (black = nodata -> alpha 0; first footprint to paint a
    # pixel wins), one PNG. The year lives in the item id
    # (`10SFJ_2024-01-01_2025-01-01`); the STAC datetime filter does not
    # constrain these items, so it is enforced on the id. The yearly footprints
    # come first, then the same year's S2_FILL_COLLECTION footprints (ids
    # suffixed `#fill`): first-to-paint-wins is the backfill.
    _store = source_coop(**S3_OPTS)
    _R = 6378137.0
    _items = {}  # item id -> {tci: path, bbox}
    _boxes = {}  # (year, rounded box) -> item ids
    _searches = {}  # rounded box -> the STAC search's future: one search answers every year
    _open = {}
    _sem = asyncio.Semaphore(32)
    _png = {}  # (year, z, x, y, scale) -> PNG bytes or None
    _arr = {}  # (year, z, x, y) -> the composited RGBA tile before the gain, or None
    _gain = {"v": float(S2_SCALE0)}  # the strip's `gamma`
    _LUT = {}  # gamma -> the 256-entry curve
    _tstat = {"served": 0, "blank": 0, "ms": 0.0}
    _fill = {}  # year -> [pixels painted by the fill collection, pixels painted]

    def _png_of(out, g):
        """The gamma (the header's `gamma`) on the composited bytes, then PNG:
        v -> 255 (v / 255) ** (1 / gamma), a lift of the midtones that keeps
        the bright end unclipped (a gain clipped it). Pure: runs on the pool."""
        if g not in _LUT:
            _LUT[g] = (255.0 * (np.arange(256, dtype=np.float64) / 255.0) ** (1.0 / g)).round().astype(np.uint8)
        rgba = out if g == 1.0 else np.concatenate([_LUT[g][out[..., :3]], out[..., 3:]], axis=2)
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgba), mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    async def _encode(key, out):
        _png[key] = await cpu(_png_of, out, key[-1])
        if len(_png) > 6000:
            _png.pop(next(iter(_png)))
        return _png[key]

    def _stac(box):
        body = json.dumps(
            {"collections": [S2_COLLECTION, S2_FILL_COLLECTION], "bbox": list(box), "limit": 200}
        ).encode()
        req = urllib.request.Request(S2_STAC, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["features"]

    # False colors from the raw bands (uint16 reflectance x 10,000, next to TCI in every item):
    # TCI clips bright ground (sand: a third of the pixels at 255 in Dubai, where the red band
    # runs to 0.56). Each band is stretched to its 98th percentile over the view (as segments-map
    # does), one stretch for every year so the years compare; set again when the colors are
    # chosen or a hold starts somewhere the stretch was not taken
    #   urban        B12 B11 B4: built ground and bare soil apart by hue
    #   blueyellow   B11 B11 B2: SWIR on red and green, blue on blue: sand yellow, concrete blue
    #   ndbi         (B11 - B8A) / (B11 + B8A): built-up and bare ground high, vegetation and water low
    #   mineral      iron oxides (B4 / B2) and clays (B11 / B12) on bare ground, vegetation (NDVI) masked
    # The two indexes are drawn only in the browser, stretched from the view's 2nd to 98th percentile
    # of each quantity (S2_INDEXES; the browser computes the same quantities, s2BandTileData)
    S2_COMPOSITES = {"tci": None, "urban": ("B12", "B11", "B04"), "blueyellow": ("B11", "B11", "B02"),
                     "ndbi": ("B11", "B8A"), "mineral": ("B02", "B03", "B04", "B11", "B12", "B8A")}
    _BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")

    def _bare(b):
        """0 under vegetation (NDVI 0.45 and up) to 1 on bare ground (NDVI 0.2 and under), and 0 on water
        (MNDWI 0.05 and up, land from -0.10), whose band ratios are noise. B11 alone does not find the
        water: the yearly medians blend rivers with their banks."""
        ndvi = (b["B8A"] - b["B04"]) / (b["B8A"] + b["B04"])
        mndwi = (b["B03"] - b["B11"]) / (b["B03"] + b["B11"])
        return np.clip((0.45 - ndvi) / 0.25, 0, 1) * np.clip((0.05 - mndwi) / 0.15, 0, 1)

    # name -> (quantities from the bands as float arrays, which pixels each one's stretch is taken over)
    S2_INDEXES = {
        "ndbi": (lambda b: [(b["B11"] - b["B8A"]) / (b["B11"] + b["B8A"])], lambda b: None),
        "mineral": (lambda b: [b["B04"] / b["B02"], b["B11"] / b["B12"], _bare(b)], lambda b: _bare(b) > 0.5),
    }
    # (above every function that uses them: marimo drops a cell's private name used before it is defined)
    _comp = {"name": "tci", "box": None, "scale": None, "gen": 0, "lock": None}


    async def _s2_items(box, year):
        key = (year, tuple(round(v, 2) for v in box))
        if key not in _boxes:
            if key[1] not in _searches:
                _searches[key[1]] = asyncio.get_running_loop().run_in_executor(None, _stac, box)
            try:
                both = await asyncio.shield(_searches[key[1]])
            except Exception:
                _searches.pop(key[1], None)
                raise
            feats = [f for f in both if f.get("collection") != S2_FILL_COLLECTION]
            ids, fill_ids = [], []
            for f in both:
                if not f["id"].endswith(f"{year}-01-01_{year + 1}-01-01"):
                    continue
                if f.get("collection") == S2_FILL_COLLECTION:
                    iid = f["id"] + "#fill"
                    _items[iid] = {"tci": f["assets"]["TCI"]["href"].split("source.coop/")[1], "bbox": f.get("bbox"), "fill": True,
                                   "bands": {k: a["href"].split("source.coop/")[1] for k, a in f["assets"].items() if k in _BANDS}}
                    fill_ids.append(iid)
                    continue
                _items[f["id"]] = {"tci": f["assets"]["TCI"]["href"].split("source.coop/")[1], "bbox": f.get("bbox"),
                                   "bands": {k: a["href"].split("source.coop/")[1] for k, a in f["assets"].items() if k in _BANDS}}
                ids.append(f["id"])
            if not ids and feats:
                # the STAC lags the bucket (2025 is there for every tile round
                # Dixie, uploaded 2026-01-31, and the search does not know it):
                # the same MGRS tile's path with the year swapped, the sibling's
                # bbox; a tile that is not there reads as empty, not an error
                seen = set()
                for f in feats:
                    tile = f["id"].split("_")[0]
                    if tile in seen:
                        continue
                    seen.add(tile)
                    iid = f"{tile}_{year}-01-01_{year + 1}-01-01"
                    base = f["assets"]["TCI"]["href"].split("source.coop/")[1].rsplit("/", 2)[0]
                    _items[iid] = {"tci": f"{base}/{iid}/TCI.tif", "bbox": f.get("bbox"), "bands": {k: f"{base}/{iid}/{k}.tif" for k in _BANDS}}
                    ids.append(iid)
            _boxes[key] = ids + fill_ids  # yearly first: the fill only paints what they left
        return _boxes[key]

    async def _get(rel):
        if rel not in _open:
            async with _sem:
                try:
                    _open[rel] = await GeoTIFF.open(rel, store=_store)
                except Exception:
                    _open[rel] = None  # not in the bucket (a synthesized year): empty
        return _open[rel]

    def s2_set_composite(name, box):
        """Choose the imagery colors for the view box. True when the tiles must be asked again."""
        if name not in S2_COMPOSITES:
            return False
        b = _comp["box"]
        inside = b is not None and box is not None and b[0] <= box[0] and b[1] <= box[1] and b[2] >= box[2] and b[3] >= box[3]
        if name == _comp["name"] and (name == "tci" or inside):
            return False
        _comp.update(name=name, box=tuple(box) if box is not None else None, scale=None)
        _comp["gen"] += 1
        return True

    def _level(g, tpx):
        """The coarsest level of g whose pixel is no coarser than tpx (m), and its pixel."""
        L, _B, R_, _T = g.bounds
        best = (g, (R_ - L) / g.shape[1])
        for lv in g.overviews:
            px = (R_ - L) / lv.shape[1]
            if px <= tpx * 1.01:
                best = (lv, px)
        return best

    async def _comp_scale():
        """Each band's 98th percentile over the view, from the latest year with imagery there."""
        box = _comp["box"]
        if box is None:
            return {}
        W_, S_, E_, N_ = box
        mx = lambda lon: _R * math.radians(lon)
        my = lambda lat: _R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
        x0, x1, y0, y1 = mx(W_), mx(E_), my(S_), my(N_)
        ids = []
        for yr in reversed(S2_YEARS):
            ids = [i for i in await _s2_items(box, yr) if not _items[i].get("fill")]
            if ids:
                break

        async def _vals(iid, band):
            path = _items[iid].get("bands", {}).get(band)
            g = await _get(path) if path else None
            if g is None:
                return None
            lv, px = _level(g, max((x1 - x0), (y1 - y0)) / 512)
            L, _B, R_, Tt = g.bounds
            H, W = lv.shape
            c0, c1 = max(0, int((x0 - L) / px)), min(W, int(math.ceil((x1 - L) / px)))
            r0, r1 = max(0, int((Tt - y1) / px)), min(H, int(math.ceil((Tt - y0) / px)))
            if c1 <= c0 or r1 <= r0:
                return None
            async with _sem:
                ra = await lv.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))
            a = np.asarray(np.ma.filled(ra.as_masked(), 0)).ravel()
            return a[a > 0]

        # every footprint and band at once (the round trips overlap)
        ub = sorted(set(S2_COMPOSITES[_comp["name"]]))
        got = await asyncio.gather(*(_vals(i, b) for b in ub for i in ids[:6]))
        out = {}
        for k, band in enumerate(ub):
            vals = [v for v in got[k * len(ids[:6]):(k + 1) * len(ids[:6])] if v is not None]
            v = np.concatenate(vals) if vals else np.zeros(0)
            out[band] = float(np.percentile(v, 98)) if len(v) > 100 else 3000.0
        return out

    async def _index_scale():
        """{lo, hi}: each quantity of the chosen index at the 2nd and 98th percentile over the view
        (the bare channel of mineral is 0 to 1 as it is), from the latest year with imagery there.
        Every band at one pixel size (20 m or coarser), where the yearly footprints' bands share a grid."""
        box = _comp["box"]
        name = _comp["name"]
        if box is None:
            return {}
        W_, S_, E_, N_ = box
        mx = lambda lon: _R * math.radians(lon)
        my = lambda lat: _R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
        x0, x1, y0, y1 = mx(W_), mx(E_), my(S_), my(N_)
        tpx = max(max(x1 - x0, y1 - y0) / 512, 19.2)
        ids = []
        for yr in reversed(S2_YEARS):
            ids = [i for i in await _s2_items(box, yr) if not _items[i].get("fill")]
            if ids:
                break
        bands = sorted(set(S2_COMPOSITES[name]))
        quant, where = S2_INDEXES[name]

        async def _win(iid, band):
            path = _items[iid].get("bands", {}).get(band)
            g = await _get(path) if path else None
            if g is None:
                return None
            lv, px = _level(g, tpx)
            L, _B, R_, Tt = g.bounds
            H, W = lv.shape
            c0, c1 = max(0, int((x0 - L) / px)), min(W, int(math.ceil((x1 - L) / px)))
            r0, r1 = max(0, int((Tt - y1) / px)), min(H, int(math.ceil((Tt - y0) / px)))
            if c1 <= c0 or r1 <= r0:
                return None
            async with _sem:
                ra = await lv.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))
            return np.asarray(np.ma.filled(ra.as_masked(), 0)).reshape(r1 - r0, c1 - c0)

        use = ids[:6]
        got = await asyncio.gather(*(_win(i, b) for i in use for b in bands))
        per = []
        for k in range(len(use)):
            arrs = got[k * len(bands):(k + 1) * len(bands)]
            if any(a is None for a in arrs):
                continue
            h, w = min(a.shape[0] for a in arrs), min(a.shape[1] for a in arrs)
            b = {n: a[:h, :w].astype(np.float32) for n, a in zip(bands, arrs)}
            ok = np.all([a > 0 for a in b.values()], 0)
            if ok.sum() < 100:
                continue
            b = {n: a[ok] for n, a in b.items()}
            m = where(b)
            per.append([q if m is None else q[m] for q in quant(b)])
        if not per:
            return {}
        lo, hi = [], []
        for j in range(len(per[0])):
            v = np.concatenate([p[j] for p in per])
            v = v[np.isfinite(v)]
            fixed = (name == "mineral" and j == 2) or len(v) <= 100
            lo.append(0.0 if fixed else float(np.percentile(v, 2)))
            hi.append(1.0 if fixed else float(np.percentile(v, 98)))
        return {"lo": lo, "hi": hi}

    async def _scale_now(cgen):
        """The stretch for the colors chosen (generation cgen), taken once."""
        if _comp["lock"] is None:
            _comp["lock"] = asyncio.Lock()
        async with _comp["lock"]:
            if _comp["gen"] != cgen:
                return {}  # other colors were chosen meanwhile
            if _comp["scale"] is None:
                v = await (_index_scale() if _comp["name"] in S2_INDEXES else _comp_scale())
                # kept only if the colors did not change while it was taken (it would be the old colors')
                if _comp["gen"] == cgen:
                    _comp["scale"] = v
                return v
            return _comp["scale"]

    async def s2_stretch():
        """The false colors' stretch for the browser, which reads the bands itself:
        {comp, gen, band: 98th percentile (reflectance x 10,000)}, or None for true color."""
        cname, cgen = _comp["name"], _comp["gen"]
        if not S2_COMPOSITES.get(cname):
            return None
        scale = await _scale_now(cgen)
        if _comp["gen"] != cgen:
            return None
        if cname in S2_INDEXES:
            return {"comp": cname, "gen": cgen, "lo": scale.get("lo", [0.0] * 3), "hi": scale.get("hi", [1.0] * 3)}
        return {"comp": cname, "gen": cgen, **{b: scale.get(b, 3000.0) for b in set(S2_COMPOSITES[cname])}}

    def _tile_ll(z, x, y):
        n = 2 ** z
        lat = lambda yy: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))
        return x / n * 360 - 180, lat(y + 1), (x + 1) / n * 360 - 180, lat(y)

    async def _items_for_tile(z, x, y, year):
        # STAC per z9 ancestor tile from z9 up; below it, per the tile itself
        d = max(0, z - S2_PYRAMID_Z)
        ids = await _s2_items(_tile_ll(z - d, x >> d, y >> d), year)
        W_, S_, E_, N_ = _tile_ll(z, x, y)
        out = []
        for i in ids:
            b = _items[i].get("bbox")
            if not b or (b[0] < E_ and b[2] > W_ and b[1] < N_ and b[3] > S_):
                out.append(i)
        return out

    async def s2_tile_png(z, x, y, year):
        """PNG bytes for Web Mercator tile (z, x, y) of the year's TCI mosaic, or
        None (below S2_TILE_MIN_Z, or no footprint under the tile)."""
        cname, cgen = _comp["name"], _comp["gen"]
        key = (year, z, x, y, cname, cgen, _gain["v"])
        if key in _png:
            return _png[key]
        akey = (year, z, x, y, cname, cgen)
        if akey in _arr:
            # composited already at another scale: re-encode, no read
            out = _arr[akey]
            return (await _encode(key, out)) if out is not None else None
        if z < S2_TILE_MIN_Z or z > S2_TCI_MAX_Z:
            _tstat["blank"] += 1
            return None
        ids = await _items_for_tile(z, x, y, year)
        if not ids:
            _tstat["blank"] += 1
            return None
        t0 = time.time()
        T = RASTER_TILE
        n = 2 ** z
        world = 2 * math.pi * _R
        tpx = world / (n * T)
        tx0, ty1 = -world / 2 + x * world / n, world / 2 - y * world / n
        xs = tx0 + (np.arange(T) + 0.5) * tpx
        ys = ty1 - (np.arange(T) + 0.5) * tpx
        li = min(S2_TCI_MAX_Z - z, S2_TCI_MAX_Z - S2_PYRAMID_Z)  # L5 (306 m) below z9: decimated

        async def _read(iid):
            """One footprint's window under the tile: (ra, c0, r0, h, w, px, L, Tt) or None."""
            g = await _get(_items[iid]["tci"])
            if g is None:
                return None
            lv = [g, *g.overviews][li]
            L, _B, R_, Tt = g.bounds
            H, W = lv.shape
            px = (R_ - L) / W
            c0, c1 = max(0, int(math.floor((tx0 - L) / px))), min(W, int(math.ceil((tx0 + T * tpx - L) / px)))
            r0, r1 = max(0, int(math.floor((Tt - ty1) / px))), min(H, int(math.ceil((Tt - (ty1 - T * tpx)) / px)))
            if c1 <= c0 or r1 <= r0:
                return None
            async with _sem:
                ra = await lv.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))
            return ra, c0, r0, r1 - r0, c1 - c0, px, L, Tt

        if cname in S2_INDEXES:
            return None  # the indexes are drawn in the browser only
        bands = S2_COMPOSITES.get(cname)
        if bands:
            scale = await _scale_now(cgen)

            async def _read_band(iid, band):
                path = _items[iid].get("bands", {}).get(band)
                g = await _get(path) if path else None
                if g is None:
                    return None
                lv, px = _level(g, tpx)
                L, _B, R_, Tt = g.bounds
                H, W = lv.shape
                c0, c1 = max(0, int(math.floor((tx0 - L) / px))), min(W, int(math.ceil((tx0 + T * tpx - L) / px)))
                r0, r1 = max(0, int(math.floor((Tt - ty1) / px))), min(H, int(math.ceil((Tt - (ty1 - T * tpx)) / px)))
                if c1 <= c0 or r1 <= r0:
                    return None
                async with _sem:
                    ra = await lv.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))
                a = np.asarray(np.ma.filled(ra.as_masked(), 0)).reshape(r1 - r0, c1 - c0)
                cols = np.floor((xs - (L + c0 * px)) / px).astype(np.int64)
                rows = np.floor(((Tt - r0 * px) - ys) / px).astype(np.int64)
                okc, okr = (cols >= 0) & (cols < c1 - c0), (rows >= 0) & (rows < r1 - r0)
                v = a[np.clip(rows, 0, r1 - r0 - 1)[:, None], np.clip(cols, 0, c1 - c0 - 1)[None, :]]
                return np.where(okr[:, None] & okc[None, :], v, 0)

            ub = sorted(set(bands))
            got = await asyncio.gather(*(_read_band(i, b) for i in ids for b in ub))

            def _composite_bands():
                out = np.zeros((T, T, 4), np.uint8)
                painted = []
                for k, iid in enumerate(ids):
                    per = dict(zip(ub, got[k * len(ub):(k + 1) * len(ub)]))
                    if any(per[b] is None for b in ub):
                        painted.append(0)
                        continue
                    rgb = np.stack([np.clip(255.0 * per[b] / max(scale.get(b, 3000.0), 1.0), 0, 255) for b in bands], -1).astype(np.uint8)
                    valid = np.all([per[b] > 0 for b in ub], 0) & (out[..., 3] == 0)
                    out[valid, :3] = rgb[valid]
                    out[valid, 3] = 255
                    painted.append(int(valid.sum()))
                return out, painted

            out, painted = await cpu(_composite_bands)
            if not out[..., 3].any():
                _tstat["blank"] += 1
                _png[key] = None
                _arr[akey] = None
                return None
            _arr[akey] = out
            if len(_arr) > 2000:
                _arr.pop(next(iter(_arr)))
            png = await _encode(key, out)
            _tstat["served"] += 1
            _tstat["ms"] += 1000 * (time.time() - t0)
            return png

        # all the footprints at once (the round trips overlap), painted in
        # their order after (first to paint a pixel wins, yearly before fill)
        reads = await asyncio.gather(*(_read(i) for i in ids))

        def _composite():
            out = np.zeros((T, T, 4), np.uint8)
            painted = []
            for rd in reads:
                if rd is None:
                    painted.append(0)
                    continue
                ra, c0, r0, h, w, px, L, Tt = rd
                a = np.asarray(np.ma.filled(ra.as_masked(), 0)).reshape(-1, h, w)[:3]
                cols = np.floor((xs - (L + c0 * px)) / px).astype(np.int64)
                rows = np.floor(((Tt - r0 * px) - ys) / px).astype(np.int64)
                okc, okr = (cols >= 0) & (cols < w), (rows >= 0) & (rows < h)
                rgb = a[:, np.clip(rows, 0, h - 1)[:, None], np.clip(cols, 0, w - 1)[None, :]].transpose(1, 2, 0)
                valid = okr[:, None] & okc[None, :] & (rgb.sum(2) > 0) & (out[..., 3] == 0)
                out[valid, :3] = rgb[valid]
                out[valid, 3] = 255
                painted.append(int(valid.sum()))
            return out, painted

        out, painted = await cpu(_composite)
        fy = _fill.setdefault(year, [0, 0])
        for iid, n_new in zip(ids, painted):
            fy[1] += n_new
            if _items[iid].get("fill"):
                fy[0] += n_new
        if not out[..., 3].any():
            _tstat["blank"] += 1
            _png[key] = None
            _arr[akey] = None
            return None
        _arr[akey] = out
        if len(_arr) > 2000:
            _arr.pop(next(iter(_arr)))
        png = await _encode(key, out)
        _tstat["served"] += 1
        _tstat["ms"] += 1000 * (time.time() - t0)
        return png

    async def s2_items_json(z, x, y, year):
        """The year's footprints under Web Mercator tile (z, x, y), for the browser to read
        itself (deck.gl-raster): JSON bytes, a list of {id, url of its TCI COG, bbox, fill,
        bands: the raw band COGs' urls}, yearly first. The same STAC search (and the same
        fallback when the STAC lags) as the kernel's own tiles."""
        ids = await _s2_items(_tile_ll(z, x, y), year)
        sc = "https://data.source.coop/"
        return json.dumps([{"id": i, "url": sc + _items[i]["tci"], "bbox": _items[i].get("bbox"), "fill": bool(_items[i].get("fill")),
                            "bands": {k: sc + v for k, v in _items[i].get("bands", {}).items()}} for i in ids]).encode()

    def s2_set_scale(v):
        """The header's `gamma`: the curve the next S2 tiles are encoded with.
        Returns True when it changed (the caller then re-asks deck for the tiles)."""
        v = float(min(4.0, max(0.1, v)))
        if v == _gain["v"]:
            return False
        _gain["v"] = v
        return True

    def s2_raster_stats():
        """The tile counters, plus `fill`: for each year whose served tiles took
        any pixels from S2_FILL_COLLECTION, the share of painted pixels that did
        (over every tile served so far, not the view)."""
        fill = {y: f / p for y, (f, p) in _fill.items() if f and p}
        return dict(_tstat, cached=len(_png), scale=_gain["v"], fill=fill)

    return S2_COMPOSITES, s2_items_json, s2_raster_stats, s2_set_composite, s2_set_scale, s2_stretch, s2_tile_png


@app.cell
def _(duckdb):
    # ---- DuckDB: the frame's join and the tables under the map --------------
    con = duckdb.connect()
    return (con,)


@app.cell
def _(ADMIN_PQ, HOME, duckdb):
    # ---- the place under a click: one point query against fused/overture ------
    # Overture's divisions theme as Fused geo-partitions it on Source
    # Cooperative, 79 GeoParquet files per type, each row with a bbox struct.
    # division_area says which polygons hold the point: country, region,
    # county, localadmin, locality, every level Overture draws, anywhere.
    # division, joined on the ids, adds local_type, the country's own word for
    # the level (city, town, village, prefecture, governorate, state). DuckDB
    # reads the footers, keeps the row groups whose bbox stats can hold the
    # point, and runs ST_Contains on what is left. Its own connection, a
    # cursor per call so a click and the warm-up can overlap, the object cache
    # on so the footers are read once: 7 s cold, 1 to 3 s after.
    import threading as _th

    _dv = {"con": None, "err": None}
    _lock = _th.Lock()
    _AREA = f"{ADMIN_PQ}/type=division_area/*.parquet"
    _DIV = f"{ADMIN_PQ}/type=division/*.parquet"
    _ORDER = {"locality": 0, "localadmin": 1, "county": 2, "region": 3, "country": 4}

    def _connect():
        with _lock:
            if _dv["con"] is None and _dv["err"] is None:
                try:
                    c = duckdb.connect()
                    for ext in ("spatial", "httpfs"):
                        try:
                            c.execute(f"LOAD {ext}")
                        except Exception:
                            c.execute(f"INSTALL {ext}; LOAD {ext}")
                    # Source Cooperative's proxy (see source_coop): its S3 API,
                    # path-style, the account as the bucket, nothing to sign with.
                    # GLOBAL, because a cursor is its own session and a plain
                    # SET would not reach it
                    c.execute("SET GLOBAL s3_endpoint='data.source.coop'; SET GLOBAL s3_url_style='path'; SET GLOBAL s3_use_ssl=true; "
                              "SET GLOBAL s3_region='us-west-2'; SET GLOBAL enable_object_cache=true")
                    _dv["con"] = c
                except Exception as e:
                    _dv["err"] = e
            return _dv["con"]

    _Q_AREA = (
        "SELECT subtype, names.primary, names.common['en'], country, division_id "
        f"FROM read_parquet('{_AREA}', hive_partitioning=0) "
        "WHERE bbox.xmin <= $x AND bbox.xmax >= $x AND bbox.ymin <= $y AND bbox.ymax >= $y "
        "AND class = 'land' AND ST_Contains(geometry, ST_Point($x, $y))"
    )
    # the country filter is what makes the join quick: the files are spatial,
    # so each row group carries a tight country range and most are skipped
    # unread (25 s cold at Wuhan against 165 s without it, 1 to 3 s warm)
    _Q_DIV = (
        "SELECT id, local_type['en'], population "
        f"FROM read_parquet('{_DIV}', hive_partitioning=0) "
        "WHERE country = $country AND list_contains($ids, id)"
    )

    def division_at(lon, lat):
        """The divisions holding the point, smallest first: a list of
        {subtype, name, name_en, local_type, population}, locality up to
        country, whichever Overture draws there. Raises on a failed read so
        the caller can say so."""
        c = _connect()
        if c is None:
            raise _dv["err"]
        cur = c.cursor()
        rows = cur.execute(_Q_AREA, {"x": float(lon), "y": float(lat)}).fetchall()
        seen, out = set(), []
        for sub, name, name_en, country, did in rows:
            if sub in seen:
                continue
            seen.add(sub)
            out.append({"subtype": sub, "name": name, "name_en": name_en, "local_type": None,
                        "population": None, "id": did, "country": country})
        out.sort(key=lambda d: _ORDER.get(d["subtype"], -1))
        ids = [d["id"] for d in out if d["id"]]
        country = next((d["country"] for d in out if d["country"]), None)
        if ids and country:
            try:
                extra = {i: (lt, pop) for i, lt, pop in cur.execute(_Q_DIV, {"country": country, "ids": ids}).fetchall()}
            except Exception:
                extra = {}
            for d in out:
                d["local_type"], d["population"] = extra.get(d["id"], (None, None))
        return out

    # the footers, read now rather than on the first click, off the main
    # thread: about 7 s for division_area and 25 s more for division
    def _warm():
        try:
            division_at(HOME["longitude"], HOME["latitude"])
        except Exception:
            pass

    _th.Thread(target=_warm, daemon=True).start()
    return (division_at,)





@app.cell
def _(change_resolution, np, os, pa):
    # ---- a FRAME: Earthwork between the window's first and last year ------------
    # CARRY THE PEAK: the years arrive folded at a finer res than the hexagons (about a pixel of the
    # read per finer cell). Every finer cell is scored, and each hexagon takes its highest-scoring
    # finer cell, so one dig is not averaged away by the quiet ground around it.
    # EARTHWORK: earthwork_model.py's logistic regression on [b, a, a * b, (a - b)^2], b and a the unit
    # AlphaEarth vectors of the window's first and last year
    # (from the repo on GitHub when the notebook runs without its folder, as in molab)
    _ewd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    _ewp = os.path.join(_ewd, "earthwork-lr.npz")
    if os.path.exists(_ewp):
        _ew = np.load(_ewp)
    else:
        import io as _io
        import urllib.request as _ur
        with _ur.urlopen("https://raw.githubusercontent.com/kentstephen/aef-earthwork/main/models/earthwork-lr.npz", timeout=60) as _r:
            _ew = np.load(_io.BytesIO(_r.read()))
    # the two models the card offers (M switches them): "current" (earthwork-lr.npz), taught on US lidar only, and
    # "abroad" (earthwork-lr-abroad.npz), taught with truth from outside the 48 states too. The other files in
    # models/ are experiments kept for the record, not offered on the map
    EW_MODELS = {"current": (_ew["w"].astype(np.float32), float(_ew["b"]))}
    _abp = os.path.join(_ewd, "earthwork-lr-abroad.npz")
    if os.path.exists(_abp):
        _m = np.load(_abp)
        EW_MODELS["abroad"] = (_m["w"].astype(np.float32), float(_m["b"]))
    # ONLY WHERE ALPHAEARTH CHANGED: the score is multiplied by clip((change - LO) / (HI - LO), 0, 1), change
    # = 1 - cos(b, a). The model reads each year's look too, and bare ground in both years (a graded pad at the
    # US sites it learned from) reads as dug: open desert, which barely changes (about 0.04), lit up wholesale.
    # Held out on 3DEP sites the gate costs no digging (mean AP .53 to .54) and takes the Cairo desert from
    # 50% to 4% lit, the Sahara from 51% to 2%
    GATE_LO, GATE_HI = 0.05, 0.15

    # THE AREA SCALE (A on the map) reads the same score as log-odds, log(p / (1 - p)), clipped to LOGIT_LO..HI:
    # in places unlike the US sites the model learned from (Bujumbura) every chance rounds to 0%, yet its
    # log-odds still rank the ground; and unlike a plain log they keep the top apart too (zoomed out most
    # hexagons carry a high peak, and a log puts everything from 50% to 100% within 0.7 of 0)
    LOGIT_LO, LOGIT_HI = -30.0, 15.0

    def _earthwork(Vf, Vl, model="current", chunk=200_000):
        """The chance the ground moved, per row of Vf (first year) and Vl (last year), gated by how far AlphaEarth
        moved, and its log-odds (clipped to LOGIT_LO..HI); NaN where either is missing. model: a key of EW_MODELS."""
        EW_W, EW_B = EW_MODELS.get(model, EW_MODELS["current"])
        out = np.full(len(Vf), np.nan, np.float32)
        logs = np.full(len(Vf), np.nan, np.float32)
        for i in range(0, len(Vf), chunk):
            f, l = Vf[i:i + chunk], Vl[i:i + chunk]
            ok = np.isfinite(f).all(1) & np.isfinite(l).all(1)
            b = np.nan_to_num(f) / np.maximum(np.linalg.norm(np.nan_to_num(f), axis=1), 1e-9)[:, None]
            a = np.nan_to_num(l) / np.maximum(np.linalg.norm(np.nan_to_num(l), axis=1), 1e-9)[:, None]
            z = np.c_[b, a, a * b, (a - b) ** 2] @ EW_W + EW_B
            gate = np.clip(((1.0 - (a * b).sum(1)) - GATE_LO) / (GATE_HI - GATE_LO), 0, 1)
            out[i:i + chunk] = np.where(ok, gate / (1 + np.exp(-z)), np.nan)
            # log p = log(gate) - log(1 + e^-z), without the overflow of exp(-z) far below the bar; log(1 - p)
            # in float64 (p runs to within 1e-12 of 1)
            with np.errstate(divide="ignore"):
                lp = np.log(gate.astype(np.float64)) - np.logaddexp(0, -z.astype(np.float64))
                lq = np.log1p(-np.minimum(np.exp(lp), 1 - 1e-12))
            logs[i:i + chunk] = np.where(ok, np.clip(lp - lq, LOGIT_LO, LOGIT_HI), np.nan)
        return out, logs

    def build_frame(aef_by_year, y0, y1, res, model="current"):
        if aef_by_year.get(y0) is None or aef_by_year.get(y1) is None:
            return None
        # both years on the first year's cells (sorted, see _compact), NaN where the last has none
        base = aef_by_year[y0]["cell"]
        nfine = len(base)
        t = aef_by_year[y1]
        if t["cell"] is base:
            Vl = t["V"]
        else:
            Vl = np.full((nfine, 64), np.nan, np.float32)
            if len(t["cell"]):
                pos = np.clip(np.searchsorted(t["cell"], base), 0, len(t["cell"]) - 1)
                m_ = t["cell"][pos] == base
                Vl[m_] = t["V"][pos[m_]]
        par = pa.array(change_resolution(base, res)).to_numpy(zero_copy_only=False).astype(np.uint64)
        cellid = np.unique(par)
        n = len(cellid)
        hix = np.searchsorted(cellid, par)
        earth_f, elog_f = _earthwork(aef_by_year[y0]["V"], Vl, model)
        earth = np.full(n, -1.0, np.float32)
        elog = np.full(n, -np.inf, np.float32)
        okf = np.isfinite(earth_f)
        if n and okf.any():
            np.maximum.at(earth, hix[okf], earth_f[okf])
            np.maximum.at(elog, hix[okf], elog_f[okf])
        earth = np.where(earth >= 0, earth, np.nan).astype(np.float32)
        elog = np.where(np.isfinite(elog), elog, np.nan).astype(np.float32)
        nkids = np.bincount(hix, minlength=n).astype(np.int32)
        scored = np.isfinite(earth)
        cells = pa.table({"cell": pa.array(cellid), "earthwork": pa.array(earth), "finer_cells": pa.array(nkids)})
        return {
            "cells": cells, "cellid": cellid, "res": res, "earth": earth, "elog": elog, "years": [y0, y1], "y0": y0, "y1": y1,
            "score": f"Earthwork {y0} to {y1}: {int(scored.sum()):,} of {n:,} hexagons scored, peak of {nfine:,} finer cells",
        }

    return EW_MODELS, LOGIT_HI, LOGIT_LO, build_frame


@app.cell
def _(anywidget, asyncio, time, traitlets):
    # every deck.gl import names the same versions (iceye-view.py's set), so deck.gl-raster's layers
    # and the map's share one deck.gl
    _L = ",".join(f"@loaders.gl/{m}@4.4.3" for m in ["core", "gis", "loader-utils", "mvt", "terrain", "tiles", "wms", "schema",
                                                      "images", "worker-utils", "compression", "crypto", "zip", "math", "textures",
                                                      "draco", "gltf", "3d-tiles", "polyfills"])
    _LUMA = "@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/webgl@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/gltf@9.3.6"
    _DECK = "@deck.gl/core@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/geo-layers@9.3.10,@deck.gl/mesh-layers@9.3.10,@deck.gl/extensions@9.3.10"
    _DGR = "@developmentseed/deck.gl-raster@0.8.0,@developmentseed/geotiff@0.8.0,@developmentseed/proj@0.8.0,@developmentseed/affine@0.8.0,@developmentseed/morecantile@0.8.0,@developmentseed/raster-reproject@0.8.0"
    _DEPS = f"deps={_DECK},{_LUMA},apache-arrow@18.1.0,{_L},{_DGR}"

    class ChangeMap(anywidget.AnyWidget):
        """The map: the AlphaEarth hexagons in viridis on a plain basemap; press
        and hold for the Sentinel-2 imagery (the hexagons go while you hold),
        scroll while holding to change its year; the year card at the top
        right with the view's changes by year and the clicked hexagon's account.

        Kernel -> browser: `cells` (uint64 LE) with `hattrs` (4 bytes per
        hexagon: the year of its biggest step, 0 none, else year - 2000; how
        much it moved 1..255, 0 none; its main land cover, 0 none, else class
        index + 1; that class's share 0..255) and `hmeta` (JSON); `card`
        (JSON); `status`; `config`. Browser -> kernel: `view`, `pick`, `ctl`.
        Tiles are custom messages: `s2`, a year's mosaic."""

        cells = traitlets.Bytes(b"").tag(sync=True)
        hattrs = traitlets.Bytes(b"").tag(sync=True)
        hmeta = traitlets.Unicode("{}").tag(sync=True)
        config = traitlets.Unicode("{}").tag(sync=True)
        status = traitlets.Unicode("").tag(sync=True)
        card = traitlets.Unicode("").tag(sync=True)
        view = traitlets.Unicode("").tag(sync=True)
        pick = traitlets.Unicode("").tag(sync=True)
        ctl = traitlets.Unicode("").tag(sync=True)

        def __init__(self, **kw):
            super().__init__(**kw)
            self.tile_fn = None  # async (src, z, x, y, year) -> PNG bytes or None
            self.tile_times = {}  # (src, z, x, y, year) -> {"wait", "run"} ms, set by tile_fn
            self.on_msg(self._on_custom)

        def _on_custom(self, widget, content, buffers):
            if not isinstance(content, dict) or content.get("kind") != "tile":
                return
            try:
                asyncio.get_running_loop().create_task(self._tile(content, time.time()))
            except RuntimeError as e:
                self.send({"kind": "tile", "id": content.get("id"), "err": f"no loop: {e}"})

        async def _tile(self, c, t_recv=None):
            """A FAILURE IS AN ERROR, never an empty tile (deck caches an empty
            tile as loaded and the area stays blank for good)."""
            if self.tile_fn is None:
                self.send({"kind": "tile", "id": c["id"], "err": "no tile_fn (re-run the wiring cell)"})
                return
            key = (c.get("src", "s2"), int(c["z"]), int(c["x"]), int(c["y"]), int(c["year"]))
            t_run = time.time()
            try:
                png = await self.tile_fn(*key)
            except Exception as e:
                self.tile_times.pop(key, None)
                self.send({"kind": "tile", "id": c["id"], "err": f"{type(e).__name__}: {e}"})
                return
            # timings for the tests: wall clock at receipt and at send, the
            # loop's delay before the tile started, and tile_fn's own split
            kt = {"recv": t_recv, "sent": time.time(), "loop": 1e3 * (t_run - t_recv) if t_recv else None, **self.tile_times.pop(key, {})}
            if png is None:
                self.send({"kind": "tile", "id": c["id"], "empty": True, "kt": kt})
            else:
                self.send({"kind": "tile", "id": c["id"], "kt": kt}, buffers=[png])

        _css = r"""
        .at{--glass:rgba(26,29,33,.92);--card:rgba(26,29,33,.8);--glass-hi:#23272c;--line:rgba(255,255,255,.12);--text:#e6e9ec;--muted:#9ba5af;--faint:rgba(255,255,255,.18);--cool:#56b4e9;--sel:rgba(255,255,255,.08);--on:#15181b;
          position:relative;width:100%;background:#0e0e0e;color:var(--text);font:14px/1.45 "Instrument Sans",ui-sans-serif,system-ui,sans-serif;font-variant-numeric:tabular-nums;overflow:hidden;border-radius:10px;-webkit-font-smoothing:antialiased}
        .at.fit{position:fixed;inset:0;z-index:9999;border-radius:0}
        .at *{box-sizing:border-box}
        .at-pane{position:relative;width:100%}
        .at-map{position:absolute;inset:0}
        .at-map.holding{cursor:ns-resize}
        .at-map.holding.key{cursor:grab}
        /* the pair (P): Sentinel-2 on the left, the map on the right, one camera */
        .at-map2{position:absolute;top:0;bottom:0;left:0;right:50%;display:none;border-right:2px solid rgba(255,255,255,.35)}
        .at.pair .at-map{left:50%}
        .at.pair .at-map2{display:block}
        .at.pair .at-msg{left:75%}
        .at-side{position:absolute;left:50%;transform:translateX(-50%);bottom:56px;z-index:6;display:none;align-items:center;gap:12px;padding:6px 10px;font-size:12.5px;color:var(--muted);white-space:nowrap;max-width:calc(100% - 24px);overflow:hidden}
        .at.pair .at-side{left:25%;bottom:40px;max-width:calc(50% - 24px);flex-wrap:wrap;justify-content:center;row-gap:4px;white-space:normal}
        .at-side > *{white-space:nowrap}
        .at-side.on{display:flex}
        .at-side b{color:var(--text);font-weight:600}
        .at-side i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
        .at-side i.ln{height:3px;vertical-align:2px}
        .at-side button{font:inherit;font-size:11.5px;padding:2px 7px;border-radius:6px;border:1px solid var(--line);background:transparent;color:inherit;cursor:pointer}
        .at-side kbd,.at-yc kbd{font:10.5px/1 ui-monospace,SFMono-Regular,Menlo,monospace;border:1px solid var(--line);border-radius:4px;padding:1px 4px}
        .at-glass{background:var(--glass);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border:1px solid var(--line);border-radius:12px;box-shadow:0 6px 22px rgba(0,0,0,.35)}
        /* the cards a little see-through; the stronger blur keeps the text clear */
        .at-panel.at-glass,.at-yc.at-glass,.at-side.at-glass{background:var(--card);backdrop-filter:blur(18px) saturate(1.15);-webkit-backdrop-filter:blur(18px) saturate(1.15)}
        .at button{font:inherit;color:inherit}
        .at button:focus-visible,.at input:focus-visible{outline:2px solid var(--cool);outline-offset:2px}
        .at-top{position:absolute;left:12px;top:12px;z-index:6;display:flex;flex-direction:column;gap:8px;align-items:flex-start;max-width:calc(100% - 420px)}
        .at-search{position:relative;z-index:2;display:flex;align-items:center;gap:8px;padding:0 12px;height:40px;width:270px}
        .at-search svg{flex:0 0 auto;opacity:.6}
        .at-search input{flex:1;min-width:0;background:none;border:0;color:var(--text);font:inherit;outline:none}
        .at-search input::placeholder{color:var(--muted)}
        .at-hits{position:absolute;left:-1px;right:-1px;top:46px;display:none;padding:4px;background:var(--glass-hi)}
        .at-hit{padding:7px 10px;border-radius:8px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .at-hit small{display:block;color:var(--muted);font-size:12px}
        .at-hit.sel{background:var(--sel)}
        .at-panel{display:flex;flex-direction:column;gap:8px;padding:9px 11px;width:360px;max-width:calc(100vw - 32px);box-sizing:border-box}
        .at-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
        .at-lab{font-size:12.5px;color:var(--muted);min-width:64px}
        .at-mkey{font:11px/1 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);border:1px solid var(--line);border-radius:4px;padding:2px 5px}
        .at-mwhy{flex-basis:100%;font-size:12px;color:var(--muted);margin:-2px 0 2px}
        .seg-s,.seg-f{display:flex;gap:2px;padding:2px;border:1px solid var(--line);border-radius:9px;width:max-content}
        .seg-s button,.seg-f button{border:0;background:none;color:var(--muted);padding:3px 10px;border-radius:7px;cursor:pointer}
        .seg-s button:hover,.seg-f button:hover{color:var(--text)}
        .seg-s button.on,.seg-f button.on{background:var(--text);color:var(--on)}
        .seg-f{margin-bottom:6px;font-size:12.5px}
        .seg-s.col{flex-direction:column;align-items:stretch}
        .at-kinds{display:flex;flex-direction:column;gap:2px;width:100%}
        .at-kinds button{display:flex;align-items:center;gap:7px;border:0;background:none;padding:2px 4px;border-radius:6px;cursor:pointer;text-align:left;color:var(--text);font-size:12.5px}
        .at-kinds button:hover{background:var(--sel)}
        .at-kinds button i{width:14px;height:14px;border-radius:3px;flex:none}
        .at-kinds button span{color:var(--muted)}
        .at-kinds button.off{opacity:.4}
        .at button.wait{opacity:.4}
        @keyframes at-ready{0%{box-shadow:0 0 0 0 rgba(86,180,233,.6)}100%{box-shadow:0 0 0 6px rgba(86,180,233,0)}}
        .seg-s button.ready{animation:at-ready 1.2s ease-out 2}
        .seg-s button.fresh{position:relative}
        .seg-s button.fresh::before{content:"";position:absolute;top:4px;left:3px;width:5px;height:5px;border-radius:50%;background:var(--cool)}
        .at-soon{justify-content:space-between;font-size:11.5px;color:var(--muted);margin-top:-4px}
        .at-soon .z{font-variant-numeric:tabular-nums;color:var(--text)}
        .at-kinds button.off i{background:none!important;border:1.5px dashed var(--muted)}
        .at-kind-dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
        .seg-s.col button{text-align:left;display:flex;align-items:center;justify-content:space-between;gap:14px}
        .seg-s kbd{font:11px/1 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);border:1px solid var(--line);border-radius:4px;padding:2px 5px}
        .seg-f kbd{font:10.5px/1 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);border:1px solid var(--line);border-radius:4px;padding:1px 4px;margin-left:3px}
        .seg-f button.on kbd{color:var(--on);border-color:rgba(0,0,0,.35)}
        .at-yc .yr .comp{display:block;font-weight:600;margin-bottom:2px}
        .at-yc .yr .comps{display:flex;gap:4px;flex-wrap:wrap;margin:2px 0 5px}
        .at-yc .yr .comps button{font:inherit;font-size:11.5px;padding:2px 7px;border-radius:6px;border:1px solid var(--line);background:transparent;color:inherit;cursor:pointer}
        .at-yc .yr .comps button.on{background:var(--text);color:var(--on);border-color:transparent}
        .yr kbd{font:10.5px/1 ui-monospace,SFMono-Regular,Menlo,monospace;border:1px solid var(--line);border-radius:4px;padding:1px 4px}
        .seg-s button.on kbd{color:var(--on);border-color:rgba(0,0,0,.35)}
        .at-row.top{align-items:flex-start}
        .at-row.top .at-lab{padding-top:5px}
        .at-hd{display:flex;align-items:center;justify-content:space-between;gap:8px;margin:-2px -4px -2px 0}
        .at-hd .t{font-size:12.5px;font-weight:600}
        .at-cb{flex:0 0 auto;border:0;background:none;color:var(--muted);cursor:pointer;width:26px;height:26px;border-radius:7px;display:inline-flex;align-items:center;justify-content:center;padding:0}
        .at-cb:hover{background:var(--sel);color:var(--text)}
        .at-cb svg{transition:transform .15s}
        .collapsed .at-cb svg{transform:rotate(-90deg)}
        .at-panel.collapsed{padding:2px 3px 2px 10px;gap:0;border-radius:10px;width:auto}
        .at-panel.collapsed .at-hd .t{font-size:12px}
        .at-panel.collapsed .at-cb{width:22px;height:22px}
        .at-panel.collapsed .at-row{display:none}
        .at-yc .yr .at-cb{margin-left:auto;align-self:flex-start;margin-top:-4px;margin-right:-8px}
        .at-yc.collapsed{width:auto}
        .at-yc.collapsed .yr span{max-width:120px}
        .at-yc.collapsed>:not(.yr){display:none}
        .at-key{display:flex;align-items:center;flex-wrap:wrap;gap:4px 8px;font-size:12.5px;color:var(--muted);flex:1 1 auto;min-width:0}
        .at-key .why{flex-basis:100%;white-space:normal;font-size:11.5px;line-height:1.35}
        .at-ramp{height:10px;border-radius:3px;width:150px}
        /* the area scale's floor (A): one thumb on the Years read slider's track, the ramp's inked part to its right */
        .at-afl-hd{flex-basis:100%}
        .at-afl-lo{min-width:52px}
        .at-afl{flex-basis:100%;display:flex;align-items:center;gap:10px;margin:2px 0 1px}
        .at-afl-win{width:190px}
        .at-afl-win input{pointer-events:auto}
        .at-afl-win input:focus-visible::-webkit-slider-thumb{box-shadow:0 0 0 2px var(--cool)}
        .at-afl-win .tks i.on{color:var(--text);font-weight:600}
        .at-win{position:relative;width:170px;height:28px;flex:0 0 auto}
        .at-win input{position:absolute;left:0;top:0;width:100%;height:22px;margin:0;background:none;pointer-events:none;-webkit-appearance:none;appearance:none}
        .at-win input:focus{outline:none}
        .at-win input::-webkit-slider-runnable-track{background:none;height:22px}
        .at-win input::-moz-range-track{background:none;height:22px}
        .at-win input::-webkit-slider-thumb{pointer-events:auto;-webkit-appearance:none;appearance:none;width:14px;height:14px;margin-top:4px;border-radius:50%;background:var(--text);border:2px solid var(--on);box-shadow:0 0 0 1px rgba(255,255,255,.3);cursor:grab}
        .at-win input::-moz-range-thumb{pointer-events:auto;width:14px;height:14px;border-radius:50%;background:var(--text);border:2px solid var(--on);cursor:grab}
        .at-win .trk{position:absolute;left:8px;right:8px;top:9px;height:4px;background:var(--faint);border-radius:2px}
        .at-win .spn{position:absolute;top:9px;height:4px;background:var(--text);border-radius:2px}
        .at-win .tks{position:absolute;left:8px;right:8px;top:19px;display:flex;justify-content:space-between;font-size:9px;color:var(--muted);line-height:1}
        .at-win .tks span{width:0;display:flex;justify-content:center}
        .at-win .tks i{font-style:normal}
        .at-wtxt{font-size:12.5px;white-space:nowrap}
        .at-tools{position:absolute;right:12px;top:12px;z-index:7;display:flex;gap:8px}
        .at-btn{height:40px;min-width:40px;padding:0 13px;display:inline-flex;align-items:center;justify-content:center;gap:7px;cursor:pointer;white-space:nowrap}
        .at-btn:hover{border-color:rgba(255,255,255,.3)}
        .at-btn.on{background:var(--text);color:var(--on);border-color:var(--text)}
        .at-bar{position:absolute;left:0;right:0;top:0;height:3px;z-index:9;overflow:hidden;pointer-events:none;opacity:0;transition:opacity .3s}
        .at-bar.busy{opacity:1}
        .at-bar i{position:absolute;top:0;height:3px;width:28%;background:linear-gradient(90deg,transparent,var(--text),transparent);animation:at-run 1.2s ease-in-out infinite}
        @keyframes at-run{0%{left:-28%}100%{left:100%}}
        .at-msg{position:absolute;left:50%;transform:translateX(-50%);bottom:16px;z-index:5;font-size:13px;color:var(--muted);padding:6px 11px;display:none;max-width:min(520px,calc(100% - 24px))}
        .at-msg.err{color:#e69f00;user-select:text}
        .at-msg-act{display:none;gap:6px;margin-left:10px;vertical-align:middle}
        .at-msg.err .at-msg-act{display:inline-flex}
        .at-msg-act button{border:1px solid var(--line);background:none;color:var(--muted);cursor:pointer;border-radius:6px;padding:0 7px;font:12px/1.5 "Instrument Sans",ui-sans-serif,system-ui,sans-serif}
        .at-msg-act button:hover{background:var(--sel);color:var(--text)}
        .at-yc{position:absolute;right:12px;top:60px;z-index:6;width:380px;max-width:calc(100% - 24px);padding:14px 16px 12px;transform-origin:top right}
        .at-yc .yr{display:flex;align-items:flex-end;gap:12px}
        .at-yc .yr b{font-size:56px;line-height:.86;font-weight:600;letter-spacing:-.035em;font-stretch:88%}
        .at-yc .yr span{font-size:12.5px;color:var(--muted);line-height:1.35;padding-bottom:2px}
        .at-yc .yr.quiet{align-items:center}
        .at-yc .yr.quiet span{padding-bottom:0}
        .at-yc .yr.quiet .at-cb{margin-top:-2px;align-self:center}
        .at-yc.holding .yr span{color:var(--text)}
        /* the imagery's year: the year and its colors on one line, the buttons and the keys under them, the
           card narrower while it shows (no tall empty corner over the year) */
        .at-yc.holding{width:300px}
        .at-yc .yr.img{display:grid;grid-template-columns:auto 1fr auto;column-gap:10px;align-items:end}
        .at-yc .yr.img b{font-size:44px}
        .at-yc .yr.img .comp{margin:0;padding-bottom:3px}
        .at-yc .yr.img .at-cb{align-self:start;margin-top:-4px}
        .at-yc .yr.img .comps{grid-column:1/-1;margin:10px 0 6px}
        .at-yc .yr.img .help{grid-column:1/-1;padding:0}
        .at-yc.collapsed .yr.img .comps,.at-yc.collapsed .yr.img .help{display:none}
        /* folded while the imagery shows: the size of the folded hint card (its padding, 120 px of text and
           the arrow), the year over its colors */
        .at-yc.collapsed.holding{width:auto}
        .at-yc.bare{width:auto}
        .at-yc.bare .yr span{max-width:120px}
        .at-yc.bare>p{max-width:152px}
        .at-yc.collapsed .yr.img{grid-template-columns:120px auto;column-gap:12px;align-items:center}
        .at-yc.collapsed .yr.img b{grid-column:1;grid-row:1;font-size:42px;line-height:.86}
        .at-yc.collapsed .yr.img .comp{grid-column:1;grid-row:2;padding:2px 0 0;font-weight:400}
        .at-yc.collapsed .yr.img .at-cb{grid-column:2;grid-row:1/3;align-self:center;margin-top:0}
        .at-yc h4{margin:14px 0 2px;font-size:13.5px;font-weight:600}
        .at-yc .sub{color:var(--muted);font-size:12.5px;margin:0 0 6px}
        .at-yc .hex{border-top:1px solid var(--line);margin-top:12px;padding-top:12px;position:relative}
        .at-yc .hex .place{color:var(--text);font-size:13px;font-weight:600;margin-bottom:2px;padding-right:28px}
        .at-yc .hex .place span{font-weight:400;color:var(--muted)}
        .at-yc .hex h3{margin:0 0 6px;font-size:16px;font-weight:600;letter-spacing:-.005em;padding-right:28px}
        .at-yc .hex p{margin:0 0 8px}
        .at-yc .coords{display:grid;grid-template-columns:1fr auto;gap:3px 8px;align-items:center;margin:6px 0 10px;padding:6px 8px;border:1px solid var(--line);border-radius:8px;font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}
        .at-yc .coords code{user-select:all;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
        .at-yc .coords button{border:1px solid var(--line);background:none;color:var(--muted);cursor:pointer;border-radius:6px;padding:1px 7px;font:12px/1.5 "Instrument Sans",ui-sans-serif,system-ui,sans-serif}
        .at-yc .coords button:hover{background:var(--sel);color:var(--text)}
        .at-yc .x{position:absolute;right:-6px;top:6px;border:0;background:none;color:var(--muted);cursor:pointer;width:28px;height:28px;border-radius:8px;font-size:17px;line-height:1}
        .at-yc .x:hover{background:var(--sel);color:var(--text)}
        /* the hexagon's card: small, floating by the click, compact until More */
        .at-yc.at-fc{display:none;right:auto;top:auto;width:300px;z-index:8;padding:10px 14px 9px;transform-origin:top left}
        .at-yc.at-fc.more{width:360px;max-height:calc(100% - 24px);overflow:auto}
        .at-fc .hex{border-top:0;margin-top:0;padding-top:0}
        .at-fc:not(.more) .hex>:not(.x):not(.place):not(h3:last-child):not(h4):not(p),.at-fc:not(.more) .hex>p.sub{display:none}
        .at-fc:not(.more) .hex h4{margin-top:2px}
        .at-yc .hexwrap .hex{margin-top:10px}
        .at-yc .hexwrap:not(.more) .hex>:not(.x):not(.place):not(h3:last-child):not(h4):not(p),.at-yc .hexwrap:not(.more) .hex>p.sub{display:none}
        .at-yc .hexwrap:not(.more) .hex h4{margin-top:6px}
        .at-yc .fcbar{display:flex;align-items:center;gap:10px;margin-top:4px;font-size:12px;color:var(--muted)}
        .at-yc .fcbar button{font:inherit;font-size:12px;padding:2px 9px;border-radius:6px;border:1px solid var(--line);background:transparent;color:var(--text);cursor:pointer}
        .at-yc .fcbar button:hover{background:var(--sel)}
        .at-fc .fcbar{display:flex;align-items:center;gap:10px;margin-top:4px;font-size:12px;color:var(--muted)}
        .at-fc .fcbar button{font:inherit;font-size:12px;padding:2px 9px;border-radius:6px;border:1px solid var(--line);background:transparent;color:var(--text);cursor:pointer}
        .at-fc .fcbar button:hover{background:var(--sel)}
        .at-yc svg text{font-size:10.5px;fill:var(--muted)}
        .at-yc svg .lbl{fill:var(--text);font-weight:600}
        .at-lc{display:grid;grid-template-columns:1fr 120px 36px;gap:4px 8px;align-items:center;font-size:12.5px;margin:4px 0 6px}
        .at-lc i{display:block;height:8px;border-radius:0 4px 4px 0;background:var(--text);opacity:.72}
        .at-lc span:nth-child(3n){text-align:right;color:var(--muted)}
        .at-tip{position:absolute;z-index:10;pointer-events:none;background:var(--glass-hi);border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:12.5px;line-height:1.4;display:none;box-shadow:0 4px 14px rgba(0,0,0,.35);max-width:260px}
        .at-more{position:absolute;right:12px;top:60px;z-index:8;width:300px;padding:8px;display:none}
        .at-more .item{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:9px 10px;border-radius:9px}
        .at-more .item:hover{background:var(--sel)}
        .at-more .item small{display:block;color:var(--muted);font-size:12px}
        .at-more hr{border:0;border-top:1px solid var(--line);margin:4px 6px}
        .at-chip{border:1px solid var(--line);background:var(--glass-hi);border-radius:999px;padding:4px 12px;cursor:pointer}
        .at-chip:hover{border-color:rgba(255,255,255,.35)}
        .at-sw{position:relative;width:34px;height:20px;flex:0 0 auto;border-radius:999px;background:rgba(255,255,255,.2);border:0;cursor:pointer;transition:background .2s}
        .at-sw::after{content:"";position:absolute;left:3px;top:3px;width:14px;height:14px;border-radius:50%;background:#fff;transition:left .2s}
        .at-sw.on{background:var(--cool)}
        .at-sw.on::after{left:17px}
        .at-more input[type=range]{width:120px;accent-color:var(--cool)}
        .at-about{position:absolute;inset:0;z-index:20;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.5)}
        .at-about .box{width:min(620px,calc(100% - 32px));max-height:calc(100% - 64px);overflow:auto;padding:22px 26px;line-height:1.55;background:var(--glass-hi)}
        .at-about h2{margin:0 0 10px;font-size:22px;font-weight:600;letter-spacing:-.01em}
        .at-about p{margin:0 0 10px;max-width:66ch}
        .at-about small{color:var(--muted)}
        .at .maplibregl-ctrl-group{border:1px solid var(--line);box-shadow:0 6px 22px rgba(0,0,0,.35);border-radius:10px;background:var(--glass)}
        .at .maplibregl-ctrl-group button+button{border-top-color:var(--line)}
        .at .maplibregl-ctrl button .maplibregl-ctrl-icon{filter:invert(1)}
        .at .maplibregl-ctrl-attrib{background:var(--card);color:var(--muted)}
        .at .maplibregl-ctrl-attrib a{color:var(--muted)}
        .at .maplibregl-ctrl-attrib-button{filter:invert(1)}
        @media (max-width:760px){.at-top{max-width:calc(100% - 24px)}.at-search{width:calc(100vw - 48px)}.at-yc{top:auto;bottom:12px;max-height:45%;overflow:auto}.at-tools{top:108px}}
        @media (prefers-reduced-motion:reduce){.at-bar i{animation:none;left:0;width:100%}}
        """

        _esm = r"""
        import maplibregl from "https://esm.sh/maplibre-gl@5.24.0";
        import {MapboxOverlay} from "https://esm.sh/@deck.gl/mapbox@9.3.10?__DEPS__";
        import {BitmapLayer, PathLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?__DEPS__";
        import {TileLayer, H3HexagonLayer} from "https://esm.sh/@deck.gl/geo-layers@9.3.10?__DEPS__";
        import {COGLayer, MosaicLayer} from "https://esm.sh/@developmentseed/deck.gl-geotiff@0.8.0?__DEPS__";
        import {CreateTexture} from "https://esm.sh/@developmentseed/deck.gl-raster@0.8.0/gpu-modules?__DEPS__";
        import {DecoderPool, GeoTIFF, PerOriginSemaphore} from "https://esm.sh/@developmentseed/geotiff@0.8.0?__DEPS__";
        import {latLngToCell, getResolution, cellToBoundary, cellToLatLng, isValidCell} from "https://esm.sh/h3-js@4.5.0";
        import {Protocol as PMProtocol} from "https://esm.sh/pmtiles@4.5.0";
        maplibregl.addProtocol("pmtiles", new PMProtocol().tile);

        // dark basemap, trying it; light was
        // https://basemaps.cartocdn.com/gl/positron-gl-style/style.json
        const STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
        const FONTS = "https://fonts.googleapis.com/css2?family=Instrument+Sans:wdth,wght@75..100,400..700&display=swap";
        const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
        const fmt = (n) => Number(n).toLocaleString("en-US");
        const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
        const INK = [230, 233, 236];

        function bytesOf(v) {
          if (!v) return null;
          if (v instanceof DataView) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
          if (v instanceof ArrayBuffer) return new Uint8Array(v);
          if (v.buffer) return new Uint8Array(v.buffer, v.byteOffset || 0, v.byteLength);
          return null;
        }
        const copyOf = (u8) => u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
        const el_ = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
        const ICON = {
          search: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
          more: '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="1.8"/><circle cx="12" cy="12" r="1.8"/><circle cx="19" cy="12" r="1.8"/></svg>',
          expand: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>',
          chev: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M6 9l6 6 6-6"/></svg>',
          pair: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 4v16"/></svg>',
          shrink: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5"/></svg>',
        };
        // how much a hexagon moved, in words, from its 0..1 level in this view
        const howMuch = (t) => t >= 0.75 ? "a lot" : t >= 0.4 ? "a fair amount" : t >= 0.15 ? "a little" : "barely";
        const FAIR = 1 + Math.round(254 * 0.4);  // the level byte at "a fair amount"
        const HB = 17;  // bytes per hexagon in hattrs (the 9th: 1 on built ground; 10 to 16 the model's; 17 earthwork, see _paint)
        // kinds of change, largest first: Okabe-Ito, made to stay apart for
        // red-weak and other color vision; quiet ground (0) faint gray
        const KIND_RGB = [[230, 159, 0], [86, 180, 233], [0, 158, 115], [240, 228, 66], [0, 114, 178], [204, 121, 167]];
        const kindCss = (k) => `rgb(${KIND_RGB[(k - 1) % KIND_RGB.length].join(",")})`;
        const lcPair = (a, b) => a === b ? `stays ${a}` : `${a} → ${b}`;
        // [[year, class], ..] in runs: "<b>cropland</b> 2021 to 2022, then <b>built-up</b> 2023 to 2025"
        function readRuns(rs) {
          const out = [];
          for (const [y, nm] of rs) {
            const last = out[out.length - 1];
            if (last && last.nm === nm && last.y1 === y - 1) last.y1 = y;
            else out.push({nm, y0: y, y1: y});
          }
          return out.map((r) => `<b>${r.nm ? r.nm : "unread"}</b> ${r.y0 === r.y1 ? r.y0 : `${r.y0} to ${r.y1}`}`).join(", then ");
        }
        // the whole history, when read (zoomed in): codes from _trajectory
        const HIST_WORD = {1: "one step, then held", 2: "came back", 3: "changes this much most years", 4: "kept changing after", 5: "too recent to tell"};
        function histText(code, k, hy, ratio) {
          if (code === 1) return `One step into ${k}, then it held through ${hy[1]}.`;
          if (code === 2) return `It changed into ${k}, then came back: by ${hy[1]} it is closer to the ground before than to ${k}. Fields, water and seasons do this; buildings rarely do.`;
          if (code === 3) return `It changes about this much most years${ratio != null ? ` (this step is ${ratio.toFixed(1)} times its usual step, ${hy[0]} to ${hy[1]})` : ""}: ground that turns over, like a field.`;
          if (code === 4) return `It kept changing after ${k}, more than the usual step most years, the way a site still building out, a quarry or a mine does.`;
          if (code === 5) return `The change is into ${hy[1]}, the last year AlphaEarth has: too recent to tell whether it holds.`;
          return "";
        }

        function render({model, el}) {
          let cfg = {};
          try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
          if (!document.getElementById("at-fonts")) {
            const f = document.createElement("link"); f.id = "at-fonts"; f.rel = "stylesheet"; f.href = FONTS; document.head.appendChild(f);
          }
          const mlcss = document.createElement("link");
          mlcss.rel = "stylesheet"; mlcss.href = "https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css";
          el.appendChild(mlcss);

          const S2Y = cfg.s2_years || [2022, 2023, 2024, 2025];
          const VIR = (cfg.viridis || "440154fde725").match(/.{6}/g).map((h) => [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)));
          const vir = (t) => { t = Math.max(0, Math.min(1, t)) * (VIR.length - 1); const i = Math.min(VIR.length - 2, Math.floor(t)), f = t - i; return VIR[i].map((v, j) => Math.round(v + (VIR[i + 1][j] - v) * f)); };
          const virCss = (n) => Array.from({length: n}, (_, i) => `rgb(${vir(i / (n - 1)).join(",")})`).join(",");
          const A_FILL = cfg.alpha_fill || 235, A_QUIET = cfg.alpha_quiet || 70, A_DIM = 45;
          const HEXZ = cfg.hex_zoom || 9, HOLD_MS = cfg.hold_ms || 200, SLOP = cfg.hold_slop || 5;
          const st = {
            gmode: "earth", want: "earth", area: false, areaFloor: 0, noPick: false, hideKinds: new Set(), focus: "all", y0: cfg.aef_from || 2022, y1: cfg.aef_to || 2025,
            imgYear: cfg.s2_year || S2Y[S2Y.length - 1], labels: true, s2scale: Number(cfg.s2_scale) || 1, s2comp: cfg.s2_comp || "tci",
            fit: !!cfg.fit, holding: false,
            // the pair (P) and its left side (Sentinel-2)
            pair: false, left: "s2",
          };

          // ---- the frame ----------------------------------------------------
          const root = el_("div", "at");
          const pane = el_("div", "at-pane");
          const mapEl = el_("div", "at-map");
          const mapEl2 = el_("div", "at-map2");
          const bar = el_("div", "at-bar", "<i></i>");
          // an error stays until the next status; copy takes its full text, x closes it
          const msg = el_("div", "at-msg at-glass", '<span class="at-msg-t"></span><span class="at-msg-act"><button data-act="copy">copy</button><button data-act="x" title="Close">\u00d7</button></span>');
          const msgTx = msg.querySelector(".at-msg-t");
          msg.addEventListener("click", (e) => {
            const b = e.target && e.target.closest && e.target.closest("button[data-act]");
            if (!b) return;
            e.stopPropagation();
            if (b.getAttribute("data-act") === "x") { msg.style.display = "none"; return; }
            const t = msgTx.textContent;
            const done = () => { b.textContent = "copied"; setTimeout(() => { b.textContent = "copy"; }, 1200); };
            const fallback = () => { const ta = document.createElement("textarea"); ta.value = t; ta.style.position = "fixed"; ta.style.opacity = "0"; root.appendChild(ta); ta.select(); try { document.execCommand("copy"); done(); } catch (e2) {} ta.remove(); };
            if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(done, fallback); else fallback();
          });
          pane.append(mapEl2, mapEl, bar, msg);
          root.appendChild(pane);
          el.appendChild(root);

          // top left: search, and the one control row for the hexagons
          const top = el_("div", "at-top");
          const search = el_("div", "at-search at-glass", ICON.search);
          const gc = el_("input"); gc.type = "search"; gc.placeholder = "Search a place or H3 string"; gc.autocomplete = "off"; gc.spellcheck = false;
          const hits = el_("div", "at-hits at-glass");
          search.append(gc, hits);
          const panel = el_("div", "at-panel at-glass");
          // every panel folds; the fold is remembered in this browser
          const keep = (k, v) => { try { if (v === undefined) return localStorage.getItem("aef-lc-" + k) === "1"; localStorage.setItem("aef-lc-" + k, v ? "1" : "0"); } catch (e) {} return false; };
          const panelHd = el_("div", "at-hd", `<span class="t">Earthwork</span>`);
          const panelCb = el_("button", "at-cb", ICON.chev);
          panelHd.appendChild(panelCb);
          panel.appendChild(panelHd);
          const foldPanel = (on) => { panel.classList.toggle("collapsed", on); panelCb.title = on ? "show the controls" : "fold the controls"; keep("panel", on); };
          panelCb.onclick = (e) => { e.stopPropagation(); foldPanel(!panel.classList.contains("collapsed")); };
          foldPanel(keep("panel"));
          const rowOf = (label) => { const r = el_("div", "at-row"); if (label) r.appendChild(el_("span", "at-lab", label)); panel.appendChild(r); return r; };
          // Kinds of change is drawn only zoomed in with its land cover read (hmeta.kinds_ready), and only
          // once asked for: the map never switches to it by itself, its button flashes when it is ready
          //
          const kindsWait = () => st.want === "kinds" && !hmeta.kinds_ready;
          const drawnMode = () => kindsWait() ? "much" : st.want;
          const segOf = (row, items, isOn, onClick) => {
            const seg = el_("div", "seg-s col");
            const bs = items.map(([k, label, title, key]) => { const b = el_("button", "", label); b.title = title ? title + (key ? " (" + key + ")" : "") : ""; if (key) b.appendChild(el_("kbd", "", key)); b.onclick = () => onClick(k); seg.appendChild(b); return b; });
            row.appendChild(seg);
            return () => items.forEach(([k], i) => bs[i].classList.toggle("on", isOn(k)));
          };
          const rFill = rowOf("Color by");
          rFill.classList.add("top");
          // KINDS OF CHANGE COMMENTED OUT: AEF Change on built ground only
          const styleSeg = segOf(rFill, [["earth", "Earthwork", "the chance the ground itself moved (dug, filled, graded) between the first and last year read: a model taught by 3DEP repeat lidar", "E"],
                                          /* ["kinds", "Kinds of change", "the ground that moved most, grouped by the way it moved: the same color changed the same way. Click a kind in the key to hide or show it", "A"], */
                                          ],
                                  (k) => k === st.gmode, (k) => { if (k === "kinds" && !hmeta.kinds_ready) return; st.want = k; st.gmode = drawnMode(); recolorHex(); styleRows(); renderYear(); update(); });
          // one layer (Earthwork): no choice to show
          rFill.style.display = "none";
          // the two models (EW_MODELS in the kernel), side by side under the title, the one drawn lit and in
          // words under them; M switches them
          const MODELS = cfg.ew_models || ["current"];
          const MODEL_INFO = {
            current: ["Current", "Taught on US lidar only. Stronger in the US."],
            abroad: ["Ex-US", "Taught with truth from outside the US too. Stronger outside it."],
          };
          st.model = cfg.model || "current";
          const rModel = rowOf("Model");
          const styleModelSeg = segOf(rModel, MODELS.map((k) => [k, (MODEL_INFO[k] || [k])[0], (MODEL_INFO[k] || [k, ""])[1], ""]), (k) => k === st.model, (k) => setModel(k));
          rModel.querySelector(".seg-s").classList.remove("col");
          const modelKey = el_("kbd", "at-mkey", "M");
          modelKey.title = "M swaps the model";
          rModel.appendChild(modelKey);
          const modelWhy = el_("span", "at-mwhy");
          rModel.appendChild(modelWhy);
          const styleModel = () => { styleModelSeg(); modelWhy.textContent = (MODEL_INFO[st.model] || ["", ""])[1]; };
          styleModel();
          function setModel(k) { if (k === st.model) return; st.model = k; send("model", {model: k}); styleModel(); note(`Model: ${(MODEL_INFO[k] || [k])[0]}`, 2500); }
          if (MODELS.length < 2) rModel.style.display = "none";
          // while Kinds of change waits: its button grayed, and a line under it with the zoom it
          // appears at and the zoom now
          const rSoon = rowOf("");
          rSoon.classList.add("at-soon");
          const soonTxt = el_("span", "t"), soonZ = el_("span", "z");
          rSoon.append(soonTxt, soonZ);
          const styleSoon = () => {
            const z = map ? map.getZoom() : 0, KZ = cfg.kinds_zoom || 11.8;
            const show = st.gmode !== "earth" && z >= HEXZ && !hmeta.kinds_ready;
            rSoon.style.display = show ? "" : "none";
            if (!show) return;
            // what reads the ground now: the view's land cover reader
            soonTxt.textContent = z < KZ ? `Built ground read from ESA WorldCover 2021 until zoom ${KZ}` : "Built ground: reading the land cover";
            soonZ.textContent = `zoom ${z.toFixed(1)}`;
          };
          // the moment Kinds of change is ready, a soft ring on its button, twice; and a small dot on it while
          // it is ready but AEF Change is the one chosen
          let kindsWaited = false;
          const styleFill = () => {
            styleSeg();
            // the Kinds of change button's wait / ready cues, out with it
            // const b = rFill.querySelector("button"), ready = !!hmeta.kinds_ready;
            // b.classList.toggle("wait", !ready);
            // if (ready && kindsWaited) { b.classList.remove("ready"); void b.offsetWidth; b.classList.add("ready"); }
            // kindsWaited = !ready;
            // b.classList.toggle("fresh", ready && st.want !== "kinds");
            styleSoon();
          };
          const rKey = rowOf("");
          rKey.classList.add("keep");
          const keyEl = el_("span", "at-key");
          rKey.appendChild(keyEl);
          // the window: the years AlphaEarth is read over; drag either end, it rereads on release
          const rWin = rowOf("Years read");
          const aefYears = cfg.aef_years || [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
          const win = el_("span", "at-win");
          const wTrk = el_("span", "trk"), wSpn = el_("span", "spn"), wTks = el_("span", "tks");
          for (const y of aefYears) wTks.appendChild(el_("span", "", `<i>’${String(y).slice(-2)}</i>`));
          const mkR = () => { const r = el_("input"); r.type = "range"; r.min = 0; r.max = aefYears.length - 1; r.step = 1; r.title = "the years AlphaEarth is read over: drag either end, it rereads when you let go"; return r; };
          const rFrom = mkR(), rTo = mkR();
          const wTxt = el_("span", "at-wtxt");
          win.append(wTrk, wSpn, wTks, rFrom, rTo);
          rWin.append(win, wTxt);
          function styleWin() {
            const i0 = Math.max(0, aefYears.indexOf(st.y0)), i1 = Math.max(0, aefYears.indexOf(st.y1)), n = Math.max(1, aefYears.length - 1);
            rFrom.value = i0; rTo.value = i1;
            rFrom.style.zIndex = i0 === n ? 3 : 2; rTo.style.zIndex = i1 === 0 ? 3 : 2;
            const usable = (win.clientWidth || 170) - 16;
            wSpn.style.left = (8 + usable * i0 / n) + "px"; wSpn.style.width = (usable * (i1 - i0) / n) + "px";
            wTxt.textContent = `${st.y0} to ${st.y1}`;
          }
          const onDrag = (which) => {
            let a = Number(rFrom.value), b = Number(rTo.value);
            if (a >= b) { if (which === "from") a = b - 1; else b = a + 1; }
            a = Math.max(0, a); b = Math.min(aefYears.length - 1, b);
            st.y0 = aefYears[a]; st.y1 = aefYears[b]; styleWin();
          };
          let winSent = [st.y0, st.y1];
          const winRelease = () => { if (st.y0 !== winSent[0] || st.y1 !== winSent[1]) { winSent = [st.y0, st.y1]; send("aef"); update(); } };
          rFrom.addEventListener("input", () => onDrag("from")); rTo.addEventListener("input", () => onDrag("to"));
          rFrom.addEventListener("change", winRelease); rTo.addEventListener("change", winRelease);
          try { new ResizeObserver(styleWin).observe(win); } catch (e) {}
          function styleKey() {
            const y0 = hmeta.y0 || st.y0, y1 = hmeta.y1 || st.y1;
            const out_ = map && map.getZoom() < HEXZ;
            // the title is what is drawn, open or folded
            panelHd.querySelector(".t").textContent = {earth: "Earthwork", kinds: "Kinds of change", much: "AEF Change"}[st.gmode] || "AEF Change";
            styleSoon();
            if (out_) {
              keyEl.innerHTML = `<span class="why">Zoom in to ${HEXZ} for the hexagons.</span>`;
              return;
            }
            if (st.gmode === "kinds") {
              const ks = hmeta.kinds || [];
              const FOC = [["all", "All", "every hexagon in full", "Q"], ["built", "Built", "full only where the land cover changed and was built-up, road or construction in some year", "W"]];
              let h = `<div class="seg-f">` + FOC.map(([k, t, tip, key]) => `<button data-focus="${k}" class="${st.focus === k ? "on" : ""}" title="${tip} (${key})">${t} <kbd>${key}</kbd></button>`).join("") + `</div><div class="at-kinds">`;
              ks.forEach((q, j) => {
                const k = j + 1, off = st.hideKinds.has(k);
                h += `<button data-kind="${k}" class="${off ? "off" : ""}" title="${off ? "show" : "hide"} kind ${k}"><i style="background:${kindCss(k)}"></i><b>${k}</b><span>${fmt(q.n)}${q.year ? `, most ${q.year}` : ""}${q.from ? `, ${lcPair(q.from, q.to)}` : ""}</span></button>`;
              });
              h += `</div><span class="why">The ground that moved most from ${y0} to ${y1}, grouped by the way it moved: one color, one way of changing. Beside each: hexagons, the most common year, and what AlphaEarth reads most of it as in ${y0} and in ${y1}, taught in this view by ${esc(hmeta.lc_source || "land cover maps")}. Click one to hide it.</span>`;
              keyEl.innerHTML = h;
              keyEl.querySelectorAll("[data-focus]").forEach((b) => { b.onclick = (e) => { e.stopPropagation(); st.focus = b.dataset.focus; recolorHex(); styleKey(); update(); }; });
              keyEl.querySelectorAll("[data-kind]").forEach((b) => { b.onclick = (e) => { e.stopPropagation(); const k = +b.dataset.kind; st.hideKinds.has(k) ? st.hideKinds.delete(k) : st.hideKinds.add(k); recolorHex(); renderYear(); styleKey(); update(); }; });
              return;
            }
            if (st.gmode === "earth" && st.area && areaSt) {
              const top = Math.round(100 * areaSt.topGlobal);
              const k0 = st.areaFloor || 0, fl = AREA_FLOORS[k0], nF = AREA_FLOORS.length - 1;
              const phrase = (f) => (f[1] === "middle" ? "middle" : f[1] + " (below it, quiet ground)");
              keyEl.innerHTML = `<span class="at-afl-hd">Unusual for this area, ${y0} to ${y1}</span><span class="at-afl-lo">${fl[1]}</span> <i class="at-ramp" style="background:linear-gradient(90deg,${virCss(8)})"></i> most unusual`
                + `<span class="at-afl"><span>Color from</span><span class="at-win at-afl-win"><span class="trk"></span><span class="spn"></span>`
                + `<span class="tks">${AREA_FLOORS.map((f, k) => `<span><i class="${k === k0 ? "on" : ""}">${f[2]}</i></span>`).join("")}</span>`
                + `<input type="range" min="0" max="${nF}" step="1" value="${k0}" aria-label="where the color starts"></span></span>`
                + `<span class="why">Each hexagon against the rest of this area (A again for the normal scale): the color starts at the area's <span class="at-afl-ph">${phrase(fl)}</span>, full ink at its top tenth of a percent. Every area has a top, even where nothing was dug: a higher start keeps only its most unusual ground. On the normal scale this area's top reaches <b>${top}%</b>${top < 5 ? ", so even its most unusual ground is quiet" : ""}.</span>`;
              // the inked part of the ramp: from the thumb to the right end
              const spn = keyEl.querySelector(".at-afl-win .spn");
              const place = (k) => { spn.style.left = `calc(8px + (100% - 16px) * ${k / nF})`; spn.style.right = "8px"; };
              place(k0);
              const rng = keyEl.querySelector(".at-afl input");
              rng.oninput = (e) => {
                e.stopPropagation();
                const k = Number(rng.value), f = AREA_FLOORS[k];
                st.areaFloor = k; recolorHex(); update(); place(k);
                keyEl.querySelectorAll(".at-afl-win .tks i").forEach((t, j) => t.classList.toggle("on", j === k));
                keyEl.querySelector(".at-afl-lo").textContent = f[1];
                keyEl.querySelector(".at-afl-ph").textContent = phrase(f);
              };
              rng.onclick = rng.onpointerdown = (e) => e.stopPropagation();
              return;
            }
            if (st.gmode === "earth") { keyEl.innerHTML = `Ground moved, ${y0} to ${y1}: unlikely <i class="at-ramp" style="background:linear-gradient(90deg,${virCss(8)})"></i> likely<span class="why">The chance the ground itself was dug, filled or graded, from AlphaEarth's ${y0} and ${y1} by a model taught on 3DEP repeat lidar. Each hexagon shows its highest-scoring patch.</span>`; return; }
            keyEl.innerHTML = `Built ground (built-up, road or construction in ${y1}): barely <i class="at-ramp" style="background:linear-gradient(90deg,${virCss(8)})"></i> a lot, ${y0} to ${y1}`;
          }
          function styleRows() { styleFill(); styleWin(); styleKey(); }
          top.append(search, panel);
          pane.appendChild(top);

          // top right: settings and fill the window, then the year card
          const tools = el_("div", "at-tools");
          const bMore = el_("button", "at-btn at-glass", ICON.more); bMore.title = "settings and about";
          const bFit = el_("button", "at-btn at-glass", ICON.expand); bFit.title = "fill the window (X); full screen (F)";
          const bPair = el_("button", "at-btn at-glass", ICON.pair); bPair.title = "pair the map with Sentinel-2 (P)";
          bPair.onclick = () => setPair(!st.pair);
          tools.append(bPair, bMore, bFit);
          pane.appendChild(tools);
          const yc = el_("div", "at-yc at-glass");
          // the old floating hexagon card: kept hidden (the hexagon drops down in the top right card)
          const fc = el_("div", "at-yc at-fc at-glass");
          pane.appendChild(yc);
          pane.appendChild(fc);
          const tip = el_("div", "at-tip");
          pane.appendChild(tip);
          // what the left side of the pair shows, with its key
          const side = el_("div", "at-side at-glass");
          pane.appendChild(side);
          const more = el_("div", "at-more at-glass");
          const item = (title, sub, ctl) => { const r = el_("div", "item"); r.append(el_("div", "", `${title}${sub ? `<small>${sub}</small>` : ""}`), ctl); more.appendChild(r); return r; };
          const sw = (get, set) => { const b = el_("button", "at-sw"); b.setAttribute("role", "switch"); const sty = () => { b.classList.toggle("on", !!get()); b.setAttribute("aria-checked", String(!!get())); }; b.onclick = () => { set(!get()); sty(); }; sty(); b.sty = sty; return b; };
          const gam = el_("input"); gam.type = "range"; gam.min = 0.3; gam.max = 2.5; gam.step = 0.1; gam.value = st.s2scale;
          gam.title = "imagery brightness (gamma); double-click for 1.0";
          let gamT = null;
          gam.oninput = () => { st.s2scale = Number(gam.value); clearTimeout(gamT); gamT = setTimeout(() => send("s2scale"), 250); };
          gam.ondblclick = () => { gam.value = 1; gam.oninput(); };
          item("Imagery brightness", "; and ' step it", gam);
          // the imagery colors (S2_COMPOSITES in the kernel): false colors from the raw bands,
          // stretched for the view, where the true color image clips bright ground
          const COMPS = [["tci", "True color", "Earth Genome's true color image"],
                         ["urban", "Urban", "B12 B11 B4 from the raw bands, each stretched to the view: built ground and bare soil apart by hue"],
                         ["blueyellow", "Blue-yellow", "B11 B11 B2 from the raw bands, each stretched to the view: sand yellow, concrete blue"],
                         // the indexes are drawn in the browser only (no kernel tiles for them)
                         ...(cfg.s2_gpu !== false ? [
                           ["ndbi", "NDBI", "(B11 - B8A) / (B11 + B8A), stretched to the view: built-up and bare ground orange, vegetation and water blue"],
                           ["mineral", "Mineral", "Bare ground by its minerals, stretched to the view: iron oxides (B4 / B2) yellow, clays (B11 / B12) blue, vegetation and water dark"]] : [])];
          // what the hold card calls each, always shown while the imagery is up
          const COMP_NAME = {tci: "True color (TCI)", urban: "False color: Urban (B12 B11 B4)", blueyellow: "False color: Blue-yellow (B11 B11 B2)",
                             ndbi: "NDBI: orange built-up or bare, blue vegetation or water", mineral: "Mineral: yellow iron oxides, blue clays, dark vegetation and water"};
          const compBox = el_("div");
          const styleComp = segOf(compBox, COMPS.map(([k, l, t]) => [k, l, t, ""]), (k) => k === st.s2comp, (k) => setComp(k));
          // new colors are new tiles: the year shown loads alone first again (s2Warm)
          function setComp(k) { st.s2comp = k; s2Warm = false; send("s2comp", {comp: k}); styleComp(); renderYear(); }
          item("Imagery colors", "C steps them", compBox);
          const swLab = sw(() => st.labels, (v) => { st.labels = v; labels(v); });
          item("Place names", "", swLab);
          more.appendChild(el_("hr"));
          const bAbout = el_("button", "at-chip", "About this map"); bAbout.style.margin = "4px 10px 6px";
          more.appendChild(bAbout);
          pane.appendChild(more);

          const about = el_("div", "at-about");
          about.innerHTML = `<div class="box at-glass">
            <h2>Earthwork</h2>
            <p><b>Earthwork</b> (E) is the chance the ground itself was dug, filled or graded between the first and last year read: a model on AlphaEarth taught where 3DEP lidar flew the same ground twice. Each hexagon shows its highest-scoring finer cell, so a single dig stands out. Pair (P) with Sentinel-2 to see what it is.</p>
            <p>Earthwork is drawn in H3 hexagons from zoom ${HEXZ}. <b>Click</b> a hexagon for its account.</p>
            <p><b>Hold space</b> to see the Sentinel-2 yearly imagery (Earth Genome, 2022 to 2025) instead of the hexagons; scroll while holding to step through the years.</p>
            <p><small>Keys: hold space for the imagery, scroll or [ and ] for its year, B its first year or its latest; P pairs the map with Sentinel-2; A the area scale (unusual for this area) and back; Q the tooltip and picking off and on; Shift + arrows turn and tilt the map; F full screen; ; and ' its brightness; - = and _ + the years read; L place names; / search (a place, or paste an H3 string); X fill the window; Esc close.</small></p>
            <p><small>AlphaEarth Foundations by Google and Google DeepMind (CC BY 4.0). ESA WorldCover 10 m 2021 v200, contains modified Copernicus Sentinel data processed by the ESA WorldCover consortium (CC BY 4.0). Impact Observatory, Microsoft and Esri 10 m annual land use and land cover v02, via Microsoft Planetary Computer (CC BY 4.0). Overture Maps transportation and land use, &copy;&nbsp;OpenStreetMap contributors (ODbL), from Overture's PMTiles. Sentinel-2 mosaics by Earth Genome (CC BY 4.0). Place names from Overture Maps divisions, &copy;&nbsp;OpenStreetMap contributors, Overture Maps Foundation (ODbL), with geoBoundaries, Esri Community Maps contributors and LINZ (CC BY 4.0): the PMTiles and, via Source Cooperative, fused/overture. Search by Photon over OpenStreetMap (ODbL). Basemap by Carto.</small></p>
            <div style="margin-top:12px"><button class="at-chip">Close</button></div></div>`;
          pane.appendChild(about);
          about.querySelector("button").onclick = () => { about.style.display = "none"; };
          about.onclick = (e) => { if (e.target === about) about.style.display = "none"; };
          bAbout.onclick = () => { more.style.display = "none"; about.style.display = "flex"; };

          const send = (act, extra) => {
            model.set("ctl", JSON.stringify(Object.assign({act, s2scale: st.s2scale, y0: st.y0, y1: st.y1, n: Date.now()}, extra || {})));
            model.save_changes();
          };

          // ---- status ------------------------------------------------------------
          const ERR = /failed|error|no match|search:|timed? ?out|zoom in|^(deck|map|load|boot|\w+ tile):/i;
          let msgT = null;
          const note = (t, ms) => { msgTx.textContent = t; msg.style.display = t ? "block" : "none"; msg.classList.toggle("err", ERR.test(t)); clearTimeout(msgT); if (ms) msgT = setTimeout(() => { msg.style.display = "none"; }, ms); };
          const say = (t) => {
            t = (t || "").replace(/​/g, "");
            if (ERR.test(t)) { note(t); bar.classList.remove("busy"); return; }
            const busy = t.split(" | ").filter((p) => p.includes("…"));
            const names = [];
            // each dataset being read, by name, once
            for (const p of busy)
              for (const [nm, re] of [["AlphaEarth", /AlphaEarth/], ["WorldCover", /WorldCover/], ["Impact Observatory", /Impact Observatory/], ["the model", /^the model|fitting/]])
                if (re.test(p) && !names.includes(nm)) names.push(nm);
            bar.classList.toggle("busy", names.length > 0);
            note(names.length ? "Loading " + (names.length > 1 ? names.slice(0, -1).join(", ") + " and " + names[names.length - 1] : names[0]) + "…" : "");
          };

          // ---- tiles ----------------------------------------------------------------
          const pending = new Map();
          let tseq = 0;
          const tstat = {asked: 0, got: 0, empty: 0, err: 0, abort: 0};
          const tlog = [];  // per tile, for the tests: asked, arrived, kernel times, bytes
          const tlogOf = new Map();
          model.on("msg:custom", (m, buffers) => {
            if (!m || m.kind !== "tile") return;
            const p = pending.get(m.id);
            if (!p) return;
            pending.delete(m.id);
            const lg = tlogOf.get(m.id);
            if (lg) { lg.got = Date.now(); lg.kt = m.kt || null; lg.bytes = buffers && buffers.length ? (buffers[0].byteLength || 0) : 0; lg.err = m.err || null; tlogOf.delete(m.id); }
            if (m.err) { tstat.err++; p.reject(new Error(m.err)); return; }
            if (m.empty || !buffers || !buffers.length) { tstat.empty++; p.resolve(null); return; }
            tstat.got++;
            p.resolve(bytesOf(buffers[0]));
          });
          const ask = (src, year, index, signal) => new Promise((resolve, reject) => {
            const id = ++tseq; tstat.asked++;
            pending.set(id, {resolve, reject});
            const lg = {id, src, year, z: index.z, x: index.x, y: index.y, asked: Date.now()};
            tlog.push(lg); tlogOf.set(id, lg); if (tlog.length > 4000) tlog.splice(0, 1000);
            model.send({kind: "tile", id, src, year, x: index.x, y: index.y, z: index.z});
            if (signal) signal.addEventListener("abort", () => { if (pending.has(id)) lg.aborted = Date.now(); tlogOf.delete(id); pending.delete(id); tstat.abort++; const e = new Error("aborted"); e.name = "AbortError"; reject(e); });
          });
          const pngBitmap = (u8) => createImageBitmap(new Blob([u8], {type: "image/png"}));

          // ---- the hexagons -----------------------------------------------------------
          let areaSt = null;  // the area scale's numbers for the frame (areaStats)
          let hexes = [], N = 0, res = -1, hexIndex = new Map(), hattrs = null, hmeta = {}, hcol = null, hcol32 = null, hexSeq = 0, hover = null, picked = null, imgPick = null;
          // THE AREA SCALE (A): each hexagon's Earthwork log-odds (3rd byte, -30..15 as 1..255) against the rest of
          // the frame's hexagons, stretched from the floor (no ink) to their top 0.1% (full ink); in log-odds
          // so places the model scores ~0% everywhere still rank, and the top stays apart. topGlobal: the normal scale's value at that top.
          // THE FLOOR is a stepped slider in the key: the frame's median by default, or its top 25, 10, 5 or 1%. A
          // frame where nothing was dug still has a top, so its ramp paints noise; a higher floor keeps only the
          // frame's most unusual ground, with the whole ramp spread over it (at the lidar sites the top fifth of
          // the ramp is mostly real digging, its middle mostly not)
          const AREA_FLOORS = [[0.5, "middle", "Middle"], [0.75, "top 25%", "25%"], [0.9, "top 10%", "10%"], [0.95, "top 5%", "5%"], [0.99, "top 1%", "1%"]];
          function areaStats() {
            const hl = new Uint32Array(256), he = new Uint32Array(256);
            let n = 0;
            for (let i = 0; i < N; i++) { const v = hattrs[HB * i + 2]; if (v) { hl[v]++; he[hattrs[HB * i + 16]]++; n++; } }
            if (n < 20) return null;
            const at = (h, q) => { let c = 0; for (let v = 1; v < 256; v++) { c += h[v]; if (c >= q * n) return v; } return 255; };
            const cdf = new Float32Array(256);
            for (let v = 1, c = 0; v < 256; v++) { c += hl[v]; cdf[v] = c / n; }
            const top = at(hl, 0.999), lo = Math.min(at(hl, AREA_FLOORS[st.areaFloor || 0][0]), top - 1);
            return {lo, top, cdf, n, topGlobal: (at(he, 0.999) - 1) / 254};
          }
          const areaT = (v) => (v ? Math.max(0, Math.min(1, (v - areaSt.lo) / (areaSt.top - areaSt.lo))) : 0);
          // "top 2% of this area" for the hover and the card
          const areaWords = (i) => { if (!(st.area && areaSt)) return ""; const v = hattrs[HB * i + 2]; if (!v) return ""; const p = 100 * (1 - areaSt.cdf[v - 1]); return `; ${p < 1 ? "top " + (p < 0.1 ? "0.1" : p.toFixed(1)) + "%" : p <= 50 ? "top " + Math.round(p) + "%" : "lower half"} of this area`; };
          function recolorHex() {
            if (!N || !hattrs || hattrs.length !== HB * N) { hcol = null; hcol32 = null; return; }
            hcol = new Uint8Array(4 * N);
            hcol32 = new Uint32Array(hcol.buffer);
            areaSt = st.gmode === "earth" ? areaStats() : null;
            for (let i = 0; i < N; i++) {
              const a8 = HB * i, o = 4 * i, lv = hattrs[a8 + 1];
              let col, a;
              if (st.gmode === "earth") {
                // the chance the ground moved, on all ground: quiet ground faint, likely earthwork in full ink
                const v = hattrs[a8 + 16];
                if (!v) continue;
                const t = st.area && areaSt ? areaT(hattrs[a8 + 2]) : (v - 1) / 254;
                col = vir(t); a = Math.round(A_QUIET + (A_FILL - A_QUIET) * t);
              } else if (st.gmode === "kinds") {
                // its kind's color, fuller the more it moved; quiet ground faint gray
                if (!lv) continue;
                const kd = hattrs[a8 + 6], t = (lv - 1) / 254;
                if (!kd) { col = [150, 156, 162]; a = 40; }
                else if (st.hideKinds.has(kd)) continue;
                else {
                  col = KIND_RGB[(kd - 1) % KIND_RGB.length]; a = Math.round(110 + (A_FILL - 110) * t);
                  // the key's Built: all but built-up, road and construction
                  // that changed made more see-through
                  if (st.focus === "built" && hattrs[a8 + 7] !== 3) a = A_DIM;
                }
              } else {
                // built ground only (BUILT_SHARE in the kernel)
                if (!lv || !hattrs[a8 + 8]) continue;
                const t = (lv - 1) / 254;
                // quiet ground faint, change in full ink
                col = vir(t); a = Math.round(A_QUIET + (A_FILL - A_QUIET) * t);
              }
              hcol[o] = col[0]; hcol[o + 1] = col[1]; hcol[o + 2] = col[2]; hcol[o + 3] = a;
            }
            hexSeq++;
          }
          // Q: no tooltip, no hover outline and no picking, so the map can be looked at (or shot) clean
          const hexAt = (ll) => { if (st.noPick || res < 0 || !map || map.getZoom() < HEXZ) return -1; try { const h = latLngToCell(ll.lat, ll.lng, res); const i = hexIndex.get(h); return i == null || (st.gmode === "much" && hattrs && !hattrs[HB * i + 8]) ? -1 : i; } catch (e) { return -1; } };
          function hexWords(i) {
            const o = HB * i, yb = hattrs[o], lv = hattrs[o + 1];
            if (!lv) return "No AlphaEarth data here.";
            if (st.gmode === "earth") {
              const v = hattrs[o + 16];
              return v ? `<b>Earthwork ${Math.round(100 * (v - 1) / 254)}%</b>${areaWords(i)}: the chance the ground itself was dug, filled or graded, ${hmeta.y0 || st.y0} to ${hmeta.y1 || st.y1}` : "No AlphaEarth data here.";
            }
            let s;
            if (st.gmode === "kinds") {
              const kd = hattrs[o + 6];
              s = kd ? `<b>Kind ${kd}</b>, changed ${howMuch((lv - 1) / 254)}${yb ? `, most in ${2000 + yb}` : ""}` : `<b>Not grouped</b>: changed ${howMuch((lv - 1) / 254)}, less than the ground that moved most`;
              if (hattrs[o + 4]) s += `<br>History: ${HIST_WORD[hattrs[o + 4]]}`;
            } else {
              s = `<b>Changed ${howMuch((lv - 1) / 254)}</b>${yb ? `, most in ${2000 + yb}` : ""}`;
              if (hattrs[o + 4]) s += `<br>History: ${HIST_WORD[hattrs[o + 4]]}`;
            }
            return s;
          }

          // ---- the year card ------------------------------------------------------------
          let cardData = null;
          // one bar per year you can scroll to, plus any other year the window
          // dates; an imagery year the window cannot date is an empty slot
          function viewCounts() {
            const out = {}, dated = {};
            const y0 = hmeta.y0 || st.y0, y1 = hmeta.y1 || st.y1;
            const ys = [...new Set([...S2Y, ...Array.from({length: Math.max(0, y1 - y0)}, (_, i) => y0 + 1 + i)])].sort((a, b) => a - b);
            for (const y of ys) { out[y] = 0; dated[y] = y > y0 && y <= y1; }
            let total = 0;
            if (hattrs) for (let i = 0; i < N; i++) {
              const o = HB * i, lv = hattrs[o + 1];
              if (lv && hattrs[o + 8]) total++;
              if (st.gmode === "kinds") { const yb = hattrs[o], kd = hattrs[o + 6]; if (yb && kd && !st.hideKinds.has(kd) && out[2000 + yb] != null) out[2000 + yb]++; }
              else { const yb = hattrs[o]; if (yb && lv >= FAIR && hattrs[o + 8] && out[2000 + yb] != null) out[2000 + yb]++; }
            }
            return {years: out, dated, total};
          }
          // bars per year: one series, ink on a baseline; the imagery year in
          // full ink with its count, the others lighter; a tooltip per bar
          function yearBars(c) {
            const ys = Object.keys(c.years).map(Number);
            if (!ys.length) return "";
            const W = 346, H = 78, base = 60, gap = 6, bw = Math.min(56, (W - gap * (ys.length - 1)) / ys.length);
            const x0 = (W - (bw * ys.length + gap * (ys.length - 1))) / 2;
            const max = Math.max(1, ...ys.map((y) => c.years[y]));
            let s = `<svg width="${W}" height="${H}" role="img" aria-label="hexagons in view that changed a fair amount or more, by the year of their biggest change">`;
            ys.forEach((y, i) => {
              const v = c.years[y], h = v ? Math.max(3, (base - 14) * v / max) : 0, x = x0 + i * (bw + gap), cur = y === st.imgYear;
              const r = Math.min(4, h / 2);
              if (h) s += `<path d="M${x},${base} v${-(h - r)} q0,${-r} ${r},${-r} h${bw - 2 * r} q${r},0 ${r},${r} v${h - r} z" fill="${rgba(INK, cur ? 0.9 : 0.28)}"/>`;
              const why = c.dated[y] ? `${fmt(v)} hexagon${v === 1 ? "" : "s"} changed most between the ${y - 1} and ${y} pictures` : `${y} is outside the years read: widen them to ${y - 1} to date changes into ${y}`;
              s += `<rect x="${x - gap / 2}" y="0" width="${bw + gap}" height="${H}" fill="transparent" data-tip="${why}"/>`;
              if (!c.dated[y]) s += `<line x1="${x}" x2="${x + bw}" y1="${base - 1}" y2="${base - 1}" stroke="rgba(230,233,236,.3)" stroke-dasharray="2 3"/>`;
              if (cur && v) s += `<text class="lbl" x="${x + bw / 2}" y="${base - h - 4}" text-anchor="middle">${fmt(v)}</text>`;
              s += `<text x="${x + bw / 2}" y="${H - 4}" text-anchor="middle"${cur ? ' class="lbl"' : ""}>${y}</text>`;
            });
            s += `<line x1="0" x2="${W}" y1="${base + 0.5}" y2="${base + 0.5}" stroke="rgba(230,233,236,.25)"/></svg>`;
            return s;
          }
          // the clicked hexagon's steps, each as a multiple of that year's
          // median step in view: the biggest in full ink, a dashed line at 1
          function stepBars(rel, steps, years, big) {
            if (!rel || !rel.length) return "";
            const W = 346, H = 74, base = 56, gap = 6, n = rel.length, bw = Math.min(46, (W - gap * (n - 1)) / n);
            const x0 = (W - (bw * n + gap * (n - 1))) / 2;
            const vmax = Math.max(2, ...rel.filter((v) => v != null));
            const yOf = (v) => base - (base - 10) * Math.min(1, v / vmax);
            let s = `<svg width="${W}" height="${H}" role="img" aria-label="AlphaEarth year-to-year change for this hexagon">`;
            rel.forEach((v, i) => {
              const x = x0 + i * (bw + gap), y = years[i];
              if (v != null) {
                const top_ = yOf(v), h = base - top_, r = Math.min(4, h / 2);
                if (h > 0.5) s += `<path d="M${x},${base} v${-(h - r)} q0,${-r} ${r},${-r} h${bw - 2 * r} q${r},0 ${r},${r} v${h - r} z" fill="${rgba(INK, y === big ? 0.9 : 0.28)}"/>`;
              }
              s += `<rect x="${x - gap / 2}" y="0" width="${bw + gap}" height="${H}" fill="transparent" data-tip="${y - 1} to ${y}: ${v == null ? "no data" : `${v.toFixed(1)} times the usual step here that year (${steps[i].toFixed(3)})`}"/>`;
              s += `<text x="${x + bw / 2}" y="${H - 4}" text-anchor="middle"${y === big ? ' class="lbl"' : ""}>${n > 5 ? `’${String(y).slice(-2)}` : `’${String(y - 1).slice(-2)} to ’${String(y).slice(-2)}`}</text>`;
            });
            const ty = yOf(1); s += `<line x1="0" x2="${W}" y1="${ty}" y2="${ty}" stroke="rgba(230,233,236,.55)" stroke-dasharray="3 3"/><text x="${W}" y="${ty - 3}" text-anchor="end">usual step here</text>`;
            s += `<line x1="0" x2="${W}" y1="${base + 0.5}" y2="${base + 0.5}" stroke="rgba(230,233,236,.25)"/></svg>`;
            return s;
          }
          function hexSection(c) {
            if (!c || !c.kind) return "";
            let h = `<div class="hex"><button class="x" title="close (Esc)" aria-label="close">×</button>`;
            if (c.place && c.place.length) h += `<div class="place">${c.place.map((q) => typeof q === "string" ? esc(q) : esc(q.name) + (q.tag ? ` <span>(${esc(q.tag)})</span>` : "")).join(", ")}</div>`;
            if (c.kind === "note") return h + `<h3>${esc(c.title || "")}</h3></div>`;
            h += `<h3>This hexagon</h3>`;
            // its H3 string and center, each copyable, however it was picked
            if (c.cell) {
              let ll = null; try { ll = cellToLatLng(c.cell); } catch (e) {}
              const lat_lon = ll ? `${ll[0].toFixed(6)}, ${ll[1].toFixed(6)}` : "";
              h += `<div class="coords"><code title="H3 string">${esc(c.cell)}</code><button data-copy="${esc(c.cell)}">copy</button>`;
              if (lat_lon) h += `<code title="lat, long of the cell's center">${lat_lon}</code><button data-copy="${lat_lon}">copy</button>`;
              h += `</div>`;
            }
            // ONE VOICE PER LAYER: the card speaks for the layer the map is colored by, nothing else
            const mode = st.gmode;
            if (c.level == null) h += `<p>No AlphaEarth data here.</p>`;
            else if (mode === "earth") {
              const i = c.cell ? hexIndex.get(c.cell) : null, v = i != null && hattrs ? hattrs[HB * i + 16] : 0;
              h += `<h4>Earthwork</h4>`;
              h += v ? `<p><b>${Math.round(100 * (v - 1) / 254)}%</b>${i != null ? areaWords(i) : ""}: the chance the ground itself was dug, filled or graded from ${c.y0} to ${c.y1} (its highest-scoring patch).</p>` : `<p>No score here.</p>`;
              h += `<p class="sub">Pair with Sentinel-2 (<kbd>P</kbd>) or hold space to see what it is.</p>`;
            } else {
              h += `<h4>AEF Change</h4>`;
              h += `<p>The ground changed <b>${howMuch(c.level)}</b> from ${c.y0} to ${c.y1}, compared with the rest of the view. Its year-to-year change stood out most between the <b>${c.big - 1} and ${c.big}</b> pictures${c.stand != null ? `, ${c.stand.toFixed(1)} times the usual step in view that year` : ""}.</p>`;
              if (c.hist && c.ccode) h += `<p class="sub">${histText(c.ccode, c.big, c.hist, c.cratio)}${c.hbig > 0 && (c.hbig <= c.y0 || c.hbig > c.y1) ? ` Its biggest step from ${c.hist[0]} to ${c.hist[1]} was into ${c.hbig}, outside the years read.` : ""}</p>`;
              h += stepBars(c.rel, c.steps, c.step_years, c.big);
              if (S2Y.includes(c.big) && S2Y.includes(c.big - 1)) h += `<p class="sub">Hold space with the pointer near it and scroll between ${c.big - 1} and ${c.big} to see what happened.</p>`;
              else if (c.big) h += `<p class="sub">The imagery starts in ${S2Y[0]}, so there is no picture from before ${c.big} to compare.</p>`;
            }
            if (c.km2) h += `<p class="sub">${c.km2 < 0.1 ? `${Math.round(c.km2 * 1e6).toLocaleString("en-US")} m²` : `${c.km2.toFixed(2)} km²`} hexagon.</p>`;
            return h + `</div>`;
          }
          let ycFolded = keep("card"), ycOpenedFor = null;
          function renderYear() {
            yc.classList.toggle("holding", st.holding);
            const c = viewCounts();
            // the imagery year only while the imagery shows; otherwise the hint
            const cbH = `<button class="at-cb" title="${ycFolded ? "show the card" : "fold the card"}">${ICON.chev}</button>`;
            let h = st.holding
              ? `<div class="yr img"><b>${st.imgYear}</b><span class="comp">${COMP_NAME[st.s2comp] || ""}</span>${cbH}<span class="comps">${COMPS.map(([k, l, t]) => `<button data-comp="${k}" class="${k === st.s2comp ? "on" : ""}" title="${esc(t)}">${l}</button>`).join("")}</span><span class="help">Scroll or <kbd>[</kbd> <kbd>]</kbd> for another year, <kbd>B</kbd> ${S2Y[0]} or ${S2Y[S2Y.length - 1]}, <kbd>C</kbd> for colors, <kbd>F</kbd> for full screen. Let go to see the hexagons.</span></div>`
              : st.pair
              ? `<div class="yr quiet"><span><kbd>P</kbd> back to one map</span>${cbH}</div>`
              : `<div class="yr quiet"><span>Hold space for the Sentinel-2 imagery; <kbd>P</kbd> pairs it with the map</span>${cbH}</div>`;
            // the view's chart by year belongs to the layers that date things (one voice per layer)
            if (N && hattrs && st.gmode === "much") {
              h += st.gmode === "kinds"
                ? `<h4>Kinds of change, by year</h4><p class="sub">Hexagons in the kinds shown, by the year their change stood out most</p>`
                : `<h4>Where it changed, by year</h4><p class="sub">Hexagons in view that changed a fair amount or more, by the year their change stood out most</p>`;
              h += yearBars(c);
            } else if (!(N && hattrs)) h += `<p class="sub" style="margin-top:12px">${map && map.getZoom() < HEXZ ? `Zoom in to ${HEXZ} for the hexagons.` : "Loading this view…"}</p>`;
            // the clicked hexagon drops down inside this card (never floating over the map): compact, More
            // for the whole account; a new pick opens the card if it was folded
            const pc = cardData && cardData.kind ? cardData : null;
            if (pc) {
              if (pc.n !== fcFor) { fcFor = pc.n; fcMore = false; ycFolded = false; }
              h += `<div class="hexwrap${fcMore ? " more" : ""}">` + hexSection(pc)
                + `<div class="fcbar">${pc.kind === "note" ? "" : `<button class="fcmore">${fcMore ? "Less" : "More"}</button>`}<span><kbd>Esc</kbd> closes</span></div></div>`;
            } else fcFor = null;
            yc.innerHTML = h;
            fc.style.display = "none";
            yc.classList.toggle("collapsed", ycFolded);
            // only the hint in it (no hexagon picked, no imagery held): as small open as folded
            yc.classList.toggle("bare", !pc && !st.holding);
            const cb = yc.querySelector(".yr .at-cb");
            if (cb) cb.onclick = (e) => { e.stopPropagation(); ycFolded = !ycFolded; keep("card", ycFolded); renderYear(); };
            const hx = yc.querySelector(".hexwrap .x");
            if (hx) hx.onclick = (e) => { e.stopPropagation(); closeCard(); };
            const hm = yc.querySelector(".hexwrap .fcmore");
            if (hm) hm.onclick = (e) => { e.stopPropagation(); fcMore = !fcMore; renderYear(); };
            fitCard();
          }
          // the card never scrolls: when its content is taller
          // than the pane below it, it is scaled down from its top right
          // corner to fit. On a narrow screen it keeps its own scroll.
          function fitCard() {
            yc.style.transform = "";
            if (window.matchMedia("(max-width:760px)").matches) return;
            const avail = pane.clientHeight - yc.offsetTop - 12, need = yc.offsetHeight;
            if (avail > 0 && need > avail) yc.style.transform = `scale(${avail / need})`;
          }
          // the hexagon's card lives in the top right card (renderYear); fc stays hidden
          let fcMore = false, fcFor = null;
          for (const el of [yc, fc]) el.addEventListener("pointermove", (e) => {
            const t = e.target && e.target.getAttribute && e.target.getAttribute("data-tip");
            if (!t) { tip.style.display = "none"; return; }
            const p = pane.getBoundingClientRect();
            tip.textContent = t;
            tip.style.display = "block";
            tip.style.left = Math.max(8, e.clientX - p.left - tip.offsetWidth - 14) + "px";
            tip.style.top = (e.clientY - p.top + 12) + "px";
          });
          for (const el of [yc, fc]) el.addEventListener("pointerleave", () => { tip.style.display = "none"; });
          // the imagery colors, from the hold card's own buttons
          yc.addEventListener("click", (e) => {
            const b = e.target && e.target.closest && e.target.closest("[data-comp]");
            if (!b) return;
            e.stopPropagation();
            setComp(b.getAttribute("data-comp"));
          });
          for (const el of [yc, fc]) el.addEventListener("click", (e) => {
            const b = e.target && e.target.closest && e.target.closest("[data-copy]");
            if (!b) return;
            e.stopPropagation();
            const t = b.getAttribute("data-copy");
            const done = () => { b.textContent = "copied"; setTimeout(() => { b.textContent = "copy"; }, 1200); };
            // the notebook may sit in an iframe without clipboard permission (molab): fall back to a selection copy
            const fallback = () => { const ta = document.createElement("textarea"); ta.value = t; ta.style.position = "fixed"; ta.style.opacity = "0"; root.appendChild(ta); ta.select(); try { document.execCommand("copy"); done(); } catch (e2) {} ta.remove(); };
            if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(done, fallback); else fallback();
          });
          function renderCard() {
            try { cardData = JSON.parse(model.get("card") || "null"); } catch (e) { cardData = null; }
            picked = cardData && cardData.cell ? cardData.cell : null;
            // a new click opens a folded card once; folding it again stays folded
            if (cardData && cardData.n != null && cardData.n !== ycOpenedFor) { ycOpenedFor = cardData.n; if (ycFolded) { ycFolded = false; keep("card", false); } }
            renderYear(); update();
          }
          function closeCard() { model.set("pick", JSON.stringify({close: true, n: ++seq})); model.save_changes(); cardData = null; picked = null; imgPick = null; renderYear(); update(); }
          // a pick: the same cell again clears it
          function pickCell(cell, ll, pt, onImagery) {
            // a searched cell goes with the next click, inside it or anywhere
            // else, and the map stays where it is; a click inside
            // it only clears it
            if (searched) {
              let inside = false; try { inside = latLngToCell(ll.lat, ll.lng, getResolution(searched)) === searched; } catch (e) {}
              searched = null; update();
              if (inside) return;
            }
            if (cell && cell === picked) { closeCard(); return; }
            imgPick = onImagery ? cell : null;
            model.set("pick", JSON.stringify({cell, lon: ll.lng, lat: ll.lat, admin: adminAt(pt), n: ++seq}));
            model.save_changes();
          }

          // ---- the layers --------------------------------------------------------------
          let map = null, ov = null, map2 = null, ov2 = null;
          const slot = (m = map) => { const want = cfg.labels_slot || "watername_ocean"; const s = m && m.getStyle && m.getStyle(); if (!s || !s.layers || s.layers.some((x) => x.id === want)) return want; const l = s.layers.find((x) => x.type === "symbol"); return (l && l.id) || want; };
          // ---- Sentinel-2 true color, read and drawn here by deck.gl-raster (Development Seed's
          // MosaicLayer + COGLayer, as iceye-view.py and segments-map's ICEYE): the kernel only names
          // the footprints under the view (its STAC search, s2i); the browser opens each one's TCI COG
          // on data.source.coop and decodes its tiles in workers, so no tile waits on the kernel. The
          // year shown loads first, the other years once it is in (a scroll while holding is then
          // instant). Yearly footprints paint over the fill ones. The false colors too, from each
          // footprint's raw band COGs; the kernel only takes their stretch for the view (cfg.s2_stretch)
          const S2GPU = cfg.s2_gpu !== false;
          const S2_WORKER = "https://esm.sh/@developmentseed/geotiff@0.8.0/es2022/dist/pool/worker.mjs";
          let s2Pool = null, s2Warm = false;
          const s2Tiffs = new Map();  // url -> Promise<GeoTIFF>: the headers, kept for the session
          // year -> {ids, batches}: a batch is one answer of the kernel (a tile's footprints, each
          // footprint in the first batch that named it) and gets its own MosaicLayer: a MosaicLayer
          // looks for its sources only when the camera moves, so footprints added to one after the
          // camera stopped would wait for the next pan; a new layer looks at once
          const s2Src = new Map();
          // the requests to data.source.coop, 16 at a time (HTTP/2: one connection; the default is 6),
          // the rest queued in order: the year shown first, yearly footprints before the fill ones,
          // then the nearest the center of the view
          const s2Limiter = new PerOriginSemaphore({maxRequests: 16});
          const s2Asked = new Set();  // "year/z/x/y" asked of the kernel
          const s2Stat = {tiles: 0, pending: 0, failed: 0, last: 0, years: {}};  // years: per year ("2025", "2025 fill") {tiles, pending, last}, for the tests
          const s2Tiff = (url, opts, year, fill) => {
            let p = s2Tiffs.get(url);
            if (!p) {
              const near = opts && opts.getPriority;
              const getPriority = () => [year === st.imgYear ? 0 : 1, fill ? 1 : 0, near ? near() : 0];
              p = GeoTIFF.fromUrl(url, {concurrencyLimiter: s2Limiter, getPriority});
              s2Tiffs.set(url, p);
              p.catch(() => { if (s2Tiffs.get(url) === p) s2Tiffs.delete(url); });  // a failed read is asked again next time
            }
            return p;
          };
          // the view and half of it again on every side: where footprints are looked for and mounted
          function s2View() {
            const bs = [map, st.pair ? map2 : null].filter(Boolean).map((m) => m.getBounds());
            const W_ = Math.min(...bs.map((b) => b.getWest())), E_ = Math.max(...bs.map((b) => b.getEast()));
            const S_ = Math.min(...bs.map((b) => b.getSouth())), N_ = Math.max(...bs.map((b) => b.getNorth()));
            const dx = (E_ - W_) / 2, dy = (N_ - S_) / 2;
            return [W_ - dx, Math.max(-85, S_ - dy), E_ + dx, Math.min(85, N_ + dy)];
          }
          function s2Discover() {
            if (!map || !S2GPU || !(st.holding || st.pair || map.getZoom() >= 9)) return;
            const z = Math.max(cfg.s2_min_z || 7, Math.min(9, Math.floor(map.getZoom()))), n = 2 ** z;
            const tx = (lon) => Math.min(n - 1, Math.max(0, Math.floor((lon + 180) / 360 * n)));
            const ty = (lat) => { const v = Math.sin(lat * Math.PI / 180); return Math.min(n - 1, Math.max(0, Math.floor((0.5 - Math.log((1 + v) / (1 - v)) / (4 * Math.PI)) * n))); };
            const [W_, S_, E_, N_] = s2View(), x0 = tx(W_), x1 = tx(E_), y0 = ty(N_), y1 = ty(S_);
            if ((x1 - x0 + 1) * (y1 - y0 + 1) > 64) return;
            for (const year of [st.imgYear, ...S2Y.filter((y) => y !== st.imgYear)])
              for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++) {
                const k = `${year}/${z}/${x}/${y}`;
                if (s2Asked.has(k)) continue;
                s2Asked.add(k);
                ask("s2i", year, {z, x, y}).then((u8) => {
                  const items = u8 ? JSON.parse(new TextDecoder().decode(u8)) : [];
                  let e = s2Src.get(year);
                  if (!e) { e = {ids: new Set(), batches: []}; s2Src.set(year, e); }
                  const bt = {key: k.replace(/\//g, "-"), yearly: [], fill: [], bbox: [180, 90, -180, -90]};
                  for (const it of items) {
                    if (!it.bbox || e.ids.has(it.id)) continue;
                    e.ids.add(it.id);
                    bt[it.fill ? "fill" : "yearly"].push({id: it.id, url: it.url, bbox: it.bbox, bands: it.bands || {}});
                    bt.bbox = [Math.min(bt.bbox[0], it.bbox[0]), Math.min(bt.bbox[1], it.bbox[1]), Math.max(bt.bbox[2], it.bbox[2]), Math.max(bt.bbox[3], it.bbox[3])];
                  }
                  if (bt.yearly.length || bt.fill.length) { e.batches.push(bt); update(); }
                }, () => { s2Asked.delete(k); });
              }
          }
          async function s2TileData(image, {device, x, y, signal, pool}, key) {
            const sy = s2Stat.years[key] || (s2Stat.years[key] = {tiles: 0, pending: 0, last: 0});
            s2Stat.pending++; sy.pending++;
            try {
              const {array} = await image.fetchTile(x, y, {boundless: false, pool, signal});
              const {width, height, data} = array, px = width * height;
              // TCI's black (0, 0, 0) is nodata: alpha 0 there, so the linear filter's blend of an edge
              // pixel with its nodata neighbor carries a falling alpha the shader can undo (a black
              // test alone let the half-dark blend through: a dark line along every footprint edge)
              let rgba = data;
              if (data.length === 3 * px) {
                rgba = new Uint8Array(4 * px);
                for (let i = 0; i < px; i++) {
                  const r = data[3 * i], g = data[3 * i + 1], b = data[3 * i + 2];
                  rgba[4 * i] = r; rgba[4 * i + 1] = g; rgba[4 * i + 2] = b; rgba[4 * i + 3] = r || g || b ? 255 : 0;
                }
              }
              s2Stat.tiles++; sy.tiles++;
              return {texture: device.createTexture({data: rgba, format: "rgba8unorm", width, height, sampler: {magFilter: "linear", minFilter: "linear"}}), width, height};
            } catch (e) {
              if (!(signal && signal.aborted)) s2Stat.failed++;
              throw e;
            } finally {
              s2Stat.pending--; sy.pending--;
              s2Stat.last = sy.last = Date.now();
              // the year shown is in: mount the others
              if (!s2Warm && s2Stat.pending === 0) { s2Warm = true; setTimeout(update, 250); }
            }
          }
          // nodata (alpha 0) is dropped; at a footprint edge the filtered color is the edge pixel's
          // color times the filtered alpha (nodata is black), so dividing by alpha gives the pixel back
          // with no dark fringe, and alpha under a half ends the footprint midway through the texel.
          // Then the strip's gamma, v -> v ** (1 / gamma), as the kernel's tiles had
          const S2Look = {
            name: "s2Look",
            fs: `uniform s2LookUniforms {
  float gamma;
} s2Look;
`,
            inject: {"fs:DECKGL_FILTER_COLOR": `
  if (color.a < 0.5) discard;
  color = vec4(pow(color.rgb / color.a, vec3(1.0 / s2Look.gamma)), 1.0);
`},
            uniformTypes: {gamma: "f32"},
            getUniforms: (q) => ({gamma: q.gamma}),
          };
          // ---- the false colors (S2_COMPOSITES in the kernel): R, G, B from the raw bands (uint16
          // reflectance x 10,000). The layer walks the first 20 m band's tiles; each other band is read
          // at the level of the same pixel size and placed by its own geotransform (the yearly
          // footprints share one grid for every band, the fill ones put the 10 m bands on another
          // origin and other blocks), from the blocks under the tile. Packed as sqrt(reflectance) in
          // 8 bits (fine steps in the dark ground, where most of it lies); nodata (a band at 0) gets
          // alpha 0 as in TCI
          // rgb: three bands, each over its stretch (S2Stretch). bands + q: quantities computed here per
          // pixel (the kernel's S2_INDEXES computes the same ones for the stretch), kept as half floats
          // and colored by S2Index (kind 0: one index on a ramp, 1: mineral)
          const s2Clamp = (x) => Math.min(1, Math.max(0, x));
          const s2Bare = (b3, b4, b11, b8a) => s2Clamp((0.45 - (b8a - b4) / (b8a + b4)) / 0.25) * s2Clamp((0.05 - (b3 - b11) / (b3 + b11)) / 0.15);
          const S2_FALSE = {
            urban: {rgb: ["B12", "B11", "B04"]},
            blueyellow: {rgb: ["B11", "B11", "B02"]},
            ndbi: {bands: ["B11", "B8A"], kind: 0, q: (v, o) => { o[0] = (v[0] - v[1]) / (v[0] + v[1]); o[1] = o[2] = 0; }},
            mineral: {bands: ["B02", "B03", "B04", "B11", "B12", "B8A"], kind: 1,
                      q: (v, o) => { o[0] = v[2] / v[0]; o[1] = v[3] / v[4]; o[2] = s2Bare(v[1], v[2], v[3], v[5]); }},
          };
          const S2_20M = new Set(["B11", "B12", "B8A"]);
          const s2BandsOf = (comp) => S2_FALSE[comp].rgb || S2_FALSE[comp].bands;
          const s2Prim = (comp) => s2BandsOf(comp).find((b) => S2_20M.has(b));
          // float -> half float bits (rgba16float takes them as a Uint16Array)
          const s2F32 = new Float32Array(1), s2U32 = new Uint32Array(s2F32.buffer);
          function s2Half(v) {
            s2F32[0] = v;
            const x = s2U32[0], sg = (x >>> 16) & 0x8000, e = ((x >>> 23) & 0xff) - 112;
            if (e <= 0 || v !== v) return sg;
            if (e >= 31) return sg | 0x7bff;
            return sg | (e << 10) | ((x & 0x7fffff) >>> 13);
          }
          const S2_HALF_ONE = s2Half(1);
          const S2_SQRT = new Uint8Array(65536);
          for (let v = 0; v < 65536; v++) S2_SQRT[v] = Math.round(255 * Math.sqrt(Math.min(v, 10000) / 10000));
          // a block on another grid lies under up to four tiles: read once for all of them (no
          // tile's abort signal, so one tile let go does not fail its neighbors)
          const s2Blocks = new Map();
          function s2Block(url, im, bx, by, pool) {
            const k = `${url}|${im.width}|${bx}|${by}`;
            let p = s2Blocks.get(k);
            if (!p) {
              p = im.fetchTile(bx, by, {boundless: false, pool}).then(({array}) => ({d: array.data || array.bands[0], w: array.width, h: array.height}), () => null);
              s2Blocks.set(k, p);
              if (s2Blocks.size > 256) s2Blocks.delete(s2Blocks.keys().next().value);
            }
            return p;
          }
          async function s2BandTileData(image, {device, x, y, signal, pool}, key, geotiff, src, comp, year, fill) {
            const sy = s2Stat.years[key] || (s2Stat.years[key] = {tiles: 0, pending: 0, last: 0});
            s2Stat.pending++; sy.pending++;
            try {
              const spec = S2_FALSE[comp], bands = s2BandsOf(comp), prim = s2Prim(comp);
              const tp = image.transform, tw = image.tileWidth, th = image.tileHeight;
              const X0 = tp[2] + x * tw * tp[0], Y0 = tp[5] + y * th * tp[4];  // the tile's corner, in meters
              const block = async (im, bx, by) => {
                try {
                  const {array} = await im.fetchTile(bx, by, {boundless: false, pool, signal});
                  return {d: array.data || array.bands[0], w: array.width, h: array.height};
                } catch (e) { if (signal && signal.aborted) throw e; return null; }  // no block there
              };
              const one = async (b) => {
                if (b === prim) return block(image, x, y);
                const g = src.bands[b] ? await s2Tiff(src.bands[b], null, year, fill) : null;
                const im = g && [g, ...g.overviews].find((q) => Math.abs(q.transform[0] / tp[0] - 1) < 0.01);
                if (!im) return null;
                const q = im.transform, c0 = Math.round((X0 - q[2]) / q[0]), r0 = Math.round((Y0 - q[5]) / q[4]);
                const bw = im.tileWidth, bh = im.tileHeight;
                if (bw === tw && bh === th && c0 === x * tw && r0 === y * th) return block(im, x, y);
                const out = new Uint16Array(tw * th), jobs = [];
                for (let by = Math.max(0, Math.floor(r0 / bh)); by <= Math.min(Math.ceil(im.height / bh) - 1, Math.floor((r0 + th - 1) / bh)); by++)
                  for (let bx = Math.max(0, Math.floor(c0 / bw)); bx <= Math.min(Math.ceil(im.width / bw) - 1, Math.floor((c0 + tw - 1) / bw)); bx++)
                    jobs.push(s2Block(src.bands[b], im, bx, by, pool).then((t) => {
                      if (!t) return;
                      // the block's pixels in the tile's frame
                      const ox = bx * bw - c0, oy = by * bh - r0;
                      for (let r = Math.max(0, -oy); r < t.h && r + oy < th; r++) {
                        const c = Math.max(0, -ox), n = Math.min(t.w, tw - ox) - c;
                        if (n > 0) out.set(t.d.subarray(r * t.w + c, r * t.w + c + n), (r + oy) * tw + c + ox);
                      }
                    }));
                await Promise.all(jobs);
                return {d: out, w: tw, h: th};
              };
              const ub = [...new Set(bands)], got = new Map(await Promise.all(ub.map(async (b) => [b, await one(b)])));
              const p = got.get(prim);
              if (!p) throw new Error("no " + prim + " tile");
              const {w: width, h: height} = p, chans = bands.map((b) => got.get(b));
              if (spec.q) {
                // the quantities as half floats, times alpha (0 at nodata, so the filter's blend at an
                // edge divides back out as in TCI)
                const px = new Uint16Array(4 * width * height), v = new Float32Array(bands.length), q = new Float32Array(3);
                if (chans.every(Boolean)) {
                  for (let r = 0; r < height; r++) for (let c = 0; c < width; c++) {
                    let ok = true;
                    for (let k = 0; k < bands.length; k++) {
                      const a = chans[k], x = r < a.h && c < a.w ? a.d[r * a.w + c] : 0;
                      if (!x) { ok = false; break; }
                      v[k] = x;
                    }
                    if (!ok) continue;
                    spec.q(v, q);
                    const o = 4 * (r * width + c);
                    px[o] = s2Half(q[0]); px[o + 1] = s2Half(q[1]); px[o + 2] = s2Half(q[2]); px[o + 3] = S2_HALF_ONE;
                  }
                }
                s2Stat.tiles++; sy.tiles++;
                return {texture: device.createTexture({data: px, format: "rgba16float", width, height, sampler: {magFilter: "linear", minFilter: "linear"}}), width, height};
              }
              const rgba = new Uint8Array(4 * width * height);
              if (chans.every(Boolean)) {
                for (let r = 0; r < height; r++) for (let c = 0; c < width; c++) {
                  const o = 4 * (r * width + c);
                  let ok = true;
                  for (let k = 0; k < 3; k++) {
                    const a = chans[k], v = r < a.h && c < a.w ? a.d[r * a.w + c] : 0;
                    if (!v) { ok = false; break; }
                    rgba[o + k] = S2_SQRT[v];
                  }
                  if (ok) rgba[o + 3] = 255; else rgba[o] = rgba[o + 1] = rgba[o + 2] = 0;
                }
              }
              s2Stat.tiles++; sy.tiles++;
              return {texture: device.createTexture({data: rgba, format: "rgba8unorm", width, height, sampler: {magFilter: "linear", minFilter: "linear"}}), width, height};
            } catch (e) {
              if (!(signal && signal.aborted)) s2Stat.failed++;
              throw e;
            } finally {
              s2Stat.pending--; sy.pending--;
              s2Stat.last = sy.last = Date.now();
              if (!s2Warm && s2Stat.pending === 0) { s2Warm = true; setTimeout(update, 250); }
            }
          }
          // as S2Look, then back from sqrt to reflectance and each band over its stretch (the
          // kernel's 98th percentile for the view, as a fraction), then the gamma
          const S2Stretch = {
            name: "s2Stretch",
            fs: `uniform s2StretchUniforms {
  float gamma;
  vec3 scale;
} s2Stretch;
`,
            inject: {"fs:DECKGL_FILTER_COLOR": `
  if (color.a < 0.5) discard;
  vec3 v = color.rgb / color.a;
  v = clamp(v * v / s2Stretch.scale, 0.0, 1.0);
  color = vec4(pow(v, vec3(1.0 / s2Stretch.gamma)), 1.0);
`},
            uniformTypes: {gamma: "f32", scale: "vec3<f32>"},
            getUniforms: (q) => ({gamma: q.gamma, scale: q.scale}),
          };
          // the indexes: each quantity over its stretch (the kernel's 2nd to 98th percentile for the view).
          // NDBI on blue (low: vegetation, water) to a light middle to orange (high: built-up, bare).
          // Mineral: iron oxides yellow, clays blue, both near white, on bare ground; vegetation dark.
          // Blue against yellow and orange, never red against green
          const S2Index = {
            name: "s2Index",
            fs: `uniform s2IndexUniforms {
  vec3 lo;
  vec3 hi;
  float kind;
} s2Index;
`,
            inject: {"fs:DECKGL_FILTER_COLOR": `
  if (color.a < 0.5) discard;
  vec3 q = color.rgb / color.a;
  vec3 t = clamp((q - s2Index.lo) / max(s2Index.hi - s2Index.lo, vec3(1e-6)), 0.0, 1.0);
  vec3 c;
  if (s2Index.kind < 0.5) {
    vec3 lo = vec3(0.13, 0.33, 0.62), mid = vec3(0.95, 0.93, 0.88), hi = vec3(0.89, 0.47, 0.08);
    c = t.r < 0.5 ? mix(lo, mid, t.r * 2.0) : mix(mid, hi, t.r * 2.0 - 1.0);
  } else {
    c = vec3(0.05, 0.06, 0.07) + t.b * (vec3(0.10) + t.r * vec3(0.95, 0.76, 0.18) + t.g * vec3(0.22, 0.50, 0.92));
  }
  color = vec4(clamp(c, 0.0, 1.0), 1.0);
`},
            uniformTypes: {lo: "vec3<f32>", hi: "vec3<f32>", kind: "f32"},
            getUniforms: (q) => ({lo: q.lo, hi: q.hi, kind: q.kind}),
          };
          const s2CogLayers = (year, left = false) => {
            if (!s2Pool) s2Pool = new DecoderPool({size: Math.min(6, navigator.hardwareConcurrency || 4),
              createWorker: () => new Worker(URL.createObjectURL(new Blob([`import "${S2_WORKER}";`], {type: "text/javascript"})), {type: "module"})});
            const e = s2Src.get(year);
            if (!e || !map) return [];
            // shown or hidden by visible, not opacity: the tile layers inside MosaicLayer and COGLayer
            // keep each tile's raster layer as it was when the tile came in (an opacity change never
            // reaches them, so a tile loaded while holding stayed after the hold), while deck.gl checks
            // every parent's visible on each draw. Hidden years still load their tiles
            const visible = (left ? st.pair && st.left === "s2" : st.holding) && year === st.imgYear;
            const gamma = Number(st.s2scale) || 1, comp = S2_FALSE[st.s2comp] ? st.s2comp : "tci";
            const pre = (left ? "s2gl-" : "s2g-") + (comp === "tci" ? "" : comp + "-");
            // the false colors wait, still loading, for the stretch of their view
            const sx = cfg.s2_stretch, stretched = comp === "tci" || (sx && sx.comp === comp);
            const spec = S2_FALSE[comp];
            const scale = !spec || !spec.rgb ? null : spec.rgb.map((b) => ((sx && sx[b]) || 3000) / 10000);
            const lo = (sx && sx.lo) || [0, 0, 0], hi = (sx && sx.hi) || [1, 1, 1];
            const look = !spec ? null : spec.rgb ? {module: S2Stretch, props: {gamma, scale}} : {module: S2Index, props: {lo, hi, kind: spec.kind}};
            const [W_, S_, E_, N_] = s2View();
            const live = e.batches.filter((bt) => bt.bbox[0] < E_ && bt.bbox[2] > W_ && bt.bbox[1] < N_ && bt.bbox[3] > S_);
            // every batch's fill under every batch's yearly footprints
            const parts = [...live.filter((bt) => bt.fill.length).map((bt) => [bt, "fill"]), ...live.filter((bt) => bt.yearly.length).map((bt) => [bt, "yearly"])];
            return parts.map(([bt, part]) => new MosaicLayer({
              id: pre + bt.key + "-" + part, sources: bt[part], maxCacheSize: 0, minZoom: cfg.s2_min_z || 7, visible: visible && stretched, beforeId: slot(left ? map2 : map),
              // a footprint that cannot be read (the STAC lists it but the bucket lacks it, or its header
              // failed) is nothing there: a rejection here leaves MosaicLayer a tile with no content,
              // which it then cannot draw ("Cannot destructure property 'source' ... null")
              getSource: (src, o) => s2Tiff(comp === "tci" ? src.url : src.bands[s2Prim(comp)], o, year, part === "fill").catch(() => null),
              renderSource: (src, {data, signal}) => !data ? null : comp === "tci" ? new COGLayer({
                id: pre + year + "-" + src.id, geotiff: data, getTileData: (img, o) => s2TileData(img, o, year + (part === "fill" ? " fill" : "")), pool: s2Pool, signal,
                refinementStrategy: "best-available", maxRequests: 16,
                renderTile: (d) => ({renderPipeline: [{module: CreateTexture, props: {textureName: d.texture}}, {module: S2Look, props: {gamma}}]}),
                updateTriggers: {renderTile: [gamma]},
              }) : new COGLayer({
                id: pre + year + "-" + src.id, geotiff: data, pool: s2Pool, signal, refinementStrategy: "best-available", maxRequests: 16,
                getTileData: (img, o) => s2BandTileData(img, o, comp + " " + year + (part === "fill" ? " fill" : ""), data, src, comp, year, part === "fill"),
                renderTile: (d) => ({renderPipeline: [{module: CreateTexture, props: {textureName: d.texture}}, look]}),
                updateTriggers: {renderTile: [gamma, ...(scale || []), ...lo, ...hi]},
              }),
            }));
          };
          // the years mounted: the one shown, the rest once it has loaded
          const s2Years = () => (s2Warm ? [st.imgYear, ...S2Y.filter((y) => y !== st.imgYear)] : [st.imgYear]);
          // every imagery year stays mounted once the view is close enough, the
          // ones not shown at opacity 0, so their tiles load ahead and a scroll
          // while holding is instant
          const s2Layer = (year, left = false) => new TileLayer({
            id: (left ? "s2l-" : "s2-") + year + "-g" + (cfg.s2_gen || 0),
            getTileData: async ({index, signal}) => { const u8 = await ask("s2", year, index, signal); return u8 ? pngBitmap(u8) : null; },
            onTileError: (e) => { if (!e || e.name !== "AbortError") say("s2 tile: " + ((e && e.message) || e)); },
            tileSize: cfg.tile || 256, minZoom: cfg.s2_min_z || 7, maxZoom: 14, refinementStrategy: "best-available", debounceTime: 120, beforeId: slot(left ? map2 : map),
            opacity: (left ? st.pair && st.left === "s2" : st.holding) && year === st.imgYear ? 1 : 0,
            renderSubLayers: (p) => { if (!p.data) return null; const {west, south, east, north} = p.tile.bbox; return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north]}); },
          });
          // the hexagons: the frame's cells drawn as geometry, colored from hcol. A zoom or pan only
          // rescales what is on screen (nothing to fetch per tile), and a new frame replaces the old
          // one whole the moment its cells arrive. hexData is made once per frame, so a hover or a
          // mode change does not rebuild the shapes
          let hexData = null;
          const hexLayer = (visible) => new H3HexagonLayer({
            id: "hexes", visible,
            data: hexData,
            getHexagon: (_, {index}) => hexes[index],
            getFillColor: (_, {index, target}) => { const o = 4 * index; target[0] = hcol[o]; target[1] = hcol[o + 1]; target[2] = hcol[o + 2]; target[3] = hcol[o + 3]; return target; },
            updateTriggers: {getFillColor: [hexSeq]},
            filled: true, stroked: false, extruded: false, highPrecision: true, pickable: false,
            beforeId: slot(),
          });
          const ring = (h) => { try { return cellToBoundary(h, true); } catch (e) { return null; } };
          const outline = (id, h, color, width, m = map) => { const r = h ? ring(h) : null; return r ? new PathLayer({id, data: [r], getPath: (d) => d, getColor: color, widthUnits: "pixels", getWidth: width, beforeId: slot(m)}) : null; };
          function layers() {
            const out = [];
            const z = map ? map.getZoom() : 0;
            // preloaded from zoom 9 only: below it the tiles are decimated from L5 (slow). Every year
            // stays mounted (the one shown loads first, a scroll is instant); the kernel's tiles (no
            // GPU path) keep only the year shown for false colors, which read three bands. Paired,
            // the imagery is the left map's, so none here
            if (!st.pair && (st.holding || z >= 9)) {
              if (S2GPU) { s2Discover(); for (const y of s2Years()) out.push(...s2CogLayers(y)); }
              else for (const y of S2Y) if (st.s2comp === "tci" || y === st.imgYear) out.push(s2Layer(y));
            }
            // while holding: the imagery, and over it only the two outlines
            //
            // kept in the stack while hidden (holding, zoomed out) so its shapes stay built
            if (hexData && hcol) out.push(hexLayer(!st.holding && z >= HEXZ));
            const hv = hover != null && hover >= 0 ? outline("hover", hexes[hover], [255, 255, 255, 235], 2) : null;
            if (hv) out.push(hv);
            // gold on the dark basemap (was near-black on the light one)
            const sc = searched ? outline("searched", searched, [255, 200, 40, 255], 3) : null;
            if (sc) out.push(sc);
            const pk = picked ? outline("picked", picked, [255, 200, 40, 255], 3) : null;
            if (pk) out.push(pk);
            return out;
          }
          // the pair's left side: the imagery years, mounted as on the map so a year step is instant,
          // and over the imagery the same outlines as on the right: the hexagon under
          // the pointer (on either side), the searched one and the picked one
          function layers2() {
            if (!st.pair || !map2) return [];
            let out;
            if (S2GPU) { s2Discover(); out = s2Years().flatMap((y) => s2CogLayers(y, true)); }
            else out = S2Y.filter((y) => st.s2comp === "tci" || y === st.imgYear).map((y) => s2Layer(y, true));
            const hv = hover != null && hover >= 0 ? outline("hover-l", hexes[hover], [255, 255, 255, 235], 2, map2) : null;
            if (hv) out.push(hv);
            const sc = searched ? outline("searched-l", searched, [255, 200, 40, 255], 3, map2) : null;
            if (sc) out.push(sc);
            const pk = picked ? outline("picked-l", picked, [255, 200, 40, 255], 3, map2) : null;
            if (pk) out.push(pk);
            return out;
          }
          function update() {
            if (ov) ov.setProps({layers: layers()});
            if (ov2) ov2.setProps({layers: layers2()});
            renderSide();
          }
          function labels(on) {
            for (const m of [map, map2]) {
              if (!m || !m.isStyleLoaded()) continue;
              (m.getStyle().layers || []).forEach((l) => { if (l.layout && l.layout["text-field"] !== undefined) m.setLayoutProperty(l.id, "visibility", on ? "visible" : "none"); });
            }
          }

          function renderSide() {
            side.classList.toggle("on", st.pair);
            if (!st.pair) return;
            side.innerHTML = `<b>Sentinel-2 ${st.imgYear}</b><span>${esc(COMP_NAME[st.s2comp] || "")}</span><span><kbd>[</kbd> <kbd>]</kbd> year, <kbd>B</kbd> ${S2Y[0]} or ${S2Y[S2Y.length - 1]}, <kbd>C</kbd> colors</span>`;
          }
          side.addEventListener("click", (e) => { const b = e.target.closest && e.target.closest("button[data-left]"); if (b) setLeft(b.getAttribute("data-left")); });
          function setLeft(v) {
            if (st.left === v) return;
            st.left = v;
            if (v === "s2" && st.s2comp !== "tci") send("s2comp", {comp: st.s2comp});
            update();
          }

          // ---- the pair (P) ---------------------------------------------------------------
          // a second map on the left, made the first time the pair opens; either map moves the
          // other. The right is the map as it was (its layer, card and clicks); the left shows
          // Sentinel-2 (the year as the hold leaves it)
          let syncing = false;
          function boot2() {
            if (map2) return;
            map2 = new maplibregl.Map({container: mapEl2, style: STYLE, center: map.getCenter(), zoom: map.getZoom(), attributionControl: {compact: true}});
            map2.keyboard.disable();
            if (root._otf) { root._otf.map2 = map2; root._otf.layers2 = () => layers2().map((l) => l.id); }
            ov2 = new MapboxOverlay({interleaved: true, layers: [], onError: (e) => say("deck, left: " + (e && e.message ? e.message : e))});
            map2.addControl(ov2);
            map2.on("load", () => { labels(st.labels); update(); });
            const follow = (a, b) => a.on("move", () => {
              if (syncing || !st.pair) return;
              syncing = true;
              b.jumpTo({center: a.getCenter(), zoom: a.getZoom(), bearing: a.getBearing(), pitch: a.getPitch()});
              syncing = false;
            });
            follow(map, map2); follow(map2, map);
            // the left side hovers and picks as the right does (no tooltip, as over the imagery held):
            // the two sides are the same size under one camera, so a point is the same place on both
            map2.on("mousemove", (e) => { const i = hexAt(e.lngLat); if (i !== hover) { hover = i; update(); } });
            map2.on("mouseout", () => { if (hover != null && hover >= 0) { hover = null; update(); } });
            map2.on("click", (e) => { if (st.noPick) return; const i = hexAt(e.lngLat); pickCell(i >= 0 ? hexes[i] : null, e.lngLat, e.point, st.left === "s2"); });
            new ResizeObserver(() => { try { map2.resize(); } catch (e) {} }).observe(mapEl2);
          }
          function setPair(on) {
            if (on === st.pair || !map) return;
            if (on && st.holding) endHold(null);
            st.pair = on;
            root.classList.toggle("pair", on);
            bPair.classList.toggle("on", on);
            bPair.title = on ? "back to one map (P)" : "pair the map with Sentinel-2 (P)";
            // the same zoom on both sides (each shows half the ground of one map), so the model and the
            // hexagons work in the pair exactly as on one map
            if (on) {
              boot2();
              map2.jumpTo({center: map.getCenter(), zoom: map.getZoom(), bearing: map.getBearing(), pitch: map.getPitch()});
              if (st.left === "s2" && st.s2comp !== "tci") send("s2comp", {comp: st.s2comp});
            }
            renderYear(); update();
          }

          // ---- hold: the imagery ------------------------------------------------------------
          // hold SPACE, only (the mouse stays free, so it can rest off the
          // building you are looking at, where the pointer would cover a small
          // one; the press and hold on the map is gone). The
          // imagery opens on the year the last
          // hold left off at (2022 the first time); while holding, the wheel anywhere over the map or the card
          // steps the imagery year instead of zooming; letting go of whichever
          // started it ends it
          let holdT = null, holdAt = null, holdBy = null, wheelAcc = 0, lastStep = 0, suppressClick = false, lastPt = null;
          const stepImg = (d) => { const i = S2Y.indexOf(st.imgYear); const n = S2Y[Math.max(0, Math.min(S2Y.length - 1, (i < 0 ? S2Y.length - 1 : i) + d))]; if (n !== st.imgYear) { st.imgYear = n; renderYear(); update(); } };
          function beginHold(x, y, by) {
            holdT = null;
            if (!map || st.holding) return;
            st.holding = true;
            holdBy = by;
            if (by === "mouse") suppressClick = true;
            mapEl.classList.add("holding");
            mapEl.classList.toggle("key", by === "key");
            // the wheel is the year while holding (the capture listener on root
            // keeps it from the map, so scrollZoom stays on: disabling it
            // mid-zoom left maplibre's zoom marked active, and scroll froze
            // after the hold); a space hold leaves the map free to drag ("i'd
            // like to be able to move the map when space is pressed"), a mouse
            // hold cannot (the press is the hold)
            if (by === "mouse") map.dragPan.disable();
            tip.style.display = "none";
            // false colors are stretched for the view: taken again if this hold is somewhere else
            if (st.s2comp !== "tci") send("s2comp", {comp: st.s2comp});
            renderYear(); update();
          }
          function endHold(by) {
            if (by !== "key") { clearTimeout(holdT); holdT = null; holdAt = null; }
            if (!st.holding || (by && by !== holdBy)) return;
            st.holding = false; holdBy = null;
            mapEl.classList.remove("holding");
            if (map) map.dragPan.enable();
            wheelAcc = 0;
            renderYear(); update();
            if (lastPt && map) showHexTip(lastPt);
          }
          mapEl.addEventListener("pointermove", (e) => {
            lastPt = {x: e.clientX, y: e.clientY};
            if (holdT && holdAt && Math.hypot(e.clientX - holdAt.x, e.clientY - holdAt.y) > SLOP) { clearTimeout(holdT); holdT = null; holdAt = null; }
          }, true);
          const endMouse = () => endHold("mouse"), endAny = () => endHold(null);
          window.addEventListener("pointerup", endMouse, true);
          window.addEventListener("pointercancel", endMouse, true);
          window.addEventListener("blur", endAny);
          // the space bar: down starts a hold at the pointer (or the map's
          // center before the pointer has been over it), up ends it
          function spaceDown() {
            // paired, the imagery is already on the left
            if (st.holding || !map || st.pair) return;
            const r = mapEl.getBoundingClientRect();
            const inMap = lastPt && lastPt.x >= r.left && lastPt.x <= r.right && lastPt.y >= r.top && lastPt.y <= r.bottom;
            const pt = inMap ? lastPt : {x: r.left + r.width / 2, y: r.top + r.height / 2};
            beginHold(pt.x, pt.y, "key");
          }
          const onKeyUp = (e) => {
            if (e.key === " ") endHold("key");
          };
          window.addEventListener("keyup", onKeyUp);
          root.addEventListener("wheel", (e) => {
            if (!st.holding) return;
            e.preventDefault(); e.stopPropagation();
            wheelAcc += e.deltaY;
            const now = performance.now();
            if (Math.abs(wheelAcc) >= 40 && now - lastStep > 140) { stepImg(wheelAcc > 0 ? 1 : -1); wheelAcc = 0; lastStep = now; }
          }, {capture: true, passive: false});
          root.addEventListener("pointermove", (e) => { lastPt = {x: e.clientX, y: e.clientY}; }, true);
          function showHexTip(pt) {
            const r = mapEl.getBoundingClientRect();
            const i = hexAt(map.unproject([pt.x - r.left, pt.y - r.top]));
            if (i !== hover) { hover = i; update(); }
            if (i < 0 || !hattrs || map.getZoom() < HEXZ) { tip.style.display = "none"; return; }
            const p = pane.getBoundingClientRect();
            tip.innerHTML = hexWords(i);
            tip.style.display = "block";
            tip.style.left = (pt.x - p.left + 14) + "px";
            tip.style.top = (pt.y - p.top + 14) + "px";
          }

          // ---- search ------------------------------------------------------------------------
          const PHOTON = "https://photon.komoot.io/api/";
          let gcHits = [], gcSel = -1, gcTimer = null, gcSeq = 0, searched = null;
          // drop the searched cell's outline, staying where the map is
          function unsearch() {
            searched = null; update();
          }
          // an H3 string in the box is a cell, not a place
          const h3Of = (q) => { const h = q.trim().toLowerCase(); try { return /^[0-9a-f]{15}$/.test(h) && isValidCell(h) ? h : null; } catch (e) { return null; } };
          const hitName = (f) => { if (f.h3) return "H3 " + f.h3; const p = f.properties || {}; return [p.name, p.street && !p.name ? p.street : null, p.city && p.city !== p.name ? p.city : null, p.state, p.country].filter(Boolean).join(", "); };
          const hitKind = (f) => { if (f.h3) { const [la, lo] = cellToLatLng(f.h3); return `res ${getResolution(f.h3)}, ${la.toFixed(5)}, ${lo.toFixed(5)}`; } const p = f.properties || {}; return [p.osm_value, p.type].filter((x) => x && x !== "yes").join(", "); };
          const gcHide = () => { hits.style.display = "none"; hits.replaceChildren(); gcSel = -1; };
          const gcShow = () => {
            hits.replaceChildren();
            if (!gcHits.length) { gcHide(); return; }
            gcHits.forEach((f, i) => { const r = el_("div", "at-hit" + (i === gcSel ? " sel" : "")); r.textContent = hitName(f); const k = el_("small"); k.textContent = hitKind(f); r.appendChild(k); r.onmousedown = (e) => { e.preventDefault(); gcFly(f); }; r.onmouseenter = () => { gcSel = i; gcShow(); }; hits.appendChild(r); });
            hits.style.display = "block";
          };
          const gcAsk = async () => {
            const q = gc.value.trim();
            if (!q && searched) { searched = null; update(); }
            if (q.length < 2) { gcHits = []; gcHide(); return; }
            const s = ++gcSeq;
            const h3 = h3Of(q);
            if (h3) { gcHits = [{h3}]; gcSel = 0; gcShow(); return; }
            const params = new URLSearchParams({q, limit: "6", lang: "en"});
            if (map) { const c = map.getCenter(); params.set("lon", c.lng.toFixed(4)); params.set("lat", c.lat.toFixed(4)); }
            try { const r = await fetch(PHOTON + "?" + params.toString()); const d = await r.json(); if (s !== gcSeq) return; gcHits = (d.features || []).filter((f) => f.geometry && f.geometry.coordinates); gcSel = gcHits.length ? 0 : -1; gcShow(); }
            catch (e) { if (s === gcSeq) note("search: " + e.message, 4000); }
          };
          const gcFly = (f) => {
            if (f.h3) {
              // to the zoom that draws hexagons of the cell's own res, the middle
              // of its zooms; finer than the finest drawn,
              // about 80 px across; never out past the hexagons
              const [lat, lon] = cellToLatLng(f.h3), r = getResolution(f.h3), L = cfg.res_ladder;
              const zoom = Math.max(HEXZ, Math.min(17, L && r <= L[3] ? L[0] + (r - L[2] + 0.5) * L[1]
                : Math.log2(78271.5 * Math.cos(lat * Math.PI / 180) * 80 / (2 * 1281256 / Math.pow(Math.sqrt(7), r)))));
              searched = f.h3; gcHits = []; gcHide(); gc.blur(); update();
              if (map) map.flyTo({center: [lon, lat], zoom, duration: 2200, essential: true});
              return;
            }
            searched = null;
            const [lon, lat] = f.geometry.coordinates;
            const ext = (f.properties || {}).extent;
            let zoom = 12;
            if (ext && ext.length === 4) { const span = Math.max(Math.abs(ext[2] - ext[0]), Math.abs(ext[1] - ext[3]) * 2, 0.01); zoom = Math.log2(360 * ((mapEl.clientWidth || 1200) / 512) / span) - 0.3; }
            zoom = Math.max(HEXZ, Math.min(16, zoom));
            gc.value = hitName(f); gcHits = []; gcHide(); gc.blur();
            if (map) map.flyTo({center: [lon, lat], zoom, duration: 2200, essential: true});
          };
          gc.addEventListener("input", () => { clearTimeout(gcTimer); gcTimer = setTimeout(gcAsk, 250); });
          gc.addEventListener("focus", () => { if (gcHits.length) gcShow(); });
          gc.addEventListener("blur", () => setTimeout(gcHide, 120));
          gc.addEventListener("keydown", (e) => {
            e.stopPropagation();
            if (e.key === "ArrowDown" && gcHits.length) { gcSel = (gcSel + 1) % gcHits.length; gcShow(); e.preventDefault(); }
            else if (e.key === "ArrowUp" && gcHits.length) { gcSel = (gcSel - 1 + gcHits.length) % gcHits.length; gcShow(); e.preventDefault(); }
            else if (e.key === "Enter") { e.preventDefault(); if (gcHits.length) gcFly(gcHits[Math.max(0, gcSel)]); else { clearTimeout(gcTimer); gcAsk().then(() => { if (gcHits.length) gcFly(gcHits[0]); else note("no match: " + gc.value.trim(), 4000); }); } }
            else if (e.key === "Escape") { gcHide(); gc.blur(); }
          });

          // ---- fill the window --------------------------------------------------------------
          const FIT_CLS = "at-fit-on";
          if (!document.getElementById("at-fit-style")) {
            const s = document.createElement("style"); s.id = "at-fit-style";
            s.textContent = ["notebook-actions-dropdown", "cell-actions-button", "drag-button", "expand-output-button", "fullscreen-output-button", "chrome-sidebar", "chrome-footer", "chrome-controls-top-right", "chrome-controls-bottom-right"].map((t) => "html." + FIT_CLS + " [data-testid='" + t + "']").join(",") + ",html." + FIT_CLS + " div[class*='top-[25vh]']{display:none!important}html." + FIT_CLS + "{overflow:hidden}";
            document.head.appendChild(s);
          }
          function sizes() {
            root.classList.toggle("fit", st.fit);
            document.documentElement.classList.toggle(FIT_CLS, st.fit);
            pane.style.height = st.fit ? "100vh" : (cfg.height || 780) + "px";
            bFit.innerHTML = st.fit ? ICON.shrink : ICON.expand; bFit.title = st.fit ? "back to the notebook (X or Esc); full screen (F)" : "fill the window (X); full screen (F)";
            setTimeout(() => { try { map && map.resize(); } catch (e) {} }, 30);
          }
          bFit.onclick = () => { st.fit = !st.fit; sizes(); };
          bMore.onclick = (e) => { e.stopPropagation(); const open = more.style.display !== "block"; more.style.display = open ? "block" : "none"; yc.style.visibility = open ? "hidden" : ""; bMore.classList.toggle("on", open); };
          more.addEventListener("click", (e) => e.stopPropagation());
          const closeMore = () => { more.style.display = "none"; yc.style.visibility = ""; bMore.classList.remove("on"); };
          root.addEventListener("click", () => { if (more.style.display === "block") closeMore(); });
          window.addEventListener("resize", () => { if (st.fit) sizes(); });

          // ---- keys ---------------------------------------------------------------------------
          root.tabIndex = 0;
          const onKey = (e) => {
            const path = e.composedPath ? e.composedPath() : [];
            if (!st.fit && !path.includes(root)) return;
            const tgt = path[0] || e.target;
            if (tgt && /^(INPUT|SELECT|TEXTAREA)$/.test(tgt.tagName)) return;
            const k = e.key, lo = st.y0, hi = st.y1;
            if (k === " ") { if (!e.repeat) spaceDown(); }
            // Shift + arrows: turn the map (left, right: 15 degrees) and tilt it (up, down: 10), as MapLibre's own
            // keys do; held, it keeps going
            else if (e.shiftKey && /^Arrow(Left|Right|Up|Down)$/.test(k)) {
              if (!map) return;
              const db = k === "ArrowLeft" ? -15 : k === "ArrowRight" ? 15 : 0, dp = k === "ArrowUp" ? 10 : k === "ArrowDown" ? -10 : 0;
              map.easeTo({bearing: map.getBearing() + db, pitch: Math.max(0, Math.min(map.getMaxPitch(), map.getPitch() + dp)), duration: e.repeat ? 120 : 250});
            }
            // Color by: E Earthwork
            else if (/^[eE]$/.test(k)) { st.want = "earth"; const m = drawnMode(); if (m !== st.gmode) { st.gmode = m; recolorHex(); renderYear(); update(); } styleRows(); }
            // Q: the tooltip and picking off and back on
            else if (k === "q" || k === "Q") { st.noPick = !st.noPick; tip.style.display = "none"; hover = null; update(); note(st.noPick ? "No tooltip or picking (Q to turn them back on)" : "Tooltip and picking on", 2500); }
            // the kinds key's Built (W)
            else if (k === "w" || k === "W") { if (st.gmode !== "kinds") return; st.focus = "built"; recolorHex(); styleKey(); update(); }
            else if (k === "p" || k === "P") setPair(!st.pair);
            // A: the area scale and back (Earthwork only)
            else if (k === "a" || k === "A") { if (st.gmode !== "earth") return; st.area = !st.area; recolorHex(); styleKey(); update(); note(st.area ? "Area scale: unusual for this area (A for the normal scale)" : "Normal scale", 2500); }
            else if (k === "[" || k === "]") stepImg(k === "]" ? 1 : -1);
            // B: the imagery's first year and its latest, back and forth (from any other year, the
            // latest), while it shows: holding space, or on the pair's left side
            else if (k === "b" || k === "B") {
              if (!(st.holding || (st.pair && st.left === "s2"))) return;
              const a = S2Y[0], b = S2Y[S2Y.length - 1];
              st.imgYear = st.imgYear === b ? a : b; renderYear(); update();
            }
            else if (k === ";" || k === "'") { st.s2scale = Math.round(10 * Math.max(0.3, Math.min(2.5, st.s2scale + (k === "'" ? 0.1 : -0.1)))) / 10; gam.value = st.s2scale; clearTimeout(gamT); gamT = setTimeout(() => send("s2scale"), 250); }
            else if (k === "-" || k === "=") { const v = Math.max(aefYears[0], Math.min(hi - 1, lo + (k === "=" ? 1 : -1))); if (v !== lo) { st.y0 = v; winSent = [st.y0, st.y1]; styleWin(); send("aef"); } }
            else if (k === "_" || k === "+") { const v = Math.max(lo + 1, Math.min(aefYears[aefYears.length - 1], hi + (k === "+" ? 1 : -1))); if (v !== hi) { st.y1 = v; winSent = [st.y0, st.y1]; styleWin(); send("aef"); } }
            else if (k === "l" || k === "L") { st.labels = !st.labels; labels(st.labels); swLab.sty(); }
            else if (k === "x" || k === "X") { st.fit = !st.fit; sizes(); }
            else if (k === "c" || k === "C") { const i = COMPS.findIndex(([c]) => c === st.s2comp); setComp(COMPS[(i + 1) % COMPS.length][0]); }
            else if ((k === "m" || k === "M") && MODELS.length > 1) setModel(MODELS[(MODELS.indexOf(st.model) + 1) % MODELS.length]);
            // full screen (the browser's own; Esc or F again leaves it), filling the window inside it
            else if (k === "f" || k === "F") {
              if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
              else { if (!st.fit) { st.fit = true; sizes(); } (root.requestFullscreen ? root.requestFullscreen() : Promise.reject()).catch(() => note("full screen is not allowed here", 2500)); }
            }
            else if (k === "/") gc.focus();
            else if (k === "Escape") { if (searched) unsearch(); else if (about.style.display === "flex") about.style.display = "none"; else if (more.style.display === "block") closeMore(); else if (cardData) closeCard(); else if (st.fit) { st.fit = false; sizes(); } }
            else return;
            e.preventDefault();
          };
          window.addEventListener("keydown", onKey);

          // ---- camera and click ----------------------------------------------------------------
          let seq = 0, lastView = "";
          function sendView() {
            if (!map) return;
            const c = map.getCenter();
            const v = {longitude: c.lng, latitude: c.lat, zoom: map.getZoom(), w: mapEl.clientWidth, h: mapEl.clientHeight};
            const key = JSON.stringify(v);
            if (key === lastView) return;
            lastView = key; v.n = ++seq;
            model.set("view", JSON.stringify(v)); model.save_changes();
          }
          const adminAt = (pt) => {
            const out = {};
            const one = (k) => { const id = "ov-div-" + k; if (!map.getLayer(id)) return null; const fs = map.queryRenderedFeatures(pt, {layers: [id]}); return fs && fs.length ? fs[0].properties : null; };
            try { const r = one("region"); if (r) out.region = r["@name"] || r.names || null; const c = one("county"); if (c) out.county = c["@name"] || c.names || null; const l = one("locality"); if (l) out.locality = l["@name"] || l.names || null; } catch (e) {}
            return out;
          };
          function boot() {
            const home = cfg.home || {longitude: 3.6, latitude: 6.46, zoom: 11};
            map = new maplibregl.Map({container: mapEl, style: STYLE, center: [home.longitude, home.latitude], zoom: home.zoom, attributionControl: {compact: true}});
            map.keyboard.disable();
            map.doubleClickZoom.enable();
            root._otf = {map, st, cfg: () => cfg, hmeta: () => hmeta, area: () => areaSt, hexShown: () => (hexData && hcol ? hmeta.seq : 0), tiles: () => tlog, s2: () => ({...s2Stat, warm: s2Warm, sources: Object.fromEntries([...s2Src].map(([y, e]) => [y, e.ids.size])), batches: Object.fromEntries([...s2Src].map(([y, e]) => [y, e.batches.length]))})};  // for headless tests
            map.addControl(new maplibregl.NavigationControl({showCompass: false}), "bottom-left");
            ov = new MapboxOverlay({interleaved: true, layers: [], onError: (e) => say("deck: " + (e && e.message ? e.message : e))});
            map.addControl(ov);
            map.on("load", () => {
              labels(st.labels);
              if (cfg.div_pm && !map.getSource("ov-div")) {
                try {
                  map.addSource("ov-div", {type: "vector", url: "pmtiles://" + cfg.div_pm});
                  for (const k of ["region", "county", "locality"]) map.addLayer({id: "ov-div-" + k, type: "fill", source: "ov-div", "source-layer": "division_area", filter: ["all", ["==", ["get", "subtype"], k], ["==", ["get", "class"], "land"]], paint: {"fill-opacity": 0}}, slot());
                } catch (e) { console.error("divisions", e); }
              }
              update(); sendView(); renderYear();
            });
            map.on("moveend", sendView);
            // a new view: the footprints under it asked for
            // (and the batches near it mounted, the far ones let go)
            map.on("moveend", () => { if (S2GPU) { s2Discover(); update(); } });
            map.on("zoomend", () => { update(); renderYear(); styleKey(); });
            // a zoom in crossing 8.3 (the kernel's _AHEAD_ZOOM) tells the kernel
            // at once, so it starts reading before the hexagons' zoom
            let aheadSent = false;
            map.on("zoom", () => { styleSoon(); const z = map.getZoom(); if (z < 8.3) aheadSent = false; else if (!aheadSent && z < HEXZ) { aheadSent = true; sendView(); } });
            map.on("mousemove", (e) => {
              if (holdT) return;
              // over the imagery: the white outline follows the pointer, no tooltip
              if (st.holding) { const i = hexAt(e.lngLat); if (i !== hover) { hover = i; update(); } return; }
              showHexTip({x: e.originalEvent.clientX, y: e.originalEvent.clientY});
            });
            map.on("mouseout", () => { tip.style.display = "none"; if (hover != null && hover >= 0) { hover = null; update(); } });
            map.on("click", (e) => {
              if (suppressClick) { suppressClick = false; return; }
              if (st.noPick) return;
              const i = hexAt(e.lngLat);
              pickCell(i >= 0 ? hexes[i] : null, e.lngLat, e.point, st.holding);
            });
            map.on("error", (ev) => { if (ev && ev.error && ev.error.message && !/tile|404/i.test(ev.error.message)) say("map: " + ev.error.message); });
            new ResizeObserver(() => { try { map.resize(); } catch (e) {} fitCard(); }).observe(mapEl);
            window.__cmMaps = () => [map];
            window.__cmState = () => ({st: Object.assign({}, st), hex: N, res, hmeta, tiles: tstat, card: cardData, status: model.get("status"),
              lc: hattrs ? Array.from({length: N}, (_, i) => hattrs[HB * i + 7]).reduce((c, v) => (c[v]++, c), [0, 0, 0, 0]) : null});
            window.__cmTiles = () => ({log: tlog, frames: flog});
            // for tests: the center of the first hexagon whose biggest step is year y and that moved a fair amount
            window.__cmHexAt = (y) => { const b = map.getBounds(); for (let i = 0; i < N; i++) if (hattrs && hattrs[HB * i] === y - 2000 && hattrs[HB * i + 1] >= FAIR) { const r = cellToBoundary(hexes[i], true); const c = r.slice(0, -1).reduce((a, p) => [a[0] + p[0] / (r.length - 1), a[1] + p[1] / (r.length - 1)], [0, 0]); if (b.contains(c) && map.project(c).x < mapEl.clientWidth - 420 && map.project(c).y > 200) return c; } return null; };
          }

          // ---- the kernel's data -----------------------------------------------------------------
          const flog = [];  // per frame received, for the tests
          const loadHex = () => {
            const tl = performance.now();
            const cb = bytesOf(model.get("cells")), ab = bytesOf(model.get("hattrs"));
            const seq0 = hmeta.seq;
            try { hmeta = JSON.parse(model.get("hmeta") || "{}"); } catch (e) { hmeta = {}; }
            // the kinds are grouped again for every new frame: what was hidden no longer means the same
            if (hmeta.seq !== seq0) st.hideKinds.clear();
            // zoomed out of Kinds of change: back to AEF Change, and it stays there when zooming in again
            if (kindsWait()) st.want = "much";
            st.gmode = drawnMode();
            styleFill();
            if (!cb || !cb.length) { hexes = []; N = 0; hexIndex = new Map(); res = -1; hattrs = null; hcol = null; hcol32 = null; hexData = null; renderYear(); styleKey(); update(); return; }
            const ids = new BigUint64Array(copyOf(cb));
            N = ids.length; hexes = new Array(N); hexIndex = new Map();
            for (let i = 0; i < N; i++) { const h = ids[i].toString(16); hexes[i] = h; hexIndex.set(h, i); }
            try { res = getResolution(hexes[0]); } catch (e) { res = -1; }
            hattrs = ab && ab.length === HB * N ? new Uint8Array(copyOf(ab)) : null;
            hexData = {length: N};
            hover = null;
            recolorHex(); renderYear(); styleKey(); update();
            flog.push({seq: hmeta.seq, n: N, t: Date.now(), ms: performance.now() - tl});
          };
          let pendHex = null;
          const hexSoon = () => { clearTimeout(pendHex); pendHex = setTimeout(loadHex, 0); };
          model.on("change:cells", hexSoon);
          model.on("change:hattrs", hexSoon);
          model.on("change:hmeta", hexSoon);
          model.on("change:card", renderCard);
          model.on("change:status", () => say(model.get("status")));
          model.on("change:config", () => { try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; } update(); });
          try {
            sizes(); boot(); styleRows(); loadHex(); renderCard(); say(model.get("status"));
          } catch (e) { say("boot: " + e.message); console.error(e); }
          return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("keyup", onKeyUp); window.removeEventListener("pointerup", endMouse, true); window.removeEventListener("pointercancel", endMouse, true); window.removeEventListener("blur", endAny); document.documentElement.classList.remove(FIT_CLS); try { map && map.remove(); } catch (e) {} };
        }
        export default {render};
        """.replace("__DEPS__", _DEPS)

    return (ChangeMap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## How it works

    **One read per view.** AlphaEarth's first and last year of the window
    are read for what is in view: an overview of the annual COGs zoomed out,
    the 10 m mosaic from about zoom 13.2. Each pixel's lon/lat goes through
    an h3ronpy UDF inside DataFusion (via xarray-sql) and the pixels are
    averaged per cell, one level finer than the hexagons drawn.

    **The score.** Each finer cell's two vectors, made unit length (b the
    first year, a the last), go into `models/earthwork-lr.npz`: a logistic
    regression on [b, a, a * b, (a - b)^2], 256 numbers. See
    `earthwork_model.py` for how it was taught and how well it does on
    ground it never saw.

    **The brightest patch, not the average.** Each hexagon takes its
    highest-scoring finer cell, so a small dig inside a hexagon of quiet
    ground keeps the hexagon lit.

    **Drawn as tiles.** The hexagons reach the browser as tiles of cell
    numbers, colored there from one byte per hexagon, so changing nothing
    but the view never reruns the kernel's work.
    """)
    return


@app.cell
def _(
    AEF_FROM0,
    AEF_TO0,
    AEF_YEARS_ALL,
    ALPHA_FILL,
    ALPHA_QUIET,
    ChangeMap,
    EW_MODELS,
    HEX_ZOOM,
    HOLD_MS,
    HOLD_SLOP_PX,
    HOME,
    LABELS_SLOT,
    BASE_RES,
    MAX_RES,
    MOSAIC_MIN_RES,
    PER_RES,
    ZOOM0,
    OV_DIV_PM,
    RASTER_TILE,
    S2_SCALE0,
    S2_TILE_MIN_Z,
    S2_YEAR0,
    S2_YEARS,
    VIEW_H,
    VIRIDIS,
    json,
    mo,
):
    # ---- the map: built ONCE, empty; never re-runs for a parameter ---------------
    # as an app (marimo run) it fills the window from the start; in the editor
    # it sits in the page (X fills the window, Esc brings it back)
    try:
        _fit = mo.app_meta().mode == "run"
    except Exception:
        _fit = False
    cmap = ChangeMap(config=json.dumps({
        "height": VIEW_H, "home": dict(HOME), "labels_slot": LABELS_SLOT, "tile": RASTER_TILE,
        "s2_year": S2_YEAR0, "s2_scale": S2_SCALE0, "s2_gen": 0, "s2_years": list(S2_YEARS), "s2_min_z": S2_TILE_MIN_Z,
        "aef_from": AEF_FROM0, "aef_to": AEF_TO0, "aef_years": list(AEF_YEARS_ALL),
        "hex_zoom": HEX_ZOOM, "div_pm": OV_DIV_PM, "fit": _fit, "hold_ms": HOLD_MS, "hold_slop": HOLD_SLOP_PX,
        "viridis": VIRIDIS, "alpha_fill": ALPHA_FILL, "alpha_quiet": ALPHA_QUIET,
        "res_ladder": [ZOOM0, PER_RES, BASE_RES, MAX_RES],
        # true color read and drawn in the browser (deck.gl-raster); False: the kernel's PNG tiles
        "s2_gpu": True,
        "ew_models": list(EW_MODELS), "model": "current",
    }))
    HOLD = {
        "frame": None, "sent": None, "box": None, "res": None, "vs": None,
        "busy": False, "pending": None, "pending_force": False, "task": None, "loop": None,
        "s2scale": S2_SCALE0, "s2gen": 0, "y0": AEF_FROM0, "y1": AEF_TO0,
        "hit": None, "pick_n": None, "card": None, "memo": {}, "aef": {},
        "h_cam": None, "h_ctl": None, "h_pick": None, "runs": 0, "hex_status": "", "place": None, "model": "current",
    }
    cmap
    return HOLD, cmap


@app.cell
def _(
    AEF_YEARS_ALL,
    CARRY_RES,
    EW_MODELS,
    CELL_KM2,
    LOGIT_HI,
    LOGIT_LO,
    HEX_UP,
    HEX_ZOOM,
    HOLD,
    HOME,
    SETTLE,
    aef_fold,
    asyncio,
    build_frame,
    cmap,
    contains,
    cpu,
    division_at,
    json,
    np,
    pad_box,
    re,
    res_for_view,
    s2_items_json,
    s2_set_composite,
    s2_set_scale,
    s2_stretch,
    s2_tile_png,
    time,
    traceback,
    view_to_bbox,
):
    # ---- wiring: the camera loop, the click and the controls. Re-runs freely. -----
    try:
        HOLD["loop"] = asyncio.get_running_loop()
    except RuntimeError:
        pass
    HOLD["runs"] += 1
    # one frame build at a time: they share the DuckDB connection's registered tables
    HOLD.setdefault("build_lock", asyncio.Lock())

    async def _tile_fn(src, z, x, y, year):
        t0 = time.time()
        if src == "s2i":
            out = await s2_items_json(z, x, y, year)
        else:
            out = await s2_tile_png(z, x, y, year)
        cmap.tile_times[(src, z, x, y, year)] = {"run": 1e3 * (time.time() - t0)}
        return out

    cmap.tile_fn = _tile_fn

    def _say(msg):
        # an unchanged status must still register in the browser, so a
        # zero-width space toggles on repeats
        try:
            if cmap.status == msg:
                msg = msg + "​" if not msg.endswith("​") else msg[:-1]
            cmap.status = msg
        except Exception:
            pass

    def _cfg(**kw):
        c = json.loads(cmap.config or "{}")
        c.update(kw)
        cmap.config = json.dumps(c)

    def _spawn(coro):
        try:
            return asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            loop = HOLD.get("loop")
            return asyncio.run_coroutine_threadsafe(coro, loop) if loop else None

    def _vsd(vs):
        if vs is None:
            return dict(HOME)
        if isinstance(vs, str):
            try:
                vs = json.loads(vs)
            except Exception:
                return dict(HOME)
        out = {"longitude": float(vs["longitude"]), "latitude": float(vs["latitude"]), "zoom": float(vs["zoom"])}
        if vs.get("w") and vs.get("h"):
            out["w"], out["h"] = float(vs["w"]), float(vs["h"])
        return out

    # ---- the hexagons -------------------------------------------------------------
    def _paint():
        """Send the frame once: 17 bytes per hexagon, the layout the browser reads. Only three are used
        here: the 2nd (AlphaEarth here: nonzero), the 3rd (Earthwork's log-odds for the area scale, LOGIT_LO..HI
        as 1..255, 0 none) and the 17th (Earthwork 1..255, 0 none); the 9th is 1 (drawn). Colored in the
        browser."""
        fr = HOLD["frame"]
        if fr is None or HOLD["sent"] is fr:
            return
        ew, el = fr["earth"], fr["elog"]
        eb = np.where(np.isnan(ew), 0, 1 + np.round(254 * np.nan_to_num(ew))).astype(np.uint8)
        lb = np.where(np.isnan(el), 0, 1 + np.round(254 * np.clip((np.nan_to_num(el) - LOGIT_LO) / (LOGIT_HI - LOGIT_LO), 0, 1))).astype(np.uint8)
        at = np.zeros((len(ew), 17), np.uint8)
        at[:, 1], at[:, 2], at[:, 8], at[:, 16] = eb, lb, 1, eb
        with cmap.hold_sync():
            cmap.cells = fr["cellid"].astype("<u8").tobytes()
            cmap.hattrs = at.tobytes()
            cmap.hmeta = json.dumps({
                "y0": int(fr["y0"]), "y1": int(fr["y1"]), "km2": float(CELL_KM2.get(HOLD["res"], 0)),
                "seq": int(fr.get("seq", 0)), "carry": CARRY_RES, "timing": fr.get("timing"),
            })
        HOLD["sent"] = fr

    _AEF_KEEP_BYTES = 512 * 1024 ** 2

    def _trim(bkey=None):
        # the kept folds: AlphaEarth's by bytes (a year in a 2x box at zoom
        # 12 is ~140 MB, float32), least recently used first and never the
        # box in use; the rest by count
        if bkey is not None:
            for k in [k for k in HOLD["aef"] if k[1] == bkey]:
                HOLD["aef"][k] = HOLD["aef"].pop(k)
        size = lambda v: v[0]["V"].nbytes if v[0] is not None else 0
        held = sum(size(v) for v in HOLD["aef"].values())
        for k in list(HOLD["aef"]):
            if held <= _AEF_KEEP_BYTES:
                break
            if k[1] != bkey:
                held -= size(HOLD["aef"].pop(k))

    async def _serve_hex(vsd, force=False):
        view = view_to_bbox(vsd)
        box = pad_box(view)
        fr0 = HOLD["frame"]
        if (fr0 is not None and HOLD["box"] is not None and contains(HOLD["box"], view) and not force
                and min(15, res_for_view(vsd, box) + HEX_UP) <= HOLD["res"] and (fr0["y0"], fr0["y1"]) == (HOLD["y0"], HOLD["y1"])):
            HOLD["hex_status"] = HOLD.get("hex_ready") or HOLD["hex_status"]
            return
        rres = res_for_view(vsd, box)
        res = min(15, rres + HEX_UP)
        fres = min(13, res + CARRY_RES)
        y0, y1 = HOLD["y0"], HOLD["y1"]
        rbox = tuple(round(v, 3) for v in box)
        key = (y0, y1, res, rbox, HOLD.get("model", "current"))
        t0 = time.time()
        # Earthwork compares the window's first and last year: only those two are read
        years = [y0, y1]
        HOLD["hex_status"] = f"reading AlphaEarth {y0} and {y1}…"
        _say(HOLD["hex_status"])
        if key in HOLD["memo"]:
            fr, stats = HOLD["memo"][key]
        else:
            bkey = (res, rbox)
            need = [y for y in years if (y, bkey) not in HOLD["aef"]]
            got = await asyncio.gather(*(aef_fold(box, fres, y, read_res=rres) for y in need))
            for y, r in zip(need, got):
                HOLD["aef"][(y, bkey)] = r
            _trim(bkey)
            # zoomed or moved on while these were read (a fast zoom in): they
            # are kept, but no frame is built or drawn for a view already left
            pend = HOLD.get("pending")
            if pend is not None and not force:
                pv = _vsd(pend)
                if pv["zoom"] < HEX_ZOOM or min(15, res_for_view(pv, pad_box(view_to_bbox(pv))) + HEX_UP) != res or not contains(box, view_to_bbox(pv)):
                    return
            aef_by_year = {y: HOLD["aef"][(y, bkey)][0] for y in years if (y, bkey) in HOLD["aef"]}
            t1 = time.time()
            async with HOLD["build_lock"]:
                fr = await cpu(build_frame, aef_by_year, y0, y1, res, HOLD.get("model", "current"))
            if fr is None:
                HOLD["hex_status"] = f"hexagons: res {res}, AlphaEarth is missing {y0} or {y1} here | " + " | ".join(HOLD["aef"][(y, bkey)][1] for y in years if (y, bkey) in HOLD["aef"])
                return
            fr["timing"] = {"reads": 1e3 * (t1 - t0), "frame": 1e3 * (time.time() - t1),
                            "aef": [HOLD["aef"][(y, bkey)][1] for y in years if (y, bkey) in HOLD["aef"]], "t_frame": time.time()}
            HOLD["fseq"] = HOLD.get("fseq", 0) + 1
            fr["seq"] = HOLD["fseq"]
            # each year's AlphaEarth read and fold, "c" where it came from memory
            _rd = []
            for y in years:
                if (y, bkey) not in HOLD["aef"]:
                    continue
                m_ = re.search(r"([\d.]+) s · fold [\d,]+ ([\d.]+) s", HOLD["aef"][(y, bkey)][1] or "")
                _rd.append(f"{y} c" if y not in need else f"{y} {m_.group(1)}+{m_.group(2)} s" if m_ else f"{y} ?")
            stats = (
                f"read res {rres}, hexagons res {res}, peak of res {fres} | AEF read+fold {', '.join(_rd)} "
                f"(all {t1 - t0:.1f} s) | frame {time.time() - t1:.1f} s"
            )
            HOLD["memo"][key] = (fr, stats)
            while len(HOLD["memo"]) > 4:
                HOLD["memo"].pop(next(iter(HOLD["memo"])))
        HOLD["frame"], HOLD["box"], HOLD["res"] = fr, box, res
        _paint()
        HOLD["hex_status"] = HOLD["hex_ready"] = f"hexagons: {stats} | {fr['score']} | {time.time() - t0:.1f} s"
        if HOLD.get("card_pick"):
            _card_send(HOLD["card_pick"])

    # READ AHEAD: from _AHEAD_ZOOM, still short of the
    # hexagons, AlphaEarth's two years for the box zoom HEX_ZOOM would read
    # here are read in the background. Crossing into the hexagons over the
    # same ground then only builds the frame; nearby, the COG bytes are kept
    # (_Kept) and the fold is quick. A new place cancels the fold, not the
    # downloads under it.
    _AHEAD_ZOOM = 8.3

    def _ahead(vsd):
        v9 = dict(vsd, zoom=HEX_ZOOM)
        box = pad_box(view_to_bbox(v9))
        rres = res_for_view(v9, box)
        res = min(15, rres + HEX_UP)
        fres = min(13, res + CARRY_RES)
        y0, y1 = HOLD["y0"], HOLD["y1"]
        bkey = (res, tuple(round(v, 3) for v in box))
        key = (y0, y1, bkey)
        at = HOLD.get("ahead")
        if at is not None and at[0] == key:
            return
        if at is not None and not at[1].done():
            at[1].cancel()
        need = [y for y in (y0, y1) if (y, bkey) not in HOLD["aef"]]

        async def _run():
            got = await asyncio.gather(*(aef_fold(box, fres, y, read_res=rres) for y in need))
            for y, r in zip(need, got):
                HOLD["aef"][(y, bkey)] = r
            _trim(bkey)

        HOLD["ahead"] = (key, _spawn(_run()))

    async def _serve(vs, force=False):
        vsd = _vsd(vs)
        if vsd["zoom"] < HEX_ZOOM:
            # the frame stays (hidden in the browser, its tiles cached there),
            # so zooming back in over the same ground is instant
            HOLD["hex_status"] = f"hexagons from zoom {HEX_ZOOM:g}"
            if vsd["zoom"] >= _AHEAD_ZOOM:
                _ahead(vsd)
        else:
            await _serve_hex(vsd, force)
        _say(HOLD["hex_status"])

    async def refresh(vs, force=False, settle=True):
        """ONE serve at a time; the latest request wins while one is in flight."""
        if HOLD["busy"]:
            HOLD["pending"] = vs
            HOLD["pending_force"] = HOLD["pending_force"] or force
            return
        HOLD["busy"] = True
        try:
            while True:
                if settle:
                    await asyncio.sleep(SETTLE)
                if HOLD["pending"] is not None:
                    vs, HOLD["pending"] = HOLD["pending"], None
                    force, HOLD["pending_force"] = HOLD["pending_force"], False
                    settle = True
                    continue
                await _serve(vs, force)
                vs = HOLD["pending"]
                if vs is None:
                    return
                force, HOLD["pending"], HOLD["pending_force"] = HOLD["pending_force"], None, False
                settle = False
        except Exception as exc:
            tb = traceback.extract_tb(exc.__traceback__)
            where = f" (line {tb[-1].lineno})" if tb else ""
            _say(f"failed: {type(exc).__name__}: {exc}{where}")
            raise
        finally:
            HOLD["busy"], HOLD["pending"], HOLD["pending_force"] = False, None, False

    def _request(force=False):
        vs = HOLD["vs"] if HOLD["vs"] is not None else dict(HOME)
        HOLD["task"] = _spawn(refresh(vs, force, settle=False))

    def _on_camera(change):
        vs = change["new"]
        if not vs:
            return
        HOLD["vs"] = vs
        HOLD["task"] = _spawn(refresh(vs))

    if HOLD.get("h_cam") is not None:
        try:
            cmap.unobserve(HOLD["h_cam"], names="view")
        except ValueError:
            pass
    cmap.observe(_on_camera, names="view")
    HOLD["h_cam"] = _on_camera

    # ---- the click: the hexagon's account, as JSON the browser lays out ----------
    def _hex_card(p):
        fr = HOLD["frame"]
        cellh = p.get("cell")
        if fr is None or not cellh:
            return None
        cell = np.uint64(int(cellh, 16))
        ids = fr["cellid"]
        i = int(np.searchsorted(ids, cell))
        if i >= len(ids) or ids[i] != cell:
            return {"kind": "note", "title": "That hexagon is not in the current view's frame."}
        ew = float(fr["earth"][i])
        return {"kind": "hex", "cell": cellh, "level": None if np.isnan(ew) else ew, "earth": None if np.isnan(ew) else ew,
                "y0": int(fr["y0"]), "y1": int(fr["y1"]), "km2": float(CELL_KM2.get(HOLD["res"], 0))}

    def _card_send(p):
        card = _hex_card(p)
        if not card:
            HOLD["card"], HOLD["card_pick"] = None, None
            cmap.card = ""
            return
        adm = p.get("admin") or {}
        got = HOLD.get("place") or {}
        card["place"] = got["levels"] if got.get("n") == p.get("n") else [{"name": x} for x in (adm.get("locality"), adm.get("county"), adm.get("region")) if x]
        card["n"] = p.get("n")
        HOLD["card"], HOLD["card_pick"] = card, p
        cmap.card = json.dumps(card)

    def _place_later(p):
        """The whole ladder of divisions under the click, from GeoParquet;
        the card is resent with it if the click is still the latest."""
        n = p.get("n")

        async def _later():
            try:
                d = await asyncio.to_thread(division_at, p["lon"], p["lat"])
            except Exception:
                return
            if HOLD.get("pick_n") != n or not d:
                return
            levels = []
            for lv in d:
                nm = lv.get("name_en") or lv.get("name")
                if not nm:
                    continue
                lt = lv.get("local_type")
                levels.append({"name": nm, "tag": lt if lt and lt != lv["subtype"] else lv["subtype"]})
            HOLD["place"] = {"n": n, "levels": levels}
            if HOLD.get("card_pick") is p:
                _card_send(p)

        _spawn(_later())

    def _on_pick(change):
        try:
            p = json.loads(change["new"] or "{}")
        except Exception:
            return
        try:
            HOLD["pick_n"] = p.get("n")
            if p.get("close"):
                HOLD["card"], HOLD["card_pick"] = None, None
                cmap.card = ""
                return
            _card_send(p)
            if HOLD.get("card") and p.get("lon") is not None:
                _place_later(p)
        except Exception as e:
            cmap.card = json.dumps({"kind": "note", "title": f"click: {type(e).__name__}: {e}"})


    if HOLD.get("h_pick") is not None:
        try:
            cmap.unobserve(HOLD["h_pick"], names="pick")
        except ValueError:
            pass
    cmap.observe(_on_pick, names="pick")
    HOLD["h_pick"] = _on_pick

    # ---- the controls -----------------------------------------------------------------
    async def _push_stretch():
        # the browser draws the false colors from the bands and stretches them itself: it needs
        # the view's stretch, the only part of them the kernel still takes
        try:
            v = await s2_stretch()
        except Exception as e:
            _say(f"imagery stretch: {type(e).__name__}: {e}")
            return
        if v is not None:
            _cfg(s2_stretch=v)

    def _on_ctl_body(change):
        try:
            c = json.loads(change["new"] or "{}")
        except Exception:
            return
        act = c.get("act")
        if act == "s2scale":
            try:
                v = float(min(3.0, max(0.2, float(c.get("s2scale", HOLD["s2scale"])))))
            except (TypeError, ValueError):
                return
            if s2_set_scale(v):
                HOLD["s2scale"] = v
                HOLD["s2gen"] += 1
                _cfg(s2_scale=v, s2_gen=HOLD["s2gen"])
            return
        if act == "s2comp":
            # the imagery colors, stretched for the view in the browser now
            try:
                box = view_to_bbox(_vsd(HOLD["vs"]))
            except Exception:
                box = HOLD.get("box")
            if s2_set_composite(str(c.get("comp", "tci")), box):
                HOLD["s2gen"] += 1
                _cfg(s2_gen=HOLD["s2gen"], s2_comp=str(c.get("comp", "tci")))
                _spawn(_push_stretch())
            return
        if act == "model":
            m = str(c.get("model", "current"))
            if m in EW_MODELS and m != HOLD.get("model"):
                HOLD["model"] = m
                _cfg(model=m)
                _request(force=True)
            return
        if act == "aef":
            a, b = int(c.get("y0", HOLD["y0"])), int(c.get("y1", HOLD["y1"]))
            if a in AEF_YEARS_ALL and b in AEF_YEARS_ALL and a < b and (a, b) != (HOLD["y0"], HOLD["y1"]):
                HOLD["y0"], HOLD["y1"] = a, b
                _cfg(aef_from=a, aef_to=b)
                _request(force=True)
            return

    def _on_ctl(change):
        try:
            _on_ctl_body(change)
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            where = f" (line {tb[-1].lineno})" if tb else ""
            _say(f"control failed: {type(e).__name__}: {e}{where}")

    if HOLD.get("h_ctl") is not None:
        try:
            cmap.unobserve(HOLD["h_ctl"], names="ctl")
        except ValueError:
            pass
    cmap.observe(_on_ctl, names="ctl")
    HOLD["h_ctl"] = _on_ctl

    # the first fold waits for the browser's own view (its real size); a
    # re-run of this cell with a view already known serves it again
    if HOLD["frame"] is None and not HOLD["busy"]:
        if HOLD["vs"] is not None:
            _request()
    else:
        HOLD["sent"] = None
        _paint()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Under the map

    Press the button once the map has settled to query the current view's
    hexagons with DuckDB, one row per hexagon, likeliest earthwork first:
    `earthwork` (its highest-scoring finer cell's chance the ground moved)
    and `finer_cells` (how many finer cells the hexagon holds).
    """)
    return


@app.cell
def _(mo):
    tables_btn = mo.ui.run_button(label="table for the current view")
    tables_btn
    return (tables_btn,)


@app.cell
def _(HOLD, con, mo, tables_btn):
    mo.stop(not tables_btn.value or HOLD["frame"] is None, mo.md("*no hexagons yet (zoom in past 9)*") if tables_btn.value else None)
    con.register("view_cells", HOLD["frame"]["cells"])
    view_table = mo.sql(
        """
        SELECT * FROM view_cells ORDER BY earthwork DESC NULLS LAST
        """,
        engine=con,
    )
    return


if __name__ == "__main__":
    app.run()
