# aef-elevation

AlphaEarth Foundations (AEF) annual embeddings draped over elevation, to see how river
corridors and watersheds change over time.

- **Elevation is the geometry.** A static DEM gives the shape of the ground:
  [Mapterhorn](https://mapterhorn.com) worldwide, USGS 3DEP lidar in the US.
- **The embeddings carry the time series.** Each 10 m pixel (or H3 cell) has a 64-dimension
  embedding for every year from 2017 to 2025. Change between years, the year of the largest
  step, and probes against land cover, water and canopy height become the color on the terrain.
- **Repeat lidar is the check.** Where 3DEP flew a place twice, the difference between surveys
  is measured ground change to compare against embedding change.

AEF was trained to reconstruct the Copernicus DEM and GEDI lidar canopy height, but at
inference it only sees Sentinel-2, Landsat and Sentinel-1. Elevation in the embeddings is
learned from imagery, so it is expected to be relative rather than precise.

## Setup

```sh
uv sync
uv run marimo edit <notebook>.py --sandbox --watch
```

## Data

- AEF annual embeddings, Source Cooperative mirror: `tge-labs/aef` (CC-BY 4.0)
- Mapterhorn terrain tiles (terrarium encoding, PMTiles)
- USGS 3DEP DEMs and WESM acquisition metadata
- GLO-30 HAND (height above nearest drainage)
