# aef-earthwork

- `earthwork.py`: where the ground was dug, filled or graded, from AlphaEarth, by a model taught
  where 3DEP lidar flew the same ground twice.
  It opens reading 2023 to 2025. Widening AEF Years (back as far as 2017) means more years of
  embeddings to fetch for every view, so the map loads more slowly.
  [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/kentstephen/aef-earthwork/blob/main/earthwork.py)
- **In the browser, no kernel:** [kentstephen.github.io/aef-earthwork-web](https://kentstephen.github.io/aef-earthwork-web/)
  ([code](https://github.com/kentstephen/aef-earthwork-web)), the same map with earthwork scored
  per 10 m pixel on the GPU.

AlphaEarth Foundations (AEF) annual embeddings draped over elevation, to see how river
corridors and watersheds change over time.

- **Elevation is the geometry.** A static DEM gives the shape of the ground: USGS 3DEP lidar
  in the US.
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

## Data and credits

What the map reads (`earthwork.py`):

- **AlphaEarth Foundations Satellite Embedding dataset**, Google and Google DeepMind (CC BY 4.0), the
  annual embeddings from Source Cooperative's mirror, `tge-labs/aef`.
- **Sentinel-2 yearly mosaics**, Earth Genome (CC BY 4.0), found through Earth Genome's STAC. Contains
  modified Copernicus Sentinel data.
- **Overture Maps divisions** for place names (ODbL; OpenStreetMap contributors, Overture Maps
  Foundation, with geoBoundaries, Esri Community Maps contributors and LINZ under CC BY 4.0): the
  PMTiles, and the GeoParquet at `fused/overture` on Source Cooperative.
- **Search**: Photon by Komoot, over OpenStreetMap data (ODbL).
- **Basemap**: CARTO dark matter, © CARTO, © OpenStreetMap contributors.

What teaches the two models (`earthwork_model.py`, `earthwork_abroad.py` and the site scripts). The
map reads none of these, only the models taught from them:

- **USGS 3D Elevation Program (3DEP)** 1 m DEMs, flown twice over the same ground, through The National
  Map and the 3DEP Elevation Index (public domain). The truth for both models, Puerto Rico and the
  US Virgin Islands included.
- **AlphaEarth Foundations** embeddings as a 10 m mosaic, Source Cooperative `tge-labs/aef-mosaic`
  (CC BY 4.0).
- **USDA NASS Cropland Data Layer**, read from the Icechunk store at
  `chill/usda-cropland-data-layer` on Source Cooperative (public domain): crop switches taught as not
  digging.

Only in the US + abroad model:

- **IGN France** RGE ALTI 1 m and LiDAR HD terrain models, Réunion and Guadeloupe, through the
  Géoplateforme (Licence Ouverte / Etalab 2.0).
- **VIRTUAL SHIZUOKA**, Shizuoka Prefecture's airborne laser surveys (CC BY 4.0 and ODbL).
- **Hong Kong airborne lidar DTMs**, Civil Engineering and Development Department, Hong Kong SAR
  Government, through the CSDI portal.
- **ICESat-2** ATL03 (through OpenAltimetry) and ATL08 (NSIDC DAAC), and **GEDI** L2A (LP DAAC), NASA.
- **Copernicus DEM GLO-30**, © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018,
  provided under COPERNICUS by the European Union and ESA; all rights reserved. Used for the slope
  correction on the satellite laser pairs.
