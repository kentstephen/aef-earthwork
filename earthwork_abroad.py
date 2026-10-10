"""The earthwork model taught with truth from outside the 48 states, side by side with the US-only recipe.

The model learned digging from US repeat lidar, and abroad its score is dim: on deep cuts in Puerto Rico, on
satellite laser pairs over mines in Indonesia and Sierra Leone, almost nothing passes 0.5 on the map, though the
ranking is often fine (the area scale, A, finds them). This adds the truth that exists abroad, all of it read
from caches other scripts build (data/earthwork/), and compares recipes on ground each was not taught on:

  tropic   IGN France, Reunion and Guadeloupe (earthwork_sites_intl.py): unchanged ground only, as not digging
           (the first terrain model there predates AlphaEarth)
  islands  Puerto Rico and the US Virgin Islands, 3DEP 2018 against 2024 (built here, build_islands): moved
           ground over 2 m teaches digging (shallower moves there are mostly lidar ground under forest), the
           unchanged ground as not
  space    ICESat-2 and GEDI pairs (earthwork_points_space.py): building and mining sites both ways, the rest
           as not digging. A site holds tens to hundreds of labels where a lidar site holds tens of thousands,
           so each weighs W_SPACE times a lidar pixel
  asia     Shizuoka and Hong Kong lidar (earthwork_sites_asia.py), when cached: sites of kind building or
           mining both ways, survey sites only scored, the rest (forest, ponds, urban, farm) as not digging

Recipes: base (the US recipe: digging, burns, dunes, Brazoria marsh), base plus each group alone, all of them,
and lidar-all (all of them on the lidar-pivots recipe: farm lidar taught quiet, no crop maps). Every site a
recipe teaches is scored by a model taught without it (5 folds over sites, stratified by group and kind);
every score is the map's, the model's times the gate on plain change. Samples are small (N_POS moved and
N_NEG unchanged per lidar site) so the six recipes run in minutes.

Held out, W_SPACE 5 (the run of 2026-10-10; moved labels in parentheses):
                                     base   lidar-all
  US building AP                     .556   .532
  US building unchanged > .5         4.8%   6.4%
  US burns, dunes, marsh unch > .5   7.7%  11.3%
  US farm lidar unchanged > .5      12.8%   3.5%
  crop switches (CDL) > .5          18.9%   6.0%
  islands deep moved > .5           40.3%  47.9%
  Shurugwi moved > .5 (13)             0%    62%   (unchanged 0% and 0%)
  Mobimbi moved > .5 (54)              0%    78%   (unchanged 0% and 8%)
  Kolwezi moved > .5 (392)             0%    23%   (unchanged 0% and 0%)
  Morowali moved > .5 (36)             0%    31%
  Sembehun moved > .5 (85)             0%    47%   (unchanged 0% and 12%)
  Xiong'an moved > .5 (405)           42%    74%   (unchanged 16% and 26%)
  2023 vs 2025, no truth, share > .5: Nusantara 2.8% to 74%, Manila Bay reclamation 0.1% to 12%,
  Cairo desert 12.9% to 4.9%, Sahara 13.9% to 8.7%, Bujumbura 0% to 2.8%
Taught alone, the tropical quiet ground dims tropical digging (islands 40% to 14%); with the islands and the
satellite pairs beside it, it keeps the banana fields quiet without that cost. The satellite pairs do most of
the brightening abroad. W_SPACE 2 to 20 trades between them: at 20 Kolwezi 40%, Sembehun 69% (but 20% of its
unchanged ground), building AP .523, crop switches 12.5%. The Shizuoka and Hong Kong sites change nothing
(their moves are mostly too small for AlphaEarth to see).

Run: uv run --with icechunk python earthwork_abroad.py [recipe ...]   (Asia recipes appear when its sites are cached)
     W_SPACE=5 FOLDS=0 SAVE=abroad uv run --with icechunk python earthwork_abroad.py lidar-all
     (FOLDS=0 fits only the final model; SAVE writes it to models/earthwork-lr-<SAVE>.npz, which the map's M
     steps through)
"""
import json
import os
import sys
import time
import warnings
import zlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP

import earthwork_model as em
import earthwork_sites_intl as intl

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
W_SPACE = float(os.environ.get("W_SPACE", "5"))
N_POS, N_NEG = int(os.environ.get("N_POS", "5000")), int(os.environ.get("N_NEG", "50000"))
FOLDS = int(os.environ.get("FOLDS", "5"))
SAVE = os.environ.get("SAVE", "")
t0 = time.time()
say = lambda *a: print(f"[{time.time() - t0:5.0f} s]", *a, flush=True)

# Puerto Rico and the US Virgin Islands: 3DEP's 2018 blocks against the 2024 island-wide project, a 1 m tile
# at each point (the 2018 blocks east of 66 W are in UTM 20, the 2024 tiles in UTM 19: the second tile is
# warped into the first's CRS)
ISLANDS_2024 = "PR_PuertoRicoUSVI_D24"
ISLAND_SITES = [
    ("pr-ponce", (-66.61, 18.01), "PR_PRVI_A_2018"), ("pr-mayaguez", (-67.14, 18.20), "PR_PRVI_A_2018"),
    ("pr-aguadilla", (-67.13, 18.49), "PR_PRVI_F_2018"), ("pr-arecibo", (-66.70, 18.46), "PR_PRVI_F_2018"),
    ("pr-juana-diaz", (-66.50, 18.03), "PR_PRVI_A_2018"), ("pr-dorado", (-66.27, 18.46), "PR_PRVI_G_2018"),
    ("pr-bayamon", (-66.16, 18.38), "PR_PRVI_G_2018"), ("pr-caguas", (-66.04, 18.24), "PR_PRVI_H_2018"),
    ("pr-carolina", (-65.95, 18.41), "PR_PRVI_E_2018"), ("pr-humacao", (-65.82, 18.15), "PR_PRVI_H_2018"),
    ("vi-st-croix", (-64.75, 17.73), "PR_PRVI_VI_B2_2018"), ("vi-st-thomas", (-64.93, 18.34), "PR_PRVI_VI_B1_2018"),
]
ISLAND_DEEP_M = 2.0
ASIA_QUIET = ("forest", "ponds", "urban", "farm")


def build_islands():
    """Each island site's cache (earthwork_model.site_data's), built where missing."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    missing = [s for s in ISLAND_SITES if not os.path.exists(f"{em.DATA}/{s[0]}.npz")]
    if not missing:
        return
    _open, first = rasterio.open, []

    def paired_open(path, *a, **k):
        s = _open(path, *a, **k)
        if not first or first[0].closed:
            first[:] = [s]
            return s
        f = first.pop()
        return WarpedVRT(s, crs=f.crs, resampling=Resampling.bilinear) if s.crs != f.crs else s
    em.rasterio.open = paired_open
    try:
        for name, pt, p1 in missing:
            em.site_data(name, "island", pt, p1, ISLANDS_2024)
    finally:
        em.rasterio.open = _open


def load_sites():
    """name -> {X, y, plain, kind, group, w}"""
    S = {}
    for name, kind, pt, p1, p2 in em.SITES + em.PIVOT_SITES:
        d = em.site_data(name, kind, pt, p1, p2)
        if d is not None:
            S[name] = {"X": d["X"], "y": d["y"], "plain": d["plain"], "kind": kind, "group": "us", "w": 1.0}
    load = lambda n: np.load(f"{em.DATA}/{n}.npz", allow_pickle=True) if os.path.exists(f"{em.DATA}/{n}.npz") else None
    for name, _, _, _ in intl.SITES_INTL:
        if (z := load(name)) is not None:
            S[name] = {"X": z["X"], "y": z["y"], "plain": z["plain"], "kind": "tropic", "group": "tropic", "w": 1.0}
    build_islands()
    for name, _, _ in ISLAND_SITES:
        if (z := load(name)) is not None:
            y = np.where((z["y"] == 1) & (np.abs(z["dz"]) <= ISLAND_DEEP_M), -1, z["y"]).astype(np.int8)
            S[name] = {"X": z["X"], "y": y, "plain": z["plain"], "kind": "island", "group": "islands", "w": 1.0}
    for f in sorted(os.listdir(em.DATA)):
        if not f.endswith(".npz"):
            continue
        if f.startswith("space-"):
            z = load(f[:-4])
            k = json.loads(str(z["meta"]))["kind"]
            S[f[:-4]] = {"X": z["X"], "y": z["y"], "plain": z["plain"], "kind": k if k in ("building", "mining") else "quiet",
                         "group": "space", "w": W_SPACE}
        elif f.startswith(("shizuoka-", "hk-")):
            z = load(f[:-4])
            k = json.loads(str(z["meta"]))["kind"]
            S[f[:-4]] = {"X": z["X"], "y": z["y"], "plain": z["plain"],
                         "kind": k if k in ("building", "mining", "survey") else "quiet" if k in ASIA_QUIET else "survey",
                         "group": "asia", "w": 1.0}
    return S


def main():
    S = load_sites()
    say(f"{len(S)} sites loaded")
    pick = lambda group, kinds: [n for n, d in S.items() if d["group"] == group and d["kind"] in kinds]
    US_DIG = pick("us", ("building", "mining"))
    US_QUIET = pick("us", ("burn", "dunes")) + [n for n in em.QUIET_SITES if n in S]
    FARM = pick("us", ("farm",))
    TROPIC, ISL = pick("tropic", ("tropic",)), pick("islands", ("island",))
    SPACE_DIG, SPACE_QUIET = pick("space", ("building", "mining")), pick("space", ("quiet",))
    ASIA_DIG, ASIA_Q, ASIA_SURVEY = pick("asia", ("building", "mining")), pick("asia", ("quiet",)), pick("asia", ("survey",))
    # (sites taught both ways, sites taught only as not digging)
    V = {
        "base": (US_DIG, US_QUIET),
        "tropic": (US_DIG, US_QUIET + TROPIC),
        "islands": (US_DIG + ISL, US_QUIET),
        "space": (US_DIG + SPACE_DIG, US_QUIET + SPACE_QUIET),
        "all": (US_DIG + ISL + SPACE_DIG, US_QUIET + TROPIC + SPACE_QUIET),
        "lidar-all": (US_DIG + ISL + SPACE_DIG, US_QUIET + FARM + TROPIC + SPACE_QUIET),
    }
    if ASIA_DIG or ASIA_Q:
        V["asia"] = (US_DIG + ASIA_DIG, US_QUIET + ASIA_Q)
        V["lidar-all-asia"] = (US_DIG + ISL + SPACE_DIG + ASIA_DIG, US_QUIET + FARM + TROPIC + SPACE_QUIET + ASIA_Q)
    want = sys.argv[1:] or list(V)

    rng = np.random.default_rng(0)
    IDX = {}
    for n, d in S.items():
        pos, neg = np.flatnonzero(d["y"] == 1), np.flatnonzero(d["y"] == 0)
        IDX[n] = (rng.choice(pos, min(len(pos), N_POS), replace=False), rng.choice(neg, min(len(neg), N_NEG), replace=False))

    def fitter(both, quiet):
        parts = [(n, np.concatenate(IDX[n])) for n in both] + [(n, IDX[n][1]) for n in quiet]
        X = np.concatenate([S[n]["X"][i].astype(np.float32) for n, i in parts])
        y = np.concatenate([S[n]["y"][i] for n, i in parts])
        w = np.concatenate([np.full(len(i), S[n]["w"], np.float32) for n, i in parts])
        return LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit(X, y, sample_weight=w)

    # 5 folds over sites, stratified by group and kind (a stable shuffle, so every recipe holds out the same)
    FOLD, groups = {}, {}
    for n, d in S.items():
        groups.setdefault((d["group"], d["kind"]), []).append(n)
    for key, ns in sorted(groups.items()):
        off = zlib.crc32("/".join(key).encode()) % 5
        for i, n in enumerate(sorted(ns, key=lambda s: zlib.crc32(s.encode()))):
            FOLD[n] = (i + off) % 5
    prob = lambda m, X: m.predict_proba(X.astype(np.float32))[:, 1]

    # ground with no truth, AlphaEarth 2023 against 2025 in a 3 km window, and the crop switches never taught
    places = {"nusantara IKN": (116.70, -0.97), "manila bay reclaim": (120.975, 14.52), "lekki/dangote": (3.98, 6.43),
              "bujumbura": (29.36, -3.38), "taylor TX": (-97.48, 30.52), "cairo desert (NAC)": (31.75, 30.02),
              "sahara (algeria)": (2.5, 28.0), "riyadh edge": (46.85, 24.85), "guadeloupe bananas": (-61.58, 16.05)}
    place_x = {}
    for nm, (lon, lat) in places.items():
        x0, y0 = int((lon - em.AEF_X0) / em.AEF_RES) - 150, int((em.AEF_Y0 - lat) / em.AEF_RES) - 150
        b, okb = em.aef_unit(2023, (y0, y0 + 300, x0, x0 + 300))
        a, oka = em.aef_unit(2025, (y0, y0 + 300, x0, x0 + 300))
        ok = (okb & oka).ravel()
        place_x[nm] = (em.features(b, a)[ok], (1 - (a * b).sum(0)).ravel()[ok])
    cdl = [em.cdl_switches(name, pt, yb, ya) for name, pt, pairs in em.CDL_TEST for yb, ya in pairs]
    say("places and crop switches read")

    res = {}
    for v in want:
        both, quiet = V[v]
        taught = set(both) | set(quiet)
        scores = {}
        for k in range(5 if FOLDS else 0):
            m = fitter([n for n in both if FOLD[n] != k], [n for n in quiet if FOLD[n] != k])
            for n in (n for n in taught if FOLD[n] == k):
                sel = S[n]["y"] >= 0
                scores[n] = (em.gate(prob(m, S[n]["X"][sel]), S[n]["plain"][sel]), S[n]["y"][sel])
        final = fitter(both, quiet)
        if SAVE:
            out = os.path.join(HERE, "models", f"earthwork-lr-{SAVE}.npz")
            np.savez(out, w=final.coef_[0].astype(np.float32), b=np.float32(final.intercept_[0]),
                     features=np.array("[b, a, a*b, (a-b)^2] of AlphaEarth unit embeddings: b the year before the first flight, a the year after"),
                     middle=np.bool_(False), sites=np.array(both), not_digging=np.array(quiet),
                     tested=np.array([n for n in S if n not in taught]), moved_m=em.MOVED_M, still_m=em.STILL_M,
                     recipe=np.array(f"earthwork_abroad.py {v}: satellite labels weighted {W_SPACE:g}, samples {N_POS}/{N_NEG} per site"))
            say(f"saved {os.path.relpath(out, HERE)}")
        if not FOLDS:
            continue
        for n, d in S.items():
            if n not in scores:
                sel = d["y"] >= 0
                scores[n] = (em.gate(prob(final, d["X"][sel]), d["plain"][sel]), d["y"][sel])
        row = {}
        for n, (g, y) in scores.items():
            row[n] = {"ap": AP(y, g) if 0 < y.sum() < len(y) else np.nan, "moved5": (g[y == 1] > .5).mean() if (y == 1).any() else np.nan,
                      "still5": (g[y == 0] > .5).mean() if (y == 0).any() else np.nan, "nm": int((y == 1).sum())}
        for nm, (X, pl) in place_x.items():
            row[f"place:{nm}"] = (em.gate(prob(final, X), pl) > .5).mean()
        row["cdl"] = (np.concatenate([em.gate(prob(final, c["X"]), c["plain"]) for c in cdl]) > .5).mean()
        res[v] = row
        say(f"{v} done")
    if not FOLDS:
        return

    mean = lambda v, ns, f: np.nanmean([res[v][n][f] for n in ns]) if ns else np.nan
    print("\nHeld out where taught, the map's score")
    print(f"{'':34s}" + "".join(f"{v:>15s}" for v in want))
    building = [n for n in US_DIG if S[n]["kind"] == "building"]
    for label, ns, f in (("US building AP", building, "ap"), ("US mining AP", [n for n in US_DIG if n not in building], "ap"),
                         ("US building moved > .5", building, "moved5"), ("US building unchanged > .5", building, "still5"),
                         ("US burns, dunes, marsh unch > .5", US_QUIET, "still5"), ("US farm lidar unchanged > .5", FARM, "still5"),
                         ("islands deep AP", ISL, "ap"), ("islands deep moved > .5", ISL, "moved5"), ("islands unchanged > .5", ISL, "still5"),
                         ("tropic quiet unchanged > .5", TROPIC, "still5"),
                         ("space digging moved > .5", SPACE_DIG, "moved5"), ("space digging unch > .5", SPACE_DIG, "still5"),
                         ("space quiet unchanged > .5", SPACE_QUIET, "still5"),
                         ("asia digging AP", ASIA_DIG, "ap"), ("asia quiet unchanged > .5", ASIA_Q, "still5"),
                         ("asia survey AP", ASIA_SURVEY, "ap"), ("asia survey moved > .5", ASIA_SURVEY, "moved5"),
                         ("asia survey unchanged > .5", ASIA_SURVEY, "still5")):
        if ns:
            print(f"  {label:32s}" + "".join(f"{mean(v, ns, f):15.3f}" for v in want))
    print(f"  {'crop switches (CDL) > .5':32s}" + "".join(f"{res[v]['cdl']:15.3f}" for v in want))
    print("\nPer site abroad: moved > .5 / unchanged > .5 (AP)")
    for n in TROPIC + ISL + SPACE_DIG + SPACE_QUIET + ASIA_DIG + ASIA_Q:
        print(f"  {n:34s} n={res[want[0]][n]['nm']:6d} " + "".join(
            f"  {res[v][n]['moved5']:4.0%}/{res[v][n]['still5']:4.0%} ({res[v][n]['ap']:.2f})" for v in want))
    print("\nNo truth, 2023 against 2025: share > .5")
    for nm in places:
        print(f"  {nm:24s}" + "".join(f"{res[v]['place:' + nm]:15.1%}" for v in want))
    json.dump(res, open(os.path.join(em.DATA, f"abroad-{'-'.join(want)}.json"), "w"), default=float)
    say("done")


if __name__ == "__main__":
    main()
