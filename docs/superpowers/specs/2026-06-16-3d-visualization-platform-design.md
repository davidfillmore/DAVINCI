# 3D Visualization Platform — Engine Selection & Staged Roadmap

**Date:** 2026-06-16
**Status:** Draft for review
**Scope:** A long-horizon design for adding **interactive 3D visualization** to DAVINCI as an
**embedded, in-process** Python layer that builds scenes directly from live `xarray` datasets —
spanning flight tracks, volumetric model fields, a globe-scale Earth, and point networks/profiles,
with a deferred path to outreach assets and immersive VR.

## Context & Problem

DAVINCI's 3D today is **static matplotlib `mplot3d`**: 3D flight-track scatter over a surface map
plane (`plots/renderers/_track3d.py`, `track_map_3d.py`), plus `curtain.py` and
`vertical_profile.py` cross-sections. These are fixed-angle PNG/PDF artifacts — you cannot rotate,
zoom, slice, or fly through a field.

The ambition is a **platform**, not a single plot type: a real GPU scene graph driven from Python
that serves four use cases over a long horizon — (1) **scientific exploration** (fly/slice
volumetric fields, follow tracks through model output), (2) **outreach/presentation**, (3)
eventually **immersive VR/AR**, treating (4) the **engine itself as an extensible substrate**.

Hard constraints (user-confirmed):

- **Open-source / permissive license** — fits NSF NCAR open-science norms; no per-seat or
  revenue-share fees.
- **Runs on researcher laptops** — macOS (the dev machine) and Linux workstations, integrated GPUs,
  **no GPU farm**.
- **Reuses DAVINCI's data layer** — consumes existing `xarray` datasets / pairing output directly;
  **no separate export-to-mesh asset pipeline**.
- **Embedded, in-process** — scenes built from live `xarray` objects inside the process, tightly
  coupled to the data layer (chosen integration model).
- Python need **not** be the engine's primary language; a solid binding/plugin layer is acceptable.

## The Decision

**Adopt PyVista (VTK) + GeoVista + `pyvista-xarray` as the embedded spine** — archetype **A**
(Python-native scientific visualization on VTK), **not** archetype **B** (a Python game engine).

This **inverts the working assumption** we entered with (a Panda3D-class game-engine spine). The
inversion was driven by evidence, not preference: a deep, adversarially-verified evaluation of 11
candidate engines + 5 cross-cutting research threads converged from independent angles on the same
conclusion. The user reviewed the inversion and **accepted PyVista-as-spine**.

### Why the game-engine spine was rejected

1. **Control inversion vs. the embedded model.** DAVINCI is import-as-library, host-owns-the-loop,
   render-to-file-from-a-pipeline-stage (`render()` → `plots_generated`, read by
   `pipeline/runner.py`). PyVista never seizes the process (`off_screen=True` / `show=False`), so
   Python stays the driver and composes with the CLI/pipeline/Jupyter. A true game engine owns
   `main()` — it **fights the architecture the user chose**. Godot is fundamentally control-inverted
   (engine hosts a Python interpreter for node scripts) and its Python bindings are non-production
   (`py4godot`/`godot-python` self-described as demo/dormant; Godot 4 support on an unmerged branch).
   O3DE's Python is editor-only automation.

2. **Build-it-yourself burden.** PyVista/VTK ship first-class GPU **volume rendering** (`add_volume`),
   **isosurfaces** (`contour`), and **slice planes** (`slice_orthogonal`) on `ImageData`/
   `RectilinearGrid` — exactly the `(time,lev,lat,lon)` operations DAVINCI needs — plus **GeoVista**,
   a purpose-built geoscience globe ("Cartopy for PyVista", UK Met Office authored) with Natural Earth
   coastlines and rectilinear/curvilinear/unstructured-swath meshes, with **real climate prior art**
   (DKRZ overlays MPI-ESM global + CORDEX regional orography on a sphere). Panda3D scores 1/5 on globe
   and volumetric: **no** first-party globe, **no** volume renderer/isosurface (verified — only
   experimental single-author snippets), **no** xarray bridge. Choosing the game engine means
   rebuilding the cartographic + volumetric + colormap + xarray layers that PyVista/GeoVista already
   provide — and there is **zero atmospheric/Earth-science interactive-3D prior art on any Python game
   engine** (Panda3D/Ursina/Harfang appear only in robotics/behavioral-VR/hobby demos). Every durable
   atmospheric 3D tool sits on the VTK/OpenGL lineage (Met.3D, PyVista/GeoVista, ParaView/trame).

3. **VR does not rescue the game engine.** The whole reason to want a game-engine spine is VR — but
   **as of mid-2026 there is no production OpenXR runtime on macOS at all** (SteamVR dropped macOS in
   2020; Monado is Linux/Android only; Apple Vision Pro is closed visionOS reachable only via CloudXR
   streaming). **No** engine — Godot, Panda3D, or PyVista — can drive hardware VR on the target Mac
   laptops. VR is an OS-runtime gap DAVINCI cannot influence; it must be a deferred, Linux-only /
   out-of-process bet. Letting a macOS-VR requirement pick the spine would trade away the
   desktop/2.5D fit that pays off immediately for a capability that cannot run on the primary platform.

### Decision matrix (eval scores, 0–100)

| Engine | Must-haves | Score | One-line verdict |
|---|---|---|---|
| **PyVista (VTK) + GeoVista + pyvista-xarray** | ✅ | **84** | **The spine.** Only candidate that is import-as-library + xarray-fed + headless-out-of-box + volume/isosurface/slice + true geoscience globe, mapping 1:1 onto DAVINCI's `render()`/`plots_generated` pipeline. Sole gap is VR (deferred for all). |
| Vispy | ✅ | 68 | Strong GPU fallback for very-large point/volume scenes (millions of cells); host-owns-loop & no-VR **confirmed**; but no globe, no xarray bridge — secondary engine, not spine. |
| Vedo (VTK) | ✅ | 68 | Lighter VTK API, offscreen-from-numpy **confirmed**, best-in-class volume/slice; no globe, no VR, single-maintainer — viable VTK-family alternative, but GeoVista makes PyVista the default. |
| Panda3D | ❌ | 52 | Embeddable (**confirmed**) and best Python-native VR escape-hatch, but fails sci-viz must-have: no globe/volume/isosurface, no xarray bridge, no atmospheric prior art. Keep **only** as a scoped Linux-VR escape hatch. |
| Godot + Python | ❌ | 34 | Best maintained cross-platform OpenXR of the field, but control-inverted, no production Python binding, no sci-viz primitives. Out-of-process VR only, consuming exported glTF. |
| Open3D | ❌ | 33 | Clean MIT bindings, good for tracks/point clouds, but no volume, no globe, hard no-VR — nothing PyVista doesn't do better. |
| Ursina | ❌ | 28 | Pleasant ergonomics, but no volume/globe/VR, weak large-grid perf, Apple-Silicon shader friction, single-maintainer. Demo toy. |
| Blender (`bpy`) | ❌ | 28 | Best offline beauty renders, but GPL copyleft poisons in-process embedding in a permissive toolkit, and headless/background-only. Out-of-process beauty backend only. |
| Harfang3D | ❌ | 22 | Genuine VR + Python binding, but OSS repo frozen since Aug 2023, Windows-only wheels, zero sci-viz — disqualifying. |
| Pyglet + moderngl | ❌ | 22 | Honest build-effort floor: maximum embeddability, but you reinvent the entire engine. Quantifies the game-engine cost. |
| O3DE | ❌ | 14 | Category mismatch: Python is editor-only automation inside the engine, no sci-viz primitives, macOS experimental, Vulkan-only VR. Wrong tool. |

### Verification refinements (honest caveats)

Adversarial verification (as of 2026-06-16) confirmed the deal-breakers above and **refined** two
PyVista claims in PyVista's favor, plus one neutral caveat:

- **macOS rendering mechanism corrected.** PyVista renders interactively **and** off-screen on
  macOS+Linux laptops with integrated GPUs using stock VTK 9.5 wheels — via the **native hardware-GL
  backend** (Cocoa on macOS, GLX/X11 on Linux), *not* an OSMesa CPU fallback. No GPU farm needed.
- **xarray ingest is closer to one-call than first stated.** There is no native `pv.from_xarray()`,
  and the **GeoVista globe** needs small hand-written `Transform.from_1d/from_2d` adapters — but the
  **gridded path** has a maintained `pyvista-xarray` accessor with CF auto-detection. Glue is small
  per-geometry mesh-builders, not a bespoke pipeline.
- **PyVista VR is genuinely absent on mac/Linux (confirmed, deal-breaker severity — but only for P5).**
  `pyvista-reality` is OpenVR-only, Windows-only, not on PyPI, dormant; VTK's OpenXR module is not
  exposed through a maintained PyVista binding. Consistent with the plan: VR is deferred for *every*
  candidate.

## Goals

1. An **embedded, in-process** 3D rendering layer that builds scenes from live `xarray` datasets and
   emits artifacts the same way today's 2D renderers do (`render()` → `plots_generated`).
2. Cover all four scene types over the roadmap: flight tracks+terrain, volumetric model fields,
   globe-scale Earth, point networks/profiles.
3. **Reuse, not reinvent,** DAVINCI's shape-driven dispatch (`detect_spatial_geometry`,
   `surface_level_index`, the `geometry` attr point/track/profile/swath/grid) and `labeling.py`.
4. A **renderer-agnostic scene abstraction** (scene = VTK data objects + geometry attrs) so an
   out-of-process VR backend can attach later without rewriting the data layer.
5. Desktop-first on macOS+Linux; VR explicitly **off the critical path**.

## Non-Goals

- Adopting a game engine as the sci-viz substrate.
- Native VR on macOS (OS-runtime gap — impossible as of mid-2026).
- Cesium-style automatic streaming LOD tiling (see Decisions — DKRZ coarse+regional pattern instead).
- Replacing the existing 2D matplotlib renderers; 3D is additive.

## Architecture

The 3D layer mirrors the existing `plots/renderers/spatial/` structure rather than inventing a
parallel one.

- **`Spatial3DPlotter`** family under `plots/renderers/spatial3d/` (proposed), each mirroring the
  `render()` contract of `spatial/field.py` and registering as optional pipeline plot types.
- **Shape-driven dispatch reused.** 3D renderers dispatch off the same `geometry` attr that
  `detect_spatial_geometry` already sets — point/track/profile → glyph/line meshes; grid/swath →
  structured/unstructured meshes + globe drape. No new dispatch taxonomy.
- **Thin xarray→mesh adapters** (~20–40 lines per geometry) convert a geometry's `xarray` slice into
  PyVista `PolyData`/`ImageData`/`RectilinearGrid`, sharing one colormap/units helper that calls the
  existing `labeling.py` (terse titles, SI units, no x/y leakage — same rules as 2D).
- **Host owns the loop.** All rendering defaults to headless (`off_screen=True` → `.screenshot()` /
  `export_*`), so the layer composes with the CLI/pipeline. An optional interactive window
  (`pyvistaqt` / `trame`) is a later, opt-in capability (see Decisions).
- **Scene abstraction for VR decoupling.** The renderable scene is kept as VTK data objects +
  geometry attrs, so a future VR backend (out-of-process, glTF-consuming) attaches without touching
  the data layer.
- **CESM vertical convention is single-sourced.** Any 3D field passes through `surface_level_index`
  (CESM `lev=-1` = surface) **before** meshing — guarding the bug class rediscovered 4×.

## Staged Roadmap

Each phase is independently shippable; VR is deferred and blocks nothing earlier.

### P0 — Spike: one flight track from a live xarray Dataset (S, 3–5 days)
Prove the embedded, host-owns-the-loop model end-to-end. A `Spatial3DPlotter` prototype (mirrors
`spatial/field.py`) builds a PyVista `PolyData` line/tube colored by a scalar from a track
`xr.Dataset` (time, lat/lon/alt); renders interactively (`show`) **and** headless
(`off_screen=True` → `.screenshot()`/PDF) on a macOS + Linux laptop; wired as an optional pipeline
plot type returning `plots_generated`, output copied to the iCloud Claude dir. Includes a
**programmatic render-mark assertion** (PolyData/PathCollection vs QuadMesh) per the repo's
verify-mark rule.

### P1 — Geometry-dispatch parity + point networks/profiles (S–M, ~1 week)
Extend the 3D path to all irregular geometries, reusing DAVINCI's shape-driven dispatch instead of a
parallel one. 3D renderers for point networks (`PolyData` points/glyphs colored by value) and
profiles, dispatched off the same `geometry` attr; thin xarray→mesh adapters per geometry sharing the
`labeling.py`-backed colormap/units helper; unit + **pipeline-path** tests (integration tests **must**
run through `PipelineRunner.run_from_config`).

### P2 — Volumetric model fields (M–L, 2–4 weeks — the headline capability)
Fly/slice 3D `(lev,lat,lon)` model fields with GPU volume rendering, isosurfaces, and slice planes via
`add_volume`/`contour`/`slice_orthogonal` on `ImageData`/`RectilinearGrid`, fed by `pyvista-xarray`.
**The real effort is preprocessing, not rendering:** a stage that (a) applies `surface_level_index`
(CESM `lev=-1`) **before** meshing and (b) regrids hybrid sigma-pressure to a clean
`(z,lat,lon[,time])` z-coordinate, then hands the result to `mesh(z=…)`. Level-ordering correctness is
a **first-class test** (guards the rediscovered-4× CESM bug class). Choose `ImageData` (uniform z) for
cheap GPU volume rendering where scientifically acceptable, `RectilinearGrid` for native non-uniform
levels.

### P3 — Globe-scale Earth + whole-Earth-to-region zoom (M–L, 2–4 weeks)
True 3D textured globe with coastlines, draping gridded/swath fields. GeoVista-backed renderer:
rectilinear/curvilinear via `Transform.from_1d/from_2d`, satellite swaths via `from_unstructured`
(matches the `swath` geometry attr), Natural Earth coastlines/base layers. Whole-Earth-to-region zoom
implemented **explicitly** (not assumed free): coarse global mesh + on-demand finer regional mesh,
`enclosed()`/`threshold()` to crop, `decimate()` to coarsen, `zlevel` offset to stack nested regions
(DKRZ pattern). Thin **vendored wrappers** around the GeoVista calls DAVINCI depends on (pin 0.5.3)
to hedge its pre-1.0 / smaller-maintainer risk; keep flat projections to the supported
cylindrical/pseudo-cylindrical set.

### P4 — Outreach: standalone interactive + web (S–M, ~1 week)
Self-contained, shareable interactive 3D assets without a game engine. One-call exports —
`export_html` (standalone, no external deps), `export_gltf` (`.glb` for downstream/Godot),
`export_vtksz`, plus movies — added as pipeline export targets alongside PNG/PDF, delivered to the
iCloud Claude dir; optional `trame`/vtk.js viewer for Jupyter/web. This doubles as the realistic
immersive delivery channel (and the only plausible future Vision-Pro path, via WebXR).

### P5 — VR/immersive (DEFERRED; behind an interface; Linux / out-of-process) (L, multi-month, explicitly deferred)
Optional native HMD walkthroughs, kept off the macOS critical path and off the spine. A
renderer-agnostic scene abstraction (scene = VTK data objects + geometry attrs) lets a VR backend
attach later without rewriting the data layer. Prototype VR on **Linux first** (Monado /
SteamVR-on-Linux) or a standalone Quest; native VR consumes **exported glTF** in a **separate
out-of-process app** (Godot 4.6 OpenXR is the strongest maintained path; `panda3d-openxr` the
Python-native fallback). Gate on VTK Vulkan/Metal + OpenXR maturity and re-evaluate macOS OpenXR
(OpenXR-OSX/MetalXR) every ~12 months. **Do not block any earlier phase on this.**

## Risks & Mitigations

- **macOS native VR is impossible (OS-runtime gap; same for every engine).** → Do **not** let
  macOS-VR drive the spine. 2D/desktop-first on macOS+Linux now (P0–P4); VR deferred behind an
  interface (P5), targeting Linux/Quest; `trame`/WebXR as the streamed channel; re-evaluate
  OpenXR-OSX/MetalXR on a 12-month cadence.
- **Control inversion vs. the embedded model.** → PyVista never seizes the process
  (`off_screen`/`show=False`); Vispy (`process_events`) and Vedo (offscreen-from-numpy) are
  same-model fallbacks; Godot/O3DE excluded from the in-process layer (Godot only ever out-of-process
  for VR, consuming glTF).
- **Large-grid performance on integrated-GPU laptops (no GPU farm).** GeoVista has no automatic
  streaming LOD. → Explicit LOD/resolution policy as a DAVINCI feature: regrid to uniform vertical for
  cheap `ImageData` GPU volume rendering; subset with `extract_subset`/`threshold`; `decimate` coarse
  global meshes, swap to finer regional on zoom (DKRZ `zlevel` pattern); batch artifacts via headless
  `off_screen`. Keep Vispy as the escape hatch for extreme point/particle scale.
- **Maintenance longevity of the geoscience layer.** GeoVista is the lower-bus-factor dependency
  (stable at 0.5.3, Oct 2024; cylindrical/pseudo-cylindrical projections only); `pyvista-xarray` is
  small. → Pin both; **vendor thin wrappers** around the specific GeoVista calls used so a break is
  contained to one adapter. Core PyVista/VTK (Kitware) and `trame` are institutionally backed and
  actively releasing (PyVista 0.48.x, VTK 9.5). Keep Vedo as a same-family fallback; put Vispy
  2.0/Datoviz on a watch-list, do not depend on it mid-rewrite.
- **Build-it-yourself burden of geospatial+volumetric on a game engine.** → The single biggest reason
  to reject the game-engine spine; PyVista/GeoVista/`pyvista-xarray` already provide
  volume/isosurface/slice, a true globe, and an xarray accessor. Reserve Panda3D **only** as a scoped,
  optional Linux-VR escape hatch.
- **CESM vertical-coordinate / sigma→height regridding correctness (the bug rediscovered 4+ times).**
  → Single-source the fix: apply `surface_level_index` **before** any 3D field reaches
  `pyvista-xarray`; make level-ordering a first-class test in the 3D path (P2); own the sigma→height
  regridding in a DAVINCI preprocessing stage (GeoCAT/MetPy/xgcm) producing a clean
  `(z,lat,lon[,time])` DataArray, with explicit extrapolation/NaN handling.

## Decisions & Rationale (defaults for the open questions)

These are chosen defaults so the roadmap is unambiguous; each is flagged for confirmation below and
materially affects effort.

- **Artifact-first, interactive opt-in.** Default to headless render-to-file (PDF/PNG/movie/standalone
  HTML) emitted from the pipeline like today's 2D plots; a live interactive window
  (`pyvistaqt`/`trame`) is an opt-in P0+ capability, not required. *Rationale:* continues the
  render-to-file-from-a-stage contract; lowest risk; interactive embedding deferred until wanted.
- **Pressure-level z first, geometric-height as a follow-on.** First volumetric cut (P2) uses
  pressure-level z; geometric-height z is a flagged enhancement. *Rationale:* sigma→pressure avoids the
  NaN/extrapolation pitfalls of sigma→height and is materially cheaper. **Domain-scientist override
  likely** — see open questions.
- **Native VR = Linux-only + out-of-process; web/streamed = the macOS-accessible immersive channel.**
  macOS native VR explicitly out of scope. *Rationale:* the only viable path (OS gap).
- **Globe LOD = DKRZ coarse-global + on-demand-regional**, not Cesium-style streaming. *Rationale:*
  PyVista/GeoVista do not provide streaming tiling for free; the DKRZ pattern is proven and
  sufficient.
- **Dependency posture: pin + vendor-wrap GeoVista 0.5.3 + `pyvista-xarray`; keep the GeoVista globe
  path optional** so grid/volume/track ship even if the globe path slips. *Rationale:* contains
  pre-1.0 risk to one adapter while the institutionally-backed core does the heavy lifting.

## Open Questions (confirm before P0)

1. **VR priority/platform.** Is native HMD VR a hard long-horizon deliverable, or is web/streamed
   immersive (`trame`/WebXR, eventually CloudXR→Vision Pro) acceptable? If native VR is required, is
   Linux-only + out-of-process acceptable (the only viable path), given macOS is impossible at the
   OS-runtime level?
2. **Interactive vs. artifact-first.** Does DAVINCI need a live interactive 3D window
   (fly/slice/scrub) embedded in Qt/Jupyter, or primarily headless render-to-file artifacts from the
   pipeline (the default above)? This sets whether `pyvistaqt`/`trame` embedding is in-scope for P0.
3. **Vertical-coordinate strategy.** Geometric-**height** z (harder; needs GeoCAT/MetPy/xgcm; NaN/
   extrapolation risk) or **pressure-level** z (the default) for the first cut? Materially changes P2.
4. **Globe LOD ambition.** Is the DKRZ coarse-global + on-demand-regional swap (the default)
   sufficient, or is Cesium-style smooth streaming required (significant custom engineering)?
5. **Dependency tolerance.** Comfortable pinning/vendoring pre-1.0 GeoVista (0.5.3) and small
   `pyvista-xarray`, given core PyVista/VTK/`trame` are institutionally backed? Or keep GeoVista
   optional and ship grid/volume/track first (the default)?

## Evidence & Provenance

This spec is grounded in a multi-agent evaluation (29 agents, ~1.17M tokens, mid-2026 web-verified):
11 engine scorecards, 5 cross-cutting research threads (VR reality, in-process embedding, volumetric
path, geospatial path, prior art), and adversarial verification of the top-4 candidates' deal-breaker
claims. Key verified findings:

- **No production OpenXR runtime on macOS** (SteamVR dropped 2020; Monado Linux/Android; Vision Pro
  closed visionOS via CloudXR) — VR is an OS gap, not an engine-binding gap.
- **Embedding/control inversion:** PyVista/Vispy/Panda3D expose host-owned-loop modes; Godot inverts
  control and its Python binding is dormant for Godot 4. (confirmed)
- **Volumetric + geospatial:** PyVista/VTK natively span xarray→VTK grids + volume/isosurface/slice;
  GeoVista is purpose-built geoscience cartography with climate prior art (DKRZ). (confirmed)
- **Prior art:** zero atmospheric interactive-3D on any Python game engine; the durable lineage is
  VTK/OpenGL (Met.3D, PyVista/GeoVista, ParaView/trame). (confirmed)
- **Panda3D** is embeddable and Apple-Silicon-supported (1.10.16, universal2 wheels) but has no
  first-party globe/volume/isosurface and no xarray bridge. (confirmed) → Linux-VR escape hatch only.
