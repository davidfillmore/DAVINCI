# 3D Geometry Export for Blender (OBJ Format)

## Overview

Add OBJ/MTL file export to DAVINCI for rendering flight tracks and model fields in Blender or other 3D software.

**Exports:**
- Flight tracks as 3D tubes colored by variable (O3, bias, etc.)
- Model fields as isosurfaces at specified concentration levels
- Horizontal slices at altitude levels
- Vertical curtains along flight paths
- Terrain mesh from DEM files (ensures coordinate alignment with other exports)

## Module Structure

```
davinci_monet/io/exporters/
├── __init__.py              # Exporter registry, public API
├── base.py                  # BaseExporter ABC
├── obj.py                   # OBJExporter class (main implementation)
├── geometry/
│   ├── __init__.py
│   ├── coordinates.py       # Lat/lon/alt → XYZ transformation
│   ├── track.py             # Tube/ribbon mesh generation
│   ├── isosurface.py        # Marching cubes wrapper
│   ├── slices.py            # Horizontal/vertical slice meshes
│   └── terrain.py           # DEM → terrain mesh generation
└── materials.py             # Colormap → MTL file generation
```

## Key Classes

### OBJExporter

```python
@exporter_registry.register("obj")
class OBJExporter(BaseExporter):
    def export_track(data, output_path, variable, tube_radius=0.5, cmap="viridis", ...) -> list[Path]
    def export_isosurface(model_data, output_path, variable, level, ...) -> list[Path]
    def export_horizontal_slice(model_data, output_path, variable, altitude, ...) -> list[Path]
    def export_vertical_curtain(model_data, output_path, variable, track=None, ...) -> list[Path]
    def export_terrain(dem_path, output_path, extent=None, resolution=100, ...) -> list[Path]
    def export_scene(output_path, tracks=[], isosurfaces=[], slices=[], terrain=None, ...) -> list[Path]
```

### CoordinateConfig

```python
@dataclass
class CoordinateConfig:
    projection: Literal["planar", "spherical"] = "planar"
    vertical_exaggeration: float = 100.0  # Z scaling
    altitude_scale: float = 0.001         # m → km
    center_lon: float | None = None
    center_lat: float | None = None
```

## Coordinate Transformation

**Planar (default):** Good for regional domains
```python
x = (lon - center_lon) * cos(center_lat) * 111.32  # km
y = (lat - center_lat) * 111.32                     # km
z = alt * 0.001 * vertical_exaggeration             # exaggerated km
```

**Spherical:** For global visualizations
```python
R = earth_radius + alt * scale * exaggeration
x = R * cos(lat) * cos(lon)
y = R * cos(lat) * sin(lon)
z = R * sin(lat)
```

## Color/Material Approach

1. Generate N discrete materials from matplotlib colormap
2. Map variable values → material indices
3. Write MTL file with Ka/Kd/Ks properties
4. OBJ file uses `usemtl color_XXX` per face group

## Terrain Generation

Generate terrain mesh from DEM (Digital Elevation Model) files using the same coordinate system as other exports for perfect alignment.

**DEM Sources:**
- SRTM (30m global, 90m outside US)
- ASTER GDEM (30m global)
- USGS 3DEP (10m or better, US only)
- Copernicus DEM (30m global)

**Implementation:**
```python
def export_terrain(
    dem_path: Path,
    output_path: Path,
    extent: tuple[float, float, float, float] | None = None,  # (lon_min, lon_max, lat_min, lat_max)
    resolution: int = 100,          # Grid points per axis (100x100 = 10k vertices)
    texture_path: Path | None = None,  # Optional satellite/topo texture
    color: str = "#8B7355",         # Default terrain brown
    coord_config: CoordinateConfig = ...,
) -> list[Path]:
    """Export terrain mesh from DEM file.

    Uses rasterio to read DEM, resamples to specified resolution,
    applies same coordinate transformation as other exports.
    """
```

**Features:**
- Auto-detect extent from DEM or use specified bounds
- Resample to manageable mesh size (full DEMs can be huge)
- Apply same vertical exaggeration as flight tracks
- Optional: color by elevation, apply satellite imagery texture
- Mesh simplification for large areas

## YAML Configuration

```yaml
export_3d:
  format: obj
  output_dir: ${ANALYSIS_DIR}/3d_export

  coordinates:
    projection: planar
    vertical_exaggeration: 100

  tracks:
    - pair: cesm_dc8_o3
      variable: bias
      tube_radius: 0.3
      cmap: RdBu_r
      vmin: -30
      vmax: 30

  isosurfaces:
    - model: cesm
      variable: O3
      level: 60
      alpha: 0.5

  slices:
    - model: cesm
      variable: O3
      altitude: 5000

  terrain:
    dem_path: ${DATA_DIR}/srtm_korea.tif
    resolution: 200              # 200x200 grid
    color: "#8B7355"             # Terrain brown
    # extent: [124.0, 132.0, 33.0, 43.0]  # Optional, auto-detect from DEM
```

## Dependencies

Add to environment.yml:
```yaml
- scikit-image>=0.19  # marching_cubes for isosurfaces
- rasterio>=1.3       # DEM file reading for terrain
```

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Create `exporters/` module with `__init__.py`, `base.py`
- [ ] Implement `CoordinateConfig` and `transform_coordinates()` in `geometry/coordinates.py`
- [ ] Implement OBJ/MTL file writers in `obj.py`
- [ ] Implement `MaterialConfig` and colormap→material mapping in `materials.py`
- [ ] Create `exporter_registry` in `core/registry.py`

### Phase 2: Track Geometry
- [ ] Implement `generate_track_tube()` in `geometry/track.py`
- [ ] Implement `generate_track_ribbon()` alternative
- [ ] Implement `OBJExporter.export_track()`
- [ ] Add unit tests with synthetic track data

### Phase 3: Model Field Geometry
- [ ] Implement `extract_isosurface()` wrapper in `geometry/isosurface.py`
- [ ] Implement `generate_horizontal_slice()` in `geometry/slices.py`
- [ ] Implement `generate_vertical_curtain()` in `geometry/slices.py`
- [ ] Implement corresponding `OBJExporter` methods
- [ ] Add unit tests with synthetic sphere/cube fields

### Phase 4: Terrain Generation
- [ ] Implement `generate_terrain_mesh()` in `geometry/terrain.py`
- [ ] Add DEM reading with rasterio, resampling to target resolution
- [ ] Implement `OBJExporter.export_terrain()`
- [ ] Add unit tests with synthetic elevation data
- [ ] Test with real SRTM tile

### Phase 5: Integration
- [ ] Add `Export3DConfig` to `config/schema.py`
- [ ] Add `export-3d` CLI command
- [ ] Implement `export_scene()` for combined exports
- [ ] Add integration tests

### Phase 6: Documentation
- [ ] Docstrings and type hints
- [ ] Example YAML config in `examples/`
- [ ] Brief Blender import instructions in docstring

## Critical Files to Modify/Create

**Create:**
- `davinci_monet/io/exporters/__init__.py`
- `davinci_monet/io/exporters/base.py`
- `davinci_monet/io/exporters/obj.py`
- `davinci_monet/io/exporters/materials.py`
- `davinci_monet/io/exporters/geometry/__init__.py`
- `davinci_monet/io/exporters/geometry/coordinates.py`
- `davinci_monet/io/exporters/geometry/track.py`
- `davinci_monet/io/exporters/geometry/isosurface.py`
- `davinci_monet/io/exporters/geometry/slices.py`
- `davinci_monet/io/exporters/geometry/terrain.py`
- `davinci_monet/tests/unit/io/exporters/test_*.py`

**Modify:**
- `davinci_monet/core/registry.py` - Add `exporter_registry`
- `davinci_monet/config/schema.py` - Add `Export3DConfig`
- `davinci_monet/cli/app.py` - Add `export-3d` command
- `environment.yml` - Add scikit-image, rasterio

## Reference Files (patterns to follow)

- `davinci_monet/io/writers.py` - Writer function patterns
- `davinci_monet/core/registry.py` - Registry pattern
- `davinci_monet/plots/renderers/track_map_3d.py` - 3D coordinate handling
- `davinci_monet/config/schema.py` - Pydantic config patterns

## Verification

1. **Unit tests:** `pytest davinci_monet/tests/unit/io/exporters/ -v`
2. **Export track:** Generate OBJ from synthetic track data, verify file structure
3. **Export isosurface:** Generate OBJ from synthetic sphere field, verify mesh
4. **Export terrain:** Generate terrain mesh from SRTM tile, verify elevation mapping
5. **Blender import:** Open all OBJs in Blender, verify:
   - Geometry and colors render correctly
   - Flight tracks align with terrain surface
   - Coordinate systems match (no manual adjustment needed)
6. **Full pipeline:** Run with YAML config containing `export_3d` section
