# Plotting Alternatives Research

**Status**: Deferred for future consideration
**Date**: 2025-01-23

## Current Stack

DAVINCI currently uses:
- **Matplotlib** - Core plotting
- **Cartopy** - Map projections and geospatial
- **Seaborn** - Statistical plot styling

## Alternatives Evaluated

### 1. HoloViz Ecosystem (hvPlot + GeoViews) ⭐ Recommended

- **hvPlot**: High-level API that works directly with xarray via `.hvplot()` accessor
- **GeoViews**: Geographic extension built on Cartopy
- **Pros**: Native xarray support, interactive by default, handles large data via Datashader
- **Cons**: Learning curve, dependency on HoloViews/Bokeh stack
- Docs: https://hvplot.holoviz.org/
- Geographic guide: https://hvplot.holoviz.org/user_guide/Geographic_Data.html

### 2. Plotly

- Interactive web-based plots, good 3D support
- `plotly.express` for quick plots, `go.Scattergeo`/`go.Choroplethmapbox` for maps
- **Pros**: Beautiful interactivity, easy sharing, Dash integration
- **Cons**: Weaker projection support vs Cartopy, large file sizes

### 3. Bokeh

- Interactive web visualizations, good for dashboards
- **Pros**: Flexible, streaming data support
- **Cons**: Lower-level API, geo support requires more effort

### 4. Folium / Leaflet

- Interactive slippy maps (like Google Maps)
- **Pros**: Easy web embedding, familiar map interface
- **Cons**: Not for scientific projections, publication-unfriendly

### 5. PyVista

- 3D scientific visualization (VTK-based)
- Great for volumetric atmospheric data, cross-sections
- **Cons**: Overkill for 2D plots

## Comparison Matrix

| Feature | Matplotlib/Cartopy | hvPlot/GeoViews | Plotly |
|---------|-------------------|-----------------|--------|
| xarray native | Moderate | Excellent | Moderate |
| Projections | Excellent | Good (uses Cartopy) | Limited |
| Interactivity | None | Built-in | Built-in |
| Publication quality | Excellent | Good | Moderate |
| Large datasets | Slow | Datashader | Slow |
| Learning curve | Known | Medium | Low |

## Recommended Approach

A **hybrid approach** for DAVINCI:

1. **Keep Matplotlib/Cartopy** for publication-quality static plots (spatial bias, maps)
2. **Add hvPlot** for exploratory/interactive analysis
3. **Optional Plotly** for web dashboards or reports

## Implementation Notes

To add hvPlot support:

```python
# In environment.yml
- hvplot
- geoviews
- datashader  # for large datasets

# Usage with xarray
import hvplot.xarray

# Interactive plot from xarray Dataset
ds.hvplot.quadmesh('longitude', 'latitude', 'variable', geo=True, cmap='viridis')
```

## References

- [hvPlot Documentation](https://hvplot.holoviz.org/)
- [xarray hvplot Tutorial](https://tutorial.xarray.dev/intermediate/hvplot.html)
- [Top 10 Python Data Visualization Libraries 2025](https://reflex.dev/blog/2025-01-27-top-10-data-visualization-libraries/)
