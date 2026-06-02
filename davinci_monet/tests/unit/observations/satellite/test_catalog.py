from davinci_monet.observations.satellite.catalog.schema import ProductEntry, VariableEntry


def test_variable_entry_minimal():
    v = VariableEntry(
        display_name="aod_550nm", sds_name="Aerosol_Optical_Depth_Land_Ocean_Mean_Mean"
    )
    assert v.display_name == "aod_550nm"
    assert v.sds_name.startswith("Aerosol_Optical_Depth")
    assert v.units == "1"  # default


def test_product_entry_resolves_variable_by_display_and_sds():
    p = ProductEntry(
        product_id="MOD08_M3",
        instrument="MODIS",
        platform="Terra",
        daac="LAADS",
        collection="061",
        level="L3",
        geometry="GRID",
        file_format="HDF4",
        time_parse="A%Y%j",
        dim_aliases={"XDim": "lon", "YDim": "lat"},
        variables=[
            VariableEntry(
                display_name="aod_550nm", sds_name="Aerosol_Optical_Depth_Land_Ocean_Mean_Mean"
            )
        ],
    )
    assert p.variable_by_display("aod_550nm").sds_name.startswith("Aerosol_Optical_Depth")
    assert (
        p.variable_by_sds("Aerosol_Optical_Depth_Land_Ocean_Mean_Mean").display_name == "aod_550nm"
    )
    assert p.variable_by_display("nope") is None
