"""Synthetic data generators for testing.

This module provides utilities to generate synthetic gridded data and
geometry data for testing the pairing, plotting, and statistics
components without requiring external datasets.

Usage:
    from davinci_monet.tests.synthetic import (
        create_dataset_dataset,
        create_point_geometries,
        create_track_geometries,
        PerfectMatchScenario,
    )

    # Generate synthetic gridded data
    dataset = create_dataset_dataset(
        variables=["O3", "PM25"],
        time_range=("2024-01-01", "2024-01-03"),
    )

    # Generate matching geometry data
    geometry = create_point_geometries(
        n_sites=10,
        variables=["O3", "PM25"],
        time_range=("2024-01-01", "2024-01-03"),
    )
"""

from davinci_monet.tests.synthetic.datasets import (
    DatasetConfig,
    create_dataset_dataset,
    create_point_geometries,
    create_profile_geometries,
    create_swath_geometries,
    create_track_geometries,
    create_variable_field,
)
from davinci_monet.tests.synthetic.generators import (
    Domain,
    TimeConfig,
    create_coordinate_grid,
    create_time_axis,
    random_locations_in_domain,
)
from davinci_monet.tests.synthetic.scenarios import (
    BiasScenario,
    MismatchScenario,
    PerfectMatchScenario,
    Scenario,
)

__all__ = [
    # Generators
    "Domain",
    "TimeConfig",
    "create_coordinate_grid",
    "create_time_axis",
    "random_locations_in_domain",
    # Datasets
    "DatasetConfig",
    "create_dataset_dataset",
    "create_variable_field",
    # Datasets
    "create_point_geometries",
    "create_track_geometries",
    "create_profile_geometries",
    "create_swath_geometries",
    # Scenarios
    "Scenario",
    "PerfectMatchScenario",
    "BiasScenario",
    "MismatchScenario",
]
