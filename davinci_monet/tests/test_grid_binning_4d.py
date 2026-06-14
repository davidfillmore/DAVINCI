import numpy as np

from davinci_monet.pairing.grid_binning import bin_points_to_grid_4d, normalize_grid


def test_bin_points_to_grid_4d_accumulates_by_cell():
    # 2 time x 2 lon x 2 lat x 2 alt grid, edges 0..2 each
    edges = np.array([0.0, 1.0, 2.0])
    nt = nx = ny = nz = 2
    count = np.zeros((nt, nx, ny, nz), dtype=np.int32)
    acc = np.zeros((nt, nx, ny, nz), dtype=np.float64)
    # two points in cell (0,0,0,0), one in (1,1,1,1)
    t = np.array([0.5, 0.5, 1.5])
    x = np.array([0.5, 0.5, 1.5])
    y = np.array([0.5, 0.5, 1.5])
    z = np.array([0.5, 0.5, 1.5])
    d = np.array([2.0, 4.0, 9.0])
    bin_points_to_grid_4d(edges, edges, edges, edges, t, x, y, z, d, count, acc)
    normalize_grid(count, acc)
    assert count[0, 0, 0, 0] == 2 and acc[0, 0, 0, 0] == 3.0  # mean(2,4)
    assert count[1, 1, 1, 1] == 1 and acc[1, 1, 1, 1] == 9.0
    assert np.isnan(acc[0, 1, 0, 1])  # empty cell
