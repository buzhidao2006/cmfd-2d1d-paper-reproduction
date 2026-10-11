"""Checks for the deliberately simplified Zhou et al. Fourier-mode experiment."""

import numpy as np
import pytest

from paper_reproduction.fourier_mode import (
    Mode,
    error_history,
    main,
    outer_step,
    scan,
    spectral_radius,
    streaming_symbols,
    transition_matrix,
)


def test_zero_spatial_mode_has_no_streaming_leakage():
    radial, axial = streaming_symbols(Mode(mode_numbers=(0, 0, 0)))
    np.testing.assert_allclose(radial, 0)
    np.testing.assert_allclose(axial, 0)


@pytest.mark.parametrize("order", ("1d2d", "2d1d"))
def test_transition_matrix_matches_direct_step(order):
    mode = Mode()
    state = np.arange(16) + 1j * np.arange(16, 32)
    matrix = transition_matrix(mode, order, 3)
    np.testing.assert_allclose(matrix @ state, outer_step(state, mode, order, 3))


def test_spectral_radius_is_eigenmode_contraction_factor():
    mode = Mode()
    matrix = transition_matrix(mode, "2d1d", 3)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    index = int(np.argmax(np.abs(eigenvalues)))
    vector = eigenvectors[:, index]
    actual = np.linalg.norm(outer_step(vector, mode, "2d1d", 3)) / np.linalg.norm(vector)
    assert actual == pytest.approx(spectral_radius(mode, "2d1d", 3))


def test_order_and_inner_count_are_observable_without_claiming_paper_values():
    mode = Mode(axial_cell_width=0.1)
    assert spectral_radius(mode, "1d2d", 3) != pytest.approx(
        spectral_radius(mode, "2d1d", 3)
    )
    assert spectral_radius(mode, "1d2d", 1) != pytest.approx(
        spectral_radius(mode, "1d2d", 3)
    )


def test_scan_and_history_are_finite_and_reproducible():
    mode = Mode()
    rows = scan(mode, [0.1, 3.0], [1, 3], steps=8)
    assert len(rows) == 8
    assert all(np.isfinite(row["spectral_radius"]) for row in rows)
    first = error_history(mode, "2d1d", 3, steps=8)
    np.testing.assert_array_equal(first, error_history(mode, "2d1d", 3, steps=8))


def test_cli_writes_csv_and_plot(tmp_path):
    csv_path = tmp_path / "scan.csv"
    plot_path = tmp_path / "scan.png"
    assert main(["--axial-widths", "3", "--inner-iterations", "1", "3",
                 "--steps", "5", "--output", str(csv_path), "--plot", str(plot_path)]) == 0
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 5
    assert plot_path.read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("kwargs", (
    {"scattering_ratio": 1.0},
    {"sigma_t": 0.0},
    {"axial_cell_width": 2.7},
))
def test_invalid_mode_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Mode(**kwargs)

