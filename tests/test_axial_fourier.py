"""Fourier factors must agree with the independently assembled SN solver."""

import numpy as np
import pytest

from paper_reproduction.axial_fourier import (
    diamond_difference_symbol,
    main,
    measured_amplification,
    mode_scan,
    scattering_amplification,
)


@pytest.mark.parametrize("cells", [5, 8])
def test_every_mode_matches_real_space_iteration(cells):
    for m in range(cells):
        expected = scattering_amplification(
            cells, m, sigma_t=1.0, sigma_s=0.8, cell_width=0.7,
        )
        actual, residual = measured_amplification(
            cells, m, sigma_t=1.0, sigma_s=0.8, cell_width=0.7,
        )
        assert actual == pytest.approx(expected, abs=1e-12)
        assert residual < 1e-12


def test_constant_mode_is_scattering_ratio_and_nyquist_is_zero():
    zero = scattering_amplification(
        8, 0, sigma_t=1.2, sigma_s=0.9, cell_width=0.4,
    )
    nyquist = scattering_amplification(
        8, 4, sigma_t=1.2, sigma_s=0.9, cell_width=0.4,
    )
    assert zero == pytest.approx(0.75)
    assert abs(nyquist) < 1e-12


def test_asymmetric_quadrature_can_have_complex_factor():
    gain = scattering_amplification(
        7, 1, sigma_t=1.0, sigma_s=0.7, cell_width=0.5,
        angles=np.array([-0.3, 0.8]), weights=np.array([0.4, 0.6]),
    )
    actual, residual = measured_amplification(
        7, 1, sigma_t=1.0, sigma_s=0.7, cell_width=0.5,
        angles=np.array([-0.3, 0.8]), weights=np.array([0.4, 0.6]),
    )
    assert abs(gain.imag) > 1e-4
    assert actual == pytest.approx(gain, abs=1e-12)
    assert residual < 1e-12


def test_scan_and_cli_export(tmp_path):
    rows = mode_scan(6)
    assert len(rows) == 6
    assert max(row["absolute_difference"] for row in rows) < 1e-12
    output = tmp_path / "modes.csv"
    assert main(["--cells", "6", "--output", str(output)]) == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 7


@pytest.mark.parametrize("kwargs", [
    {"mu": 0.0},
    {"sigma_t": 0.0},
    {"cell_width": -0.5},
])
def test_invalid_directional_symbol_inputs(kwargs):
    parameters = {"mu": 0.5, "sigma_t": 1.0, "cell_width": 0.5}
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        diamond_difference_symbol(8, 1, **parameters)
