"""Current reconstruction and fixed-source consistency checks for 1D CMFD."""

import numpy as np
import pytest

from paper_reproduction.axial_sn import solve_axial_sn
from paper_reproduction.cmfd_1d import (
    CurrentCorrection,
    cmfd_matrix,
    current_correction,
    interface_current,
    interface_diffusion,
    solve_fixed_source,
)


def test_uniform_interface_diffusion_and_theta():
    standard = interface_diffusion(np.ones(4), 0.5)
    stabilized = interface_diffusion(np.ones(4), 0.5, theta=0.25)
    np.testing.assert_allclose(standard, 2 / 3)
    np.testing.assert_allclose(stabilized, 11 / 12)


@pytest.mark.parametrize("theta", [0.0, 0.25])
def test_manufactured_flux_and_current_are_reconstructed(theta):
    flux = np.array([1.0, 1.3, 0.9, 1.1, 1.2])
    widths = np.array([0.4, 0.5, 0.6, 0.5, 0.4])
    sigma_t = np.array([1.0, 1.2, 1.1, 0.9, 1.3])
    diffusion = interface_diffusion(sigma_t, widths, theta)
    known = CurrentCorrection(diffusion, np.array([0.01, -0.02, 0.03, 0.0, -0.01]))
    current = interface_current(flux, known)
    recovered = current_correction(flux, current, sigma_t, widths, theta)
    np.testing.assert_allclose(recovered.diffusion, diffusion)
    np.testing.assert_allclose(recovered.correction, known.correction, atol=1e-15)
    absorption = np.full(5, 0.4)
    source = (current - np.roll(current, 1)) / widths + absorption * flux
    np.testing.assert_allclose(solve_fixed_source(source, absorption, widths, recovered),
                               flux, atol=1e-13)
    np.testing.assert_allclose(
        np.dot(widths, source - absorption * flux), 0, atol=1e-14
    )


@pytest.mark.parametrize("theta", [0.0, 0.25])
def test_reconstructs_uncoupled_axial_sn_transport_solution(theta):
    source = np.array([0.8, 1.0, 0.7, 1.2, 0.9])
    sigma_t, width = 1.1, 0.5
    result = solve_axial_sn(
        source, sigma_t=sigma_t, sigma_s=0.0,
        cell_width=width, inner_iterations=1,
    )
    angles = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    weights = np.array([0.5, 0.5])
    right_faces = np.roll(result.face_flux, -1, axis=1)
    transport_current = np.tensordot(weights * angles, right_faces, axes=(0, 0))
    coefficients = current_correction(
        result.scalar_flux, transport_current, np.full(source.size, sigma_t),
        width, theta,
    )
    np.testing.assert_allclose(
        interface_current(result.scalar_flux, coefficients), transport_current,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        cmfd_matrix(sigma_t, width, coefficients) @ result.scalar_flux,
        source, atol=1e-13,
    )
    np.testing.assert_allclose(
        solve_fixed_source(source, sigma_t, width, coefficients),
        result.scalar_flux, atol=1e-13,
    )


@pytest.mark.parametrize("kwargs", [
    {"transport_flux": np.array([1.0, 0.0])},
    {"transport_current": np.array([0.0])},
    {"theta": -0.1},
])
def test_invalid_current_correction_inputs(kwargs):
    parameters = {
        "transport_flux": np.array([1.0, 1.2]),
        "transport_current": np.array([0.0, 0.1]),
        "sigma_t": np.ones(2),
        "cell_width": 0.5,
    }
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        current_correction(**parameters)
