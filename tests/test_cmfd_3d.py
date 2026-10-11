"""Three-dimensional current preservation and axial-transport reduction."""

import numpy as np
import pytest

from paper_reproduction.axial_sn import solve_axial_sn
from paper_reproduction.cmfd_1d import interface_diffusion
from paper_reproduction.cmfd_3d import (
    CurrentCorrection3D,
    balance_operator,
    cmfd_matrix,
    current_correction,
    interface_currents,
    interface_diffusions,
    solve_fixed_source,
)


@pytest.mark.parametrize("theta", [0.0, 0.25])
def test_manufactured_3d_currents_are_reconstructed(theta):
    shape = (2, 3, 2)
    widths = (0.4, 0.5, 0.6)
    positions = np.indices(shape)
    flux = 1.0 + 0.08 * positions[0] + 0.05 * positions[1] + 0.03 * positions[2]
    sigma_t = 1.0 + 0.1 * positions[0] + 0.03 * positions[2]
    diffusion = interface_diffusions(sigma_t, widths, theta)
    known_hats = tuple(np.full(shape, (axis - 1) * 0.01) for axis in range(3))
    known = CurrentCorrection3D(diffusion, known_hats)
    transport_currents = interface_currents(flux, known)
    recovered = current_correction(flux, transport_currents, sigma_t, widths, theta)
    for actual, expected in zip(recovered.correction, known_hats):
        np.testing.assert_allclose(actual, expected, atol=1e-15)
    for actual, expected in zip(interface_currents(flux, recovered), transport_currents):
        np.testing.assert_allclose(actual, expected, atol=1e-15)
    absorption = 0.3 + 0.02 * positions[1]
    source = balance_operator(flux, absorption, widths, recovered)
    np.testing.assert_allclose(
        solve_fixed_source(source, absorption, widths, recovered), flux, atol=1e-13
    )
    np.testing.assert_allclose(
        source.sum(), (absorption * flux).sum(), atol=1e-13
    )


def test_axial_sn_reduces_to_3d_cmfd_with_flat_radial_flux():
    source_z = np.array([0.8, 1.0, 0.7, 1.2, 0.9])
    sigma_t, width = 1.1, 0.5
    result = solve_axial_sn(
        source_z, sigma_t=sigma_t, sigma_s=0,
        cell_width=width, inner_iterations=1,
    )
    shape = (2, 2, source_z.size)
    flux = np.broadcast_to(result.scalar_flux, shape).copy()
    angles = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    right_faces = np.roll(result.face_flux, -1, axis=1)
    axial_current = np.tensordot(0.5 * angles, right_faces, axes=(0, 0))
    currents = (
        np.zeros(shape), np.zeros(shape),
        np.broadcast_to(axial_current, shape).copy(),
    )
    widths = (0.7, 0.8, width)
    coefficients = current_correction(
        flux, currents, np.full(shape, sigma_t), widths, theta=0.25,
    )
    np.testing.assert_allclose(coefficients.correction[0], 0, atol=1e-14)
    np.testing.assert_allclose(coefficients.correction[1], 0, atol=1e-14)
    source = np.broadcast_to(source_z, shape).copy()
    np.testing.assert_allclose(
        balance_operator(flux, sigma_t, widths, coefficients), source, atol=1e-13
    )
    np.testing.assert_allclose(
        solve_fixed_source(source, sigma_t, widths, coefficients), flux, atol=1e-13
    )
    np.testing.assert_allclose(
        coefficients.diffusion[2][0, 0],
        interface_diffusion(np.full(source_z.size, sigma_t), width, theta=0.25),
    )


def test_dense_matrix_matches_operator_and_constant_balance():
    shape = (2, 2, 2)
    uniform = np.full(shape, 1.25)
    coefficients = current_correction(
        uniform, (np.zeros(shape),) * 3, np.ones(shape),
        (0.5, 0.5, 0.5),
    )
    matrix = cmfd_matrix(0.4, (0.5, 0.5, 0.5), coefficients)
    np.testing.assert_allclose(
        (matrix @ uniform.ravel()).reshape(shape),
        balance_operator(uniform, 0.4, (0.5, 0.5, 0.5), coefficients),
    )
    np.testing.assert_allclose(
        solve_fixed_source(np.full(shape, 0.5), 0.4,
                           (0.5, 0.5, 0.5), coefficients), uniform,
    )


@pytest.mark.parametrize("bad_flux,bad_widths", [
    (np.zeros((2, 2, 2)), (0.5, 0.5, 0.5)),
    (np.ones((2, 2, 2)), (0.5, 0.0, 0.5)),
    (np.ones((2, 2)), (0.5, 0.5, 0.5)),
])
def test_invalid_inputs_are_rejected(bad_flux, bad_widths):
    shape = (2, 2, 2)
    with pytest.raises(ValueError):
        current_correction(
            bad_flux, (np.zeros(shape),) * 3,
            np.ones(shape), bad_widths,
        )
