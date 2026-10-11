"""Independent checks of the axial diamond-difference SN stage."""

import numpy as np
import pytest

from paper_reproduction.axial_sn import diamond_difference_sweep, solve_axial_sn


@pytest.mark.parametrize("mu", [-1 / np.sqrt(3), 1 / np.sqrt(3)])
def test_periodic_manufactured_angular_flux(mu):
    n, h, sigma_t = 7, 0.4, 1.2
    target_faces = np.exp(2j * np.pi * np.arange(n) / n)
    target_cells = (target_faces + np.roll(target_faces, -1)) / 2
    source = sigma_t * target_cells + mu * (
        np.roll(target_faces, -1) - target_faces
    ) / h
    faces, centers = diamond_difference_sweep(source, mu, sigma_t, h)
    np.testing.assert_allclose(faces, target_faces, atol=1e-13)
    np.testing.assert_allclose(centers, target_cells, atol=1e-13)


def test_constant_source_converges_to_absorption_balance():
    result = solve_axial_sn(
        np.full(5, 0.4), sigma_t=1.0, sigma_s=0.6,
        cell_width=0.5, inner_iterations=80,
    )
    np.testing.assert_allclose(result.scalar_flux, 1.0, atol=1e-12)
    assert result.scalar_history.shape == (81, 5)


def test_periodic_global_balance_with_frozen_transverse_leakage():
    source = np.array([0.4, 0.8, 0.2, 0.6])
    leakage = np.array([[0.02, 0.04, 0.01, 0.03],
                        [0.03, 0.01, 0.04, 0.02]])
    weights = np.array([0.25, 0.75])
    result = solve_axial_sn(
        source, sigma_t=1.1, sigma_s=0.3, cell_width=0.7,
        angles=np.array([-0.5, 0.5]), weights=weights,
        transverse_leakage=leakage, inner_iterations=3,
        initial_scalar=np.full(4, 0.1),
    )
    previous = result.scalar_history[-2]
    weighted_leakage = np.tensordot(weights, leakage, axes=(0, 0))
    np.testing.assert_allclose(
        1.1 * result.scalar_flux.sum(),
        (source + 0.3 * previous - weighted_leakage).sum(),
    )


def test_one_cell_periodic_solution_is_local_balance():
    faces, centers = diamond_difference_sweep(np.array([2.0]), -0.4, 4.0, 0.1)
    np.testing.assert_allclose(faces, [0.5])
    np.testing.assert_allclose(centers, [0.5])


@pytest.mark.parametrize("kwargs", [
    {"sigma_s": 1.0},
    {"inner_iterations": 0},
    {"weights": np.array([0.2, 0.2])},
    {"transverse_leakage": np.zeros((3, 4))},
])
def test_invalid_inputs_are_rejected(kwargs):
    parameters = {"sigma_t": 1.0, "sigma_s": 0.3, "cell_width": 0.5}
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        solve_axial_sn(np.ones(4), **parameters)
