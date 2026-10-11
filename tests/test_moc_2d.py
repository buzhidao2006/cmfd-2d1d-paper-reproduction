"""Independent checks of the minimal periodic diagonal-track MOC prototype."""

import numpy as np
import pytest

from paper_reproduction.moc_2d import diagonal_track_sweep, solve_moc_2d


@pytest.mark.parametrize("direction", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
def test_manufactured_periodic_track_flux(direction):
    shape, width = (2, 3), 0.4
    positions = np.indices(shape)
    target_incoming = 1.0 + 0.03 * positions[0] + 0.02 * positions[1]
    total = 1.0 + 0.1 * positions[0] + 0.05 * positions[1]
    length = np.sqrt(2) * width
    attenuation = np.exp(-total * length)
    next_incoming = np.roll(
        target_incoming, shift=(-direction[0], -direction[1]), axis=(0, 1)
    )
    source = total * (next_incoming - attenuation * target_incoming) / (
        1 - attenuation
    )
    incoming, outgoing, average = diagonal_track_sweep(
        source, total, width, direction
    )
    np.testing.assert_allclose(incoming, target_incoming, atol=1e-13)
    np.testing.assert_allclose(outgoing, next_incoming, atol=1e-13)
    np.testing.assert_allclose(
        outgoing - incoming + total * length * average,
        source * length, atol=1e-13,
    )
    np.testing.assert_allclose((outgoing - incoming).sum(), 0, atol=1e-13)


def test_constant_source_and_scattering_limit():
    result = solve_moc_2d(
        np.full((3, 2), 0.4), sigma_t=1.0, sigma_s=0.6,
        cell_width=0.5, inner_iterations=80,
    )
    np.testing.assert_allclose(result.scalar_flux, 1.0, atol=1e-12)
    assert result.scalar_history.shape == (81, 3, 2)


def test_angle_dependent_axial_leakage_obeys_segment_balance():
    shape, width = (2, 3), 0.5
    source = np.array([[0.8, 0.9, 1.0], [1.1, 0.7, 0.6]])
    leakage = np.stack([np.full(shape, 0.01 * (p + 1)) for p in range(4)])
    result = solve_moc_2d(
        source, sigma_t=1.2, sigma_s=0.3, cell_width=width,
        axial_leakage=leakage, inner_iterations=3,
    )
    previous = result.scalar_history[-2]
    length = np.sqrt(2) * width
    for p in range(4):
        q = source + 0.3 * previous - leakage[p]
        np.testing.assert_allclose(
            result.outgoing_flux[p] - result.incoming_flux[p]
            + 1.2 * length * result.angular_flux[p],
            q * length, atol=1e-13,
        )
        np.testing.assert_allclose(
            (result.outgoing_flux[p] - result.incoming_flux[p]).sum(),
            0, atol=1e-13,
        )
    np.testing.assert_allclose(
        result.scalar_flux, result.angular_flux.mean(axis=0), atol=1e-13
    )


def test_one_cell_periodic_track_reduces_to_local_balance():
    incoming, outgoing, average = diagonal_track_sweep(
        np.array([[0.8]]), 2.0, 0.5, (1, -1)
    )
    np.testing.assert_allclose(incoming, [[0.4]])
    np.testing.assert_allclose(outgoing, [[0.4]])
    np.testing.assert_allclose(average, [[0.4]])


@pytest.mark.parametrize("kwargs", [
    {"sigma_s": 1.0},
    {"cell_width": 0.0},
    {"axial_leakage": np.zeros((3, 2, 2))},
    {"weights": np.array([0.25, 0.25, 0.25, 0.1])},
    {"inner_iterations": 0},
])
def test_invalid_inputs_are_rejected(kwargs):
    parameters = {"sigma_t": 1.0, "sigma_s": 0.2, "cell_width": 0.5}
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        solve_moc_2d(np.ones((2, 2)), **parameters)
