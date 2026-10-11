"""Small periodic 3D CMFD current-consistency reference implementation.

This extends the fixed-source interface-current check to three dimensions.
It does not implement the paper's MOC coupling or eigenvalue iteration.
The dense matrix is intentionally limited to small verification meshes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CurrentCorrection3D:
    """Right-face coefficients in x, y, z order."""

    diffusion: tuple[np.ndarray, np.ndarray, np.ndarray]
    correction: tuple[np.ndarray, np.ndarray, np.ndarray]


def _shape(values: np.ndarray, name: str) -> tuple[int, int, int]:
    array = np.asarray(values)
    if array.ndim != 3 or any(size < 1 for size in array.shape):
        raise ValueError(f"{name} must be a nonempty three-dimensional array")
    return array.shape


def _grid(values: np.ndarray | float, shape: tuple[int, int, int], name: str) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    if array.ndim == 0:
        array = np.full(shape, float(array))
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and match the three-dimensional grid")
    return array.astype(float, copy=False)


def _widths(cell_widths: tuple[float, float, float]) -> tuple[float, float, float]:
    array = np.asarray(cell_widths, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("cell_widths must contain three finite positive widths")
    return tuple(float(value) for value in array)


def interface_diffusions(
    sigma_t: np.ndarray,
    cell_widths: tuple[float, float, float],
    theta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute periodic x/y/z right-face coefficients from Eq. (7)."""
    shape = _shape(sigma_t, "sigma_t")
    total = _grid(sigma_t, shape, "sigma_t")
    widths = _widths(cell_widths)
    if np.any(total <= 0):
        raise ValueError("sigma_t must be positive")
    if not np.isfinite(theta) or theta < 0:
        raise ValueError("theta must be finite and nonnegative")
    coefficients = []
    for axis, width in enumerate(widths):
        cell_diffusion = 1 / (3 * total) + theta * width
        neighbor = np.roll(cell_diffusion, -1, axis=axis)
        coefficients.append(
            2 * cell_diffusion * neighbor / (width * (cell_diffusion + neighbor))
        )
    return tuple(coefficients)


def current_correction(
    transport_flux: np.ndarray,
    transport_currents: tuple[np.ndarray, np.ndarray, np.ndarray],
    sigma_t: np.ndarray,
    cell_widths: tuple[float, float, float],
    theta: float = 0.0,
) -> CurrentCorrection3D:
    """Fit each face-current correction to the supplied transport moments."""
    shape = _shape(transport_flux, "transport_flux")
    flux = _grid(transport_flux, shape, "transport_flux")
    if np.any(flux <= 0):
        raise ValueError("transport_flux must be strictly positive")
    if len(transport_currents) != 3:
        raise ValueError("transport_currents must contain x, y, z arrays")
    diffusion = interface_diffusions(sigma_t, cell_widths, theta)
    if diffusion[0].shape != shape:
        raise ValueError("sigma_t must match transport_flux")
    correction = []
    for axis in range(3):
        current = _grid(transport_currents[axis], shape, "transport_current")
        neighbor = np.roll(flux, -1, axis=axis)
        correction.append(-(
            current + diffusion[axis] * (neighbor - flux)
        ) / (neighbor + flux))
    return CurrentCorrection3D(diffusion, tuple(correction))


def interface_currents(
    flux: np.ndarray, coefficients: CurrentCorrection3D
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positive-axis right-face currents for a coarse-grid flux."""
    shape = _shape(flux, "flux")
    phi = _grid(flux, shape, "flux")
    if len(coefficients.diffusion) != 3 or len(coefficients.correction) != 3:
        raise ValueError("coefficients must contain three axes")
    currents = []
    for axis in range(3):
        diffusion = _grid(coefficients.diffusion[axis], shape, "diffusion")
        correction = _grid(coefficients.correction[axis], shape, "correction")
        if np.any(diffusion <= 0):
            raise ValueError("interface diffusion must be positive")
        neighbor = np.roll(phi, -1, axis=axis)
        currents.append(
            -diffusion * (neighbor - phi) - correction * (neighbor + phi)
        )
    return tuple(currents)


def balance_operator(
    flux: np.ndarray,
    sigma_a: np.ndarray | float,
    cell_widths: tuple[float, float, float],
    coefficients: CurrentCorrection3D,
) -> np.ndarray:
    """Apply sum_axis (J_right-J_left)/h + sigma_a*flux."""
    shape = _shape(flux, "flux")
    phi = _grid(flux, shape, "flux")
    absorption = _grid(sigma_a, shape, "sigma_a")
    if np.any(absorption <= 0):
        raise ValueError("sigma_a must be positive for this fixed-source solve")
    widths = _widths(cell_widths)
    currents = interface_currents(phi, coefficients)
    result = absorption * phi
    for axis in range(3):
        result = result + (
            currents[axis] - np.roll(currents[axis], 1, axis=axis)
        ) / widths[axis]
    return result


def cmfd_matrix(
    sigma_a: np.ndarray | float,
    cell_widths: tuple[float, float, float],
    coefficients: CurrentCorrection3D,
) -> np.ndarray:
    """Build a dense reference matrix for at most 216 coarse cells."""
    shape = _shape(coefficients.diffusion[0], "diffusion")
    cells = int(np.prod(shape))
    if cells > 216:
        raise ValueError("dense reference matrix is limited to 216 coarse cells")
    basis = np.eye(cells)
    return np.column_stack([
        balance_operator(basis[:, column].reshape(shape), sigma_a,
                         cell_widths, coefficients).ravel()
        for column in range(cells)
    ])


def solve_fixed_source(
    source: np.ndarray,
    sigma_a: np.ndarray | float,
    cell_widths: tuple[float, float, float],
    coefficients: CurrentCorrection3D,
) -> np.ndarray:
    """Solve the small periodic 3D fixed-source coarse-mesh system."""
    shape = _shape(source, "source")
    rhs = _grid(source, shape, "source")
    if coefficients.diffusion[0].shape != shape:
        raise ValueError("source and coefficients must have matching grids")
    matrix = cmfd_matrix(sigma_a, cell_widths, coefficients)
    return np.linalg.solve(matrix, rhs.ravel()).reshape(shape)
