"""One-dimensional periodic CMFD current consistency for fixed sources.

This is a small, independently testable subset of the three-dimensional
current correction in Zhou et al. (2022), Eq. (7).  It is not the paper's
eigenvalue iteration, 2D MOC coupling, or pCMFD stability analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CurrentCorrection:
    """Coefficients on right faces; face i lies between cells i and i+1."""

    diffusion: np.ndarray
    correction: np.ndarray


def _cell_values(values: np.ndarray | float, cells: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    if array.ndim == 0:
        array = np.full(cells, float(array))
    if array.shape != (cells,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and have one value per cell")
    return array.astype(float, copy=False)


def interface_diffusion(
    sigma_t: np.ndarray, cell_width: np.ndarray | float, theta: float = 0.0
) -> np.ndarray:
    """Periodic interface coefficient from Eq. (7), with theta stabilization.

    D_i = 1/(3 sigma_t_i) + theta*h_i and
    D_tilde[i] = 2 D_i D_(i+1)/(D_i h_(i+1) + D_(i+1) h_i).
    """
    sigma_t = np.asarray(sigma_t)
    if sigma_t.ndim != 1 or sigma_t.size < 1:
        raise ValueError("sigma_t must be a nonempty one-dimensional array")
    sigma_t = _cell_values(sigma_t, sigma_t.size, "sigma_t")
    widths = _cell_values(cell_width, sigma_t.size, "cell_width")
    if np.any(sigma_t <= 0) or np.any(widths <= 0):
        raise ValueError("sigma_t and cell_width must be positive")
    if not np.isfinite(theta) or theta < 0:
        raise ValueError("theta must be finite and nonnegative")
    diffusion = 1 / (3 * sigma_t) + theta * widths
    next_diffusion = np.roll(diffusion, -1)
    next_width = np.roll(widths, -1)
    return 2 * diffusion * next_diffusion / (
        diffusion * next_width + next_diffusion * widths
    )


def current_correction(
    transport_flux: np.ndarray,
    transport_current: np.ndarray,
    sigma_t: np.ndarray,
    cell_width: np.ndarray | float,
    theta: float = 0.0,
) -> CurrentCorrection:
    """Choose the correction that exactly preserves transport face currents.

    The right-face current is J = -D_tilde*(phi_next-phi)
    -D_hat*(phi_next+phi).  Transport cell fluxes must be positive;
    this routine rejects a near-zero denominator rather than hiding it.
    """
    transport_flux = np.asarray(transport_flux)
    if transport_flux.ndim != 1 or transport_flux.size < 1:
        raise ValueError("transport_flux must be a nonempty one-dimensional array")
    cells = transport_flux.size
    flux = _cell_values(transport_flux, cells, "transport_flux")
    current = _cell_values(transport_current, cells, "transport_current")
    if np.any(flux <= 0):
        raise ValueError("transport_flux must be strictly positive")
    pair_sum = flux + np.roll(flux, -1)
    if np.any(pair_sum <= 1e-12 * max(1.0, float(np.max(flux)))):
        raise ValueError("neighboring transport flux sum is too small")
    diffusion = interface_diffusion(sigma_t, cell_width, theta)
    correction = -(
        current + diffusion * (np.roll(flux, -1) - flux)
    ) / pair_sum
    return CurrentCorrection(diffusion, correction)


def interface_current(flux: np.ndarray, coefficients: CurrentCorrection) -> np.ndarray:
    """Compute right-face currents for a given coarse-mesh scalar flux."""
    flux = np.asarray(flux)
    if flux.ndim != 1 or flux.size < 1:
        raise ValueError("flux must be a nonempty one-dimensional array")
    cells = flux.size
    phi = _cell_values(flux, cells, "flux")
    diffusion = _cell_values(coefficients.diffusion, cells, "diffusion")
    correction = _cell_values(coefficients.correction, cells, "correction")
    if np.any(diffusion <= 0):
        raise ValueError("interface diffusion must be positive")
    neighbor = np.roll(phi, -1)
    return -diffusion * (neighbor - phi) - correction * (neighbor + phi)


def cmfd_matrix(
    sigma_a: np.ndarray | float,
    cell_width: np.ndarray | float,
    coefficients: CurrentCorrection,
) -> np.ndarray:
    """Assemble (J_right-J_left)/h + sigma_a*phi = source."""
    diffusion = np.asarray(coefficients.diffusion)
    if diffusion.ndim != 1 or diffusion.size < 1:
        raise ValueError("diffusion must be a nonempty one-dimensional array")
    cells = diffusion.size
    diffusion = _cell_values(diffusion, cells, "diffusion")
    correction = _cell_values(coefficients.correction, cells, "correction")
    absorption = _cell_values(sigma_a, cells, "sigma_a")
    widths = _cell_values(cell_width, cells, "cell_width")
    if np.any(diffusion <= 0) or np.any(absorption <= 0) or np.any(widths <= 0):
        raise ValueError("diffusion, sigma_a, and cell_width must be positive")
    matrix = np.zeros((cells, cells))
    for i in range(cells):
        left = (i - 1) % cells
        right = (i + 1) % cells
        matrix[i, i] += (diffusion[i] - correction[i]) / widths[i]
        matrix[i, right] += (-diffusion[i] - correction[i]) / widths[i]
        matrix[i, i] += (diffusion[left] + correction[left]) / widths[i]
        matrix[i, left] += (-diffusion[left] + correction[left]) / widths[i]
        matrix[i, i] += absorption[i]
    return matrix


def solve_fixed_source(
    source: np.ndarray,
    sigma_a: np.ndarray | float,
    cell_width: np.ndarray | float,
    coefficients: CurrentCorrection,
) -> np.ndarray:
    """Solve a frozen-coefficient 1D periodic fixed-source CMFD system."""
    source = np.asarray(source)
    if source.ndim != 1 or source.size < 1:
        raise ValueError("source must be a nonempty one-dimensional array")
    rhs = _cell_values(source, source.size, "source")
    matrix = cmfd_matrix(sigma_a, cell_width, coefficients)
    if matrix.shape[0] != rhs.size:
        raise ValueError("source and coefficients must have matching cell counts")
    return np.linalg.solve(matrix, rhs)
