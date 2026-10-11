"""Periodic one-group axial SN transport with diamond-difference cells.

This implements an independently testable part of the axial discretization
described by Zhou et al. (2022), not the paper's coupled MOC/CMFD solver.
The radial transverse leakage is supplied as a frozen, angle-dependent input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxialResult:
    """Fluxes after the requested number of source iterations."""

    scalar_flux: np.ndarray
    angular_flux: np.ndarray
    face_flux: np.ndarray
    scalar_history: np.ndarray


def diamond_difference_sweep(
    source: np.ndarray, mu: float, sigma_t: float, cell_width: float
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one direction with periodic faces; return face and cell fluxes.

    Face j is the left face of cell j.  The discrete balance is
    mu * (face[j+1] - face[j]) / h + sigma_t * cell[j] = source[j],
    with cell[j] = (face[j] + face[j+1]) / 2.
    """
    source = np.asarray(source)
    if source.ndim != 1 or source.size < 1 or not np.all(np.isfinite(source)):
        raise ValueError("source must be a finite, nonempty one-dimensional array")
    if not np.isfinite(mu) or mu == 0:
        raise ValueError("mu must be finite and nonzero")
    if not np.isfinite(sigma_t) or sigma_t <= 0:
        raise ValueError("sigma_t must be finite and positive")
    if not np.isfinite(cell_width) or cell_width <= 0:
        raise ValueError("cell_width must be finite and positive")

    cells = source.size
    matrix = np.zeros((cells, cells), dtype=np.result_type(source, float))
    indices = np.arange(cells)
    matrix[indices, indices] += sigma_t / 2 - mu / cell_width
    matrix[indices, (indices + 1) % cells] += sigma_t / 2 + mu / cell_width
    faces = np.linalg.solve(matrix, source)
    centers = (faces + np.roll(faces, -1)) / 2
    return faces, centers


def solve_axial_sn(
    source: np.ndarray,
    *,
    sigma_t: float,
    sigma_s: float,
    cell_width: float,
    angles: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    transverse_leakage: np.ndarray | None = None,
    inner_iterations: int = 1,
    initial_scalar: np.ndarray | None = None,
) -> AxialResult:
    """Perform fixed-source iterations with frozen radial transverse leakage.

    ``source`` is the isotropic external source per angle, and
    ``transverse_leakage`` has shape (number of angles, number of cells).
    At each iteration, the scattering source uses the previous scalar flux.
    This is not an eigenvalue solve and does not include CMFD correction.
    """
    source = np.asarray(source)
    if source.ndim != 1 or source.size < 1 or not np.all(np.isfinite(source)):
        raise ValueError("source must be a finite, nonempty one-dimensional array")
    if not np.isfinite(sigma_t) or sigma_t <= 0:
        raise ValueError("sigma_t must be finite and positive")
    if not np.isfinite(sigma_s) or not 0 <= sigma_s < sigma_t:
        raise ValueError("sigma_s must be in [0, sigma_t)")
    if not np.isfinite(cell_width) or cell_width <= 0:
        raise ValueError("cell_width must be finite and positive")
    if not isinstance(inner_iterations, int) or inner_iterations < 1:
        raise ValueError("inner_iterations must be a positive integer")

    if angles is None:
        angles = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    else:
        angles = np.asarray(angles, dtype=float)
    if angles.ndim != 1 or angles.size < 1 or not np.all(np.isfinite(angles)):
        raise ValueError("angles must be a finite, nonempty one-dimensional array")
    if np.any(angles == 0) or np.any(np.abs(angles) > 1):
        raise ValueError("angles must be nonzero direction cosines in [-1, 1]")

    if weights is None:
        weights = np.full(angles.size, 1 / angles.size)
    else:
        weights = np.asarray(weights, dtype=float)
    if (weights.shape != angles.shape or not np.all(np.isfinite(weights))
            or np.any(weights <= 0) or not np.isclose(weights.sum(), 1)):
        raise ValueError("weights must be positive, match angles, and sum to one")

    shape = (angles.size, source.size)
    if transverse_leakage is None:
        transverse_leakage = np.zeros(shape)
    else:
        transverse_leakage = np.asarray(transverse_leakage)
    if transverse_leakage.shape != shape or not np.all(np.isfinite(transverse_leakage)):
        raise ValueError("transverse_leakage must be finite with shape (angles, cells)")

    if initial_scalar is None:
        initial_scalar = np.zeros(source.shape)
    else:
        initial_scalar = np.asarray(initial_scalar)
    if initial_scalar.shape != source.shape or not np.all(np.isfinite(initial_scalar)):
        raise ValueError("initial_scalar must be finite and match source")

    dtype = np.result_type(source, transverse_leakage, initial_scalar, float)
    scalar = initial_scalar.astype(dtype, copy=True)
    history = [scalar.copy()]
    angular = np.empty(shape, dtype=dtype)
    faces = np.empty(shape, dtype=dtype)
    for _ in range(inner_iterations):
        scatter_source = sigma_s * scalar
        for p, mu in enumerate(angles):
            q = source + scatter_source - transverse_leakage[p]
            faces[p], angular[p] = diamond_difference_sweep(
                q, float(mu), sigma_t, cell_width
            )
        scalar = np.tensordot(weights, angular, axes=(0, 0))
        history.append(scalar.copy())
    return AxialResult(scalar, angular, faces, np.stack(history))
