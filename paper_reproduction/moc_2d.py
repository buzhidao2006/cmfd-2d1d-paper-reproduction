"""Minimal periodic 2D method-of-characteristics fixed-source prototype.

Only square Cartesian cells and four diagonal track directions are modeled.
Each direction has one periodic corner-to-corner track segment per cell.
The segment average is a track sample, not a full spatial/angular quadrature.
There is no axial coupling, CMFD acceleration, or eigenvalue solve here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_DIRECTIONS = np.array([
    [-1, -1], [-1, 1], [1, -1], [1, 1],
], dtype=int)


@dataclass(frozen=True)
class MOC2DResult:
    """Track-sampled fluxes after fixed-source scattering iterations."""

    scalar_flux: np.ndarray
    angular_flux: np.ndarray
    incoming_flux: np.ndarray
    outgoing_flux: np.ndarray
    scalar_history: np.ndarray


def _grid(values: np.ndarray | float, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = np.full(shape, array)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and match the 2D grid")
    return array


def _direction(direction: tuple[int, int] | np.ndarray) -> tuple[int, int]:
    array = np.asarray(direction)
    if array.shape != (2,) or not np.all(np.isin(array, (-1, 1))):
        raise ValueError("direction must contain x/y signs, each -1 or +1")
    return int(array[0]), int(array[1])


def diagonal_track_sweep(
    source: np.ndarray,
    sigma_t: np.ndarray | float,
    cell_width: float,
    direction: tuple[int, int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve exact flat-source transport along closed diagonal tracks.

    For a segment of length L = sqrt(2)*cell_width, the outgoing flux is
    exp(-sigma_t*L)*incoming + (1-exp(-sigma_t*L))*source/sigma_t.
    Periodicity equates each outgoing flux with the next cell's incoming flux.
    """
    source = np.asarray(source)
    if source.ndim != 2 or min(source.shape) < 1:
        raise ValueError("source must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(source)):
        raise ValueError("source must be finite")
    total = _grid(sigma_t, source.shape, "sigma_t")
    if np.iscomplexobj(total) or np.any(total <= 0):
        raise ValueError("sigma_t must be real and positive")
    if not np.isfinite(cell_width) or cell_width <= 0:
        raise ValueError("cell_width must be finite and positive")
    sign_x, sign_y = _direction(direction)

    cells = source.size
    shape = source.shape
    length = np.sqrt(2.0) * cell_width
    optical_length = total * length
    attenuation = np.exp(-optical_length)
    source_fraction = -np.expm1(-optical_length)
    particular = source / total
    source_term = source_fraction * particular

    matrix = np.eye(cells, dtype=np.result_type(source, float))
    rhs = np.zeros(cells, dtype=matrix.dtype)
    for x, y in np.ndindex(shape):
        current = np.ravel_multi_index((x, y), shape)
        following = np.ravel_multi_index(
            ((x + sign_x) % shape[0], (y + sign_y) % shape[1]), shape
        )
        matrix[following, current] -= attenuation[x, y]
        rhs[following] = source_term[x, y]
    incoming = np.linalg.solve(matrix, rhs).reshape(shape)
    outgoing = attenuation * incoming + source_term
    track_average = particular + (incoming - particular) * (
        source_fraction / optical_length
    )
    return incoming, outgoing, track_average


def solve_moc_2d(
    source: np.ndarray,
    *,
    sigma_t: np.ndarray | float,
    sigma_s: np.ndarray | float,
    cell_width: float,
    directions: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    axial_leakage: np.ndarray | None = None,
    inner_iterations: int = 1,
    initial_scalar: np.ndarray | None = None,
) -> MOC2DResult:
    """Iterate isotropic scattering with frozen angle-dependent axial leakage."""
    source = np.asarray(source)
    if source.ndim != 2 or min(source.shape) < 1:
        raise ValueError("source must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(source)):
        raise ValueError("source must be finite")
    total = _grid(sigma_t, source.shape, "sigma_t")
    scatter = _grid(sigma_s, source.shape, "sigma_s")
    if (np.iscomplexobj(total) or np.iscomplexobj(scatter)
            or np.any(total <= 0) or np.any(scatter < 0)
            or np.any(scatter >= total)):
        raise ValueError("cross sections must satisfy 0 <= sigma_s < sigma_t")
    if not isinstance(inner_iterations, int) or inner_iterations < 1:
        raise ValueError("inner_iterations must be a positive integer")
    if directions is None:
        directions = DEFAULT_DIRECTIONS.copy()
    else:
        directions = np.asarray(directions)
    if directions.ndim != 2 or directions.shape[1] != 2 or directions.shape[0] < 1:
        raise ValueError("directions must contain x/y signs")
    for direction in directions:
        _direction(direction)
    count = directions.shape[0]
    if weights is None:
        weights = np.full(count, 1 / count)
    else:
        weights = np.asarray(weights, dtype=float)
    if (weights.shape != (count,) or not np.all(np.isfinite(weights))
            or np.any(weights <= 0) or not np.isclose(weights.sum(), 1)):
        raise ValueError("weights must be positive, match directions, and sum to one")

    angular_shape = (count, *source.shape)
    if axial_leakage is None:
        axial_leakage = np.zeros(angular_shape)
    else:
        axial_leakage = np.asarray(axial_leakage)
    if axial_leakage.shape != angular_shape or not np.all(np.isfinite(axial_leakage)):
        raise ValueError("axial_leakage must be finite with shape (directions, x, y)")
    if initial_scalar is None:
        initial_scalar = np.zeros(source.shape)
    else:
        initial_scalar = _grid(initial_scalar, source.shape, "initial_scalar")

    dtype = np.result_type(source, axial_leakage, initial_scalar, float)
    scalar = initial_scalar.astype(dtype, copy=True)
    history = [scalar.copy()]
    incoming = np.empty(angular_shape, dtype=dtype)
    outgoing = np.empty(angular_shape, dtype=dtype)
    angular = np.empty(angular_shape, dtype=dtype)
    for _ in range(inner_iterations):
        isotropic_source = source + scatter * scalar
        for p, direction in enumerate(directions):
            incoming[p], outgoing[p], angular[p] = diagonal_track_sweep(
                isotropic_source - axial_leakage[p], total,
                cell_width, direction,
            )
        scalar = np.tensordot(weights, angular, axes=(0, 0))
        history.append(scalar.copy())
    return MOC2DResult(scalar, angular, incoming, outgoing, np.stack(history))
