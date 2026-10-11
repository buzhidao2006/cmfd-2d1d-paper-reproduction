"""Fourier validation of the standalone periodic axial SN discretization.

The mode factors here belong only to the one-dimensional source iteration.
They are not the spectral radii of the paper's coupled 2D/1D-CMFD scheme.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from paper_reproduction.axial_sn import solve_axial_sn


def fourier_mode(cells: int, mode_number: int) -> np.ndarray:
    """Return a unit-amplitude cell-centered mode on a periodic mesh."""
    if not isinstance(cells, int) or cells < 1:
        raise ValueError("cells must be a positive integer")
    if not isinstance(mode_number, int) or not 0 <= mode_number < cells:
        raise ValueError("mode_number must be an integer in [0, cells)")
    return np.exp(2j * np.pi * mode_number * np.arange(cells) / cells)


def diamond_difference_symbol(
    cells: int, mode_number: int, mu: float, sigma_t: float, cell_width: float
) -> complex:
    """Cell-flux response to a unit cell-source Fourier mode in one direction.

    With z = exp(2 pi i m / N), periodic face flux is F z**j.  Inserting
    it into the diamond-difference balance gives the center/source factor
    (1+z)/2 / [sigma_t(1+z)/2 + mu(z-1)/h].
    """
    fourier_mode(cells, mode_number)
    if not np.isfinite(mu) or mu == 0 or abs(mu) > 1:
        raise ValueError("mu must be a nonzero finite direction cosine")
    if not np.isfinite(sigma_t) or sigma_t <= 0:
        raise ValueError("sigma_t must be finite and positive")
    if not np.isfinite(cell_width) or cell_width <= 0:
        raise ValueError("cell_width must be finite and positive")
    phase = np.exp(2j * np.pi * mode_number / cells)
    center_factor = (1 + phase) / 2
    return complex(center_factor / (
        sigma_t * center_factor + mu * (phase - 1) / cell_width
    ))


def scattering_amplification(
    cells: int,
    mode_number: int,
    *,
    sigma_t: float,
    sigma_s: float,
    cell_width: float,
    angles: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> complex:
    """One-step amplification of a scalar-flux error Fourier mode."""
    if not np.isfinite(sigma_s) or not 0 <= sigma_s < sigma_t:
        raise ValueError("sigma_s must be in [0, sigma_t)")
    if angles is None:
        angles = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    else:
        angles = np.asarray(angles, dtype=float)
    if angles.ndim != 1 or angles.size < 1 or not np.all(np.isfinite(angles)):
        raise ValueError("angles must be a finite, nonempty one-dimensional array")
    if weights is None:
        weights = np.full(angles.size, 1 / angles.size)
    else:
        weights = np.asarray(weights, dtype=float)
    if (weights.shape != angles.shape or not np.all(np.isfinite(weights))
            or np.any(weights <= 0) or not np.isclose(weights.sum(), 1)):
        raise ValueError("weights must be positive, match angles, and sum to one")
    return complex(sigma_s * np.dot(weights, [
        diamond_difference_symbol(cells, mode_number, float(mu), sigma_t, cell_width)
        for mu in angles
    ]))


def measured_amplification(
    cells: int, mode_number: int, *, sigma_t: float, sigma_s: float,
    cell_width: float, angles: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[complex, float]:
    """Apply the real-space SN solver once and return gain and mode residual."""
    mode = fourier_mode(cells, mode_number)
    after = solve_axial_sn(
        np.zeros(cells), sigma_t=sigma_t, sigma_s=sigma_s,
        cell_width=cell_width, angles=angles, weights=weights,
        inner_iterations=1, initial_scalar=mode,
    ).scalar_flux
    gain = complex(np.vdot(mode, after) / np.vdot(mode, mode))
    residual = float(np.linalg.norm(after - gain * mode))
    return gain, residual


def mode_scan(
    cells: int, *, sigma_t: float = 1.0, scattering_ratio: float = 0.9,
    cell_width: float = 0.5,
) -> list[dict[str, int | float]]:
    """Compare the derived factor with the real-space operator for each mode."""
    if not np.isfinite(scattering_ratio) or not 0 <= scattering_ratio < 1:
        raise ValueError("scattering_ratio must be in [0, 1)")
    rows = []
    for mode_number in range(cells):
        theory = scattering_amplification(
            cells, mode_number, sigma_t=sigma_t,
            sigma_s=scattering_ratio * sigma_t, cell_width=cell_width,
        )
        measured, residual = measured_amplification(
            cells, mode_number, sigma_t=sigma_t,
            sigma_s=scattering_ratio * sigma_t, cell_width=cell_width,
        )
        rows.append({
            "mode_number": mode_number,
            "theory_real": theory.real,
            "theory_imag": theory.imag,
            "amplification_abs": abs(theory),
            "measured_real": measured.real,
            "measured_imag": measured.imag,
            "absolute_difference": abs(theory - measured),
            "mode_residual": residual,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--sigma-t", type=float, default=1.0)
    parser.add_argument("--scattering-ratio", type=float, default=0.9)
    parser.add_argument("--cell-width", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="Optional CSV output")
    args = parser.parse_args(argv)
    if args.cells < 1:
        parser.error("--cells must be positive")
    rows = mode_scan(
        args.cells, sigma_t=args.sigma_t,
        scattering_ratio=args.scattering_ratio, cell_width=args.cell_width,
    )
    for row in rows:
        print(f"m={row['mode_number']:2d} "
              f"|gain|={row['amplification_abs']:.6f} "
              f"difference={row['absolute_difference']:.2e} "
              f"residual={row['mode_residual']:.2e}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
