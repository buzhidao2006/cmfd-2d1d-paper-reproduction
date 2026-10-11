"""Single-group 2D/1D transport-splitting Fourier-mode experiment.

This is a pedagogical error-propagation model, not the paper's NECP-X solver:
it has no MOC ray tracing, CMFD correction, fission eigenvalue, or benchmarks.
The two transport blocks exchange transverse leakage, so their update order
and the number of axial inner iterations can be studied independently.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

ORDERS = ("1d2d", "2d1d")
_DIRECTIONS = np.asarray(list(product((-1.0, 1.0), repeat=3))) / np.sqrt(3.0)


@dataclass(frozen=True)
class Mode:
    """One spatial Fourier mode in a periodic, homogeneous medium."""

    sigma_t: float = 1.0
    scattering_ratio: float = 0.9
    wavelengths: tuple[float, float, float] = (3.0, 3.0, 60.0)
    mode_numbers: tuple[int, int, int] = (1, 0, 1)
    radial_cell_width: float = 1.0 / 3.0
    axial_cell_width: float = 3.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.sigma_t) or self.sigma_t <= 0:
            raise ValueError("sigma_t must be finite and positive")
        if not np.isfinite(self.scattering_ratio) or not 0 <= self.scattering_ratio < 1:
            raise ValueError("scattering_ratio must be in [0, 1)")
        if any(not np.isfinite(v) or v <= 0 for v in self.wavelengths):
            raise ValueError("wavelengths must be finite and positive")
        if any(not isinstance(v, int) for v in self.mode_numbers):
            raise ValueError("mode_numbers must be integers")
        if any(not np.isfinite(v) or v <= 0 for v in
               (self.radial_cell_width, self.axial_cell_width)):
            raise ValueError("cell widths must be finite and positive")
        if self.radial_cell_width > min(self.wavelengths[:2]) / 2:
            raise ValueError("radial cell width must allow at least two cells per period")
        if self.axial_cell_width > self.wavelengths[2] / 2:
            raise ValueError("axial cell width must allow at least two cells per period")
        for length, width in zip(self.wavelengths,
                                 (self.radial_cell_width, self.radial_cell_width,
                                  self.axial_cell_width)):
            cells = length / width
            if not np.isclose(cells, round(cells), rtol=0, atol=1e-9):
                raise ValueError("each cell width must evenly divide its periodic length")


def _upwind_symbol(direction: np.ndarray, wave_number: float, width: float) -> np.ndarray:
    """Fourier symbol of a first-order upwind streaming derivative."""
    return np.abs(direction) * (1.0 - np.exp(-1j * np.sign(direction) * wave_number * width)) / width


def streaming_symbols(mode: Mode) -> tuple[np.ndarray, np.ndarray]:
    """Return radial and axial leakage symbols for eight symmetric directions."""
    kx, ky, kz = (2.0 * np.pi * n / length
                  for n, length in zip(mode.mode_numbers, mode.wavelengths))
    radial = (_upwind_symbol(_DIRECTIONS[:, 0], kx, mode.radial_cell_width)
              + _upwind_symbol(_DIRECTIONS[:, 1], ky, mode.radial_cell_width))
    axial = _upwind_symbol(_DIRECTIONS[:, 2], kz, mode.axial_cell_width)
    return radial, axial


def outer_step(state: np.ndarray, mode: Mode, order: str, axial_iterations: int) -> np.ndarray:
    """Apply one split outer iteration to [axial angles, radial angles].

    Each block solves its own one-group fixed-source transport balance with
    the other block's transverse leakage held fixed during that block update.
    Isotropic scattering uses the angular mean of the block being updated.
    """
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    if axial_iterations < 1:
        raise ValueError("axial_iterations must be at least 1")
    state = np.asarray(state, dtype=complex)
    if state.shape != (16,):
        raise ValueError("state must contain eight axial and eight radial angular fluxes")

    axial, radial = state[:8].copy(), state[8:].copy()
    radial_stream, axial_stream = streaming_symbols(mode)
    sigma_s = mode.sigma_t * mode.scattering_ratio

    def update_axial() -> None:
        nonlocal axial
        axial = (sigma_s * axial.mean() - radial_stream * radial) / (mode.sigma_t + axial_stream)

    def update_radial() -> None:
        nonlocal radial
        radial = (sigma_s * radial.mean() - axial_stream * axial) / (mode.sigma_t + radial_stream)

    if order == "1d2d":
        for _ in range(axial_iterations):
            update_axial()
        update_radial()
    else:
        update_radial()
        for _ in range(axial_iterations):
            update_axial()
    # A deliberately simple common coarse-scalar reconciliation.  It is not
    # CMFD: there is no coarse current balance or transport-current correction.
    # It makes the order of the two sweeps observable in the full outer map.
    mismatch = radial.mean() - axial.mean()
    axial += 0.5 * mismatch
    radial -= 0.5 * mismatch
    return np.concatenate((axial, radial))


def transition_matrix(mode: Mode, order: str, axial_iterations: int) -> np.ndarray:
    """Construct the exact linear error-propagation matrix of this toy model."""
    basis = np.eye(16, dtype=complex)
    return np.column_stack(
        [outer_step(basis[:, column], mode, order, axial_iterations)
         for column in range(16)]
    )


def spectral_radius(mode: Mode, order: str, axial_iterations: int) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(transition_matrix(mode, order, axial_iterations)))))


def error_history(mode: Mode, order: str, axial_iterations: int, steps: int = 60,
                  seed: int = 7) -> np.ndarray:
    """Norm of a deterministic random initial error after each outer iteration."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    rng = np.random.default_rng(seed)
    state = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    state /= np.linalg.norm(state)
    history = [1.0]
    for _ in range(steps):
        state = outer_step(state, mode, order, axial_iterations)
        history.append(float(np.linalg.norm(state)))
    return np.asarray(history)


def scan(mode: Mode, axial_widths: list[float], inner_counts: list[int],
         steps: int = 60) -> list[dict[str, float | int | str]]:
    """Compare order and axial iteration count over axial mesh widths."""
    rows = []
    for width in axial_widths:
        case = Mode(mode.sigma_t, mode.scattering_ratio, mode.wavelengths,
                    mode.mode_numbers, mode.radial_cell_width, width)
        for order in ORDERS:
            for count in inner_counts:
                rho = spectral_radius(case, order, count)
                history = error_history(case, order, count, steps)
                rows.append({
                    "axial_width": width,
                    "axial_optical_thickness": case.sigma_t * width,
                    "order": order,
                    "axial_iterations": count,
                    "spectral_radius": rho,
                    "error_after_steps": float(history[-1]),
                    "steps": steps,
                })
    return rows


def plot_error_histories(mode: Mode, axial_widths: list[float], inner_counts: list[int],
                         steps: int, output: Path) -> None:
    """Save observed error-norm curves; these are not plots from the paper."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(axial_widths),
                                figsize=(5 * len(axial_widths), 4), squeeze=False)
    for axis, width in zip(axes[0], axial_widths):
        case = Mode(mode.sigma_t, mode.scattering_ratio, mode.wavelengths,
                    mode.mode_numbers, mode.radial_cell_width, width)
        for order in ORDERS:
            for count in inner_counts:
                history = error_history(case, order, count, steps)
                axis.semilogy(np.maximum(history, np.finfo(float).tiny),
                              label=f"{order}, M={count}")
        axis.set_title(f"Axial optical thickness = {mode.sigma_t * width:g}")
        axis.set_xlabel("Outer iteration")
        axis.set_ylabel("Error norm")
        axis.grid(True, which="both", alpha=0.25)
    axes[0, -1].legend(fontsize="small", loc="best")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scattering-ratio", type=float, default=0.9)
    parser.add_argument("--axial-widths", type=float, nargs="+", default=[0.1, 3.0, 20.0])
    parser.add_argument("--inner-iterations", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--output", type=Path, help="Optional CSV path")
    parser.add_argument("--plot", type=Path, help="Optional PNG of error histories")
    args = parser.parse_args(argv)
    if any(width <= 0 for width in args.axial_widths):
        parser.error("all axial widths must be positive")
    if any(count < 1 for count in args.inner_iterations):
        parser.error("all inner iteration counts must be positive")
    mode = Mode(scattering_ratio=args.scattering_ratio)
    rows = scan(mode, args.axial_widths, args.inner_iterations, args.steps)
    for row in rows:
        print(f"tau={row['axial_optical_thickness']:5.2f} "
              f"order={row['order']} M={row['axial_iterations']:2d} "
              f"rho={row['spectral_radius']:.6f} "
              f"error={row['error_after_steps']:.3e}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.plot:
        plot_error_histories(mode, args.axial_widths, args.inner_iterations,
                             args.steps, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

