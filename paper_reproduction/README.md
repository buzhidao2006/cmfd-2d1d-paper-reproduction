# Zhou et al. (2022): first-stage numerical experiment

Reference: X. Zhou, Z. Liu, L. Cao, H. Wu, *Convergence analysis for the CMFD
accelerated 2D/1D neutron transport calculation method based on Fourier
analysis*, Annals of Nuclear Energy 170 (2022) 108982,
https://doi.org/10.1016/j.anucene.2022.108982.

## What this experiment is

`fourier_mode.py` is a small **single-group, fixed-source error-propagation
model**, not a reproduction of the paper's full algorithm. Its 8 angular
directions use first-order upwind streaming symbols on a periodic grid. A
radial (2D) block and an axial (1D) block exchange transverse leakage. After
each outer iteration, a simple shared-scalar reconciliation aligns their
angular means. That reconciliation is **not CMFD**: it has no coarse-grid
current balance, transport-current correction, or eigenvalue solve. There is
also no MOC ray tracing or NECP-X implementation here.

The goal is narrower: make the iteration order and the number of axial inner
iterations explicit, then compare the exact spectral radius of this *model's*
16-by-16 transition matrix against an observed error-norm history. Its
numbers and trends must not be presented as reproductions of the paper's
Figures 5-9 or Tables 3-6.

## Run

From the repository root:

```powershell
python -m paper_reproduction.fourier_mode
python -m paper_reproduction.fourier_mode --scattering-ratio 0.99 --steps 80 --output scan.csv --plot scan.png
python -m pytest tests/test_paper_reproduction.py -v
```

The default scan varies axial cell width (0.1, 3, 20), sweep order (`1d2d`
or `2d1d`), and axial inner iterations (1, 3, 5, 10). With total cross section
1, the cell widths equal axial optical thicknesses. The periodic axial
length is 60, so all default widths divide it evenly. `rho < 1` means
asymptotic contraction for a mode; `rho > 1` means asymptotic growth.
Observed error norms may initially behave differently from the spectral
radius because the transition matrix can be non-normal.

## Next stage for a paper-level reproduction

Replace this model with a 2D MOC / 1D SN transport solver and a true 3D CMFD
correction; implement the paper's convergence thresholds and cross sections;
then compare the simplified VERA, KUCA, and KAIST benchmarks against the
published tables. Keep those results separate from the current experiment.

