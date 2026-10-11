# 第三阶段：一维轴向 Fourier 因子的独立交叉验证

本阶段只研究[第二阶段](STAGE2_AXIAL_SN.md)的**单群、周期、均匀介质轴向 SN 源迭代**。它不是论文的 3D 2D/1D-CMFD 误差传播矩阵，也不对应论文图 5–9 的谱半径。

对 `N` 个等宽单元，令 `θ = 2πm/N`、`z = exp(iθ)`，并设第 `j` 个周期面的角通量为 `F z^j`。将它代入菱形差分方程，可得到单方向从单元源到中心角通量的因子：

```text
G(θ, μ) = [(1 + z)/2] / [Σt (1 + z)/2 + μ (z - 1)/h]
```

对每个角方向加权，单步散射源迭代的标量通量误差因子为：

```text
λ(θ) = Σs Σ_p w_p G(θ, μ_p)
```

这是从本仓库的离散方程直接推得的公式；不能代替论文包含二维 MOC、横向泄漏耦合与 CMFD 修正的完整 Fourier 分析。程序将公式计算的 `λ` 与实空间 SN 求解器对同一复指数模式的单步输出作比较。`absolute_difference` 比较两种算法给出的因子，`mode_residual` 检查该模式是否仍是本算子的特征向量。

运行：

```powershell
python -m paper_reproduction.axial_fourier --cells 8 --output results/axial_modes.csv
python -m pytest tests/test_axial_fourier.py -v
```

零波数模式的放大因子等于 `Σs/Σt`；偶数单元网格的 Nyquist 模式中心通量因子为零。后者是菱形差分和周期网格的离散性质，不应解释为物理上的高频完全消失。后续要比较论文的收敛趋势，仍需实现自洽的二维 MOC / 一维 SN 横向泄漏交换和真正的 CMFD 误差传播矩阵。
