# CMFD 2D/1D 论文复现记录

这是一个独立的、可运行的数值实验仓库，用来逐步研究 Zhou 等人（2022）的 2D/1D 中子输运收敛分析。**当前版本仅是第一阶段的简化 Fourier 模式误差传播实验，不是论文算法或结果的完整复现。**

> X. Zhou, Z. Liu, L. Cao, H. Wu, “Convergence analysis for the CMFD accelerated 2D/1D neutron transport calculation method based on Fourier analysis,” *Annals of Nuclear Energy* 170 (2022), 108982. [DOI: 10.1016/j.anucene.2022.108982](https://doi.org/10.1016/j.anucene.2022.108982)

## 当前完成

- 用单群、定源、周期均匀介质建立 16 维误差传播矩阵。
- 比较 1D→2D 与 2D→1D 的更新顺序，以及轴向内迭代次数、轴向网格宽度对本模型谱半径的影响。
- 输出可重复的 CSV 数据和误差范数曲线；用自动化测试检查矩阵与直接迭代的一致性。

本模型使用八个角方向的一阶迎风 Fourier 符号，以及简单的共享标量调和。**共享标量调和不是 CMFD**；目前没有 2D MOC 特征线追踪、1D SN 输运、3D CMFD 电流修正、裂变本征值求解，也没有复现论文的 VERA、KUCA、KAIST 算例。因此本仓库的数值和趋势不可当作论文图表的复现结果。

## 运行

建议 Python 3.10 或更新版本：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m paper_reproduction.fourier_mode --output results/scan.csv --plot results/scan.png
python -m pytest tests/ -v
```

在 macOS/Linux 上，把激活命令换成 `source .venv/bin/activate`。默认扫描三档轴向网格宽度、两种迭代顺序及四档内迭代次数；也可通过 `--help` 查看参数。输出的 `rho < 1` 表示该模式渐近收缩，`rho > 1` 表示渐近放大。

## 复现路线

1. 保存第一阶段模型、输入和测试，明确与论文方法的差异。
2. 实现真正的 2D MOC / 1D SN 输运耦合与 CMFD 修正，并逐项验证离散方程和迭代矩阵。
3. 按论文给出的截面、网格、收敛阈值和计算顺序构建算例，对照论文图表；无法取得的数据需明确标为缺失。
4. 将独立复现结果、偏差与失败案例写入可追踪的实验记录。

更详细的第一阶段假设见 [实验说明](paper_reproduction/README.md)。仓库不包含论文 PDF；请通过出版方或学校图书馆合法获取。

## 后续进展

已加入可独立验证的周期轴向 [1D SN 菱形差分子模块](docs/STAGE2_AXIAL_SN.md)，覆盖制造解、均匀介质极限与离散守恒。它仍未与 2D MOC 或 CMFD 耦合，不改变上文“尚非完整复现”的结论。

随后增加了[轴向 Fourier 因子交叉验证](docs/STAGE3_AXIAL_FOURIER.md)：逐个周期模式比较解析离散符号与实空间求解器的一次迭代，仍不将其称为论文完整谱半径复现。

现已增加[一维 CMFD 界面电流一致性](docs/STAGE4_CMFD_1D.md)检查，用轴向 SN 面电流验证粗网格修正公式与定源平衡；它不是完整的三维 pCMFD 加速求解。

每次推送与拉取请求会在 Python 3.10、3.12 下自动运行代码检查和测试。

另已加入[三维周期粗网格 CMFD 电流一致性](docs/STAGE5_CMFD_3D.md)验证，并检查它在径向均匀情形下退化为现有轴向 SN 子问题。该模块仍只解定源问题，未覆盖论文的二维 MOC 与本征值计算。

现已新增[最小二维特征线定源求解器](docs/STAGE6_MOC_2D.md)，验证平源轨迹积分与周期轨迹平衡。其轨迹/角度采样极简，尚未与轴向 SN、CMFD 自洽耦合，也不能用作论文数值结果的直接对照。
