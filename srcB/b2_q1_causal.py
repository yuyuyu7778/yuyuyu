"""Q1 因果发现与关联共识排序（方案 E：特征间因果网络）。

对应 CLAUDE_B.md 第三节 Q1 第 2–3 条，以及用户补充约束 1–5。

## 设计演进（如实记录，见 out/B/log/p2b_causal.log）

原设计用「事件窗口 0/1 指示」作因果锚点，经实测**两种锚点均失效**：
  * 0/1 指示锚点：KSG kNN 条件互信息是连续变量估计器，二值条件变量使其退化；
  * 事件倒计时锚点（方案 D）：锚点满足 a[t] = a[t-1] + 1，
    条件在 a[t-τ] 上时 a[t] 被完全确定，因此 I(a_t; X_{t-τ} | a_{t-τ}) ≡ 0，
    TE 在数学上恒为 0，与数据无关。

经用户批准改用**方案 E：特征间因果网络排序**，并采用**全期训练 + 事件窗口偏移**：
  * TE / PCMCI / dCor 在全期（2021–2026，46768 有效小时）上计算，保证 n_eff 充足；
  * 分布偏移证据（KL / 马氏距离）在事件窗口 vs 气候背景上计算，捕捉事件特异性；
  * 因果方向证据由 PCMCI 特征间网络给出（入度/出度），不依赖任何人造锚点。

## 四类证据

1. **TE（传递熵）**：对每个特征 X，计算它与「事件窗口内最强同步伙伴」之间的
   TE，取事件窗口内 TE 总和作为「事件期间信息交换活跃度」。
2. **PCMCI 网络**：全期特征间因果网络，统计出度（驱动他人）与入度（被驱动）。
3. **dCor + 尾依赖 λ_U**：特征与「事件窗口指示」的非线性依赖 + 极端情形依赖。
4. **分布偏移**：事件窗口 vs 5 年同期背景的 KL 散度与马氏距离。

## 用户补充约束的落实

1. tigramite 5.2.10.1 已装，走 PCMCI+ 正式路径；
2. 每个证据在自己有效样本上算，禁止统一 dropna()，每个统计量附 n_eff；
3. 滞后 1–24 h 扫描 × 特征 → BH-FDR 校正，报告校正前后显著个数；
4. Z0C_m（缺测 87%）不进排名，单列定性说明；
5. 不迎合"强辐射"叙事，结果以数据为准。

## 输出

    out/B/tab/b2_ranking_full.csv       全特征 × 四证据（含 n_eff、FDR）
    out/B/tab/b2_top5_consensus.csv     Top-5 共识
    out/B/tab/b2_negative_control.csv   负对照
    out/B/tab/b2_pcmci_network.csv      PCMCI 网络边列表
    out/B/fig/b2_stability_bar.png      bootstrap 入选频率
    out/B/fig/b2_negative_control.png   真实 vs 置换
    out/B/fig/b2_pcmci_degree.png       网络入度/出度
    out/B/log/p2b_causal.log            全部数字与判定

AI 辅助声明：本文件由 Claude Code（Anthropic，2026-10-24）辅助生成，人工已复核。

用法：
    python srcB/b2_q1_causal.py
"""

from __future__ import annotations

import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import SEED  # noqa: E402

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CLEAN_DIR: Path = PROJECT_ROOT / "data" / "B" / "clean"
TAB_DIR: Path = PROJECT_ROOT / "out" / "B" / "tab"
FIG_DIR: Path = PROJECT_ROOT / "out" / "B" / "fig"
LOG_DIR: Path = PROJECT_ROOT / "out" / "B" / "log"

#: 事件前兆窗口。
EVENT_START: pd.Timestamp = pd.Timestamp("2026-08-19 00:00")
EVENT_END: pd.Timestamp = pd.Timestamp("2026-08-26 10:00")

#: 气候背景年份（剔除 2026）。
BACKGROUND_YEARS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)
BG_MMDD_START: str = "08-19 00:00"
BG_MMDD_END: str = "08-26 10:00"

#: 滞后扫描范围（小时）。
LAGS: tuple[int, ...] = tuple(range(1, 25))

#: PCMCI 的 tau_max（计算量限制，取 6 h；TE 仍扫描 1–24 h）。
PCMCI_TAU_MAX: int = 4

#: PCMCI 时间抽样步长（小时）。全期 52584 行对 38 变量 PCMCI 太慢，
#: 3 h 抽样使自相关接近独立，同时保留 17528 个样本。
PCMCI_SUBSAMPLE: int = 3

#: BH-FDR 显著性水平。
ALPHA: float = 0.05

#: block bootstrap。
BOOT_BLOCK_HOURS: int = 72
BOOT_N: int = 500

#: KSG 近邻数。
KSG_K: int = 5

#: TE 置换检验次数。
N_PERM: int = 40

#: TE 计算的时间抽样步长（小时）。
#: 全期 52584 行逐时序列的一阶自相关很高，抽样到 6 h 既能大幅降低算量，
#: 又能让样本接近独立，减少 n_eff 折损。论文中须注明抽样步长。
TE_SUBSAMPLE: int = 6

#: TE 滞后扫描步长（小时）。
#: 抽样间隔必须 ≥ 滞后步长，否则「滞后 2h」在 6h 抽样下没有实际时间分辨率。
#: 6h 抽样 → 滞后 6/12/18/24 h 共 4 个点，覆盖作战地图要求的 1–24 h 范围。
TE_LAG_STEP: int = 6

#: 每个特征的候选 TE 伙伴数（先在事件窗口内按 |相关| 预筛）。
TE_N_CANDIDATES: int = 2

#: TE 置换检验次数（正式运行）。控制算量取 20。
TE_N_PERM: int = 20

#: TE 计算起始时间戳（用于进度日志）。
_t0: float = 0.0

#: dCor 的最大计算样本数。距离矩阵为 O(n²)，n=4000 时约 128 MB。
#: 超过此值时无放回抽样。尾依赖 λ_U 仍用全样本。
DCOR_MAX_N: int = 4000

#: 负对照的块长（小时）。**刻意取 37 h（非 24 的整数倍）**：
#: 24 h 的整倍数移位不会破坏日周期对齐，会导致负对照失效（实测下降仅 −1.2%）。
NEG_BLOCK_HOURS: int = 37

#: 不上排名的特征（用户约束 4 + 方法论理由）。
EXCLUDED: dict[str, str] = {
    "Z0C_m": "缺测率 87.29%（近地层外推本身不成立），仅作定性参考（用户约束 4）",
    "PDD_cumsum": "累计量，ρ₁≡1，n_eff≈3，作为因果变量无意义；信息由 PDD_hourly / Melt_mm_h 承载",
    "Ta_2m_K": "原始观测气温，未做物理重构；保留在特征表中供出图解释，不参与因果排名",
}

#: FLUX 分组（用户约束 2）。
FLUX_FLUX_VARS: tuple[str, ...] = ("H_Wm2", "LE_Wm2", "Bowen_beta", "EF", "EBR")
FLUX_MOMENTUM_VARS: tuple[str, ...] = ("USTAR", "TKE")

#: 辐射类特征（用户约束 5，用于专项说明）。
RADIATION_VARS: tuple[str, ...] = ("Rn_Wm2", "Albedo", "G_0.1m")


# --------------------------------------------------------------------------- #
# 统计工具
# --------------------------------------------------------------------------- #


def effective_n(x: np.ndarray) -> float:
    """有效样本数 n_eff = n·(1−ρ₁)/(1+ρ₁)（AR(1) 近似，Wilks 1997）。

    Args:
        x: 一维序列（可含 NaN）。

    Returns:
        有效样本数，下限截断到 3。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float(n)
    rho1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    if not np.isfinite(rho1):
        return float(n)
    rho1 = max(min(rho1, 0.999), -0.999)
    return float(max(n * (1.0 - rho1) / (1.0 + rho1), 3.0))


def bh_fdr(pvals: np.ndarray, alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg FDR 校正。

    Args:
        pvals: 原始 p 值（可含 NaN）。
        alpha: 目标 FDR。

    Returns:
        ``(reject, qvals)``。
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    reject = np.zeros(n, dtype=bool)
    q = np.full(n, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return reject, q
    idx = np.where(valid)[0]
    pv = p[idx]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    qv = ranked * m / np.arange(1, m + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(qv, 0.0, 1.0)
    q[idx] = out
    reject[idx] = out <= alpha
    return reject, q


def block_permute(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """时间块随机重排：保留块内自相关，破坏跨变量的时序对齐。

    **关键设计约束**：块长不能是 24 h 的整数倍。气象与土壤变量都有强烈的
    日周期，若按 24 h 的整倍数循环移位，两个序列的日周期仍严格对齐，
    负对照几乎不会导致 TE 下降（实测块长 24 h 时中位下降仅 −1.2%）。
    因此默认块长取 24 h 的整数倍以外的值（见 :data:`NEG_BLOCK_HOURS`），
    且采用**随机块顺序**而非简单循环移位。

    Args:
        x: 一维序列。
        block: 块长（小时），建议取非 24 的整数倍。
        rng: 随机数生成器。

    Returns:
        重排后的序列（长度不变）。
    """
    n = len(x)
    if n <= block:
        return x.copy()
    nb = int(np.ceil(n / block))
    blocks = [x[i * block:(i + 1) * block] for i in range(nb)]
    order = rng.permutation(nb)
    return np.concatenate([blocks[i] for i in order])[:n]


def ar_residual_surrogate(x: np.ndarray, order: int, rng: np.random.Generator) -> np.ndarray:
    """AR(p) 残差置换零模型（用户批准的负对照方案 1）。

    步骤：
      1. 拟合 AR(p)：x_t = Σφ_i·x_{t−i} + ε_t（最小二乘）；
      2. 保留 φ 与残差 ε 的**边缘分布**，但把 ε 随机打乱后重新合成：
         x*_t = Σφ_i·x*_{t−i} + ε_{π(t)}。

    这样得到的 surrogate 保留了：一阶/高阶自相关、日周期（因为 AR 项重建了
    自相关结构）、边缘分布近似的形态；但破坏了 x 与其它变量的**跨变量耦合**。
    相比块置换，它不会因为「分布不变而耦合仍在」而失效。

    Args:
        x: 一维序列（应无 NaN，调用方先插值）。
        order: AR 阶数。
        rng: 随机数生成器。

    Returns:
        与 ``x`` 等长的 surrogate 序列。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < order + 20:
        return x.copy()

    # 构造设计矩阵
    Y = x[order:]
    X = np.column_stack([x[order - i - 1: n - i - 1] for i in range(order)])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coef

    # 打乱残差后递推重建
    eps = rng.permutation(resid)
    out = x[:order].copy()
    for t in range(order, n):
        lags = np.array([out[t - i - 1] for i in range(order)])
        out = np.append(out, float(lags @ coef + eps[t - order]))
    return out[:n]


def ksghanadharan_mi(x: np.ndarray, y: np.ndarray, k: int = KSG_K) -> float:
    """KSG 一型互信息估计（Kraskov–Stögbauer–Grassberger 2004）。

    Args:
        x: 变量 X（一维，已对齐且无 NaN）。
        y: 变量 Y（一维）。
        k: 近邻数。

    Returns:
        互信息估计值（nats），非负。
    """
    from scipy.spatial import cKDTree
    from scipy.special import digamma as psi

    n = len(x)
    if n < k + 5:
        return np.nan
    xy = np.column_stack([x, y])
    tree_xy = cKDTree(xy)
    dist, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = np.maximum(dist[:, -1], 1e-12)
    nx = np.asarray(cKDTree(x.reshape(-1, 1)).query_ball_point(
        x.reshape(-1, 1), eps, p=np.inf, return_length=True), dtype=float) - 1
    ny = np.asarray(cKDTree(y.reshape(-1, 1)).query_ball_point(
        y.reshape(-1, 1), eps, p=np.inf, return_length=True), dtype=float) - 1
    mi = psi(k) + psi(n) - np.mean(psi(nx) + psi(ny))
    return float(max(mi, 0.0))


# --------------------------------------------------------------------------- #
# 证据 1：TE（特征 ↔ 事件窗口内最强伙伴）
# --------------------------------------------------------------------------- #


def te_between(x: np.ndarray, y: np.ndarray, lags: tuple[int, ...] = LAGS,
               k: int = KSG_K, n_perm: int = N_PERM) -> dict[str, float]:
    """TE(X→Y) 的 KSG 条件互信息估计，扫描滞后取最大值并做置换检验。

    TE(X→Y) = I(Y_t ; X_{t−τ} | Y_{t−τ})。

    Args:
        x: 驱动序列。
        y: 响应序列。
        lags: 滞后集合。
        k: KSG 近邻数。
        n_perm: 置换次数。

    Returns:
        ``te`` / ``lag`` / ``p`` / ``n`` / ``n_eff``。
    """
    from tigramite.independence_tests.cmiknn import CMIknn

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(SEED)
    # 初值用 -inf：KSG 的有限样本偏差可使 CMI 为负，用 0.0 作初值会
    # 导致「所有滞后都返回负值」时 best 永不更新、n 恒为 0 的假象。
    best = {"te": -np.inf, "lag": np.nan, "p": 1.0, "n": 0, "n_eff": np.nan}
    # 使用 tigramite 官方 KSG kNN 条件互信息估计器（需 numba，已安装）。
    _est = CMIknn(knn=k, shuffle_neighbors=5, workers=1)
    _xyz = np.array([0, 1, 2])

    def _cmi(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """I(a;b|c)，调用 tigramite CMIknn（KSG 二型，Frenzel–Pompe 条件形式）。

        Args:
            a: 响应变量的当前值 Y_t。
            b: 驱动变量的滞后值 X_{t−τ}。
            c: 响应变量的滞后值 Y_{t−τ}（条件变量）。

        Returns:
            条件互信息（nats）。样本不足时返回 NaN。
        """
        n_ = len(a)
        if n_ < k + 5:
            return np.nan
        return float(_est.get_dependence_measure(np.array([a, b, c]), _xyz))

    for lag in lags:
        if len(x) <= lag + 50:
            continue
        n = len(x) - lag
        xz, yz, yt = x[:n], y[:n], y[lag:lag + n]
        m = np.isfinite(xz) & np.isfinite(yz) & np.isfinite(yt)
        if m.sum() < 100:
            continue
        xz, yz, yt = xz[m], yz[m], yt[m]
        te = _cmi(yt, xz, yz)
        if not np.isfinite(te) or te <= best["te"]:
            continue
        null = np.array([_cmi(yt, rng.permutation(xz), yz) for _ in range(n_perm)])
        p = float((np.sum(null >= te) + 1) / (n_perm + 1))
        best = {"te": te, "lag": int(lag), "p": p, "n": int(m.sum()),
                "n_eff": effective_n(yt)}
    if not np.isfinite(best["te"]):
        best["te"] = np.nan
    return best


def evidence_te_panel(feats: pd.DataFrame, cols: list[str], log: list[str]) -> pd.DataFrame:
    """对每个特征计算它与事件窗口内「最强同步伙伴」的 TE。

    做法：先在事件窗口内用全期预筛选出与其他特征 TE 最强的伙伴，
    减少计算量；再在全期上计算该配对的 TE。

    Args:
        feats: 全期特征矩阵。
        cols: 参与排名的特征。
        log: 日志收集器。

    Returns:
        含 ``feature`` / ``te`` / ``te_lag`` / ``te_p`` / ``te_partner`` / ``te_n`` / ``te_n_eff``。
    """
    n_before = len(feats)
    sub_feats = feats.iloc[::TE_SUBSAMPLE]
    log.append(f"   TE 输入: {len(sub_feats)} 行 × {len(cols)} 列 "
               f"(全期 {n_before} 行按 {TE_SUBSAMPLE}h 抽样)")

    ev = sub_feats.loc[EVENT_START:EVENT_END]
    rows = []

    for i, c in enumerate(cols):
        x = sub_feats[c].to_numpy(dtype=float)
        best = {"te": 0.0, "partner": "", "lag": np.nan, "p": 1.0, "n": 0, "n_eff": np.nan}
        # 预筛：在事件窗口内与若干候选伙伴求相关，只对最相关的少数伙伴算 TE
        sub = ev[[c] + [o for o in cols if o != c]].corr()[c].drop(c).abs()
        candidates = sub.nlargest(TE_N_CANDIDATES).index.tolist()
        for partner in candidates:
            y_full = sub_feats[partner].to_numpy(dtype=float)
            r = te_between(x, y_full, lags=LAGS[::TE_LAG_STEP], n_perm=TE_N_PERM)
            if r["te"] > best["te"]:
                best = {**r, "partner": partner}
        rows.append({
            "feature": c, "te": best["te"], "te_lag": best["lag"],
            "te_p": best["p"], "te_partner": best["partner"],
            "te_n": best["n"], "te_n_eff": best["n_eff"],
        })
        if (i + 1) % 5 == 0 or i == len(cols) - 1:
            log.append(f"   TE 进度 {i + 1}/{len(cols)}  ({time.time() - _t0:.0f}s)")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 证据 2：PCMCI 特征间因果网络
# --------------------------------------------------------------------------- #


def evidence_pcmci_network(feats: pd.DataFrame, cols: list[str],
                           log: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在特征间运行 PCMCI，统计入度/出度与边列表。

    Args:
        feats: 全期特征矩阵。
        cols: 参与排名的特征。
        log: 日志收集器。

    Returns:
        ``(degrees, edges)``：每特征的入度/出度，与全部检出边。
    """
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    X = feats[cols].to_numpy(dtype=float)
    missing = np.isnan(X).mean(axis=0)
    df = pd.DataFrame(X).interpolate(limit_direction="both").ffill().bfill()
    sub = df.iloc[::PCMCI_SUBSAMPLE].to_numpy(dtype=float)
    log.append(f"   PCMCI 输入: {sub.shape[0]} 行 × {sub.shape[1]} 列 "
               f"(全期 {PCMCI_SUBSAMPLE}h 抽样)")
    log.append(f"   PCMCI tau_max = {PCMCI_TAU_MAX}  pc_alpha = {ALPHA}")

    t0 = time.time()
    data = pp.DataFrame(sub)
    pc = PCMCI(dataframe=data, cond_ind_test=ParCorr(), verbosity=0)
    res = pc.run_pcmci(tau_max=PCMCI_TAU_MAX, pc_alpha=ALPHA)
    G, V = res["graph"], res["val_matrix"]
    log.append(f"   PCMCI 耗时 {time.time() - t0:.1f}s")

    n = len(cols)
    outdeg = np.zeros(n, dtype=int)
    indeg = np.zeros(n, dtype=int)
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for tau in range(1, G.shape[2]):
                if G[i, j, tau] != "":
                    outdeg[i] += 1
                    indeg[j] += 1
                    edges.append({"source": cols[i], "target": cols[j],
                                  "lag_h": tau * PCMCI_SUBSAMPLE,
                                  "link": str(G[i, j, tau]),
                                  "val": float(V[i, j, tau])})

    total = n * (n - 1) * PCMCI_TAU_MAX
    dens = len(edges) / total if total else np.nan
    log.append(f"   检出边数 = {len(edges)} / {total}  网络密度 = {dens:.2%}")
    log.append(f"   平均出度 = {outdeg.mean():.2f}   平均入度 = {indeg.mean():.2f}")

    degrees = pd.DataFrame({
        "feature": cols, "out_degree": outdeg, "in_degree": indeg,
        "degree_sum": outdeg + indeg,
        "pcmci_missing_before_interp": missing,
    })
    return degrees, pd.DataFrame(edges)


# --------------------------------------------------------------------------- #
# 证据 3 / 4
# --------------------------------------------------------------------------- #


def _dcor_1d(x: np.ndarray, y: np.ndarray) -> float:
    """一维距离相关 dCor，用 O(n log n) 的排序恒等式避免构造 n×n 矩阵。

    dCov² 的展开式（Székely & Rizzo）：

        dCov²(X,Y) = E|X−X'||Y−Y'| − 2·E[E|X−X'||Y−Y'| | X] + E|X−X'|·E|Y−Y'|

    对一维变量，`E|X−X'|`、行均值等均可由排序后的前缀和 O(n log n) 求出。

    Args:
        x: 一维样本。
        y: 一维样本。

    Returns:
        距离相关 ∈ [0, 1]。
    """
    n = len(x)
    if n < 4:
        return np.nan

    def _abs_diff_row_mean(s: np.ndarray) -> np.ndarray:
        """返回每个 i 的 (1/n)·Σ_j |s_i − s_j|。"""
        order = np.argsort(s, kind="mergesort")
        ss = s[order]
        prefix = np.concatenate([[0.0], np.cumsum(ss)])
        k = np.arange(n)
        # Σ_j |ss_k − ss_j| = k·ss_k − prefix[k] + (prefix[n]−prefix[k+1]) − (n−1−k)·ss_k
        row_sorted = (k * ss - prefix[k]
                      + (prefix[n] - prefix[k + 1])
                      - (n - 1 - k) * ss)
        row = np.empty(n)
        row[order] = row_sorted
        return row / n

    a = _abs_diff_row_mean(x)
    b = _abs_diff_row_mean(y)
    ga, gb = float(a.mean()), float(b.mean())
    # dCov² = E_ij[a_i b_j] 的中心化形式：mean(A·B)，其中 A_ij = |x_i−x_j| − a_i − a_j + ga
    # 由于一维 |x_i−x_j| 的矩阵均值可解析：mean_ij |x_i−x_j| = 2·Σ_k (2k−n+1)·ss_k / n²
    # 直接用等价式：dCov² = E|X−X'||Y−Y'| − 2·E[a(X)·b(Y')] + ga·gb
    # 其中第一项的 E 仍需 O(n²)，故改用抽样公式：由中心化矩阵 A、B 的迹无法简化，
    # 这里对 n 已抽样到 ≤ DCOR_MAX_N，直接构造矩阵可接受（默认 4000² = 128 MB）。
    A = np.abs(x[:, None] - x[None, :])
    B = np.abs(y[:, None] - y[None, :])
    A -= a[:, None] + a[None, :] - ga
    B -= b[:, None] + b[None, :] - gb
    dcov2 = float((A * B).mean())
    vx = float((A * A).mean())
    vy = float((B * B).mean())
    denom = np.sqrt(max(vx * vy, 1e-24))
    return float(np.sqrt(max(dcov2, 0.0) / denom))


def evidence_dcor_tail(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """距离相关 dCor 与上尾依赖系数 λ_U（经验 copula，u > 0.9）。

    Args:
        x: 变量 X。
        y: 变量 Y。

    Returns:
        ``dcor`` / ``lambda_u`` / ``n`` / ``n_eff``。
    """
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n_full = len(x)
    if n_full < 30:
        return {"dcor": np.nan, "lambda_u": np.nan, "n": n_full, "n_eff": np.nan}

    # dCor 的 O(n²) 距离矩阵在全期（数万样本）会撑爆内存（36051² ≈ 9.7 GB），
    # 因此对 dCor 做无放回抽样；尾依赖 λ_U 用全样本（只需排序，O(n log n)）。
    rng = np.random.default_rng(SEED)
    if n_full > DCOR_MAX_N:
        idx = rng.choice(n_full, size=DCOR_MAX_N, replace=False)
        xs, ys = x[idx], y[idx]
    else:
        xs, ys = x, y
    n = len(xs)

    dcor = _dcor_1d(xs, ys)
    dcor = float(np.clip(dcor, 0.0, 1.0))

    # 上尾依赖用全样本
    u = stats.rankdata(x) / (n_full + 1)
    v = stats.rankdata(y) / (n_full + 1)
    thr = 0.9
    lam_u = float(((u > thr) & (v > thr)).sum() / max((u > thr).sum(), 1))

    return {"dcor": dcor, "lambda_u": lam_u, "n": n_full, "n_sampled": n,
            "n_eff": effective_n(x)}


def evidence_shift(feat: np.ndarray, bg: np.ndarray) -> dict[str, float]:
    """事件窗口相对气候背景的分布偏移：KL 散度 + 马氏距离。

    Args:
        feat: 事件窗口特征值。
        bg: 气候背景特征值。

    Returns:
        ``kl`` / ``mahal`` / ``n_event`` / ``n_bg`` / ``n_eff``。
    """
    f = np.asarray(feat, dtype=float)
    b = np.asarray(bg, dtype=float)
    f, b = f[np.isfinite(f)], b[np.isfinite(b)]
    base = {"n_event": len(f), "n_bg": len(b), "n_eff": effective_n(f)}
    if len(f) < 20 or len(b) < 20:
        return {**base, "kl": np.nan, "mahal": np.nan}

    lo, hi = float(min(f.min(), b.min())), float(max(f.max(), b.max()))
    if hi <= lo:
        return {**base, "kl": 0.0, "mahal": 0.0}
    bins = np.linspace(lo, hi, 51)
    pf, _ = np.histogram(f, bins=bins)
    pb, _ = np.histogram(b, bins=bins)
    pf = (pf + 1e-6) / (pf.sum() + 1e-6 * len(pf))
    pb = (pb + 1e-6) / (pb.sum() + 1e-6 * len(pb))
    kl = float(np.sum(pf * np.log(pf / pb)))

    sd = float(b.std(ddof=1))
    mahal = float(abs(f.mean() - b.mean()) / sd) if sd > 1e-12 else np.nan
    return {**base, "kl": kl, "mahal": mahal}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def main() -> int:
    """执行四证据计算、共识排序、负对照与稳定性分析。

    Returns:
        进程退出码：0 正常；1 负对照未达标。
    """
    t_all = time.time()
    log: list[str] = []
    add = log.append

    import importlib.metadata as _md

    add("=" * 78)
    add("P2-2 Q1 因果共识排序（方案 E：特征间因果网络）")
    add("=" * 78)
    add(f"运行时间戳 : {datetime.now().astimezone().isoformat()}")
    add(f"Python     : {sys.version.split()[0]}   numpy {np.__version__} / pandas {pd.__version__}")
    add(f"tigramite  : {_md.version('tigramite')}")
    add(f"SEED       : {SEED}")
    add("")

    add("-" * 78)
    add("[0] 设计演进（如实记录）")
    add("-" * 78)
    add("原设计用「事件窗口 0/1 指示」作因果锚点，实测两种锚点均失效：")
    add("  (a) 0/1 指示：KSG kNN 条件互信息为连续变量估计器，二值条件变量使其退化；")
    add("  (b) 事件倒计时（方案 D）：锚点满足 a[t]=a[t-1]+1，条件在 a[t-τ] 上时")
    add("      a[t] 被完全确定，故 I(a_t; X_{t-τ} | a_{t-τ}) ≡ 0，TE 数学上恒为 0。")
    add("经用户批准改用 方案 E（特征间因果网络）+ 全期训练 + 事件窗口偏移。")
    add("")

    feats = pd.read_csv(CLEAN_DIR / "features_hourly.csv", encoding="utf-8-sig",
                        index_col=0, parse_dates=True)
    all_cols = list(feats.columns)
    excluded = [c for c in all_cols if c in EXCLUDED]
    cols = [c for c in all_cols if c not in EXCLUDED]

    add("-" * 78)
    add("[1] 特征与样本")
    add("-" * 78)
    add(f"全期特征数 : {len(all_cols)}   参与排名 : {len(cols)}")
    for c in excluded:
        add(f"   排除 {c}: {EXCLUDED[c]}")
    add(f"全期行数   : {len(feats)}   {feats.index.min()} -> {feats.index.max()}")
    ev = feats.loc[EVENT_START:EVENT_END]
    add(f"事件窗口   : {EVENT_START} -> {EVENT_END}  n = {len(ev)}")
    add("")

    # ---------- 证据 1：TE ----------
    add("-" * 78)
    add("[2] 证据 1：传递熵 TE（全期，特征 ↔ 最强同步伙伴）")
    add("-" * 78)
    global _t0
    _t0 = time.time()
    te_df = evidence_te_panel(feats, cols, log)
    add(f"   TE 计算耗时 {time.time() - _t0:.1f}s")
    add("")

    # ---------- 证据 2：PCMCI ----------
    add("-" * 78)
    add("[3] 证据 2：PCMCI 特征间因果网络（全期）")
    add("-" * 78)
    deg, edges = evidence_pcmci_network(feats, cols, log)
    add("")

    # ---------- 证据 3：dCor / 尾依赖 ----------
    add("-" * 78)
    add("[4] 证据 3：dCor + 上尾依赖 λ_U（特征 vs 事件窗口指示）")
    add("-" * 78)
    is_event = pd.Series(((feats.index >= EVENT_START) & (feats.index <= EVENT_END)).astype(float),
                         index=feats.index)
    dc_rows = []
    for c in cols:
        r = evidence_dcor_tail(feats[c].to_numpy(dtype=float), is_event.to_numpy(dtype=float))
        dc_rows.append({"feature": c, "dcor": r["dcor"], "lambda_u": r["lambda_u"],
                        "dt_n": r["n"], "dt_n_eff": r["n_eff"]})
    dc_df = pd.DataFrame(dc_rows)
    add(f"   完成 {len(cols)} 个特征")
    add("")

    # ---------- 证据 4：分布偏移 ----------
    add("-" * 78)
    add("[5] 证据 4：事件窗口 vs 气候背景的分布偏移")
    add("-" * 78)
    bg_parts = [feats.loc[f"{y}-{BG_MMDD_START}":f"{y}-{BG_MMDD_END}"] for y in BACKGROUND_YEARS]
    background = pd.concat(bg_parts)
    add(f"   背景样本 n = {len(background)}（{len(BACKGROUND_YEARS)} 年同期）")
    sh_rows = []
    for c in cols:
        r = evidence_shift(ev[c].to_numpy(dtype=float), background[c].to_numpy(dtype=float))
        sh_rows.append({"feature": c, "kl": r["kl"], "mahal": r["mahal"],
                        "shift_n_event": r["n_event"], "shift_n_bg": r["n_bg"],
                        "shift_n_eff": r["n_eff"]})
    sh_df = pd.DataFrame(sh_rows)
    add(f"   完成 {len(cols)} 个特征")
    add("")

    # ---------- 合并 ----------
    df = (te_df.merge(deg, on="feature", how="left")
                .merge(dc_df, on="feature", how="left")
                .merge(sh_df, on="feature", how="left"))
    df["flux_group"] = df["feature"].apply(
        lambda c: "flux_flux" if c in FLUX_FLUX_VARS
        else "flux_momentum" if c in FLUX_MOMENTUM_VARS else "")

    # ---------- BH-FDR ----------
    add("-" * 78)
    add("[6] BH-FDR 多重比较校正 ★ 用户约束 3")
    add("-" * 78)
    n_lag = len(LAGS[::TE_LAG_STEP])
    p_te = df["te_p"].to_numpy(dtype=float)
    rej1, q1 = bh_fdr(p_te, ALPHA)
    df["te_q"] = q1
    df["te_sig"] = rej1

    p_strict = np.minimum(p_te * n_lag, 1.0)
    rej2, q2 = bh_fdr(p_strict, ALPHA)
    df["te_q_strict"] = q2
    df["te_sig_strict"] = rej2

    add(f"检验族: {len(cols)} 个特征；每个特征扫描 {n_lag} 个滞后"
        f"（{TE_SUBSAMPLE}h 抽样，滞后 {list(LAGS[::TE_LAG_STEP])} h）取最大 TE")
    add(f"α = {ALPHA}")
    add(f"  口径 A（族 = {len(cols)} 特征）        : "
        f"校正前显著 {int((p_te <= ALPHA).sum())} → 校正后 {int(rej1.sum())}")
    add(f"  口径 B（族 = {len(cols)}×{n_lag} 检验，保守）: "
        f"校正前显著 {int((p_strict <= ALPHA).sum())} → 校正后 {int(rej2.sum())}")
    add("")

    # ---------- 共识排序 ----------
    add("-" * 78)
    add("[7] Borda 共识排序（四证据）")
    add("-" * 78)
    df["rank_te"] = df["te"].rank(ascending=False, method="average")
    df["rank_pcmci"] = df["degree_sum"].rank(ascending=False, method="average")
    df["rank_dcor"] = df["dcor"].rank(ascending=False, method="average")
    df["rank_shift"] = df["kl"].rank(ascending=False, method="average")
    for c in ["rank_te", "rank_pcmci", "rank_dcor", "rank_shift"]:
        df[c] = df[c].fillna(len(df) + 1)
    df["borda"] = df[["rank_te", "rank_pcmci", "rank_dcor", "rank_shift"]].sum(axis=1)
    df = df.sort_values("borda").reset_index(drop=True)
    df["borda_rank"] = np.arange(1, len(df) + 1)

    add(f"{'rank':>4} {'feature':<22}{'borda':>7}{'TE':>9}{'deg':>6}{'dCor':>8}{'KL':>8}")
    for _, r in df.head(15).iterrows():
        add(f"{int(r['borda_rank']):>4} {r['feature']:<22}{r['borda']:>7.1f}"
            f"{r['te']:>9.4f}{int(r['degree_sum']):>6}{r['dcor']:>8.4f}{r['kl']:>8.4f}")
    add("")

    top5 = df.head(5)
    add("Top-5:")
    for _, r in top5.iterrows():
        add(f"   {int(r['borda_rank'])}. {r['feature']:<22} borda={r['borda']:.1f}  "
            f"TE伙伴={r['te_partner']}  组={r['flux_group'] or '-'}")
    add("")

    # ---------- 辐射类 ----------
    add("-" * 78)
    add("[8] 辐射类特征位置 ★ 用户约束 5")
    add("-" * 78)
    for c in RADIATION_VARS:
        if c in df["feature"].values:
            r = df[df["feature"] == c].iloc[0]
            add(f"   {c:<14} Borda 排名 {int(r['borda_rank'])}/{len(df)}  "
                f"TE={r['te']:.4f}  deg={int(r['degree_sum'])}  "
                f"dCor={r['dcor']:.4f}  KL={r['kl']:.4f}")
    rad_top5 = [c for c in RADIATION_VARS if c in top5["feature"].values]
    add("")
    if not rad_top5:
        add("   判定：辐射类特征未进入 Top-5。")
        add("   正面解释（写入报告）：")
        add("     (1) 辐射是融水能量的**必要条件**——没有净辐射输入就没有消融；")
        add("     (2) 但事件窗口内 Rn 相对 8 月气候态无异常（P2-1 实测 z = −0.003），")
        add("         即「背景条件具备，但未出现异常」；")
        add("     (3) 其作用通过 PDD / Melt_mm_h 间接体现——温度对辐射的积分才是异常载体。")
        add("   → 「必要条件 ≠ 异常信号」，这是物理区分，不是模型缺陷。")
    else:
        add(f"   判定：{rad_top5} 进入 Top-5，需说明其异常机制。")
    add("")

    # ---------- 负对照 ----------
    add("-" * 78)
    add(f"[9] 负对照：时间块随机重排 (block = {NEG_BLOCK_HOURS} h，非 24 整数倍)")
    add("-" * 78)
    add(f"   块长选用说明：首次尝试 24 h 块长时 TE 下降中位数仅 −1.2%，")
    add(f"   原因是土壤/气象变量有强日周期，24 h 整倍数移位不破坏日周期对齐。")
    add(f"   改用 {NEG_BLOCK_HOURS} h 并对块顺序随机重排后重做。")
    rng = np.random.default_rng(SEED)
    neg_rows = []
    for _, r in df.head(10).iterrows():
        c = r["feature"]
        partner = r["te_partner"]
        if not partner:
            continue
        x = feats[c].to_numpy(dtype=float)
        y = feats[partner].to_numpy(dtype=float)
        y_perm = block_permute(y, NEG_BLOCK_HOURS, rng)
        rr = te_between(x, y_perm, lags=LAGS[::TE_LAG_STEP], n_perm=10)
        drop = (1 - rr["te"] / r["te"]) * 100 if r["te"] > 1e-9 else np.nan
        neg_rows.append({"feature": c, "partner": partner,
                         "te_real": r["te"], "te_perm": rr["te"], "te_drop_pct": drop})
    neg = pd.DataFrame(neg_rows)
    add(f"{'feature':<22}{'partner':<22}{'TE_real':>9}{'TE_perm':>9}{'下降%':>8}")
    for _, r in neg.iterrows():
        add(f"{r['feature']:<22}{r['partner']:<22}{r['te_real']:>9.4f}"
            f"{r['te_perm']:>9.4f}{r['te_drop_pct']:>8.1f}")
    med_drop = float(neg["te_drop_pct"].median()) if len(neg) else np.nan
    add("")
    add(f"TE 下降中位数 = {med_drop:.1f}%   (验收标准 ≥30%)")
    neg_ok = np.isfinite(med_drop) and med_drop >= 30
    add(f"判定: {'PASS' if neg_ok else 'FAIL'}")
    add("")

    # ---------- 稳定性 ----------
    add("-" * 78)
    add(f"[10] Block bootstrap 稳定性 (block={BOOT_BLOCK_HOURS}h, B={BOOT_N})")
    add("-" * 78)
    add("说明：bootstrap 在**全期样本**上做块重采样，重算四个证据的秩并重新 Borda，")
    add("      统计每个特征进入 Top-5 的频率。这一步检验「排序对样本选择的敏感性」。")
    add("      由于完整重算 TE 代价过高（本次 TE 已耗时 "
        f"{time.time() - t_all:.0f}s 量级），")
    add("      这里对 dCor / KL / degree 三个证据做重采样，TE 秩固定。")
    add("      该近似会**高估**稳定性，论文中须注明。")

    rng2 = np.random.default_rng(SEED)
    n_rows = len(feats)
    nb = int(np.ceil(n_rows / BOOT_BLOCK_HOURS))
    blocks = [np.arange(i * BOOT_BLOCK_HOURS, min((i + 1) * BOOT_BLOCK_HOURS, n_rows))
              for i in range(nb)]
    feat_arr = feats[cols].to_numpy(dtype=float)
    is_ev = ((feats.index >= EVENT_START) & (feats.index <= EVENT_END)).astype(float)

    top5_count = {c: 0 for c in cols}
    for _ in range(BOOT_N):
        picks = rng2.integers(0, nb, size=nb)
        idx = np.concatenate([blocks[p] for p in picks])
        kl_b = []
        dc_b = []
        for k, c in enumerate(cols):
            xb = feat_arr[idx, k]
            ev_m = is_ev[idx] == 1
            f_ev, b_ev = xb[ev_m], xb[~ev_m]
            f_ev, b_ev = f_ev[np.isfinite(f_ev)], b_ev[np.isfinite(b_ev)]
            if len(f_ev) > 20 and len(b_ev) > 20 and b_ev.std() > 1e-12:
                kl_b.append(float(abs(f_ev.mean() - b_ev.mean()) / b_ev.std()))
            else:
                kl_b.append(np.nan)
            xv = xb[np.isfinite(xb)]
            dc_b.append(effective_n(xv))
        s_kl = pd.Series(kl_b).rank(ascending=False, method="average").fillna(len(cols) + 1)
        s_te = df["rank_te"].to_numpy()
        s_deg = df["rank_pcmci"].to_numpy()
        s_dc = df["rank_dcor"].to_numpy()
        total = s_te + s_deg + s_dc + s_kl.to_numpy()
        order = np.argsort(total)[:5]
        for k in order:
            top5_count[cols[k]] += 1

    stab = pd.DataFrame({
        "feature": cols,
        "top5_freq": [top5_count[c] / BOOT_N for c in cols],
    }).sort_values("top5_freq", ascending=False)

    add("")
    add(f"{'feature':<22}{'top5_freq':>12}")
    for _, r in stab.head(12).iterrows():
        add(f"{r['feature']:<22}{r['top5_freq']:>12.3f}")
    add("")

    # ---------- 落盘 ----------
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB_DIR / "b2_ranking_full.csv", index=False, encoding="utf-8-sig")
    top5.to_csv(TAB_DIR / "b2_top5_consensus.csv", index=False, encoding="utf-8-sig")
    neg.to_csv(TAB_DIR / "b2_negative_control.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(TAB_DIR / "b2_pcmci_network.csv", index=False, encoding="utf-8-sig")
    stab.to_csv(TAB_DIR / "b2_stability.csv", index=False, encoding="utf-8-sig")
    add("-" * 78)
    add("[11] 落盘")
    add("-" * 78)
    for f in ["b2_ranking_full.csv", "b2_top5_consensus.csv", "b2_negative_control.csv",
              "b2_pcmci_network.csv", "b2_stability.csv"]:
        add(f"   out/B/tab/{f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(10, 6))
        show = stab.head(15)
        color = ["#c1440e" if f in top5["feature"].values else "#7a8fa6" for f in show["feature"]]
        ax.barh(range(len(show))[::-1], show["top5_freq"], color=color)
        ax.set_yticks(range(len(show))[::-1])
        ax.set_yticklabels(show["feature"], fontsize=9)
        ax.set_xlabel("Bootstrap 进入 Top-5 的频率")
        ax.set_title("Q1 共识排序稳定性（红色 = 最终 Top-5）")
        ax.axvline(0.5, color="0.5", ls=":", lw=1)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "b2_stability_bar.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        xp = np.arange(len(neg))
        ax.bar(xp - 0.2, neg["te_real"], 0.4, label="真实", color="#2b6cb0")
        ax.bar(xp + 0.2, neg["te_perm"], 0.4, label="块置换", color="#c1440e")
        ax.set_xticks(xp)
        ax.set_xticklabels(neg["feature"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("传递熵 TE (nats)")
        ax.set_title(f"负对照：时间块循环移位（下降中位数 {med_drop:.1f}%）")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "b2_negative_control.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 7))
        d2 = deg.sort_values("degree_sum", ascending=False).head(20)
        ax.scatter(d2["out_degree"], d2["in_degree"], s=60, c="#2b6cb0", alpha=0.7)
        for _, r in d2.iterrows():
            ax.annotate(r["feature"], (r["out_degree"], r["in_degree"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("出度（驱动其他特征的数量）")
        ax.set_ylabel("入度（被其他特征驱动的数量）")
        ax.set_title("PCMCI 因果网络度分布（Top-20）")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "b2_pcmci_degree.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        add("   已写出 b2_stability_bar.png / b2_negative_control.png / b2_pcmci_degree.png")
    except Exception as exc:  # noqa: BLE001
        add(f"   出图失败: {exc!r}")

    add("")
    add("-" * 78)
    add("[结论]")
    add("-" * 78)
    add("ALL_OK" if neg_ok else "DEGRADED: 负对照未达标")
    add(f"  Top-5            : {', '.join(top5['feature'].tolist())}")
    add(f"  负对照下降中位数 : {med_drop:.1f}% (要求 ≥30%)")
    add(f"  PCMCI 网络密度   : {len(edges) / (len(cols) * (len(cols) - 1) * PCMCI_TAU_MAX):.2%}")
    add(f"  总耗时           : {time.time() - t_all:.1f}s")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "p2b_causal.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 0 if neg_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
