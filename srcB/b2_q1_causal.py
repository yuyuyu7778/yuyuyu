"""Q1：客观响应锚点上的四证据排序、稳定性与 24 h 负对照。

响应变量不使用人工综合指数。按本轮确认，TE/PCMCI+ 的连续响应是
``A_t = dTheta_dt_0.1m = VWC_0.1m(t) - VWC_0.1m(t-1)``，即实测浅层
土壤含水量的逐时变化。窗口指示只用于 dCor/尾依赖；KL/马氏距离直接比较：

* 前兆窗：2026-08-19 00:00 至 2026-08-26 10:00，含首尾，共 179 h；
* 背景窗：2021--2025 每年相同的 179 h，不跨年构造滞后。

不能沿用旧的窗口二值响应做 TE：在每个独立窗口内 A_t=A_{t-tau}，
故 I(A_t;X_{t-tau}|A_{t-tau})=0。连续响应是为消除该数学退化而调整的口径，
不是事件灾害过程的因果证明。双变量 PCMCI+ 仍可能受未观测混杂影响。

四类证据：
1. KSG/Frenzel--Pompe kNN 条件互信息 TE，k=5，逐时滞后 1--24；
   条件局部置换检验，另给全滞后 maxT 校正；仅在显著滞后中取最大 TE。
2. 双变量 PCMCI+/ParCorr，alpha=0.05，六个独立时段共同估计；
   强度为显著 X(t-tau)->A(t) 边的最大 |MCI|。tigramite 不可用时，
   回退为三阶符号化 TE + 滞后偏相关，并在结果中明确标记。
3. 精确一维 dCor（Fenwick 树 O(n log n)）与 u>0.9 经验 copula 尾依赖。
4. 50 箱 KL(P_event||P_background) 与一维马氏距离 |mu_e-mu_b|/sigma_b。

Borda 使用 TE、PCMCI、(dCor,lambda_U)、(KL,Mahalanobis) 四组秩。
500 次、72 h 块 bootstrap 中，dCor/尾依赖/KL/马氏距离均真正重算；
TE 使用固定原邻域的逐点贡献重采样，PCMCI 使用同滞后偏相关代理，
这两项是计算近似，不能写成“每次完整重跑 KSG/PCMCI+”。

负对照把特征值相对共同时间轴整体循环移动 24 h（不是仅 roll 时间列），
再重算所有证据。24 h 移位保留日周期，因此没有先验理由保证指标下降；
不达标会明确输出 FAIL，不得改用其他块长来追求通过。

正式用法：python srcB/b2_q1_causal.py
函数自检：python srcB/b2_q1_causal.py --check-only
小样本测试必须指定独立输出目录，例如：
  python srcB/b2_q1_causal.py --features Rn_Wm2,PDD_hourly --bootstrap 5 \
      --te-perms 9 --output-root out/B/smoke_b2

AI 辅助声明：Claude Code（Anthropic，2026-10-24）辅助生成；统计口径及
上述近似须在论文中如实披露。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from scipy.special import digamma

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import SEED  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "B" / "clean" / "features_hourly.csv"
DEFAULT_OUT = ROOT / "out" / "B"
RESPONSE = "dTheta_dt_0.1m"
EVENT_START = pd.Timestamp("2026-08-19 00:00")
EVENT_END = pd.Timestamp("2026-08-26 10:00")
BACKGROUND_YEARS = (2021, 2022, 2023, 2024, 2025)
LAGS = tuple(range(1, 25))
KSG_K = 5
TE_PERMUTATIONS = 99
PCMCI_ALPHA = 0.05
BOOTSTRAP_N = 500
BOOT_BLOCK = 72
NEGATIVE_SHIFT = 24
TAIL_Q = 0.9
KL_BINS = 50


@dataclass
class TEData:
    """一个特征的全部滞后样本与 KSG 局部贡献缓存。"""

    observed: np.ndarray
    p_lag: np.ndarray
    p_max_t: float
    null_max: np.ndarray
    local: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]
    n_eff: int
    raw_lag: int
    raw_value: float
    selected_lag: int | None
    selected_value: float


def windows_of(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame, bool]]:
    """返回六个独立连续时段，第三项标记是否为 2026 前兆窗。"""
    expected = pd.date_range(EVENT_START, EVENT_END, freq="h")
    result = []
    for year in (*BACKGROUND_YEARS, 2026):
        start = EVENT_START.replace(year=year)
        end = EVENT_END.replace(year=year)
        part = frame.loc[start:end]
        if not part.index.equals(pd.date_range(start, end, freq="h")):
            raise ValueError(f"{year} 窗口不是完整逐时 179 点: {part.index.min()}--{part.index.max()}, n={len(part)}")
        result.append((str(year), part, year == 2026))
    return result


def _group_triples(windows, feature: str, lag: int, response: str = RESPONSE):
    """构造 (A_t,X_{t-lag},A_{t-lag})，segment 防止跨年邻居与置换。"""
    xs, ys, zs, groups, positions = [], [], [], [], []
    for group, (_, part, _) in enumerate(windows):
        x = part[feature].to_numpy(float)
        y = part[response].to_numpy(float)
        pos = np.arange(lag, len(part))
        keep = np.isfinite(x[pos - lag]) & np.isfinite(y[pos]) & np.isfinite(y[pos - lag])
        pos = pos[keep]
        xs.append(x[pos - lag])
        ys.append(y[pos])
        zs.append(y[pos - lag])
        groups.append(np.full(len(pos), group, dtype=int))
        positions.append(pos - lag)
    return tuple(np.concatenate(a) for a in (xs, ys, zs, groups, positions))


def _rank_columns(a: np.ndarray) -> np.ndarray:
    return np.column_stack([stats.rankdata(a[:, j], method="average") for j in range(a.shape[1])])


def ksg_cmi(x: np.ndarray, y: np.ndarray, z: np.ndarray, groups: np.ndarray,
            k: int = KSG_K, seed: int = SEED, return_local: bool = False):
    """Frenzel--Pompe 条件 KSG，邻居严格限制在各自连续时段内。

    估计 I(X;Y|Z)=psi(k)-mean[psi(n_XZ)+psi(n_YZ)-psi(n_Z)]。
    使用 Chebyshev 距离、k 邻域半径的严格开球及微小固定种子 jitter 处理并列。
    """
    x, y, z, groups = (np.asarray(v) for v in (x, y, z, groups))
    if len(x) != len(y) or len(x) != len(z) or len(x) != len(groups):
        raise ValueError("KSG 输入长度不一致")
    rng = np.random.default_rng(seed)
    local = {}
    contributions = []
    for group in np.unique(groups):
        take = np.flatnonzero(groups == group)
        if len(take) <= k + 2:
            continue
        a = _rank_columns(np.column_stack([x[take], y[take], z[take]]))
        scale = a.std(axis=0)
        if np.any(scale < 1e-12):
            continue
        a = a + rng.normal(0.0, 1e-7 * scale, a.shape)
        xyz_tree = cKDTree(a)
        eps = xyz_tree.query(a, k=k + 1, p=np.inf)[0][:, -1]
        eps = np.maximum(eps * (1 - 1e-10), 1e-12)
        nxz = cKDTree(a[:, [0, 2]]).query_ball_point(a[:, [0, 2]], eps, p=np.inf, return_length=True) - 1
        nyz = cKDTree(a[:, [1, 2]]).query_ball_point(a[:, [1, 2]], eps, p=np.inf, return_length=True) - 1
        nz = cKDTree(a[:, [2]]).query_ball_point(a[:, [2]], eps, p=np.inf, return_length=True) - 1
        if np.any(np.minimum.reduce([nxz, nyz, nz]) < 1):
            continue
        c = digamma(nxz) + digamma(nyz) - digamma(nz)
        contributions.append(c)
        if return_local:
            local[int(group)] = (take.astype(int), c)  # take: 段内位置索引
    if not contributions:
        value = np.nan
    else:
        value = float(digamma(k) - np.mean(np.concatenate(contributions)))
    return (value, local) if return_local else value


def restricted_permutation(x: np.ndarray, z: np.ndarray, groups: np.ndarray,
                           rng: np.random.Generator, neighbors: int = 5) -> np.ndarray:
    """每个连续时段独立循环移位；同一 b 在各滞后共用零实现。"""
    out = np.empty(len(x), dtype=float)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        shift = int(rng.integers(1, len(idx))) if len(idx) > 1 else 0
        out[idx] = np.roll(x[idx], shift)
    return out


def te_for_feature(windows, feature: str, n_perm: int, seed: int) -> TEData:
    """扫描 1--24 h；逐滞后条件置换及跨滞后 maxT 校正。"""
    observed, local = [], {}
    triples = {}
    for lag in LAGS:
        x, y, z, group, pos = _group_triples(windows, feature, lag)
        triples[lag] = (x, y, z, group, pos)
        value, contributions = ksg_cmi(x, z, y, group, seed=seed + lag, return_local=True)
        observed.append(value)
        # ksg_cmi 返回的 idx 是段内三元组数组的局部位置；直接保留即可。
        local[lag] = {g: (np.asarray(idx, dtype=int), c)
                      for g, (idx, c) in contributions.items()}
    observed = np.asarray(observed, dtype=float)
    null = np.full((n_perm, len(LAGS)), np.nan)
    # 同一 b 的所有滞后共用一次移位，才构成合法的 maxT 零分布。
    for b in range(n_perm):
        shifts = {g: int(np.random.default_rng(seed + 100003 + b).integers(1, 179))
                  for g in range(len(windows))}
        for lag_i, lag in enumerate(LAGS):
            x, y, z, group, _ = triples[lag]
            if len(x) < 30 or not np.isfinite(observed[lag_i]):
                continue
            xp = np.empty(len(x))
            for g, shift in shifts.items():
                take = group == g
                if take.sum() > 1:
                    xp[take] = np.roll(x[take], shift)
                else:
                    xp[take] = x[take]
            null[b, lag_i] = ksg_cmi(xp, z, y, group, seed=seed + 7919 + b)
    valid_lags = np.isfinite(observed) & np.isfinite(null).all(axis=0)
    p_lag = np.full(len(LAGS), np.nan)
    p_lag[valid_lags] = ((1 + np.sum(null[:, valid_lags] >= observed[None, valid_lags], axis=0))
                         / (n_perm + 1))
    valid_max = np.isfinite(null).all(axis=1)
    null_max = np.full(n_perm, np.nan)
    null_max[valid_max] = np.max(null[valid_max][:, valid_lags], axis=1) if valid_lags.any() else np.nan
    finite = np.isfinite(observed)
    if not finite.any():
        raise ValueError(f"{feature}: 所有滞后均不可估计")
    raw_i = int(np.nanargmax(observed))
    raw_value = float(observed[raw_i])
    p_max_t = float((1 + np.sum(null_max >= raw_value)) / (np.isfinite(null_max).sum() + 1))
    significant = np.flatnonzero(finite & (p_lag < 0.05))
    if len(significant):
        selected_i = int(significant[np.argmax(observed[significant])])
        selected_lag, selected_value = LAGS[selected_i], float(observed[selected_i])
    else:
        selected_lag, selected_value = None, np.nan
    n_eff = min(len(triples[lag][0]) for lag in LAGS)
    return TEData(observed, p_lag, p_max_t, null_max, local, n_eff,
                  LAGS[raw_i], raw_value, selected_lag, selected_value)


def pcmci_for_feature(windows, feature: str):
    """六独立时段上的双变量 PCMCI+；返回最强显著 X->A 边与边表。"""
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    datasets, masks = {}, {}
    for name, part, _ in windows:
        a = part[[feature, RESPONSE]].to_numpy(float)
        mask = ~np.isfinite(a)
        datasets[name] = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        masks[name] = mask
    data = pp.DataFrame(datasets, mask=masks, analysis_mode="multiple")
    model = PCMCI(dataframe=data, cond_ind_test=ParCorr(), verbosity=0)
    result = model.run_pcmciplus(tau_min=1, tau_max=24, pc_alpha=PCMCI_ALPHA,
                                 max_conds_dim=2, max_combinations=2)
    graph, values, pvals = result["graph"], result["val_matrix"], result["p_matrix"]
    significant_lags = [tau for tau in LAGS
                        if graph[0, 1, tau] == "-->" and pvals[0, 1, tau] < PCMCI_ALPHA]
    if significant_lags:
        lag = max(significant_lags, key=lambda tau: abs(values[0, 1, tau]))
        strength = float(abs(values[0, 1, lag]))
        p = float(pvals[0, 1, lag])
    else:
        lag, strength, p = None, 0.0, np.nan
    edges = [{"source": feature, "target": RESPONSE, "lag_h": tau,
              "mci": values[0, 1, tau], "p_value": pvals[0, 1, tau],
              "method": "PCMCI+/ParCorr"} for tau in significant_lags]
    return {"pcmci_strength": strength, "pcmci_raw_strength": strength,
            "pcmci_lag": lag, "pcmci_p": p,
            "pcmci_significant": bool(significant_lags)}, edges


def partial_corr_segments(windows, feature: str, lag: int, indices=None):
    """逐段 r(A_t,X_lag|A_lag)，按 Fisher z 与自由度加权合并。"""
    estimates, weights = [], []
    for seg, (_, part, _) in enumerate(windows):
        x = part[feature].to_numpy(float)
        y = part[RESPONSE].to_numpy(float)
        pos = np.arange(lag, len(part)) if indices is None else indices[seg]
        pos = pos[(pos >= lag) & (pos < len(part))]
        a = np.column_stack([y[pos], x[pos - lag], y[pos - lag]])
        a = a[np.isfinite(a).all(axis=1)]
        if len(a) < 12 or np.any(a.std(axis=0) < 1e-14):
            continue
        corr = np.corrcoef(a, rowvar=False)
        den = np.sqrt((1 - corr[0, 2] ** 2) * (1 - corr[1, 2] ** 2))
        r = (corr[0, 1] - corr[0, 2] * corr[1, 2]) / den if den > 1e-14 else np.nan
        if np.isfinite(r) and abs(r) < 1:
            estimates.append(np.arctanh(np.clip(r, -0.999999, 0.999999)))
            weights.append(len(a) - 4)
    if not estimates or sum(weights) <= 0:
        return np.nan, 1.0, 0
    z = np.average(estimates, weights=weights)
    r = float(np.tanh(z))
    p = float(2 * stats.norm.sf(abs(z) * np.sqrt(sum(weights))))
    return r, p, int(sum(weights))


def symbolic_te(windows, feature: str, lag: int, indices=None):
    """回退用三阶符号化 TE，秩分箱；不跨时段创建转移。"""
    total, weight = 0.0, 0
    for seg, (_, part, _) in enumerate(windows):
        x = part[feature].to_numpy(float)
        y = part[RESPONSE].to_numpy(float)
        pos = np.arange(lag, len(part)) if indices is None else indices[seg]
        pos = pos[(pos >= lag) & (pos < len(part))]
        a = np.column_stack([y[pos], x[pos - lag], y[pos - lag]])
        a = a[np.isfinite(a).all(axis=1)]
        if len(a) < 30:
            continue
        # 分箱仅依据本段样本，避免不同年份量纲漂移影响符号状态。
        edges = [np.unique(np.quantile(a[:, j], np.linspace(0, 1, 5))) for j in range(3)]
        if min(map(len, edges)) < 3:
            continue
        states = [np.clip(np.searchsorted(e[1:-1], a[:, j]), 0, 3) for j, e in enumerate(edges)]
        joint = (states[0] * 16 + states[1] * 4 + states[2]).astype(int)
        # H(Yt,Xlag,Ylag)+H(Ylag)-H(Xlag,Ylag)-H(Yt,Ylag)。
        def entropy(codes):
            count = np.bincount(codes, minlength=int(codes.max()) + 1).astype(float)
            p = count[count > 0] / len(codes)
            return -np.sum(p * np.log(p))
        value = (entropy(joint) + entropy(states[2])
                 - entropy(states[1] * 4 + states[2]) - entropy(states[0] * 4 + states[2]))
        total += max(value, 0.0) * len(a)
        weight += len(a)
    return total / weight if weight else np.nan


def _absolute_moments(x: np.ndarray):
    """sum_j |x_i-x_j| 及其全局均值，O(n log n)。"""
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    prefix = np.concatenate([[0.0], np.cumsum(sx)])
    k = np.arange(len(x))
    row_sorted = k * sx - prefix[k] + (prefix[-1] - prefix[k + 1]) - (len(x) - 1 - k) * sx
    row = np.empty(len(x))
    row[order] = row_sorted
    return row, row.mean() / len(x)


def _fenwick_sums(tree: np.ndarray, i: int):
    total = np.zeros(tree.shape[0])
    while i:
        total += tree[:, i - 1]
        i -= i & -i
    return total


def _fenwick_add(tree: np.ndarray, i: int, value: np.ndarray):
    i += 1
    while i <= tree.shape[1]:
        tree[:, i - 1] += value
        i += i & -i


def dcov2(x: np.ndarray, y: np.ndarray) -> float:
    """一维偏差距离协方差，Fenwick 树精确计算交叉项，O(n log n)。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 4:
        return np.nan
    ax, mean_ax = _absolute_moments(x)
    ay, mean_ay = _absolute_moments(y)
    order = np.argsort(x, kind="mergesort")
    sx, sy = x[order], y[order]
    levels = np.unique(y)
    tree = np.zeros((4, len(levels)))
    prefix_x = np.concatenate([[0.0], np.cumsum(sx)])
    prefix_y = prefix_xy = 0.0
    cross_sum = 0.0
    for i in range(n):
        rank = int(np.searchsorted(levels, sy[i], side="right"))
        count, sum_x, sum_y, sum_xy = _fenwick_sums(tree, rank)
        dx_le = sx[i] * count - sum_x
        dxy_le = sx[i] * sum_y - sum_xy
        dx_all = i * sx[i] - prefix_x[i]
        dxy_all = sx[i] * prefix_y - prefix_xy
        cross_sum += (sy[i] * dx_le - dxy_le
                      + (dxy_all - dxy_le) - sy[i] * (dx_all - dx_le))
        _fenwick_add(tree, rank - 1, np.array([1.0, sx[i], sy[i], sx[i] * sy[i]]))
        prefix_y += sy[i]
        prefix_xy += sx[i] * sy[i]
    value = 2 * cross_sum / n**2 - 2 * np.dot(ax, ay) / n**3 + mean_ax * mean_ay
    return float(value)


def dcor(x: np.ndarray, y: np.ndarray) -> float:
    """偏差距离相关；常数变量或无效分母返回 NaN，不伪造零相关。"""
    if len(x) < 4:
        return np.nan
    xy, xx, yy = dcov2(x, y), dcov2(x, x), dcov2(y, y)
    den = np.sqrt(max(xx, 0.0) * max(yy, 0.0))
    return float(np.sqrt(max(xy, 0.0) / den)) if den > 1e-24 else np.nan


def tail_dependence(x: np.ndarray, anchor: np.ndarray, q: float = TAIL_Q):
    """lambda_U=P(V>q|U>q)，秩采用平均秩处理二值锚点并列。"""
    n = len(x)
    if n < 20:
        return np.nan, np.nan, 0
    u = stats.rankdata(x, method="average") / (n + 1)
    v = stats.rankdata(anchor, method="average") / (n + 1)
    ux, vx = u > q, v > q
    denom = int(ux.sum())
    lam = float(np.sum(ux & vx) / denom) if denom else np.nan
    event_rate = float(np.mean(anchor[ux])) if denom else np.nan
    return lam, event_rate, denom


def distribution_shift(event: np.ndarray, background: np.ndarray):
    """50 箱 KL(event||background) 与标量马氏距离；Jeffreys 0.5 平滑。"""
    event = event[np.isfinite(event)]
    background = background[np.isfinite(background)]
    out = {"kl": np.nan, "mahalanobis": np.nan, "n_event": len(event), "n_background": len(background)}
    if len(event) < 20 or len(background) < 20:
        return out
    lo, hi = min(event.min(), background.min()), max(event.max(), background.max())
    sd = background.std(ddof=1)
    if hi <= lo or sd <= 1e-14:
        return out
    bins = np.linspace(lo, hi, KL_BINS + 1)
    p = np.histogram(event, bins)[0].astype(float) + 0.5
    q = np.histogram(background, bins)[0].astype(float) + 0.5
    p, q = p / p.sum(), q / q.sum()
    out["kl"] = float(np.sum(p * np.log(p / q)))
    out["mahalanobis"] = float(abs(event.mean() - background.mean()) / sd)
    return out


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """从同一时段内有放回抽取固定长度时间块，不跨年拼接。"""
    starts = np.arange(0, n, block)
    chosen = rng.choice(starts, size=int(np.ceil(n / block)), replace=True)
    return np.concatenate([np.arange(s, min(s + block, n)) for s in chosen])[:n]


def eligible_features(frame: pd.DataFrame):
    """保留所有特征行；只将响应自身与完全无信息列排除出 Borda。"""
    reasons = {}
    for col in frame.columns:
        if col == RESPONSE:
            reasons[col] = "连续响应自身：不作为自身原因参与 Borda"
        elif frame[col].notna().sum() < 30 or frame[col].nunique(dropna=True) < 2:
            reasons[col] = "有效样本不足或为常数"
    return [c for c in frame.columns if c not in reasons], reasons


def compute_evidence(frame: pd.DataFrame, features: list[str], n_perm: int,
                     seed: int, with_te: bool = True):
    """计算全部证据；负对照复用时执行同一套函数。"""
    windows = windows_of(frame)
    rows, lag_rows, edges, caches = [], [], [], {}
    pcmci_available = True
    pcmci_import_error = ""
    try:
        import tigramite  # noqa: F401
    except ImportError as exc:
        pcmci_available = False
        pcmci_import_error = repr(exc)
    for i, feature in enumerate(features, 1):
        row = {"feature": feature}
        if with_te:
            try:
                te = te_for_feature(windows, feature, n_perm, seed + i * 1000003)
                caches[feature] = te
                row.update(te_raw=te.raw_value, te_raw_lag=te.raw_lag,
                           te=te.selected_value, te_lag=te.selected_lag,
                           te_p=te.p_lag[te.raw_lag - 1], te_maxT_p=te.p_max_t,
                           te_n=te.n_eff,
                           te_score=max(te.raw_value, 0.0) if te.p_max_t < 0.05 else 0.0)
                for j, lag in enumerate(LAGS):
                    lag_rows.append({"feature": feature, "lag_h": lag, "te": te.observed[j],
                                     "p_lag": te.p_lag[j], "maxT_p_at_raw_best": te.p_max_t})
            except (ValueError, FloatingPointError) as exc:
                row.update(te_raw=np.nan, te_raw_lag=np.nan, te=np.nan, te_lag=np.nan,
                           te_p=np.nan, te_maxT_p=np.nan, te_n=0, te_score=np.nan,
                           te_error=str(exc))
        if feature != RESPONSE and frame[feature].notna().sum() >= 30:
            if pcmci_available:
                try:
                    p, e = pcmci_for_feature(windows, feature)
                    edges.extend(e)
                    row.update(p)
                    row["pcmci_method"] = "PCMCI+/ParCorr"
                except Exception as exc:  # 失败立即启用显式回退，不用缺失掩盖。
                    fallback_error = repr(exc)
                    row.update(pcmci_strength=np.nan, pcmci_raw_strength=np.nan,
                               pcmci_lag=np.nan, pcmci_p=np.nan,
                               pcmci_significant=False, pcmci_method="symbolic_TE+partial_corr",
                               pcmci_error=fallback_error)
                    candidates = []
                    for lag in LAGS:
                        r, p, n = partial_corr_segments(windows, feature, lag)
                        sym = symbolic_te(windows, feature, lag)
                        candidates.append((lag, r, p, n, sym))
                    valid = [c for c in candidates if np.isfinite(c[1]) and c[2] < PCMCI_ALPHA]
                    if valid:
                        best = max(valid, key=lambda c: abs(c[1]))
                        row.update(pcmci_strength=abs(best[1]), pcmci_raw_strength=abs(best[1]),
                                   pcmci_lag=best[0], pcmci_p=best[2], pcmci_significant=True,
                                   symbolic_te=best[4])
                        edges.append({"source": feature, "target": RESPONSE, "lag_h": best[0],
                                      "mci": best[1], "p_value": best[2], "method": "partial_corr_fallback"})
            else:
                candidates = []
                for lag in LAGS:
                    r, p, n = partial_corr_segments(windows, feature, lag)
                    sym = symbolic_te(windows, feature, lag)
                    candidates.append((lag, r, p, n, sym))
                valid = [c for c in candidates if np.isfinite(c[1])]
                if valid:
                    best = max(valid, key=lambda c: abs(c[1]))
                    row.update(pcmci_strength=abs(best[1]) if best[2] < 0.05 else 0.0,
                               pcmci_raw_strength=abs(best[1]), pcmci_lag=best[0], pcmci_p=best[2],
                               pcmci_significant=best[2] < 0.05, pcmci_method="symbolic_TE+partial_corr",
                               symbolic_te=best[4])
                    edges.append({"source": feature, "target": RESPONSE, "lag_h": best[0],
                                  "mci": best[1], "p_value": best[2], "method": "partial_corr_fallback"})
        else:
            row["pcmci_method"] = "not_applicable_response" if feature == RESPONSE else "insufficient_data"
        stacked = pd.concat([part[[feature]] for _, part, _ in windows])
        labels = np.concatenate([np.full(len(part), is_event, dtype=float) for _, part, is_event in windows])
        x = stacked[feature].to_numpy(float)
        keep = np.isfinite(x)
        row["dcor"] = dcor(x[keep], labels[keep]) if keep.sum() >= 20 else np.nan
        lam, event_rate, tail_n = tail_dependence(x[keep], labels[keep])
        row.update(lambda_u=lam, tail_event_rate=event_rate, tail_n=tail_n, dependence_n=int(keep.sum()))
        event = windows[-1][1][feature].to_numpy(float)
        background = np.concatenate([part[feature].to_numpy(float) for _, part, is_event in windows if not is_event])
        row.update(distribution_shift(event, background))
        rows.append(row)
        print(f"  证据 {i:2d}/{len(features)} {feature}", flush=True)
    return pd.DataFrame(rows).set_index("feature"), pd.DataFrame(lag_rows), pd.DataFrame(edges), caches


def add_ranks(result: pd.DataFrame) -> pd.DataFrame:
    """四组等权 Borda；原始证据全部保留，秩只用于排序。"""
    out = result.copy()
    out["dependence_score"] = out[["dcor", "lambda_u"]].rank(pct=True).mean(axis=1)
    out["shift_score"] = out[["kl", "mahalanobis"]].rank(pct=True).mean(axis=1)
    metrics = {"te": "te_score", "pcmci": "pcmci_strength", "dependence": "dependence_score", "shift": "shift_score"}
    n = len(out)
    for name, col in metrics.items():
        out[f"rank_{name}"] = out[col].rank(ascending=False, method="average", na_option="bottom")
    out["borda_rank_sum"] = out[[f"rank_{x}" for x in metrics]].sum(axis=1)
    out["borda_score"] = (n + 1) * 4 - out["borda_rank_sum"]
    out["borda_rank"] = out["borda_rank_sum"].rank(method="min").astype(int)
    return out.sort_values(["borda_rank_sum", "te_raw"], ascending=[True, False])


def bootstrap_stability(frame: pd.DataFrame, features: list[str], caches: dict,
                        main: pd.DataFrame, n_boot: int, seed: int):
    """72 h 块自助法；精确重算依赖/偏移，近似重算 TE/PCMCI（见模块说明）。"""
    windows = windows_of(frame)
    rng = np.random.default_rng(seed)
    counts = pd.Series(0, index=features, dtype=float)
    valid_boot = 0
    for b in range(n_boot):
        indices = [block_indices(len(part), BOOT_BLOCK, rng) for _, part, _ in windows]
        rows = []
        for feature in features:
            row = {"feature": feature}
            if feature in caches:
                te = caches[feature]
                lag_scores = []
                for lag in LAGS:
                    values, weights = [], []
                    for seg, local in te.local[lag].items():
                        idx, contribution = local
                        # idx 是段内三元组局部位置；bootstrap 重采样后用 indices[seg] 定位。
                        # 若该位置未被抽到则跳过；权重按抽中次数。
                        boot_idx = indices[seg]
                        # 把段内 idx 映射到 bootstrap 抽样后的贡献：取 idx 与 boot_idx 的交集计数。
                        count = np.zeros(len(contribution), dtype=int)
                        unique, cnt = np.unique(boot_idx, return_counts=True)
                        in_range = unique < len(contribution)
                        count[unique[in_range]] = cnt[in_range]
                        mask = count > 0
                        if mask.any():
                            values.append(np.average(contribution[mask], weights=count[mask]))
                            weights.append(int(count[mask].sum()))
                    lag_scores.append(np.average(values, weights=weights) if values else np.nan)
                lag_scores = np.asarray(lag_scores)
                best = np.nanmax(lag_scores) if np.isfinite(lag_scores).any() else np.nan
                threshold = np.nanquantile(te.null_max, 0.95) if np.isfinite(te.null_max).any() else np.inf
                row["te_score"] = max(best, 0.0) if best > threshold else 0.0
                row["te_raw"] = best
            else:
                row["te_raw"], row["te_score"] = np.nan, np.nan
            if feature != RESPONSE:
                main_lag = main.loc[feature, "pcmci_lag"]
                lag = int(main_lag) if main_lag is not None and not (isinstance(main_lag, float) and np.isnan(main_lag)) else 1
                if "partial_corr" in str(main.loc[feature, "pcmci_method"]):
                    candidates = [(abs(r), p) for r, p, _ in [partial_corr_segments(windows, feature, l, indices) for l in LAGS]]
                    score, p = max(candidates, key=lambda v: v[0] if np.isfinite(v[0]) else -np.inf)
                    row["pcmci_strength"] = score if p < 0.05 else 0.0
                else:
                    r, p, _ = partial_corr_segments(windows, feature, lag, indices)
                    row["pcmci_strength"] = abs(r) if np.isfinite(r) and p < 0.05 else 0.0
            else:
                row["pcmci_strength"] = np.nan
            x_parts, labels = [], []
            for seg, (_, part, is_event) in enumerate(windows):
                x_parts.append(part[feature].to_numpy(float)[indices[seg]])
                labels.append(np.full(len(indices[seg]), is_event, dtype=float))
            x, label = np.concatenate(x_parts), np.concatenate(labels)
            keep = np.isfinite(x)
            row["dcor"] = dcor(x[keep], label[keep]) if keep.sum() >= 20 else np.nan
            row["lambda_u"] = tail_dependence(x[keep], label[keep])[0]
            event = windows[-1][1][feature].to_numpy(float)[indices[-1]]
            background = np.concatenate([part[feature].to_numpy(float)[indices[seg]]
                                         for seg, (_, part, is_event) in enumerate(windows) if not is_event])
            row.update(distribution_shift(event, background))
            rows.append(row)
        ranked = add_ranks(pd.DataFrame(rows).set_index("feature"))
        counts.loc[ranked.head(5).index] += 1
        valid_boot += 1
        if (b + 1) % 25 == 0 or b == 0:
            print(f"  bootstrap {b + 1}/{n_boot}", flush=True)
    stability = (counts / valid_boot).rename("top5_frequency").to_frame()
    stability["n_bootstrap"] = valid_boot
    stability["block_hours"] = BOOT_BLOCK
    stability["method"] = "exact dCor/tail/KL/Mahalanobis; frozen-neighborhood TE; lag-partial-corr PCMCI proxy"
    return stability.sort_values("top5_frequency", ascending=False)


def negative_control(frame: pd.DataFrame, features: list[str], main: pd.DataFrame,
                     n_perm: int, seed: int):
    """全时间轴 24 h 循环移位特征值，响应与窗口标签不动，重算全部证据。"""
    n = len(frame)
    shifted = frame.iloc[np.roll(np.arange(n), NEGATIVE_SHIFT)].copy()
    shifted.index = frame.index
    perm, _, _, _ = compute_evidence(shifted, features, n_perm, seed + 88000001, with_te=True)
    rows = []
    comparisons = {"te": ("te_raw", "TE raw max"), "pcmci": ("pcmci_raw_strength", "PCMCI |MCI|"),
                   "dcor": ("dcor", "dCor"), "lambda_u": ("lambda_u", "lambda_U"),
                   "kl": ("kl", "KL"), "mahalanobis": ("mahalanobis", "Mahalanobis")}
    for feature in features:
        row = {"feature": feature}
        for name, (col, _) in comparisons.items():
            real, null = main.loc[feature, col], perm.loc[feature, col]
            row[f"{name}_real"], row[f"{name}_shifted"] = real, null
            row[f"{name}_drop_pct"] = 100 * (real - null) / abs(real) if np.isfinite(real) and abs(real) > 1e-14 else np.nan
        rows.append(row)
    summary = []
    for name, (_, label) in comparisons.items():
        pair = pd.DataFrame({"real": main[comparisons[name][0]], "shifted": perm[comparisons[name][0]]}).dropna()
        # 排除真实值/置换值同时为零的不可辨识比较，不把 0/0 当 100% 下降。
        pair = pair[(pair.real.abs() > 1e-14) | (pair.shifted.abs() > 1e-14)]
        delta = pair.real - pair.shifted
        if len(pair) >= 6 and np.any(delta.abs() > 1e-14):
            test = stats.wilcoxon(pair.real, pair.shifted, alternative="greater", zero_method="wilcox")
            p = float(test.pvalue)
        else:
            p = np.nan
        drop = 100 * delta / pair.real.abs().replace(0, np.nan)
        summary.append({"metric": name, "label": label, "n": len(pair),
                        "real_median": pair.real.median(), "shifted_median": pair.shifted.median(),
                        "median_drop_pct": drop.median(), "wilcoxon_p_greater": p,
                        "significant_decrease": bool(np.isfinite(p) and p < 0.05 and delta.median() > 0)})
    return pd.DataFrame(rows).set_index("feature"), pd.DataFrame(summary), perm


def configure_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "text.color": "#0b0b0b",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    return plt


def plot_network(edges: pd.DataFrame, path: Path, method: str):
    """只画显著 X->连续响应边；无边时明确写无，不画装饰性网络。"""
    import networkx as nx
    plt = configure_plotting()
    fig, ax = plt.subplots(figsize=(11, 8))
    blue, orange = "#2a78d6", "#eb6834"
    if edges.empty:
        ax.text(0.5, 0.55, "α=0.05 下未检出 X → 响应的滞后边", ha="center", va="center", fontsize=14)
        ax.text(0.5, 0.45, "空图是检验结果，不以相关性补边", ha="center", color="#52514e")
        ax.axis("off")
    else:
        show = edges.reindex(edges.mci.abs().sort_values(ascending=False).index).head(20)
        graph = nx.DiGraph()
        graph.add_node(RESPONSE)
        for _, edge in show.iterrows():
            graph.add_edge(edge.source, edge.target, lag=edge.lag_h, weight=abs(edge.mci))
        pos = nx.spring_layout(graph, seed=SEED, k=1.4)
        nx.draw_networkx_nodes(graph, pos, nodelist=[n for n in graph if n != RESPONSE],
                               node_color=blue, node_size=900, edgecolors="#fcfcfb", linewidths=2, ax=ax)
        nx.draw_networkx_nodes(graph, pos, nodelist=[RESPONSE], node_color=orange,
                               node_size=1800, edgecolors="#fcfcfb", linewidths=2, ax=ax)
        widths = [1 + 5 * abs(graph[u][v]["weight"]) / max(show.mci.abs().max(), 1e-12) for u, v in graph.edges]
        nx.draw_networkx_edges(graph, pos, width=widths, edge_color="#52514e",
                               arrows=True, arrowsize=18, connectionstyle="arc3,rad=0.06", ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
        nx.draw_networkx_edge_labels(graph, pos, {(u, v): f"{d['lag']}h" for u, v, d in graph.edges(data=True)},
                                     font_size=7, ax=ax)
        ax.set_axis_off()
    ax.set_title(f"X → 逐时浅层土壤含水变化（{method}，α=0.05）\n最多显示 20 条最强边；探索性双变量结果", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stability(stability: pd.DataFrame, top5: list[str], path: Path, n_boot: int):
    plt = configure_plotting()
    show = stability.head(15).sort_values("top5_frequency")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#2a78d6" if f in top5 else "#c3c2b7" for f in show.index]
    bars = ax.barh(show.index, show.top5_frequency, color=colors, height=0.66)
    for bar, value in zip(bars, show.top5_frequency):
        ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    ax.set(xlim=(0, 1.12), xlabel=f"进入 Borda Top-5 的频率（{n_boot} 次，72 h 块）",
           title="Q1 共识排序稳定性\n蓝色为全样本 Top-5；TE/PCMCI 为重采样近似")
    ax.grid(axis="x", color="#e1e0d9", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_negative(detail: pd.DataFrame, summary: pd.DataFrame, path: Path):
    plt = configure_plotting()
    metrics = [("te", "TE raw max"), ("pcmci", "PCMCI |MCI|"),
               ("dcor", "dCor"), ("kl", "KL")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        real, null = detail[f"{metric}_real"], detail[f"{metric}_shifted"]
        keep = real.notna() & null.notna()
        for a, b in zip(real[keep], null[keep]):
            ax.plot([0, 1], [a, b], color="#c3c2b7", lw=0.6, alpha=0.65, zorder=1)
        ax.scatter(np.zeros(keep.sum()), real[keep], s=20, color="#2a78d6", label="真实", zorder=2)
        ax.scatter(np.ones(keep.sum()), null[keep], s=20, color="#eb6834", marker="s", label="24 h 移位", zorder=2)
        info = summary.loc[summary.metric == metric].iloc[0]
        p = info.wilcoxon_p_greater
        ax.set_title(f"{label}：中位下降 {info.median_drop_pct:.1f}%，p={p:.3g}" if np.isfinite(p)
                     else f"{label}：有效配对不足")
        ax.set_xticks([0, 1], ["真实", "24 h 移位"])
        ax.grid(axis="y", color="#e1e0d9", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle("负对照：特征值相对共同时间轴循环移动 24 h\n逐特征灰线配对；显著性为单侧 Wilcoxon", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def self_check(frame: pd.DataFrame):
    """不依赖正式输出的统计与窗口回归测试。"""
    windows = windows_of(frame)
    assert len(windows) == 6 and all(len(p) == 179 for _, p, _ in windows)
    assert sum(not e for _, _, e in windows) == 5
    rng = np.random.default_rng(SEED)
    for n in (30, 80):
        x, y = rng.normal(size=n), rng.normal(size=n)
        a = np.abs(x[:, None] - x[None, :])
        b = np.abs(y[:, None] - y[None, :])
        ac = a - a.mean(0)[None, :] - a.mean(1)[:, None] + a.mean()
        bc = b - b.mean(0)[None, :] - b.mean(1)[:, None] + b.mean()
        expected = np.mean(ac * bc)
        assert np.isclose(dcov2(x, y), expected, atol=1e-10), (n, dcov2(x, y), expected)
        expected_r = np.sqrt(max(expected, 0) / np.sqrt(np.mean(ac * ac) * np.mean(bc * bc)))
        assert np.isclose(dcor(x, y), expected_r, atol=1e-10)
    # KSG 独立均匀样本应接近零；完全函数关系应大于零。
    x, y, z = rng.normal(size=(3, 240))
    independent = ksg_cmi(x, y, z, np.zeros(len(x), int), seed=SEED)
    dependent = ksg_cmi(x, x + 0.01 * y, z, np.zeros(len(x), int), seed=SEED)
    assert dependent > independent, (dependent, independent)
    block = block_indices(179, 72, rng)
    assert len(block) == 179 and np.all((block >= 0) & (block < 179))
    shifted = frame.iloc[np.roll(np.arange(len(frame)), 24)]
    assert len(shifted) == len(frame)
    print("SELF_CHECK: PASS（窗口、dCor 矩阵对照、KSG 基本行为、72h 块、24h 移位）")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="只运行统计自检，不写文件")
    parser.add_argument("--features", help="逗号分隔特征子集；必须同时改 --output-root")
    parser.add_argument("--te-perms", type=int, default=TE_PERMUTATIONS)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.te_perms < 1 or args.bootstrap < 1:
        raise ValueError("置换与 bootstrap 次数必须为正")
    started = time.time()
    frame = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    if RESPONSE not in frame:
        raise KeyError(f"缺少连续响应 {RESPONSE}")
    self_check(frame)
    if args.check_only:
        return 0
    features, exclusions = eligible_features(frame)
    if args.features:
        requested = args.features.split(",")
        missing = set(requested) - set(frame.columns)
        if missing:
            raise KeyError(f"未知特征 {sorted(missing)}")
        if args.output_root.resolve() == DEFAULT_OUT.resolve():
            raise ValueError("特征子集测试必须指定非正式 --output-root，禁止覆盖正式结果")
        features = requested
    out = args.output_root
    tab, fig, log_dir = out / "tab", out / "fig", out / "log"
    for directory in (tab, fig, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    log = ["=" * 78, "P2 Q1 客观连续响应的四证据排序", "=" * 78,
           f"运行时间戳: {datetime.now().astimezone().isoformat()}",
           f"Python {sys.version.split()[0]} / numpy {np.__version__} / pandas {pd.__version__}",
           f"SEED={SEED}; TE perms={args.te_perms}; bootstrap={args.bootstrap}",
           f"连续响应: {RESPONSE}（实测 VWC_0.1m 的逐时差分，不是人工综合指数）",
           "原二值窗口响应在分段内满足 A_t=A_lag，TE 条件互信息恒零；故不作为 TE/PCMCI 响应。",
           "前兆窗=2026-08-19 00:00--2026-08-26 10:00（含首尾 179h）",
           "背景窗=2021--2025 每年同历日同时段，每年 179h，总计 895h；不跨年构造滞后。",
           "TE: KSG k=5，1--24h；条件局部置换；逐滞后 p<0.05 取最大，并另给 maxT 校正。",
           "PCMCI+: 双变量 ParCorr，alpha=0.05；仅计显著 X->A 边。",
           "dCor/lambda: 六窗内 X 与前兆窗指示；lambda_U=P(V>.9|U>.9)。",
           "KL: 50 箱 P_event||P_background；马氏距离为一维 |均值差|/背景标准差。",
           "因果限制: 双变量探索性统计，不排除未观测混杂，不等于灾害因果识别。"]
    try:
        tigramite_version = importlib.metadata.version("tigramite")
        log.append(f"tigramite={tigramite_version}（已安装，未执行安装）")
    except importlib.metadata.PackageNotFoundError:
        log.append("tigramite 未安装：回退为三阶符号化 TE + 偏相关")
    for col, reason in exclusions.items():
        log.append(f"Borda 排除 {col}: {reason}")
    log.append(f"参与计算特征 {len(features)}；主表保留全部输入特征。")

    print("[1/5] 全样本四证据", flush=True)
    evidence, lag_table, edges, caches = compute_evidence(frame, features, args.te_perms, SEED)
    ranked = add_ranks(evidence)
    top5 = ranked.head(5).copy()
    print("[2/5] 72h block bootstrap", flush=True)
    stability = bootstrap_stability(frame, features, caches, evidence, args.bootstrap, SEED + 500)
    top5 = top5.join(stability[["top5_frequency"]])
    print("[3/5] 24h 循环移位负对照（重算全部证据）", flush=True)
    negative, negative_summary, perm_evidence = negative_control(frame, features, evidence, args.te_perms, SEED)
    negative_pass = bool(negative_summary.significant_decrease.all())
    print("[4/5] 写表与图", flush=True)

    ranked.insert(0, "analysis_role", np.where(ranked.index == RESPONSE, "response", "candidate"))
    ranked["seed"] = SEED
    ranked.to_csv(tab / "b2_ranking_full.csv", encoding="utf-8-sig")
    top5.to_csv(tab / "b2_top5_consensus.csv", encoding="utf-8-sig")
    stability.to_csv(tab / "b2_stability.csv", encoding="utf-8-sig")
    negative.to_csv(tab / "b2_negative_control.csv", encoding="utf-8-sig")
    negative_summary.to_csv(tab / "b2_negative_control_summary.csv", index=False, encoding="utf-8-sig")
    lag_table.to_csv(tab / "b2_te_lags.csv", index=False, encoding="utf-8-sig")
    if edges.empty:
        edges = pd.DataFrame(columns=["source", "target", "lag_h", "mci", "p_value", "method"])
    edges.to_csv(tab / "b2_pcmci_network.csv", index=False, encoding="utf-8-sig")
    plot_network(edges, fig / "b2_causal_network.png", str(evidence.pcmci_method.dropna().iloc[0]))
    plot_stability(stability, top5.index.tolist(), fig / "b2_stability_bar.png", args.bootstrap)
    plot_negative(negative, negative_summary, fig / "b2_negative_control.png")

    log.extend(["", "Top-5（Borda 四组等权秩，越小越好）:"])
    for feature, row in top5.iterrows():
        log.append(f"  {int(row.borda_rank)}. {feature}: rank_sum={row.borda_rank_sum:.2f}, "
                   f"TE={row.te_raw:.5g}, PCMCI={row.pcmci_raw_strength:.5g}, "
                   f"dCor={row.dcor:.5g}, KL={row.kl:.5g}, boot={row.top5_frequency:.3f}")
    log.extend(["", "24h 负对照（单侧 Wilcoxon，real > shifted）:"])
    for _, row in negative_summary.iterrows():
        log.append(f"  {row.metric}: n={row.n}, real={row.real_median:.6g}, shifted={row.shifted_median:.6g}, "
                   f"median_drop={row.median_drop_pct:.3f}%, p={row.wilcoxon_p_greater:.6g}, "
                   f"{'PASS' if row.significant_decrease else 'FAIL'}")
    log.append("注意：24h 移位保留日周期；窗内循环移位还不改变 KL/马氏距离的边缘分布，"
               "因此负对照失败不能通过改块长掩盖。")
    log.extend(["", "bootstrap 口径：依赖与分布偏移逐次重算；TE 固定原邻域仅重采样局部贡献；"
                "PCMCI 用主分析所选滞后上的偏相关代理。后两项为近似，稳定性可能偏乐观。",
                "", f"NEGATIVE_CONTROL: {'PASS' if negative_pass else 'FAIL'}",
                "结论口径：四证据 Top-5 已生成，但仅作探索性排序；负对照失败时不作已验证因果结论。",
                f"总耗时: {time.time() - started:.1f}s",
                "输出: b2_ranking_full.csv / b2_top5_consensus.csv / b2_stability.csv / "
                "b2_negative_control.csv / b2_te_lags.csv / b2_pcmci_network.csv",
                "图: b2_causal_network.png / b2_stability_bar.png / b2_negative_control.png"])
    (log_dir / "p2b_causal.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 0 if negative_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
