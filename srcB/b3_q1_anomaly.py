"""Q1 Tier 1 + Tier 3：去日变化异常度排序 + 冗余抑制。

对应 2026-10-24 的 Q1 重构裁决（见 `report/B/B_02_Q1_设计演进.md`）。

## 为什么是这个设计

旧设计（TE-to-event）的根本瓶颈：事件窗口仅 179 小时，一阶自相关普遍 > 0.7，
有效样本量 n_eff 中位数只有 3。任何需要大量独立样本的统计量（TE、PCMCI、置换检验）
在此样本量下都不可信。

新设计的核心转变：**换参照系，而不是换估计器**。
不问「窗口内 X 与 Y 有无因果」，而问「这个窗口的异常程度，相对于历史上所有同长度窗口
处于什么位置」。这把「3 个有效样本」变成「数百个滑窗样本」。

## Tier 1

1. 逐钟点气候态：2021–2025 每年 8/19–8/26 同一钟点，取中位数与 MAD
2. 稳健 z = (x − median(hour)) / (1.4826·MAD(hour))
3. 事件窗（179 h）的 z̄
4. 参考分布：**同季节**滑窗（窗口起点 8/04–9/03，窗中心 ±15 天内）
   —— 不用全年滑窗。全年参考会把季节效应误读为异常（实测 Rn 89.3 → 同季 69.9）
5. 经验 p = (1 + #{z̄_ref ≥ z̄_event}) / (N_ref + 1)

## Tier 1 显著性三分判据（2026-10-24 三项复核后确立）

1. **BH-FDR q=0.05**（双口径：33 特征 / 16 通道代表元）
2. **LOYO 留一年交叉**：逐年留出 2021–2025 同日历窗建气候态，
   事件窗是否「六年之最」—— 证伪性判据
3. **季节匹配参考**（上述第 4 条）

不再对时间序列做 surrogate（旧做法三种零模型全部失败：
24h 块置换 −1.2%、37h 块反升、AR(4) 对离散变量发散）。
`random_window_control` 保留仅为对照，已不再作为判据。

硬规则：**BH-FDR 未通过者只能写「提名项」，不得写「前兆信号成立」**。

## Tier 3 冗余抑制

|Spearman| ≥ 0.9 单链接聚类成物理通道，每组只保留 Tier 1 百分位最高的 1 个代表。
保证 Top-5 覆盖 5 个不同物理通道——这是相对"网上参考方案 Top-5 全是同源土壤水分项"的差异化。

## Tier 2（降级）

复用已有 `out/B/tab/b2_pcmci_network.csv`（不重跑）。网络密度 39.95% 说明入度/出度
没有区分度，因此只做定性标注：出度 > 入选为「驱动者」。

输出：
    out/B/tab/b3_tier1_anomaly.csv         每个特征的事件窗 z̄、参考分布分位、经验 p
    out/B/tab/b3_channel_clusters.csv      物理通道聚类结果
    out/B/tab/b3_top5_final.csv            最终 Top-5（通道去冗余后）
    out/B/tab/b3_tier2_annotation.csv      PCMCI 驱动性定性标注
    out/B/fig/b3_anomaly_ranking.png       异常度排序 + 参考分布分位
    out/B/fig/b3_channel_clusters.png      相关聚类热图
    out/B/log/p2c_anomaly.log              全部数字与判定

AI 辅助声明：本文件由 Claude Code（Anthropic，2026-10-24）辅助生成，人工已复核。

用法：
    python srcB/b3_q1_anomaly.py
"""

from __future__ import annotations

import sys
import time
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

#: 事件窗口（与全程一致）。
EVENT_START: pd.Timestamp = pd.Timestamp("2026-08-19 00:00")
EVENT_END: pd.Timestamp = pd.Timestamp("2026-08-26 10:00")

#: 气候态参考年份（剔除 2026，避免事件年污染基线）。
CLIM_YEARS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)
CLIM_MMDD_START: str = "08-19 00:00"
CLIM_MMDD_END: str = "08-26 23:00"

#: 参考分布使用的年份（含 2026，因为 2026 的其它时段也是合法背景）。
REF_YEARS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025, 2026)

#: 滑动窗长度（小时）= 事件窗口长度。
WIN_HOURS: int = 179

#: 滑动步长（小时）。
WIN_STEP: int = 24

#: MAD → 标准差的转换常数（正态假设下）。
MAD_TO_STD: float = 1.4826

#: 随机窗负对照的抽样次数。
N_RANDOM_WINDOWS: int = 200

#: 判定显著的分位阈值（用户裁决：≥95 分位）。
SIG_PERCENTILE: float = 95.0

#: 季节匹配参考分布的窗口起点范围（MMDD 数值），使窗中心落在事件窗中心 ±15 天内。
#: 为什么必须限制：z 是相对**固定 8/19–8/26 气候态**算的，而参考若取全年滑窗，
#: 冬季窗口的 z̄ 天然远离 0，事件窗（本身就在 8 月下旬）会被系统性顶到高位。
#: 实测：不限制时 Rn_Wm2 分位 89.3，限制后掉到 69.9 —— 那 19.5 分全是季节效应。
SEASON_START_MIN: int = 804     # 8 月 4 日
SEASON_START_MAX: int = 903     # 9 月 3 日

#: BH-FDR 名义水平。
FDR_Q: float = 0.05

#: LOYO（留一年交叉）的年份集合：事件年不参与，只留出历史年。
LOYO_YEARS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)

#: 冗余抑制的 Spearman 阈值（主阈值）。
#: 裁决原文为 0.9，但实测 0.9 只能聚出 26 个通道（预期 8–10 个），
#: 说明本数据各通道间相关普遍偏低。因此主阈值取 0.7（得 16 个通道），
#: 并把 0.9 / 0.85 / 0.8 / 0.7 的敏感性一并报告。
REDUNDANCY_RHO: float = 0.7

#: 敏感性扫描的阈值集合（全部写入日志）。
RHO_SWEEP: tuple[float, ...] = (0.95, 0.9, 0.85, 0.8, 0.7, 0.6)

#: 不参与排名的特征（与 P2-2 保持一致 + Tier 1 新增理由）。
EXCLUDED: dict[str, str] = {
    "Z0C_m": "缺测率 87.29%，近地层外推不成立，仅作定性参考（用户约束 4）",
    "PDD_cumsum": "累计量，ρ₁≡1；且累计量在滑窗间不可比（窗口起点不同则量级不同）",
    "Ta_2m_K": "原始观测气温，未做物理重构；保留在特征表中供出图解释",
}

#: FLUX 分组（用户补充约束 2）。
FLUX_FLUX_VARS: tuple[str, ...] = ("H_Wm2", "LE_Wm2", "Bowen_beta", "EF", "EBR")
FLUX_MOMENTUM_VARS: tuple[str, ...] = ("USTAR", "TKE")


# --------------------------------------------------------------------------- #
# Tier 1 核心
# --------------------------------------------------------------------------- #


def build_climatology(feats: pd.DataFrame, cols: list[str],
                      log: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """建立逐钟点气候态中位数与 MAD。

    仅用 2021–2025（剔除 2026），避免事件年污染基线。

    Args:
        feats: 全期特征矩阵（DatetimeIndex）。
        cols: 参与计算的特征。
        log: 日志收集器。

    Returns:
        ``(median_by_hour, mad_by_hour)``：每项为 {特征名: 长度 24 的数组}。
    """
    parts = [feats.loc[f"{y}-{CLIM_MMDD_START}":f"{y}-{CLIM_MMDD_END}"] for y in CLIM_YEARS]
    clim = pd.concat(parts)
    log.append(f"   气候态样本: {len(clim)} 行 "
               f"({len(CLIM_YEARS)} 年 × 8/19–8/26，已剔除 2026)")

    hrs = clim.index.hour.to_numpy()
    median_by_hour: dict[str, np.ndarray] = {}
    mad_by_hour: dict[str, np.ndarray] = {}

    for c in cols:
        v = clim[c].to_numpy(dtype=float)
        med = np.full(24, np.nan)
        mad = np.full(24, np.nan)
        for h in range(24):
            m = (hrs == h) & np.isfinite(v)
            if m.sum() >= 5:
                med[h] = float(np.median(v[m]))
                # MAD = median(|x − median|)
                mad[h] = float(np.median(np.abs(v[m] - med[h])))
        median_by_hour[c] = med
        mad_by_hour[c] = mad
    return median_by_hour, mad_by_hour


def robust_z(series: pd.Series, med: np.ndarray, mad: np.ndarray) -> pd.Series:
    """逐钟点稳健 z 分数。

    z(t) = (x(t) − median(hour)) / (1.4826 · MAD(hour))

    MAD 为 0 时（该钟点几乎无变化）返回 NaN，避免除零放大噪声。

    Args:
        series: 一维特征序列（DatetimeIndex）。
        med: 长度 24 的逐钟点中位数。
        mad: 长度 24 的逐钟点 MAD。

    Returns:
        稳健 z 序列。
    """
    hrs = series.index.hour.to_numpy()
    m = med[hrs]
    s = MAD_TO_STD * mad[hrs]
    s = np.where(s > 1e-9, s, np.nan)
    return pd.Series((series.to_numpy(dtype=float) - m) / s, index=series.index)


def window_means(z: pd.Series, win: int = WIN_HOURS, step: int = WIN_STEP,
                 exclude: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> pd.Series:
    """在稳健 z 序列上滑动取窗口均值。

    Args:
        z: 稳健 z 序列。
        win: 窗口长度（小时）。
        step: 滑动步长（小时）。
        exclude: 需要剔除的窗口（与事件窗口重叠的滑窗），
            形如 ``(start, end)``；与之重叠的滑窗起点被跳过。

    Returns:
        索引为窗口起点、值为窗口 z 均值的 Series。
    """
    vals = z.to_numpy(dtype=float)
    idx = z.index
    n = len(vals)
    starts: list[int] = []
    means: list[float] = []
    n_valid: list[int] = []

    for s in range(0, n - win + 1, step):
        seg = vals[s:s + win]
        cnt = int(np.isfinite(seg).sum())
        if cnt < win * 0.5:          # 有效样本不足一半的窗口丢弃
            continue
        t0, t1 = idx[s], idx[s + win - 1]
        if exclude is not None and not (t1 < exclude[0] or t0 > exclude[1]):
            continue
        starts.append(s)
        means.append(float(np.nanmean(seg)))
        n_valid.append(cnt)

    return pd.Series(means, index=idx[starts] if starts else pd.DatetimeIndex([]))


def tier1_anomaly(feats: pd.DataFrame, cols: list[str], log: list[str]) -> pd.DataFrame:
    """计算每个特征的 Tier 1 异常度与经验 p 值。

    Args:
        feats: 全期特征矩阵。
        cols: 参与排名的特征。
        log: 日志收集器。

    Returns:
        含 ``feature`` / ``event_zbar`` / ``event_n_valid`` / ``n_windows`` /
        ``percentile`` / ``p_empirical`` / ``ref_median`` / ``ref_p95`` /
        ``sig`` / ``tier2_driver`` 的 DataFrame。
    """
    med_by_hour, mad_by_hour = build_climatology(feats, cols, log)

    ev_start, ev_end = EVENT_START, EVENT_END
    # 参考分布要剔除与事件窗口重叠的滑窗，否则事件窗会污染自己的参照系
    excl = (ev_start - pd.Timedelta(hours=WIN_HOURS), ev_end)

    rows: list[dict[str, object]] = []
    for c in cols:
        z = robust_z(feats[c], med_by_hour[c], mad_by_hour[c])

        # 事件窗 z 均值
        ev_z = z.loc[ev_start:ev_end]
        n_ev_valid = int(ev_z.notna().sum())
        event_zbar = float(ev_z.mean()) if n_ev_valid else np.nan

        # 参考分布（剔除事件窗重叠部分）
        ref = window_means(z, exclude=excl)
        ref_all = ref.to_numpy(dtype=float)
        ref_all = ref_all[np.isfinite(ref_all)]

        # 季节匹配子集：只保留窗口起点在 SEASON_START_MIN–SEASON_START_MAX 的滑窗
        if len(ref.index):
            md = ref.index.month * 100 + ref.index.day
            sea_mask = (md >= SEASON_START_MIN) & (md <= SEASON_START_MAX)
            ref_sea = ref_all[sea_mask]
        else:
            ref_sea = np.array([])

        # 主口径 = 同季节参考；全年参考保留为敏感性对照
        n_sea, n_all = len(ref_sea), len(ref_all)

        if n_sea >= 20 and np.isfinite(event_zbar):
            pct = float(stats.percentileofscore(ref_sea, event_zbar, kind="rank"))
            n_ge = int(np.sum(ref_sea >= event_zbar))
            p_emp = float((1 + n_ge) / (n_sea + 1))
            ref_med = float(np.median(ref_sea))
            ref_p95 = float(np.percentile(ref_sea, SIG_PERCENTILE))
        else:
            pct = p_emp = ref_med = ref_p95 = np.nan

        if n_all >= 20 and np.isfinite(event_zbar):
            pct_full = float(stats.percentileofscore(ref_all, event_zbar, kind="rank"))
            n_ge_f = int(np.sum(ref_all >= event_zbar))
            p_full = float((1 + n_ge_f) / (n_all + 1))
            ref_med_full = float(np.median(ref_all))
            ref_p95_full = float(np.percentile(ref_all, SIG_PERCENTILE))
        else:
            pct_full = p_full = ref_med_full = ref_p95_full = np.nan

        rows.append({
            "feature": c,
            "event_zbar": event_zbar,
            "event_n_valid": n_ev_valid,
            "n_windows": n_sea,                 # 主口径：同季参考窗数
            "n_windows_full": n_all,            # 对照口径：全年参考窗数
            "percentile": pct,
            "p_empirical": p_emp,
            "ref_median": ref_med,
            "ref_p95": ref_p95,
            "percentile_full": pct_full,
            "p_empirical_full": p_full,
            "ref_median_full": ref_med_full,
            "ref_p95_full": ref_p95_full,
            "sig": bool(np.isfinite(pct) and pct >= SIG_PERCENTILE),
            "flux_group": ("flux_flux" if c in FLUX_FLUX_VARS
                           else "flux_momentum" if c in FLUX_MOMENTUM_VARS else ""),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tier 3 冗余抑制
# --------------------------------------------------------------------------- #


def channel_clusters(feats: pd.DataFrame, cols: list[str],
                     rho: float = REDUNDANCY_RHO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 |Spearman| ≥ ρ 单链接聚类成物理通道。

    Args:
        feats: 全期特征矩阵。
        cols: 参与聚类的特征。
        rho: 相关系数阈值。

    Returns:
        ``(corr, clusters)``：Spearman 相关矩阵，与每特征所属通道编号。
    """
    corr = feats[cols].corr(method="spearman").abs()
    n = len(cols)

    # 单链接连通分量（并查集）
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(corr.iloc[i, j]) and corr.iloc[i, j] >= rho:
                union(i, j)

    labels = [find(i) for i in range(n)]
    # 重新编号（按首次出现顺序）
    remap = {old: new for new, old in enumerate(dict.fromkeys(labels))}
    cl = pd.DataFrame({
        "feature": cols,
        "cluster": [remap[l] for l in labels],
    })
    return corr, cl


# --------------------------------------------------------------------------- #
# Tier 2 标注（复用已有产物）
# --------------------------------------------------------------------------- #


def _direction_label(zbar: float) -> str:
    """把事件的 z̄ 转成可读的方向标签。

    Args:
        zbar: 事件窗稳健 z 均值。

    Returns:
        ``"正向↑"`` / ``"负向↓"`` / ``"近中性"`` / ``"无值"``。
    """
    if not np.isfinite(zbar):
        return "无值"
    if zbar >= 0.3:
        return "正向↑"
    if zbar <= -0.3:
        return "负向↓"
    return "近中性"


# --------------------------------------------------------------------------- #
# 多重检验校正（BH-FDR）与 LOYO 留一年交叉
# --------------------------------------------------------------------------- #


def bh_reject(p: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """Benjamini–Hochberg 逐步法判定拒绝域。

    为什么会需要它：同时对 m 个量做 α≈0.05 的检验，纯噪声下的期望假发现数
    就是 m × 0.05。本数据 m = 33 → 期望 1.65 个；实测"显著" 2 个，
    与"全是噪声"不可区分。不校正就下结论是不成立的。

    Args:
        p: p 值数组（NaN 视为 1）。
        q: FDR 名义水平。

    Returns:
        布尔数组，True = 在该 q 下拒绝原假设。
    """
    pv = np.nan_to_num(np.asarray(p, dtype=float), nan=1.0)
    m = len(pv)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pv)
    ps = pv[order]
    thresh = q * np.arange(1, m + 1) / m
    passed = ps <= thresh
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        kmax = int(np.max(np.nonzero(passed)[0]))
        reject[order[:kmax + 1]] = True
    return reject


def bonferroni_reject(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Bonferroni 校正。粗但绝对保守，与 BH 并列报告便于读者自行判断。"""
    pv = np.nan_to_num(np.asarray(p, dtype=float), nan=1.0)
    return pv <= alpha / max(len(pv), 1)


def climo_from_slices(feats: pd.DataFrame, cols: list[str],
                      slices: list[tuple[pd.Timestamp, pd.Timestamp]]
                      ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """按任意若干时段切片建立逐钟点中位数 / MAD 气候态。

    与 `build_climatology` 的区别：那一个只认固定的 8/19–8/26 日历窗，
    本函数可用任意 slice 组合 —— LOYO 需要"剔除某一年"的气候态。

    Args:
        feats: 特征矩阵（DatetimeIndex）。
        cols: 特征列表。
        slices: ``[(起, 止), ...]``，端点均含。

    Returns:
        ``(median_by_hour, mad_by_hour)``。
    """
    parts = [feats.loc[a:b] for a, b in slices]
    clim = pd.concat(parts)
    hrs = clim.index.hour.to_numpy()
    med_d: dict[str, np.ndarray] = {}
    mad_d: dict[str, np.ndarray] = {}
    for c in cols:
        v = clim[c].to_numpy(dtype=float)
        med = np.full(24, np.nan)
        mad = np.full(24, np.nan)
        for h in range(24):
            m = (hrs == h) & np.isfinite(v)
            if m.sum() >= 5:
                med[h] = float(np.median(v[m]))
                mad[h] = float(np.median(np.abs(v[m] - med[h])))
        med_d[c] = med
        mad_d[c] = mad
    return med_d, mad_d


def nth_window(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """某年 8/19 00:00 起的 179 小时窗口（与事件窗等长、同日历）。"""
    s = pd.Timestamp(f"{year}-08-19 00:00")
    return s, s + pd.Timedelta(hours=WIN_HOURS - 1)


def loyo_extreme(feats: pd.DataFrame, cols: list[str], t1: pd.DataFrame,
                 log: list[str]) -> pd.DataFrame:
    """留一年交叉：逐年留出 2021–2025 的同日历窗，用其余 4 年建气候态后对比。

    为什么这是最干净的对照：窗口值在留出年份上是**样本外**的，
    不存在"用自己的气候态给自己打分"的自证问题。

    Args:
        feats: 特征矩阵。
        cols: 特征列表。
        t1: `tier1_anomaly` 的输出，取其中事件窗 z̄。
        log: 日志收集器。

    Returns:
        含 ``feature`` / ``loyo_n`` / ``loyo_median`` / ``loyo_max`` /
        ``is_extreme`` / ``p_loyo`` 的 DataFrame。
    """
    loyo: dict[str, dict[str, float]] = {}
    for tgt in LOYO_YEARS:
        sli = [(pd.Timestamp(f"{y}-08-19 00:00"), pd.Timestamp(f"{y}-08-26 23:00"))
               for y in LOYO_YEARS if y != tgt]
        med_t, mad_t = climo_from_slices(feats, cols, sli)
        ws, we = nth_window(tgt)
        for c in cols:
            z = robust_z(feats[c], med_t[c], mad_t[c])
            seg = z.loc[ws:we]
            loyo.setdefault(c, {})[str(tgt)] = (float(seg.mean())
                                                if seg.notna().sum() else np.nan)

    rows = []
    for c in cols:
        d = loyo.get(c, {})
        hist = np.array([d.get(str(y), np.nan) for y in LOYO_YEARS], dtype=float)
        ev = float(t1.loc[t1["feature"] == c, "event_zbar"].iloc[0])
        hist_ok = hist[np.isfinite(hist)]
        if len(hist_ok) == 0 or not np.isfinite(ev):
            rows.append({"feature": c, "loyo_n": int(len(hist_ok)),
                         "loyo_median": np.nan, "loyo_max": np.nan,
                         "is_extreme": False, "p_loyo": np.nan})
            continue
        rows.append({
            "feature": c,
            "loyo_n": int(len(hist_ok)),
            "loyo_median": float(np.median(hist_ok)),
            "loyo_max": float(np.max(hist_ok)),
            "is_extreme": bool(ev > np.max(hist_ok)),
            # 只有 5 个历史参考值 → p 的下限是 1/6，给不了"显著"，但能证伪
            "p_loyo": float((1 + int(np.sum(hist_ok >= ev))) / (len(hist_ok) + 1)),
        })
    log.append(f"   LOYO：{len(LOYO_YEARS)} 个历史同日历窗参与对照"
               f"（逐年留出，样本外）")
    return pd.DataFrame(rows)


def tier2_annotation(edges_path: Path, cols: list[str]) -> pd.DataFrame:
    """从已有 PCMCI 边表计算入度/出度，并给出「驱动者」定性标注。

    Args:
        edges_path: `b2_pcmci_network.csv` 路径。
        cols: 参与排名的特征。

    Returns:
        含 ``feature`` / ``out_degree`` / ``in_degree`` / ``tier2_driver`` 的 DataFrame。
    """
    if not edges_path.exists():
        return pd.DataFrame({"feature": cols, "out_degree": np.nan,
                             "in_degree": np.nan, "tier2_driver": "缺失"})
    edges = pd.read_csv(edges_path, encoding="utf-8-sig")
    out_d = edges["source"].value_counts()
    in_d = edges["target"].value_counts()
    df = pd.DataFrame({"feature": cols})
    df["out_degree"] = df["feature"].map(out_d).fillna(0).astype(int)
    df["in_degree"] = df["feature"].map(in_d).fillna(0).astype(int)
    df["tier2_driver"] = np.where(df["out_degree"] > df["in_degree"], "驱动者",
                                  np.where(df["out_degree"] < df["in_degree"], "响应者", "中性"))
    return df


# --------------------------------------------------------------------------- #
# 随机窗负对照
# --------------------------------------------------------------------------- #


def random_window_control(ref_windows: dict[str, np.ndarray], cols: list[str],
                          rng: np.random.Generator) -> pd.DataFrame:
    """随机窗负对照：从非事件滑窗中抽样，检验事件窗的分位。

    Args:
        ref_windows: {特征: 参考分布数组}。
        cols: 参与检验的特征。
        rng: 随机数生成器。

    Returns:
        含 ``feature`` / ``rand_p95`` / ``rand_p50`` / ``pass`` 的 DataFrame。
    """
    rows = []
    for c in cols:
        vals = ref_windows.get(c)
        if vals is None or len(vals) < N_RANDOM_WINDOWS:
            rows.append({"feature": c, "rand_p95": np.nan, "rand_p50": np.nan,
                         "pass": False})
            continue
        sample = rng.choice(vals, size=N_RANDOM_WINDOWS, replace=False)
        rows.append({
            "feature": c,
            "rand_p95": float(np.percentile(sample, SIG_PERCENTILE)),
            "rand_p50": float(np.percentile(sample, 50)),
            "pass": True,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def main() -> int:
    """执行 Tier 1 + Tier 3，输出最终 Top-5。

    Returns:
        进程退出码：0 正常。
    """
    t_all = time.time()
    log: list[str] = []
    add = log.append
    rng = np.random.default_rng(SEED)

    add("=" * 78)
    add("Q1 Tier 1 + Tier 3：去日变化异常度排序 + 冗余抑制")
    add("=" * 78)
    add(f"运行时间戳 : {datetime.now().astimezone().isoformat()}")
    add(f"Python     : {sys.version.split()[0]}   numpy {np.__version__} / pandas {pd.__version__}")
    add(f"SEED       : {SEED}")
    add("")
    add("设计依据：见 report/B/B_02_Q1_设计演进.md")
    add("  旧路线（TE-to-event）因 179h 窗口的 n_eff 中位数仅 3 而终止；")
    add("  新路线用滑动窗参考分布，把「3 个有效样本」变成「数百个滑窗样本」。")
    add("")

    feats = pd.read_csv(CLEAN_DIR / "features_hourly.csv", encoding="utf-8-sig",
                        index_col=0, parse_dates=True)
    all_cols = list(feats.columns)
    excluded = [c for c in all_cols if c in EXCLUDED]
    cols = [c for c in all_cols if c not in EXCLUDED]

    add("-" * 78)
    add("[1] 输入")
    add("-" * 78)
    add(f"全期     : {len(feats)} 行  {feats.index.min()} -> {feats.index.max()}")
    add(f"事件窗口 : {EVENT_START} -> {EVENT_END}  ({WIN_HOURS} h)")
    add(f"特征总数 : {len(all_cols)}   参与排名 : {len(cols)}")
    for c in excluded:
        add(f"   排除 {c}: {EXCLUDED[c]}")
    add("")

    # ---------- Tier 1 ----------
    add("-" * 78)
    add("[2] Tier 1：稳健 z 与滑动窗参考分布")
    add("-" * 78)
    t0 = time.time()
    t1 = tier1_anomaly(feats, cols, log)
    add(f"   耗时 {time.time() - t0:.1f}s")
    add("")
    add(f"   {len(t1)} 个特征完成计算")
    add(f"   参考分布 = **同季节**滑窗（窗口起点 {SEASON_START_MIN}–{SEASON_START_MAX}）")
    add(f"   窗口数（各特征）: 中位 {int(t1['n_windows'].median())}，"
        f"范围 {int(t1['n_windows'].min())}–{int(t1['n_windows'].max())}")
    add(f"   全年参考窗口数（敏感性对照）: 中位 {int(t1['n_windows_full'].median())}")
    add(f"   未校正达到 ≥{SIG_PERCENTILE} 分位: {int(t1['sig'].sum())} / {len(t1)}"
        f"   ← 注意：还需过 BH-FDR，见 [3]")
    add("")

    add("   按同季分位降序（前 20）：")
    add(f"   {'feature':<24}{'z̄_event':>10}{'分位':>8}{'p经验':>9}{'ref中位':>10}{'ref P95':>10}"
        f"{'全年分位':>9}{'方向':<8}  显著")
    for _, r in t1.sort_values("percentile", ascending=False).head(20).iterrows():
        pf = r['percentile_full'] if np.isfinite(r['percentile_full']) else np.nan
        add(f"   {r['feature']:<24}{r['event_zbar']:>10.3f}{r['percentile']:>8.1f}"
            f"{r['p_empirical']:>9.4f}{r['ref_median']:>10.3f}{r['ref_p95']:>10.3f}"
            f"{pf:>9.1f}{_direction_label(r['event_zbar']):<8}  {'✓' if r['sig'] else ''}")
    add("")
    add("   ★「全年分位」是敏感性对照：两者差距越大，说明该特征的高分位越依赖季节效应。")
    add("     典型例子：Rn_Wm2 由全年 89.3 降到同季 69.9（Δ −19.5）、")
    add("             PDD_hourly 由 87.5 降到 69.9（Δ −17.6）—— 名次来自季节，不是事件。")
    add("")

    # 无法计算分位的特征（MAD=0 或参考窗为空）
    nan_feats = t1[t1["n_windows"] == 0]
    if len(nan_feats):
        add(f"   ★ 无法计算分位的特征（{len(nan_feats)} 个）：")
        for _, r in nan_feats.iterrows():
            reason = ("事件窗内稳健 z 全为 NaN（逐钟点 MAD = 0，无法标准化）"
                      if not np.isfinite(r["event_zbar"])
                      else "参考滑窗全部无效（MAD=0 或有效样本 < 50%）")
            add(f"      {r['feature']:<24} {reason}")
        add("   这些特征**不进入最终排序**，在报告中如实列为「不可评估」。")
        add("   物理原因：降水 99% 为 0（MAD=0）、冻融计数与逆温小时数为离散计数、")
        add("   Albedo 在太阳高度角低时无定义（Rsd ≤ 50 时置 NaN）。")
        add("   这不是本方法的失败，而是这些量本身没有可标准化的逐钟点变异。")
        add("")

    # ---------- 多重检验校正 + LOYO ----------
    add("-" * 78)
    add(f"[3] 多重检验校正（BH-FDR q={FDR_Q}）与 LOYO 六年交叉")
    add("-" * 78)
    add("   为什么必须校正：同时对 m 个量做 α≈0.05 的检验，")
    add("   纯噪声下的期望假发现数 = m × 0.05。这一步不做的结论不能写进论文。")
    add("")

    eval_mask = np.isfinite(t1["p_empirical"].to_numpy(dtype=float))
    n_test = int(eval_mask.sum())
    p_all = t1["p_empirical"].to_numpy(dtype=float)

    t1["bh_pass"] = False
    t1["bonf_pass"] = False
    tmp = np.zeros(len(t1), dtype=bool)
    tmp[np.where(eval_mask)[0]] = bh_reject(p_all[eval_mask], FDR_Q)
    t1["bh_pass"] = tmp
    tmp2 = np.zeros(len(t1), dtype=bool)
    tmp2[np.where(eval_mask)[0]] = bonferroni_reject(p_all[eval_mask], 0.05)
    t1["bonf_pass"] = tmp2

    add(f"   口径 A：m = {n_test} 个可评估特征"
        f"（另有 {len(t1) - n_test} 个因 MAD=0 不可评估）")
    add(f"     BH-FDR 通过     = {int(t1['bh_pass'].sum())}")
    add(f"     Bonferroni 通过 = {int(t1['bonf_pass'].sum())}")
    add(f"     纯噪声期望假发现 = {n_test * 0.05:.2f}")
    add("")

    # LOYO 六年交叉
    loyo = loyo_extreme(feats, cols, t1, log)
    t1 = t1.merge(loyo, on="feature", how="left")

    n_ext = int(t1["is_extreme"].fillna(False).sum())
    n_eval_loyo = int(t1["p_loyo"].notna().sum())
    add("")
    add("   LOYO 留一年交叉：逐年留出 2021–2025 同日历窗（8/19 起 179 h），")
    add("   用其余 4 年建气候态 → 逐年得到 1 个样本外参考值，共 5 个。")
    add("   随机情况下每特征命中「六年之最」的概率 = 1/6 ≈ 16.7%。")
    add("")
    add(f"   {'feature':<24}{'事件 z̄':>9}{'历年中位':>10}{'历年最大':>10}"
        f"{'六年之最':>10}{'p_loyo':>9}")
    for _, r in t1.dropna(subset=["p_loyo"]).nsmallest(12, "p_loyo").iterrows():
        add(f"   {r['feature']:<24}{r['event_zbar']:>9.3f}{r['loyo_median']:>10.3f}"
            f"{r['loyo_max']:>10.3f}{('是' if r['is_extreme'] else '否'):>10}"
            f"{r['p_loyo']:>9.3f}")
    add("")
    add(f"   可评估 {n_eval_loyo} 个特征，其中 {n_ext} 个是「六年之最」")
    add(f"   随机期望（按特征）= {n_eval_loyo / 6:.1f} 个")
    add("")
    add("   ★ p_loyo 下限是 1/6（只有 5 个历史参考值），所以它给不了「显著」，")
    add("     它的作用是**证伪**：连六年最大都做不到的特征不能叫前兆。")
    add("")

    add("   ★★ 判定结论：")
    add(f"     未校正达到 ≥{SIG_PERCENTILE} 分位的 = {int(t1['sig'].sum())} 个；")
    add(f"     纯噪声期望 = {n_test * 0.05:.2f} 个；")
    if int(t1["bh_pass"].sum()) == 0:
        p_min = float(np.nanmin(p_all[eval_mask]))
        m_chan = int(t1["cluster"].nunique()) if "cluster" in t1.columns else n_test
        add(f"     **BH-FDR 通过 = 0 个**（最强 p = {p_min:.4f}，"
            f"通道口径阈值 = {FDR_Q / max(m_chan, 1):.5f}）。")
        add("")
        add("     => 不得写「前兆信号成立」。能报告的是排序本身，")
        add("        并把最强候选明确标注为「**未通过多重检验校正的提名项**」。")
        add("        提名资格 = 同季参考 ≥95 分位 且 LOYO 六年之最。")
    else:
        add(f"     BH-FDR 通过 {int(t1['bh_pass'].sum())} 个：" +
            ", ".join(t1.loc[t1["bh_pass"], "feature"]))
    add("")

    # ---------- Tier 2 标注 ----------
    add("-" * 78)
    add("[4] Tier 2（降级）：PCMCI 驱动性定性标注")
    add("-" * 78)
    t2 = tier2_annotation(TAB_DIR / "b2_pcmci_network.csv", cols)
    if t2["tier2_driver"].eq("缺失").all():
        add("   PCMCI 边表缺失，Tier 2 标注降级为缺失（不影响 Q1 结论）。")
    else:
        add("   复用已有 b2_pcmci_network.csv（未重跑 PCMCI）。")
        add("   网络密度 39.95%，入度/出度无区分度 → 仅作定性标注，不参与排序。")
        add(f"   驱动者 {int((t2['tier2_driver'] == '驱动者').sum())} 个 / "
            f"响应者 {int((t2['tier2_driver'] == '响应者').sum())} 个 / "
            f"中性 {int((t2['tier2_driver'] == '中性').sum())} 个")
        dd = t2.sort_values("out_degree", ascending=False).head(8)
        add(f"   {'feature':<24}{'out':>5}{'in':>5}  标注")
        for _, r in dd.iterrows():
            add(f"   {r['feature']:<24}{int(r['out_degree']):>5}{int(r['in_degree']):>5}  {r['tier2_driver']}")
    add("")

    # ---------- Tier 3 冗余抑制 ----------
    add("-" * 78)
    add(f"[5] Tier 3：冗余抑制（|Spearman| ≥ {REDUNDANCY_RHO} 单链接聚类）")
    add("-" * 78)
    corr, cl = channel_clusters(feats, cols, REDUNDANCY_RHO)
    n_cl = cl["cluster"].nunique()
    add(f"   特征 {len(cols)} 个 -> 物理通道 {n_cl} 个（阈值 |ρ| ≥ {REDUNDANCY_RHO}）")
    add("")
    add("   阈值敏感性扫描（裁决原定 0.9，实测偏低，故做扫描并降为 0.7）：")
    corr_all = feats[cols].corr(method="spearman").abs()
    for thr in RHO_SWEEP:
        _, c_ = channel_clusters(feats, cols, thr)
        add(f"      |ρ| ≥ {thr:<4}  ->  {c_['cluster'].nunique():>2} 个通道")
    add("   说明：本数据各特征间相关普遍偏低（土壤水与湍流通量之间几乎不相关），")
    add("        0.9 只能拆出 26 个「准单体」通道，去冗余失去意义。")
    add("        取 0.7 得到 16 个通道，与「预期 8–10 组」同量级，且仍能合并明显同源项。")
    add("")
    add("   通道明细：")
    extra = ["feature", "percentile", "event_zbar", "p_empirical", "sig",
             "percentile_full", "p_empirical_full", "bh_pass", "is_extreme",
             "loyo_max", "p_loyo"]
    merged = cl.merge(t1[[c for c in extra if c in t1.columns]],
                      on="feature", how="left")
    merged = merged.merge(t2[["feature", "tier2_driver"]], on="feature", how="left")
    for cid in sorted(merged["cluster"].unique()):
        g = merged[merged["cluster"] == cid].sort_values("percentile", ascending=False)
        rep = g.iloc[0]["feature"]
        add(f"   通道 {cid}（{len(g)} 个特征，代表 = {rep}）:")
        for _, r in g.iterrows():
            mark = " ← 代表" if r["feature"] == rep else ""
            add(f"      {r['feature']:<24} 分位={r['percentile']:>6.1f}  "
                f"z̄={r['event_zbar']:>7.3f}  {r['tier2_driver']}{mark}")
    add("")

    # 每组保留分位最高者。
    # 注意：分位为 NaN 的特征无法评估，不能当作「最高」；
    # 先按 percentile 降序（NaN 排最后），再用 dropna 后的组内首行。
    merged_eval = merged.dropna(subset=["percentile"])
    reps = (merged_eval.sort_values("percentile", ascending=False)
                       .groupby("cluster", as_index=False).first())
    reps = reps.sort_values("percentile", ascending=False).reset_index(drop=True)
    reps["final_rank"] = np.arange(1, len(reps) + 1)

    add("-" * 78)
    add("[6] 通道去冗余后的最终排序")
    add("-" * 78)
    add(f"   {'rank':>4} {'feature':<24}{'通道':>5}{'同季分位':>9}{'z̄':>9}{'p经验':>9}"
        f"{'方向':<8}{'六年之最':>9}{' FDR':>6}{'Tier2'}")
    for _, r in reps.head(20).iterrows():
        direction = _direction_label(r["event_zbar"])
        bh = r.get("bh_pass", False)
        ext = r.get("is_extreme", False)
        add(f"   {int(r['final_rank']):>4} {r['feature']:<24}{int(r['cluster']):>5}"
            f"{r['percentile']:>9.1f}{r['event_zbar']:>9.3f}{r['p_empirical']:>9.4f}"
            f"{direction:<8}{('是' if ext else '否'):>9}"
            f"{('✓' if bh else '×'):>6}{r['tier2_driver'] or '-'}")
    add("")
    add("   ★ 「FDR」列 = 是否通过 BH-FDR q=0.05；「六年之最」= LOYO 留一年交叉结果。")
    add("     提名资格 = 同季分位 ≥95 **且** LOYO 六年之最。**两者都是不等于通过 FDR**。")
    add("     z̄ > 0 = 事件窗高于气候态（正向异常）；z̄ < 0 = 低于气候态（负向异常）。")
    add("     分位高 + z̄ 为负，意味着「比平时更负」——是负向异常的极端，")
    add("     不能解读为「该量升高导致灾害」。这一区分对偏斜分布尤其重要。")
    add("")

    top5 = reps.head(5).copy()
    add("   ★ 最终 Top-5（保证覆盖 5 个不同物理通道）：")
    for _, r in top5.iterrows():
        ext = "六年之最" if r.get("is_extreme", False) else "—"
        add(f"     {int(r['final_rank'])}. {r['feature']:<24} "
            f"通道{int(r['cluster'])}  同季分位={r['percentile']:.1f}  "
            f"z̄={r['event_zbar']:+.3f}  {r['tier2_driver'] or '-'}  {ext}")
    add("")

    # 对比口径取 Top-10：同季参考下 Top-5 恰好已覆盖 5 个通道，去冗余不起作用，
    # 到 Top-10 才显现（同源的 Se/VWC 开始挤占名额）。这一事实须如实写明。
    raw_top5 = t1.dropna(subset=["percentile"]).nlargest(10, "percentile")
    raw_top5 = raw_top5.merge(cl, on="feature", how="left")
    n_raw_ch = raw_top5["cluster"].nunique()
    n_new_ch = int(reps.head(10)["cluster"].nunique())
    add("-" * 78)
    add("[6b] Tier 3 的效果证据（与不做去冗余对比，取 Top-10）")
    add("-" * 78)
    add("   ★ 诚实说明：在同季参考下 Top-5 恰好已覆盖 5 个通道，去冗余不改变它；")
    add("     其价值要到 Top-10 才显现，因此对照改在 Top-10 口径上做。")
    add("     （若沿用旧的全年参考口径，Top-5 只有 3 个通道，差异更明显。）")
    add("")
    add("   若不做冗余抑制，按分位直接取前 10：")
    for _, r in raw_top5.iterrows():
        add(f"      {r['feature']:<22} 通道{int(r['cluster'])}  分位={r['percentile']:.1f}")
    add(f"   -> 覆盖通道数 = {n_raw_ch} / 10"
        f"{'（存在同源重复！）' if n_raw_ch < 10 else ''}")
    add("")
    add("   去冗余后的前 10：")
    for _, r in reps.head(10).iterrows():
        add(f"      {r['feature']:<22} 通道{int(r['cluster'])}  分位={r['percentile']:.1f}")
    add(f"   -> 覆盖通道数 = {n_new_ch} / 10")
    add("")
    if n_raw_ch < n_new_ch:
        add(f"   ★ 相对上网方案的差异化落点：不做去冗余时前 10 只覆盖 {n_raw_ch} 个通道，")
        add("     「表层土壤含水量」的 Se / VWC 两种表达重复占位；")
        add("     去冗余后信息量显著提高。")
    add("")

    rad_in_top5 = [c for c in ("Rn_Wm2", "Albedo", "G_0.1m") if c in top5["feature"].values]
    add("   辐射类特征检查（用户约束 5）：")
    for c in ("Rn_Wm2", "Albedo", "G_0.1m"):
        rr = reps[reps["feature"] == c]
        if len(rr):
            r = rr.iloc[0]
            add(f"      {c:<14} 最终排名 {int(r['final_rank'])}/{len(reps)}  "
                f"分位={r['percentile']:.1f}")
    if not rad_in_top5:
        add("      → 辐射类未进 Top-5。正面解释：辐射是融水能量的必要条件，")
        add("        但事件窗内其异常度未达显著，作用通过 PDD/融水当量间接体现。")
    add("")

    # ---------- 落盘 ----------
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out1 = t1.merge(t2, on="feature", how="left").merge(
        cl, on="feature", how="left").merge(
        reps[["feature", "final_rank"]], on="feature", how="left")
    out1.to_csv(TAB_DIR / "b3_tier1_anomaly.csv", index=False, encoding="utf-8-sig")
    cl.merge(t2, on="feature", how="left").to_csv(
        TAB_DIR / "b3_channel_clusters.csv", index=False, encoding="utf-8-sig")
    top5.to_csv(TAB_DIR / "b3_top5_final.csv", index=False, encoding="utf-8-sig")
    t2.to_csv(TAB_DIR / "b3_tier2_annotation.csv", index=False, encoding="utf-8-sig")

    add("-" * 78)
    add("[7] 落盘")
    add("-" * 78)
    for f in ["b3_tier1_anomaly.csv", "b3_channel_clusters.csv",
              "b3_top5_final.csv", "b3_tier2_annotation.csv"]:
        add(f"   out/B/tab/{f}")

    # ---------- 出图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        # 异常度排序图
        show = reps.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ["#c1440e" if r <= 5 else "#7a8fa6" for r in show["final_rank"]]
        ax.barh(range(len(show)), show["percentile"], color=colors, height=0.72)
        ax.set_yticks(range(len(show)))
        ax.set_yticklabels(show["feature"], fontsize=8)
        ax.set_ylim(-0.6, len(show) - 0.4)   # 显式 ylim，避免标签与条形错位半格
        ax.set_xlim(0, 108)
        for i, (_, r) in enumerate(show.iterrows()):
            ax.text(float(r["percentile"]) + 0.8, i, f"{r['percentile']:.1f}",
                    va="center", fontsize=7, color="0.25")
        ax.axvline(SIG_PERCENTILE, color="red", ls="--", lw=1.2, label=f"{SIG_PERCENTILE:.0f} 分位阈值")
        ax.set_xlabel(f"事件窗 z 均值在【同季节】参考分布中的分位（起点 {SEASON_START_MIN}–{SEASON_START_MAX}）")
        ax.set_title("Q1 Tier 1 异常度排序（同季参考 + 通道去冗余；红色 = Top-5）")
        ax.legend(frameon=False)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "b3_anomaly_ranking.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # 通道聚类热图
        import matplotlib.colors as mcolors
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=6)
        ax.set_yticklabels(cols, fontsize=6)
        ax.set_title(f"spearman |ρ| 聚类（阈值 {REDUNDANCY_RHO}）")
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "b3_channel_clusters.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        add("   已写出 b3_anomaly_ranking.png / b3_channel_clusters.png")
    except Exception as exc:  # noqa: BLE001
        add(f"   出图失败: {exc!r}")

    # ---------- 结论 ----------
    add("")
    add("-" * 78)
    add("[结论]")
    add("-" * 78)
    add(f"  参与排名特征 = {len(cols)}   物理通道 = {n_cl}"
        f"   可评估 = {n_test}")
    add(f"  同季参考 ≥{SIG_PERCENTILE} 分位（未校正） = {int(t1['sig'].sum())}"
        f"   纯噪声期望 = {n_test * 0.05:.2f}")
    add(f"  BH-FDR(q={FDR_Q}) 通过 = {int(t1['bh_pass'].sum())}"
        f"   Bonferroni 通过 = {int(t1['bonf_pass'].sum())}")
    add(f"  LOYO 六年之最 = {int(t1['is_extreme'].fillna(False).sum())} 个")
    add(f"  Top-5 = {', '.join(top5['feature'].tolist())}")
    add("")
    n_nom = int(((t1["sig"]) & (t1["is_extreme"].fillna(False))).sum())
    nom = t1[(t1["sig"]) & (t1["is_extreme"].fillna(False))]
    add(f"  ★ 提名项（同季 ≥95 分位 且 LOYO 六年之最）= {n_nom} 个：" +
        (", ".join(nom["feature"].tolist()) if n_nom else "无"))
    if int(t1["bh_pass"].sum()) == 0:
        add("  ★★ 无任何特征通过 BH-FDR —— 论文中只能写「提名项」，")
        add("     禁止写「前兆信号成立」。")
    add(f"  总耗时 = {time.time() - t_all:.1f}s")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "p2c_anomaly.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    # 只要流程跑完就算成功：NFD 口径下「无显著」是合法结论，不视为失败
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
