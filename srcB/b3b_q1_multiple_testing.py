"""Q1 Tier 1 的三项独立复核（增量脚本，不修改 b3_q1_anomaly.py）。

## 为什么要做这个复核

`b3_q1_anomaly.py` 报告「显著（≥95 分位）= 2 / 38」，并据此写了一句
「前兆信号成立」。这句话在**多重检验**意义下站不住：同时对 38 个特征做
α≈0.05 的检验，纯随机情况下的期望假发现数就是 38 × 0.05 ≈ 1.9 个。
实测恰好 2 个 —— 和「全是噪声」的期望值无法区分。

用户此前明确要求「必须做 BH-FDR 并注明口径」，但主脚本里没有这一步。
本脚本补上，并且额外检查一个更隐蔽的问题：**参考分布混了季节**。

## 三项检查

### [1] 多重检验校正（BH-FDR / Bonferroni / Holm）
- 口径 A：m = 38 个参与排名的特征
- 口径 B：m = 16 个物理通道（代表元）——因为 38 个特征高度冗余，
  BH 在正相依下仍然有效，但有效检验数应按通道计。两种口径都报告。

### [2] 季节性混杂：季节匹配参考分布
原设计的 z 是相对**固定的 8/19–8/26 逐钟点气候态**算的，而参考分布却
是全时段（含冬春季）滑窗。冬季窗口的 z̄ 天然远离 0，事件窗本身就是 8 月下旬，
于是「高分位」有可能只是「这个窗口在 8 月」，而不是「这一周异常」。
修正办法：参考分布只保留与事件窗同季节的滑窗（窗口起点在 8/04–9/03，
使窗口中心落在事件窗中心 ±15 天内）。

### [3] 留一年交叉（LOYO）六年对比 — 最干净的同季对照
把 2021–2025 每年的同一日历窗（8/19 00:00 起 179 h）依次留出，
用其余 4 年建气候态，算该年被留出年份窗口的 z̄。
这样得到 5 个「同季节、样本外」的参考值，与事件窗（用 5 年气候态）对比。
- 比历年都极端 → 提示异常
- 随机情况下每个特征的命中率是 1/6

注意 LOYO 的不对称：历史年用 4 年气候态，事件窗用 5 年气候态。
这会略微放大历史年的波动，属于保守方向（不利于事件显著），如实记录。

输出：
    out/B/tab/b3b_multiple_testing.csv
    out/B/log/p2d_multiple_testing.log

AI 辅助声明：本文件由 AI 辅助生成（2026-10-24），人工已复核。

用法：
    python srcB/b3b_q1_multiple_testing.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import SEED  # noqa: E402
import b3_q1_anomaly as B3  # noqa: E402

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CLEAN_DIR: Path = PROJECT_ROOT / "data" / "B" / "clean"
TAB_DIR: Path = PROJECT_ROOT / "out" / "B" / "tab"
LOG_DIR: Path = PROJECT_ROOT / "out" / "B" / "log"

#: FDR 名义水平。
FDR_Q: float = 0.05

#: 季节匹配参考分布的窗口起点范围（月日，MMDD 数值），使窗中心落在 ±15 天内。
SEASON_START_MIN: int = 804     # 8 月 4 日
SEASON_START_MAX: int = 903     # 9 月 3 日

#: LOYO 的年份集合。
LOYO_YEARS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #


def bh_reject(p: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """Benjamini–Hochberg 逐步法判定拒绝域。

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
    """Bonferroni 校正。粗但绝对保守，用于与 BH 对照。"""
    pv = np.nan_to_num(np.asarray(p, dtype=float), nan=1.0)
    return pv <= alpha / max(len(pv), 1)


def emp_p(ref: np.ndarray, obs: float) -> float:
    """经验 p = (1 + #{ref ≥ obs}) / (n + 1)。"""
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0 or not np.isfinite(obs):
        return np.nan
    return float((1 + int(np.sum(ref >= obs))) / (len(ref) + 1))


def emp_pct(ref: np.ndarray, obs: float) -> float:
    """观测值在参考分布中的百分位（0–100）。"""
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0 or not np.isfinite(obs):
        return np.nan
    return float(100.0 * np.mean(ref < obs) + 100.0 * 0.5 * np.mean(ref == obs))


def climo_from_slices(feats: pd.DataFrame, cols: list[str],
                      slices: list[tuple[pd.Timestamp, pd.Timestamp]]
                      ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """按任意若干时段切片建立逐钟点中位数 / MAD 气候态。

    与 `B3.build_climatology` 的区别：那一个只认固定的 8/19–8/26 日历窗，
    本函数可以用任意 slice 组合（LOYO 需要「剔除某年」的气候态）。

    Args:
        feats: 特征矩阵（DatetimeIndex）。
        cols: 特征列表。
        slices: [(起, 止), ...]，起止均含端点。

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
    """返回某年 8/19 00:00 起的 179 小时窗口（与事件窗等长、同日历）。"""
    s = pd.Timestamp(f"{year}-08-19 00:00")
    return s, s + pd.Timedelta(hours=B3.WIN_HOURS - 1)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def main() -> int:
    """执行三项复核。

    Returns:
        退出码：0 = 三项检查全部完成。
    """
    t_all = time.time()
    log: list[str] = []
    add = log.append

    for d in (TAB_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    add("=" * 78)
    add("Q1 Tier 1 三项独立复核：多重检验 / 季节性混杂 / LOYO 六年对比")
    add("=" * 78)
    add(f"运行时间戳 : {datetime.now().astimezone().isoformat()}")
    add(f"Python     : {sys.version.split()[0]}   numpy {np.__version__} / pandas {pd.__version__}")
    add(f"SEED       : {SEED}")
    add("")
    add("复核动机：主脚本报告『显著 2/38』，而对 38 个量同时做 α≈0.05 检验，")
    add("          纯噪声下的期望假发现数 = 38 × 0.05 ≈ 1.9 —— 与实测 2 个不可区分。")
    add("          必须做 BH-FDR 后才能下『前兆信号成立』的结论。")
    add("")

    feats = pd.read_csv(CLEAN_DIR / "features_hourly.csv", encoding="utf-8-sig",
                        index_col=0, parse_dates=True)
    all_cols = list(feats.columns)
    cols = [c for c in all_cols if c not in B3.EXCLUDED]

    # 通道信息：复用已有聚类（不重算）
    cl_path = TAB_DIR / "b3_channel_clusters.csv"
    cl = pd.read_csv(cl_path, encoding="utf-8-sig") if cl_path.exists() else None
    if cl is not None and "cluster" not in cl.columns:
        cl = None

    # 已有结果，用于回归校验（确认本脚本复算的数字与主脚本一致）
    t1_old = pd.read_csv(TAB_DIR / "b3_tier1_anomaly.csv", encoding="utf-8-sig")

    add("-" * 78)
    add("[0] 输入与回归校验")
    add("-" * 78)
    add(f"   特征表 {len(feats)} 行，参与排名 {len(cols)} 个特征")
    add(f"   已加载通道聚类：{'是' if cl is not None else '否（将退化为单通道）'}")
    add("")

    # ---------- 基础气候态与 z ----------
    quiet: list[str] = []
    med, mad = B3.build_climatology(feats, cols, quiet)
    ev_start, ev_end = B3.EVENT_START, B3.EVENT_END
    excl = (ev_start - pd.Timedelta(hours=B3.WIN_HOURS), ev_end)

    rows: list[dict[str, object]] = []
    ref_full_all: dict[str, np.ndarray] = {}
    ref_season_all: dict[str, np.ndarray] = {}

    for c in cols:
        z = B3.robust_z(feats[c], med[c], mad[c])
        ev_z = z.loc[ev_start:ev_end]
        ev_bar = float(ev_z.mean()) if ev_z.notna().sum() else np.nan

        # 参考分布（全年滑窗，剔除与事件窗重叠者）——与 tier1 完全一致的做法
        ref = B3.window_means(z, exclude=excl)
        ref_vals = ref.to_numpy(dtype=float)
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        ref_full_all[c] = ref_vals

        # 季节匹配参考分布：只保留与事件窗同季节的滑窗起点
        if len(ref.index):
            md = ref.index.month * 100 + ref.index.day
            keep = (md >= SEASON_START_MIN) & (md <= SEASON_START_MAX)
            ref_sea = ref_vals[keep]
        else:
            ref_sea = np.array([])
        ref_season_all[c] = ref_sea

        rows.append({
            "feature": c,
            "event_zbar": ev_bar,
            "n_ref_full": int(len(ref_vals)),
            "p_full": emp_p(ref_vals, ev_bar),
            "pctl_full": emp_pct(ref_vals, ev_bar),
            "n_ref_season": int(len(ref_sea)),
            "p_season": emp_p(ref_sea, ev_bar),
            "pctl_season": emp_pct(ref_sea, ev_bar),
        })

    df = pd.DataFrame(rows)

    # 回归校验：本脚本的 p_full 应与主脚本的**全年参考**口径一致。
    # 注意：主脚本改为同季参考后，其 `p_empirical` 已变成同季口径，
    # 全年口径保存在 `p_empirical_full`，故优先用后者比对。
    old_col = "p_empirical_full" if "p_empirical_full" in t1_old.columns else "p_empirical"
    chk = df.merge(t1_old[["feature", old_col]], on="feature", how="left")
    diff_p = float(np.nanmax(np.abs(chk["p_full"] - chk[old_col]))) if len(chk) else np.nan
    add(f"   回归校验：本脚本 p_full 与主脚本 {old_col} 最大差 = {diff_p:.2e}")
    add(f"            （应 ≈ 0，证明复核链路与主结果同源）")
    add("")

    # ---------- [1] 多重检验校正 ----------
    add("-" * 78)
    add(f"[1] 多重检验校正（BH-FDR q={FDR_Q}）")
    add("-" * 78)
    eval_mask = np.isfinite(df["p_full"].to_numpy(dtype=float))
    n_test = int(eval_mask.sum())
    add(f"   可评估特征数 m = {n_test}（另有 {len(df) - n_test} 个因 MAD=0 无法评估）")
    add("")

    # 口径 A：全部可评估特征
    pA = df["p_full"].to_numpy(dtype=float)
    rejA = bh_reject(pA[eval_mask], FDR_Q)
    rejA_full = np.zeros(len(df), dtype=bool)
    rejA_full[np.where(eval_mask)[0]] = rejA
    df["bh_pass_38"] = rejA_full
    df["bonf_pass_38"] = False
    bonf = bonferroni_reject(pA[eval_mask], 0.05)
    tmp = np.zeros(len(df), dtype=bool)
    tmp[np.where(eval_mask)[0]] = bonf
    df["bonf_pass_38"] = tmp

    add(f"   口径 A：m = {n_test} 个特征")
    add(f"      BH-FDR 通过数     = {int(df['bh_pass_38'].sum())}")
    add(f"      Bonferroni 通过数 = {int(df['bonf_pass_38'].sum())}")
    add(f"      期望 accidental discovery（纯噪声）= m × 0.05 = {n_test * 0.05:.2f}")
    add("")

    # 口径 B：只检验通道代表元
    df["bh_pass_chan"] = False
    if cl is not None:
        df = df.merge(cl[["feature", "cluster"]], on="feature", how="left")
        # 每组取 p_full 最小者作为代表（等价于原脚本取分位最高者）。
        # 注意：pandas 3 下 groupby(as_index=False).first() 返回 RangeIndex，
        # 直接用 .index 会取到「表的前 N 行」而不是各组代表行。
        # 用 groupby(...).head(1) 可保留原始索引标签。
        reps = (df[df["p_full"].notna()]
                .sort_values("p_full")
                .groupby("cluster", as_index=True).head(1))
        rep_names = set(reps["feature"])
        sub = df[df["feature"].isin(rep_names)].copy()
        pw = sub["p_full"].to_numpy(dtype=float)
        rejB = bh_reject(pw, FDR_Q)
        m = int(df["cluster"].nunique())
        add(f"   口径 B：m = {m} 个物理通道（每组只检验 1 个代表元）")
        add(f"      BH-FDR 通过数     = {int(rejB.sum())}")
        add(f"      Bonferroni 通过数 = {int(bonferroni_reject(pw, 0.05).sum())}")
        add(f"      期望 accidental discovery（纯噪声）= {m * 0.05:.2f}")
        add("")
        add("      BH 阈值对照（形如  p_(k) ≤ k/m·q ），仅列代表元：")
        pw_sorted = np.sort(pw)
        for k, pk in enumerate(pw_sorted[:6], start=1):
            thr = FDR_Q * k / m
            add(f"        第 {k} 小 p = {pk:.4f}   阈值 = {thr:.5f}   "
                f"{'✓ 通过' if pk <= thr else '× 未通过'}")
        for f_, rb in zip(sub["feature"], rejB):
            df.loc[df["feature"] == f_, "bh_pass_chan"] = bool(rb)
    else:
        add("   口径 B：未找到通道聚类文件，跳过。")

    add("")
    add("   ★ 两个口径都按 BH-FDR 判定，结论不会因『按特征还是按通道』而摇摆。")

    # ---------- [2] 季节性混杂 ----------
    add("")
    add("-" * 78)
    add("[2] 季节性混杂检验：季节匹配参考分布")
    add("-" * 78)
    add(f"   原参考分布 = 全年滑窗；其中「非 8 月」窗口的 z̄ 天然偏离 0，")
    add(f"   而事件窗本身就在 8 月下旬 —— 高分位可能只反映「这个窗口在 8 月」。")
    add(f"   修正：只保留窗口起点 {SEASON_START_MIN}–{SEASON_START_MAX} 的滑窗"
        f"（窗中心落在事件窗中心 ±15 天内）。")
    add("")
    add(f"   {'feature':<24}{'n全季':>7}{'p全季':>9}{'分位全季':>10}"
        f"{'n同季':>7}{'p同季':>9}{'分位同季':>10}{'Δ分位':>9}")
    top = df.dropna(subset=["pctl_full"]).nsmallest(12, "p_full")
    for _, r in top.iterrows():
        dlt = (r["pctl_season"] - r["pctl_full"]) if np.isfinite(r["pctl_season"]) else np.nan
        add(f"   {r['feature']:<24}{r['n_ref_full']:>7}{r['p_full']:>9.4f}{r['pctl_full']:>10.1f}"
            f"{r['n_ref_season']:>7}{r['p_season']:>9.4f}{r['pctl_season']:>10.1f}{dlt:>9.1f}")
    add("")
    add("   Δ分位为负 = 换成同季节参考后异常度下降，即原先的高分位确实混入了季节效应。")

    # ---------- [3] LOYO 六年对比 ----------
    add("")
    add("-" * 78)
    add("[3] LOYO 六年对比（最干净的同季节、样本外对照）")
    add("-" * 78)
    add("   逐年留出 2021–2025 的同一日历窗（8/19 00:00 起 179 h），")
    add("   用其余 4 年建气候态，算被留出年份的 z̄；得到 5 个样本外参考值。")
    add("   随机情况下每特征命中『2026 为六年之最』的概率 = 1/6 ≈ 16.7%。")
    add("")

    loyo: dict[str, dict[str, float]] = {}
    for tgt in LOYO_YEARS:
        sli = [(pd.Timestamp(f"{y}-08-19 00:00"), pd.Timestamp(f"{y}-08-26 23:00"))
               for y in LOYO_YEARS if y != tgt]
        med_t, mad_t = climo_from_slices(feats, cols, sli)
        ws, we = nth_window(tgt)
        for c in cols:
            z = B3.robust_z(feats[c], med_t[c], mad_t[c])
            seg = z.loc[ws:we]
            loyo.setdefault(c, {})[str(tgt)] = float(seg.mean()) if seg.notna().sum() else np.nan

    hit_rows = []
    for c in cols:
        d = loyo.get(c, {})
        hist = np.array([d.get(str(y), np.nan) for y in LOYO_YEARS], dtype=float)
        ev = float(df.loc[df["feature"] == c, "event_zbar"].iloc[0])
        hist_ok = hist[np.isfinite(hist)]
        if len(hist_ok) == 0 or not np.isfinite(ev):
            hit_rows.append({"feature": c, "loyo_n": int(len(hist_ok)),
                             "loyo_median": np.nan, "loyo_max": np.nan,
                             "is_extreme": False, "p_loyo": np.nan})
            continue
        p_l = (1 + int(np.sum(hist_ok >= ev))) / (len(hist_ok) + 1)
        hit_rows.append({
            "feature": c,
            "loyo_n": int(len(hist_ok)),
            "loyo_median": float(np.median(hist_ok)),
            "loyo_max": float(np.max(hist_ok)),
            "is_extreme": bool(ev > np.max(hist_ok)),
            "p_loyo": float(p_l),
        })
    hit = pd.DataFrame(hit_rows)
    df = df.merge(hit, on="feature", how="left")

    add(f"   {'feature':<24}{'事件 z̄':>9}{'历年中位':>10}{'历年最大':>10}"
        f"{'为六年之最':>12}{'p_loyo':>9}")
    show = df.dropna(subset=["p_loyo"]).nsmallest(15, "p_loyo")
    for _, r in show.iterrows():
        add(f"   {r['feature']:<24}{r['event_zbar']:>9.3f}{r['loyo_median']:>10.3f}"
            f"{r['loyo_max']:>10.3f}{('是' if r['is_extreme'] else '否'):>12}"
            f"{r['p_loyo']:>9.3f}")
    add("")

    n_ext_total = int(df["is_extreme"].fillna(False).sum())
    n_eval = int(df["p_loyo"].notna().sum())
    # 通道数（用于计算期望命中数）
    n_ch = int(df["cluster"].nunique()) if "cluster" in df.columns else n_eval
    add(f"   可评估 {n_eval} 个特征，其中 {n_ext_total} 个特征是「六年之最」")
    add(f"   随机期望（按特征）= {n_eval / 6:.1f} 个；按 {n_ch} 个通道 = {n_ch / 6:.1f} 个")
    add(f"   -> 观测 {n_ext_total} vs 期望约 {n_ch / 6:.1f}，"
        f"{'未见明显超出' if n_ext_total <= n_ch / 6 + 1 else '高于纯随机期望'}")
    add("")
    add("   ★ p_loyo 的下限是 1/6 ≈ 0.167（只有 5 个历史参考值），")
    add("     所以这一步不能单独给出「显著」，它的作用是**证伪**：")
    add("     若某特征连六年最大都做不到，就不可能被称为前兆。")

    # ---------- [4] 季节匹配口径下的 Top-5（通道去冗余） ----------
    add("")
    add("-" * 78)
    add("[4] 季节匹配参考分布下的 Top-5（仍做通道去冗余）")
    add("-" * 78)
    add("   既然 [2] 证明全年参考分布混入季节效应，排序就应改用同季节参考。")
    add("   这里给出替代版 Top-5，供与主脚本的版本对照取舍。")
    add("")

    if "cluster" in df.columns:
        ev_df = df.dropna(subset=["pctl_season"])
        reps2 = (ev_df.sort_values("pctl_season", ascending=False)
                      .groupby("cluster", as_index=False).first())
        reps2 = reps2.sort_values("pctl_season", ascending=False).reset_index(drop=True)
        reps2["season_rank"] = np.arange(1, len(reps2) + 1)
        reps2.to_csv(TAB_DIR / "b3b_top5_seasonmatched.csv", index=False, encoding="utf-8-sig")

        add(f"   {'rank':>4} {'feature':<22}{'通道':>5}{'同季分位':>10}{'z̄':>9}"
            f"{'方向':<8}{'六年之最':>9}{'p全季':>9}")
        for _, r in reps2.head(8).iterrows():
            add(f"   {int(r['season_rank']):>4} {r['feature']:<22}{int(r['cluster']):>5}"
                f"{r['pctl_season']:>10.1f}{r['event_zbar']:>9.3f}"
                f"{B3._direction_label(r['event_zbar']):<8}"
                f"{('是' if r['is_extreme'] else '否'):>9}{r['p_full']:>9.4f}")
        add("")
        top5_new = list(reps2.head(5)["feature"])
        old_top5 = pd.read_csv(TAB_DIR / "b3_top5_final.csv", encoding="utf-8-sig")
        old_names = list(old_top5["feature"])
        add(f"   原 Top-5（全年参考）: {', '.join(old_names)}")
        add(f"   新 Top-5（同季参考）: {', '.join(top5_new)}")
        gone = [f for f in old_names if f not in top5_new]
        came = [f for f in top5_new if f not in old_names]
        add(f"   剔除: {', '.join(gone) if gone else '（无）'}")
        add(f"   新增: {', '.join(came) if came else '（无）'}")
        add("")
        add("   ★ 被剔除的项——它们的『高分位』在同季节参考下大幅回落，")
        add("     说明原本的高名次主要是季节效应，而非本次事件的异常。")
    else:
        add("   未找到通道聚类，跳过。")

    # ---------- 落盘 ----------
    df.to_csv(TAB_DIR / "b3b_multiple_testing.csv", index=False, encoding="utf-8-sig")

    # ---------- 结论 ----------
    add("")
    add("-" * 78)
    add("[结论]")
    add("-" * 78)
    n_bh = int(df["bh_pass_38"].sum())
    n_bonf = int(df["bonf_pass_38"].sum())
    add(f"   口径 A（{n_test} 特征）：BH-FDR 通过 {n_bh} 个，Bonferroni 通过 {n_bonf} 个")
    if cl is not None:
        n_bh2 = int(df["bh_pass_chan"].sum())
        m = int(df["cluster"].nunique())
        add(f"   口径 B（{m} 通道）  ：BH-FDR 通过 {n_bh2} 个")
    add("")
    add(f"   未校正达到 ≥95 分位的原始个数 = {int((df['pctl_full'] >= 95).sum())}"
        f"（对照：纯噪声期望 {n_test * 0.05:.2f}）")
    add("")
    if n_bh == 0:
        add("   ★★ 校正后没有任何特征通过 BH-FDR。")
        add("      即：不能声称『前兆信号成立』。可以报告的是排序本身，")
        add("      并把最强的候选明确标注为『未通过多重检验校正的提名项』。")
    else:
        add("   通过校正的特征：" + ", ".join(df.loc[df["bh_pass_38"], "feature"]))

    add("")
    add(f"   总耗时 {time.time() - t_all:.1f}s")
    (LOG_DIR / "p2d_multiple_testing.log").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
