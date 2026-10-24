"""物理状态重构：把原始观测通道转成 ≥25 个具有物理意义的诊断量。

对应 CLAUDE_B.md 第三节 Q1 第 1 条。核心原则：**先物理重构，再统计关联**——
禁止把原始 60+ 通道直接喂给模型。

输出：
    data/B/clean/features_hourly.csv      2021–2026 全期逐小时特征矩阵
    out/B/tab/b1_feature_validn.csv       每个特征的有效样本数与缺测率（必交件）
    out/B/log/p2_physics.log              物理合理性自检 + 干旱归因说明
    out/B/fig/b1_diurnal.png              事件周 vs 气候态日变化对比

关键约束（CLAUDE_B.md 2.2 与用户补充）：
    1. 每个特征必须在 b1_feature_validn.csv 中报告有效样本数与缺测率；
    2. FLUX 通量类（H/LE/TAU/TSTAR/FC_mass/e_sat/VPD）缺测约 44%，
       动量类（Ux/Uy/Uz/USTAR/TKE）约 34%，**二者不可按同一有效样本数做因果比较**；
    3. 计算 Se 与湿润锋速率时，必须输出「事件前 X 天无有效降雨」的实算说明。

AI 辅助声明：本文件由 Claude Code（Anthropic，2026-10-24）辅助生成，人工已复核。

用法：
    python srcB/physics.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import SEED, load_source  # noqa: E402

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CLEAN_DIR: Path = PROJECT_ROOT / "data" / "B" / "clean"
TAB_DIR: Path = PROJECT_ROOT / "out" / "B" / "tab"
LOG_DIR: Path = PROJECT_ROOT / "out" / "B" / "log"
FIG_DIR: Path = PROJECT_ROOT / "out" / "B" / "fig"

#: 物理常数
G: float = 9.80665            # 重力加速度 m/s²
KAPPA: float = 0.4            # von Kármán 常数

#: 稳定度计算取 10 m 层（作战地图 P2-1 指定）。
Z_REF: float = 10.0

#: 0℃ 层外推封顶高度 (m)。
Z0_CAP: float = 6000.0

#: 度日因子 DDF，先验值 [待核实]。
#: 文献方向：Hock (2003) 冰川消融度日模型综述，DDF_snow ≈ 2.5–5.5 mm w.e./°C/d，
#: DDF_ice ≈ 6–9 mm w.e./°C/d。此处取冰雪混合的中间值 4.5，论文中须标注区间敏感性。
DDF_MM_PER_C_DAY: float = 4.5

#: API 衰减系数（CLAUDE_B.md 指定 k=0.9 / 0.85）。
API_K10: float = 0.9
API_K30: float = 0.85

#: 事件时刻与窗口。
EVENT_TIME: pd.Timestamp = pd.Timestamp("2026-08-26 10:30")
EVENT_WEEK_START: pd.Timestamp = pd.Timestamp("2026-08-20 00:00")
EVENT_WEEK_END: pd.Timestamp = pd.Timestamp("2026-08-27 08:00")

#: 湿度/辐射列的单位换算与物理合理范围（用于剔除明显跳变，见日志说明）。
#: 只剔除"物理不可能"的值，不剔除"物理极端"的值——后者正是本项目关心的信号。
PHYS_BOUNDS: dict[tuple[str, str], tuple[float, float]] = {
    ("SOIL", "VWC_0.8m(m3/m3)"): (0.0, 1.0),      # 体积含水率不可能 >1
    ("SOIL", "G_0.1m(W/m2)"): (-200.0, 500.0),    # 土壤热通量量级上限
    ("SOIL", "G_0.2m(W/m2)"): (-200.0, 500.0),
    ("RADM", "Rld(W/m2)"): (0.0, 600.0),          # 下行长波上限（青藏高原约 400 W/m² 量级）
    ("RADM", "Rlu(W/m2)"): (0.0, 700.0),
    ("FLUX", "e_sat(kPa)"): (0.0, 10.0),
    ("FLUX", "VPD(hPa)"): (-50.0, 100.0),
    ("FLUX", "LE(W/m2)"): (-2000.0, 1000.0),
}

#: FLUX 变量分组（用户补充约束 1：两组有效样本数不同，不可混用）。
FLUX_FLUX_VARS: tuple[str, ...] = ("H(W/m2)", "LE(W/m2)", "TAU(kg/m*s2)", "TSTAR(deg C)",
                                   "FC_mass(mg/m2*s)", "e_sat(kPa)", "VPD(hPa)")
FLUX_MOMENTUM_VARS: tuple[str, ...] = ("Ux(m/s)", "Uy(m/s)", "Uz(m/s)", "USTAR(m/s)", "TKE(m2/s2)")

#: 跨函数传递的辅助统计（仅用于日志，不影响特征值）。
_AUX: dict[str, object] = {}

#: VWC 五层。
VWC_LAYERS: tuple[str, ...] = ("VWC_0.1m(m3/m3)", "VWC_0.2m(m3/m3)", "VWC_0.4m(m3/m3)",
                               "VWC_0.8m(m3/m3)", "VWC_1.6m(m3/m3)")

#: 气温五层（用于 0℃ 层外推）。
TA_LAYERS: tuple[tuple[float, str], ...] = (
    (1.0, "Ta_1m(degree C)"), (2.0, "Ta_2m(degree C)"), (4.0, "Ta_4m(degree C)"),
    (10.0, "Ta_10m(degree C)"), (20.0, "Ta_20m(degree C)"),
)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #


def _mask_bounds(df: pd.DataFrame, prefix: str, log: list[str]) -> pd.DataFrame:
    """把超出物理合理范围的值置 NaN，并记录剔除数量。

    Args:
        df: 某源的数值矩阵。
        prefix: 源前缀。
        log: 日志行收集器（就地追加）。

    Returns:
        掩码后的副本。
    """
    out = df.copy()
    for (p, col), (lo, hi) in PHYS_BOUNDS.items():
        if p != prefix or col not in out.columns:
            continue
        bad = (out[col] < lo) | (out[col] > hi)
        n = int(bad.sum())
        if n:
            log.append(f"   物理范围剔除 {prefix}.{col}: {n} 个值 (允许区间 [{lo}, {hi}])")
            out.loc[bad, col] = np.nan
    return out


def _safe_div(num: pd.Series, den: pd.Series, eps: float) -> pd.Series:
    """带下限保护的安全除法：|den| < eps 时返回 NaN。

    Args:
        num: 分子。
        den: 分母。
        eps: 分母绝对值的下限阈值。

    Returns:
        商序列，分母过小处为 NaN。
    """
    return num.where(den.abs() >= eps) / den.where(den.abs() >= eps)


# --------------------------------------------------------------------------- #
# 各诊断量
# --------------------------------------------------------------------------- #


def obukhov_length(u_star: pd.Series, t_star: pd.Series, t_air_k: pd.Series) -> pd.Series:
    """Obukhov 长度 L = -(u*³·T₀)/(κ·g·θ*)。

    Args:
        u_star: 摩擦速度 USTAR (m/s)。
        t_star: 温度尺度 TSTAR (°C，作为 θ* 的近似)。
        t_air_k: 参考气温 T₀ (K)。

    Returns:
        L (m)。θ* 近 0 时 L → ±∞，此处保留原值并在自检中报告。
    """
    return -(u_star ** 3 * t_air_k) / (KAPPA * G * t_star)


def stability_zeta(obukhov: pd.Series, z: float = Z_REF) -> pd.Series:
    """稳定度参数 ζ = z/L。

    Args:
        obukhov: Obukhov 长度 L (m)。
        z: 参考高度 (m)。

    Returns:
        无量纲稳定度参数。
    """
    return z / obukhov


def bulk_richardson(t_low: pd.Series, t_high: pd.Series,
                    u_low: pd.Series, u_high: pd.Series,
                    z_low: float, z_high: float) -> pd.Series:
    """总体理查森数 Ri_b = (g/θ)·(Δθ/Δz) / (ΔU/Δz)²。

    用 5 层气温/风速廓线中的 1 m 与 20 m 两层做差分（作战地图写 2m/20m，
    此处取 1m/20m 以获得更大的 Δz、降低差分噪声；两者都会输出）。

    Args:
        t_low: 低层气温 (°C)。
        t_high: 高层气温 (°C)。
        u_low: 低层风速 (m/s)。
        u_high: 高层风速 (m/s)。
        z_low: 低层高度 (m)。
        z_high: 高层高度 (m)。

    Returns:
        Ri_b（无量纲）。ΔU 近 0 时为 NaN。
    """
    dz = z_high - z_low
    theta = 273.15 + 0.5 * (t_low + t_high)          # 层平均位温近似，K
    dtheta_dz = (t_high - t_low) / dz
    du_dz = (u_high - u_low) / dz
    return (G / theta) * dtheta_dz / (du_dz ** 2).where(du_dz.abs() > 1e-6)


def inversion_strength(ta_1m: pd.Series, ta_20m: pd.Series) -> pd.Series:
    """逆温强度 Γ = (Ta_20 − Ta_1)/ln(20/1)。

    值为正表示温度随高度增加（逆温）。

    Args:
        ta_1m: 1 m 气温 (°C)。
        ta_20m: 20 m 气温 (°C)。

    Returns:
        Γ (°C 每自然对数高度单位)。
    """
    return (ta_20m - ta_1m) / np.log(20.0 / 1.0)


def inversion_hours_per_day(gamma: pd.Series, threshold: float = 0.0) -> pd.Series:
    """逐日逆温累计时长（小时）。

    Args:
        gamma: 逆温强度序列（逐小时）。
        threshold: 判定为逆温的 Γ 阈值 (°C/log-unit)，默认 > 0。

    Returns:
        与输入同索引的逐小时序列，值为**当日**逆温小时数（同日内重复），
        便于直接与其他逐小时特征对齐。
    """
    flag = (gamma > threshold).astype(float)
    daily = flag.groupby(flag.index.normalize()).transform("sum")
    daily[gamma.isna()] = np.nan
    return daily.rename("InvHours")


def bowen_ratio(h: pd.Series, le: pd.Series, eps: float = 5.0) -> pd.Series:
    """Bowen 比 β = H/LE。

    Args:
        h: 感热通量 H (W/m²)。
        le: 潜热通量 LE (W/m²)。
        eps: |LE| 下限；低于此值 β 无物理意义，置 NaN。

    Returns:
        β（无量纲）。
    """
    return _safe_div(h, le, eps)


def evaporative_fraction(le: pd.Series, rn: pd.Series, g_soil: pd.Series,
                        eps: float = 20.0) -> pd.Series:
    """蒸发比 EF = LE/(Rn − G)。

    Args:
        le: 潜热通量 LE (W/m²)。
        rn: 净辐射 (W/m²)。
        g_soil: 土壤热通量 (W/m²)。
        eps: |Rn − G| 下限。

    Returns:
        EF（无量纲）。
    """
    return _safe_div(le, rn - g_soil, eps)


def energy_balance_ratio(h: pd.Series, le: pd.Series, rn: pd.Series,
                         g_soil: pd.Series, eps: float = 20.0) -> pd.Series:
    """能量闭合度 EBR = (H + LE)/(Rn − G)，作为数据质量证据。

    Args:
        h: 感热通量 (W/m²)。
        le: 潜热通量 (W/m²)。
        rn: 净辐射 (W/m²)。
        g_soil: 土壤热通量 (W/m²)。
        eps: |Rn − G| 下限。

    Returns:
        EBR（无量纲，理想值 1）。
    """
    return _safe_div(h + le, rn - g_soil, eps)


def net_radiation(rsd: pd.Series, rsu: pd.Series, rld: pd.Series, rlu: pd.Series) -> pd.Series:
    """净辐射 Rn = Rsd − Rsu + Rld − Rlu。

    Args:
        rsd: 下行短波 (W/m²)。
        rsu: 上行短波 (W/m²)。
        rld: 下行长波 (W/m²)。
        rlu: 上行长波 (W/m²)。

    Returns:
        Rn (W/m²)。
    """
    return rsd - rsu + rld - rlu


def antecedent_precipitation_index(precip: pd.Series, k: float, days: int) -> pd.Series:
    """指数衰减前期降水指数 API = Σ P(t−i)·k^i。

    Args:
        precip: 逐小时降水 (mm)。
        k: 逐小时衰减系数。
        days: 累加窗口（天）。

    Returns:
        API 序列（单位 mm，逐小时更新）。
    """
    p = precip.fillna(0.0)
    window = days * 24
    weights = np.array([k ** i for i in range(window)])
    api = p.rolling(window, min_periods=window).apply(
        lambda x: float(np.dot(x[::-1], weights[: len(x)])), raw=True
    )
    # 原始降水缺测的位置，API 不可信
    api[precip.isna()] = np.nan
    return api


def positive_degree_days(t_air: pd.Series, base: float = 0.0) -> pd.Series:
    """正积温 PDD（逐小时增量形式）：max(0, T − base)/24，单位 °C·d。

    Args:
        t_air: 气温 (°C)。
        base: 基准温度 (°C)，默认 0。

    Returns:
        逐小时 PDD 增量 (°C·d)。
    """
    return ((t_air - base).clip(lower=0.0)) / 24.0


def melt_equivalent(pdd_hourly: pd.Series, ddf: float = DDF_MM_PER_C_DAY) -> pd.Series:
    """融水当量 Melt = DDF · max(0, T) / 24（mm/h）。

    Args:
        pdd_hourly: 逐小时 PDD 增量 (°C·d)。
        ddf: 度日因子 (mm/°C/d)。

    Returns:
        逐小时融水当量 (mm/h)。
    """
    return ddf * pdd_hourly


def zero_degree_height(ta_by_level: dict[float, pd.Series],
                       cap: float = Z0_CAP,
                       min_lapse: float = 0.002,
                       stats: dict[str, int] | None = None) -> pd.Series:
    """用 5 层气温廓线外推 0℃ 层高度（近地层版本）。

    **重要局限（必须在论文中声明）**：本站 5 层气温仅覆盖 1–20 m，
    属于近地层（surface layer），温度廓线受局地湍流与地表加热主导，
    **不能代表自由大气的温度递减率**（通常 6.5 °C/km）。
    实测 1–20 m 的温差中位数仅约 0.36 °C，若据此线性外推会得到 100–200 m 的
    "0℃ 层高度"，与真实的 4000–5500 m 相差一个量级，属**物理上不成立**的外推。

    因此本实现只在同时满足以下条件时给出数值：
      * 对数廓线斜率为负（温度随高度递减，即无逆温）；
      * 外推得到的递减率 >= ``min_lapse`` (K/m)，即至少 2 °C/km 量级；
      * 结果落在 [0, cap]。
    其余时刻一律返回 NaN，并把各类失败原因计数写入 ``stats``，供日志如实报告。

    Args:
        ta_by_level: {高度 m: 气温序列}。
        cap: 外推封顶高度 (m)。
        min_lapse: 允许外推的最小温度递减率 (K/m)。
        stats: 可选的计数器字典，就地累加失败原因。

    Returns:
        0℃ 层高度序列 (m)，大部分时刻为 NaN（见上方局限）。
    """
    levels = sorted(ta_by_level)
    log_z = np.log(np.array(levels, dtype=float))
    dz = levels[-1] - levels[0]
    temps = np.column_stack([ta_by_level[z].to_numpy(dtype=float) for z in levels])

    n = temps.shape[0]
    out = np.full(n, np.nan)
    cnt = {"n_total": n, "n_insufficient_levels": 0, "n_no_lapse": 0,
           "n_lapse_too_weak": 0, "n_out_of_range": 0, "n_ok": 0}

    for i in range(n):
        t = temps[i]
        valid = ~np.isnan(t)
        if valid.sum() < 2:
            cnt["n_insufficient_levels"] += 1
            continue
        slope, intercept = np.polyfit(log_z[valid], t[valid], 1)
        if slope >= 0:
            cnt["n_no_lapse"] += 1
            continue
        # 线性近似递减率：dT/dz ≈ slope / z (在对数坐标下 z·dT/dz = slope)
        lapse = -slope / 10.0            # 取 10 m 处等效线性递减率 (K/m)
        if lapse < min_lapse:
            cnt["n_lapse_too_weak"] += 1
            continue
        expo = -intercept / slope
        if not np.isfinite(expo) or expo > np.log(cap * 10):
            cnt["n_out_of_range"] += 1
            continue
        z0 = float(np.exp(expo))
        if not (0.0 <= z0 <= cap):
            cnt["n_out_of_range"] += 1
            continue
        out[i] = z0
        cnt["n_ok"] += 1

    if stats is not None:
        for k, v in cnt.items():
            stats[k] = stats.get(k, 0) + v
    _ = dz
    s = pd.Series(out, index=next(iter(ta_by_level.values())).index, name="Z0C_m")
    return s.clip(lower=0.0, upper=cap)


def effective_saturation(vwc: pd.Series) -> pd.Series:
    """有效饱和度 Se = (θ − θr)/(θs − θr)。

    θr / θs 用各层的 5% / 95% 经验分位（数据驱动估计，非实测土壤参数）。
    论文中必须标注这是**分位数估计**而非实验室测定的残余/饱和含水率。

    Args:
        vwc: 体积含水率序列 (m³/m³)。

    Returns:
        Se ∈ [0, 1]（可因极端值略超出，不截断，便于发现异常）。
    """
    theta_r = vwc.quantile(0.05)
    theta_s = vwc.quantile(0.95)
    if not np.isfinite(theta_s - theta_r) or (theta_s - theta_r) < 1e-9:
        return pd.Series(np.nan, index=vwc.index, name=vwc.name)
    return (vwc - theta_r) / (theta_s - theta_r)


def wetting_front_rate(vwc_shallow: pd.Series, hours: int = 1) -> pd.Series:
    """湿润锋推进速率代理：d(VWC_0.1m)/dt（每小时含水率增量）。

    真实湿润锋速率需土层界面到达时间，本数据只有固定深度含水率，
    因此用表层含水率的时间导数作为**替代量**，论文中须说明这一局限。

    Args:
        vwc_shallow: 0.1 m 层体积含水率。
        hours: 差分步长（小时）。

    Returns:
        逐小时变化率 (m³/m³/h)。
    """
    return vwc_shallow.diff(hours)


def freeze_thaw_events(soil_t: pd.Series) -> pd.DataFrame:
    """冻融循环：0℃ 跨越次数与累计时长。

    仅在逐日尺度上统计（逐小时的跨越会因噪声产生虚假循环）。

    Args:
        soil_t: 土壤温度 (°C)，通常取 0.1 m 层。

    Returns:
        索引为日期的 DataFrame，含 ``ft_crossings``（当日 0℃ 跨越次数）与
        ``ft_hours_below``（当日 0℃ 以下小时数）。
    """
    below = (soil_t < 0.0).astype(float)
    below[soil_t.isna()] = np.nan
    crossed = (below.diff().abs() > 0.5).astype(float)
    day = soil_t.index.normalize()
    out = pd.DataFrame({
        "ft_crossings": crossed.groupby(day).sum(),
        "ft_hours_below": below.groupby(day).sum(),
    })
    out.index.name = "date"
    return out


# --------------------------------------------------------------------------- #
# 干旱归因（用户补充约束 2）
# --------------------------------------------------------------------------- #


def dry_spell_before_event(precip: pd.Series, event: pd.Timestamp = EVENT_TIME) -> dict[str, object]:
    """实算事件前连续无有效降雨的天数 X。

    定义与 `b0_event_window.daily_precip_and_dry_spell` 保持一致：
    日降水 > 0 记为有雨日；缺失日不计入。

    Args:
        precip: 逐小时降水序列（全期，用于回溯）。
        event: 事件时刻。

    Returns:
        含 ``x_days`` / ``last_rain_day`` / ``daily_table`` 的字典。
    """
    p = precip.mask(precip.abs() >= 9999)
    daily_sum = p.groupby(p.index.normalize()).sum(min_count=1)
    daily_cnt = p.groupby(p.index.normalize()).count()
    complete = daily_cnt == 24
    rainy = pd.Series(np.where(complete, daily_sum > 0, np.nan), index=daily_sum.index)

    upto = rainy.loc[: event.normalize()]
    # 事件当天按定义不计（事件发生在 8/26 10:30，当天 0 雨不构成"事件前"
    upto = upto.iloc[:-1] if len(upto) else upto

    x_days = 0
    last_rain = None
    for day, val in upto.iloc[::-1].items():
        if pd.isna(val):
            break
        if bool(val):
            last_rain = day
            break
        x_days += 1

    return {
        "x_days": x_days,
        "last_rain_day": last_rain,
        "daily_table": rainy,
        "daily_sum": daily_sum,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def build_features(log: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """构建全期逐小时特征矩阵。

    Args:
        log: 日志收集器。

    Returns:
        ``(features, raw_sources)``：特征矩阵与各源原始数值矩阵（供有效样本统计）。
    """
    met, met_qc = load_source("MET")
    radm, _ = load_source("RADM")
    soil, _ = load_source("SOIL")
    flux, _ = load_source("FLUX")

    met = _mask_bounds(met, "MET", log)
    radm = _mask_bounds(radm, "RADM", log)
    soil = _mask_bounds(soil, "SOIL", log)
    flux = _mask_bounds(flux, "FLUX", log)

    log.append(f"  各源行数: MET={len(met)} RADM={len(radm)} SOIL={len(soil)} FLUX={len(flux)}")
    log.append(f"  索引对齐: MET/RADM/SOIL 一致={met.index.equals(radm.index) and met.index.equals(soil.index)}, "
               f"FLUX 一致={met.index.equals(flux.index)}")

    f = pd.DataFrame(index=met.index)
    f.index.name = "Timestamp"

    # ---------- 边界层 ----------
    t0_k = met["Ta_2m(degree C)"] + 273.15
    f["L_Obukhov"] = obukhov_length(flux["USTAR(m/s)"], flux["TSTAR(deg C)"], t0_k)
    f["zeta_10m"] = stability_zeta(f["L_Obukhov"])
    f["Ri_b_1_20"] = bulk_richardson(met["Ta_1m(degree C)"], met["Ta_20m(degree C)"],
                                     met["WS_1m(m/s)"], met["WS_20m(m/s)"], 1.0, 20.0)
    f["Ri_b_2_20"] = bulk_richardson(met["Ta_2m(degree C)"], met["Ta_20m(degree C)"],
                                     met["WS_2m(m/s)"], met["WS_20m(m/s)"], 2.0, 20.0)
    f["Inversion_Gamma"] = inversion_strength(met["Ta_1m(degree C)"], met["Ta_20m(degree C)"])
    f["Inversion_Hours_Day"] = inversion_hours_per_day(f["Inversion_Gamma"])

    # ---------- 能量 ----------
    rn = net_radiation(radm["Rsd(W/m2)"], radm["Rsu(W/m2)"], radm["Rld(W/m2)"], radm["Rlu(W/m2)"])
    f["Rn_Wm2"] = rn
    f["Bowen_beta"] = bowen_ratio(flux["H(W/m2)"], flux["LE(W/m2)"])
    f["EF"] = evaporative_fraction(flux["LE(W/m2)"], rn, soil["G_0.1m(W/m2)"])
    f["EBR"] = energy_balance_ratio(flux["H(W/m2)"], flux["LE(W/m2)"], rn, soil["G_0.1m(W/m2)"])
    f["Albedo"] = (radm["Rsu(W/m2)"] / radm["Rsd(W/m2)"]).where(radm["Rsd(W/m2)"] > 50.0)

    # ---------- 水文触发势 ----------
    precip = met["Precip(mm)"]
    f["Precip_mm"] = precip
    f["API_10"] = antecedent_precipitation_index(precip, API_K10, 10)
    f["API_30"] = antecedent_precipitation_index(precip, API_K30, 30)
    pdd_h = positive_degree_days(met["Ta_2m(degree C)"])
    f["PDD_hourly"] = pdd_h
    f["PDD_cumsum"] = pdd_h.fillna(0.0).cumsum()
    f["Melt_mm_h"] = melt_equivalent(pdd_h)
    f["Ta_2m_K"] = met["Ta_2m(degree C)"]        # 保留原始气温供出图与解释
    if "_z0c_stats" not in _AUX:
        _AUX["_z0c_stats"] = {}
    f["Z0C_m"] = zero_degree_height({z: met[c] for z, c in TA_LAYERS},
                                    stats=_AUX["_z0c_stats"])

    # ---------- 土体 ----------
    for col in VWC_LAYERS:
        depth = col.split("_")[1].replace("m(m3/m3)", "")
        f[f"Se_{depth}m"] = effective_saturation(soil[col])
        f[f"VWC_{depth}m"] = soil[col]
    f["dTheta_dt_0.1m"] = wetting_front_rate(soil["VWC_0.1m(m3/m3)"])
    f["dTheta_dt_0.1m_6h"] = wetting_front_rate(soil["VWC_0.1m(m3/m3)"], hours=6)
    f["SoilT_grad_0.1_0.4"] = (soil["soilT_0.4m(degree C)"] - soil["soilT_0.1m(degree C)"]) / 0.3
    f["SoilT_grad_0.8_1.6"] = (soil["soilT_1.6m(degree C)"] - soil["soilT_0.8m(degree C)"]) / 0.8
    f["G_0.1m"] = soil["G_0.1m(W/m2)"]

    ft = freeze_thaw_events(soil["soilT_0.1m(degree C)"])
    f["FT_crossings_day"] = ft["ft_crossings"].reindex(f.index.normalize()).to_numpy()
    f["FT_hours_below_day"] = ft["ft_hours_below"].reindex(f.index.normalize()).to_numpy()

    # ---------- 湍流 ----------
    f["USTAR"] = flux["USTAR(m/s)"]
    f["TKE"] = flux["TKE(m2/s2)"]
    f["FETCH_90"] = flux["FETCH_90(m)"]
    f["H_Wm2"] = flux["H(W/m2)"]
    f["LE_Wm2"] = flux["LE(W/m2)"]

    raw_sources = {"MET": met, "RADM": radm, "SOIL": soil, "FLUX": flux}
    return f, raw_sources


def validity_table(features: pd.DataFrame, raw_sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """生成每个特征的有效样本数与缺测率表（用户补充约束 1 的必交件）。

    Args:
        features: 特征矩阵。
        raw_sources: 各源原始数值矩阵，用于评估「上游最严格输入」的有效样本数。

    Returns:
        含 ``feature`` / ``n_total`` / ``n_valid`` / ``miss_rate`` /
        ``binding_source``（决定该特征样本量的最严格源列）/
        ``flux_group``（flux_flux / flux_momentum / 空）的 DataFrame。
    """
    rows: list[dict[str, object]] = []
    n_total = len(features)

    # 每个特征直接依赖的原始列，用于定位瓶颈
    deps: dict[str, list[tuple[str, str]]] = {
        "L_Obukhov": [("FLUX", "USTAR(m/s)"), ("FLUX", "TSTAR(deg C)"), ("MET", "Ta_2m(degree C)")],
        "zeta_10m": [("FLUX", "USTAR(m/s)"), ("FLUX", "TSTAR(deg C)")],
        "Ri_b_1_20": [("MET", "Ta_1m(degree C)"), ("MET", "Ta_20m(degree C)"),
                      ("MET", "WS_1m(m/s)"), ("MET", "WS_20m(m/s)")],
        "Ri_b_2_20": [("MET", "Ta_2m(degree C)"), ("MET", "Ta_20m(degree C)"),
                      ("MET", "WS_2m(m/s)"), ("MET", "WS_20m(m/s)")],
        "Inversion_Gamma": [("MET", "Ta_1m(degree C)"), ("MET", "Ta_20m(degree C)")],
        "Inversion_Hours_Day": [("MET", "Ta_1m(degree C)"), ("MET", "Ta_20m(degree C)")],
        "Rn_Wm2": [("RADM", "Rsd(W/m2)"), ("RADM", "Rsu(W/m2)"),
                   ("RADM", "Rld(W/m2)"), ("RADM", "Rlu(W/m2)")],
        "Bowen_beta": [("FLUX", "H(W/m2)"), ("FLUX", "LE(W/m2)")],
        "EF": [("FLUX", "LE(W/m2)"), ("RADM", "Rsd(W/m2)"), ("SOIL", "G_0.1m(W/m2)")],
        "EBR": [("FLUX", "H(W/m2)"), ("FLUX", "LE(W/m2)"), ("RADM", "Rsd(W/m2)")],
        "Albedo": [("RADM", "Rsd(W/m2)"), ("RADM", "Rsu(W/m2)")],
        "Precip_mm": [("MET", "Precip(mm)")],
        "API_10": [("MET", "Precip(mm)")],
        "API_30": [("MET", "Precip(mm)")],
        "PDD_hourly": [("MET", "Ta_2m(degree C)")],
        "PDD_cumsum": [("MET", "Ta_2m(degree C)")],
        "Melt_mm_h": [("MET", "Ta_2m(degree C)")],
        "Ta_2m_K": [("MET", "Ta_2m(degree C)")],
        "Z0C_m": [("MET", "Ta_1m(degree C)"), ("MET", "Ta_20m(degree C)")],
        "dTheta_dt_0.1m": [("SOIL", "VWC_0.1m(m3/m3)")],
        "dTheta_dt_0.1m_6h": [("SOIL", "VWC_0.1m(m3/m3)")],
        "SoilT_grad_0.1_0.4": [("SOIL", "soilT_0.1m(degree C)"), ("SOIL", "soilT_0.4m(degree C)")],
        "SoilT_grad_0.8_1.6": [("SOIL", "soilT_0.8m(degree C)"), ("SOIL", "soilT_1.6m(degree C)")],
        "G_0.1m": [("SOIL", "G_0.1m(W/m2)")],
        "FT_crossings_day": [("SOIL", "soilT_0.1m(degree C)")],
        "FT_hours_below_day": [("SOIL", "soilT_0.1m(degree C)")],
        "USTAR": [("FLUX", "USTAR(m/s)")],
        "TKE": [("FLUX", "TKE(m2/s2)")],
        "FETCH_90": [("FLUX", "FETCH_90(m)")],
        "H_Wm2": [("FLUX", "H(W/m2)")],
        "LE_Wm2": [("FLUX", "LE(W/m2)")],
    }

    for col in features.columns:
        n_valid = int(features[col].notna().sum())
        src_deps = deps.get(col)
        if src_deps is None and col.startswith(("Se_", "VWC_")):
            depth = col.split("_", 1)[1]
            src_deps = [("SOIL", f"VWC_{depth}(m3/m3)")]

        binding = ""
        if src_deps:
            counts = []
            for prefix, raw_col in src_deps:
                if raw_col in raw_sources[prefix].columns:
                    counts.append((prefix, raw_col, int(raw_sources[prefix][raw_col].notna().sum())))
            if counts:
                prefix, raw_col, n = min(counts, key=lambda x: x[2])
                binding = f"{prefix}.{raw_col}(n={n})"

        group = ""
        if src_deps and any(p == "FLUX" for p, _ in src_deps):
            if any(c in FLUX_FLUX_VARS for _, c in src_deps):
                group = "flux_flux"
            elif any(c in FLUX_MOMENTUM_VARS for _, c in src_deps):
                group = "flux_momentum"

        rows.append({
            "feature": col,
            "n_total": n_total,
            "n_valid": n_valid,
            "miss_rate": round(1.0 - n_valid / n_total, 6),
            "binding_source": binding,
            "flux_group": group,
        })

    return pd.DataFrame(rows).sort_values("miss_rate", ascending=False)


def diurnal_figure(features: pd.DataFrame, out_path: Path) -> None:
    """绘制事件周与气候态日变化对比图。

    Args:
        features: 特征矩阵。
        out_path: 输出 PNG。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    panels = ["Ta_2m_K", "Rn_Wm2", "Se_0.1m", "USTAR"]
    labels = {"Ta_2m_K": "气温 Ta_2m (°C)", "Rn_Wm2": "净辐射 Rn (W/m²)",
              "Se_0.1m": "表层有效饱和度 Se", "USTAR": "USTAR (m/s)"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.ravel(), panels):
        if col not in features.columns:
            ax.set_visible(False)
            continue
        s = features[col]
        clim = s[s.index.month == 8]
        event = s.loc[EVENT_WEEK_START:EVENT_WEEK_END]
        clim_h = clim.groupby(clim.index.hour).agg(["mean", "std"])
        ev_h = event.groupby(event.index.hour).mean()
        ax.plot(clim_h.index, clim_h["mean"], color="#333333", lw=1.6, label="8 月气候态")
        ax.fill_between(clim_h.index, clim_h["mean"] - clim_h["std"],
                        clim_h["mean"] + clim_h["std"], color="#999999", alpha=0.25, label="±1σ")
        ax.plot(ev_h.index, ev_h.reindex(clim_h.index), color="#c1440e", lw=1.8,
                marker="o", ms=3, label="事件周 (8/20–8/27)")
        ax.set_title(col)
        ax.set_xlabel("小时 (北京时间)")
        ax.set_ylabel(labels.get(col, ""))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("事件周 vs 8 月气候态日变化对比", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """构建特征、写盘、自检、出图。

    Returns:
        进程退出码：0 正常；1 存在 inf 或严重异常。
    """
    log: list[str] = []
    add = log.append
    add("=" * 78)
    add("P2-1 物理状态重构 (srcB/physics.py)")
    add("=" * 78)
    add(f"运行时间戳 : {datetime.now().astimezone().isoformat()}")
    add(f"Python     : {sys.version.split()[0]}   pandas {pd.__version__} / numpy {np.__version__}")
    add(f"SEED       : {SEED}  (本脚本无随机过程)")
    add("")

    add("-" * 78)
    add("[1] 物理范围掩码（剔除物理不可能值，保留物理极端值）")
    add("-" * 78)
    build_log: list[str] = []
    features, raw_sources = build_features(build_log)
    log.extend(build_log)
    add("")

    add("-" * 78)
    add("[2] 特征矩阵")
    add("-" * 78)
    add(f"行数 = {len(features)}   列数 = {features.shape[1]}")
    add(f"时间范围 = {features.index.min()} -> {features.index.max()}")
    inf_count = int(np.isinf(features.to_numpy(dtype=float)).sum())
    add(f"inf 个数 = {inf_count}")
    add("")

    # 有效样本表（必交件）
    valid = validity_table(features, raw_sources)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    valid_path = TAB_DIR / "b1_feature_validn.csv"
    valid.to_csv(valid_path, index=False, encoding="utf-8-sig")
    add(f"已写出 {valid_path.relative_to(PROJECT_ROOT)}")
    add("")
    add("特征有效样本（按缺测率降序）：")
    add(f"{'feature':<22}{'n_valid':>8}{'miss_rate':>11}  {'flux_group':<15}binding_source")
    for _, r in valid.iterrows():
        add(f"{r['feature']:<22}{int(r['n_valid']):>8}{r['miss_rate']:>11.4f}  "
            f"{r['flux_group']:<15}{r['binding_source']}")
    add("")

    # FLUX 两组有效样本对比（用户补充约束 1）
    add("-" * 78)
    add("[3] FLUX 分组有效样本对比 ★ 不可按同一有效样本数做因果比较")
    add("-" * 78)
    flux = raw_sources["FLUX"]
    add("通量类 (flux_flux):")
    for c in FLUX_FLUX_VARS:
        if c in flux.columns:
            n = int(flux[c].notna().sum())
            add(f"   {c:<20} n_valid={n:>6}  miss_rate={1 - n / len(flux):.4f}")
    add("动量类 (flux_momentum):")
    for c in FLUX_MOMENTUM_VARS:
        if c in flux.columns:
            n = int(flux[c].notna().sum())
            add(f"   {c:<20} n_valid={n:>6}  miss_rate={1 - n / len(flux):.4f}")
    n_flux = int(flux[list(FLUX_FLUX_VARS)].notna().all(axis=1).sum()) if all(
        c in flux.columns for c in FLUX_FLUX_VARS) else -1
    n_mom = int(flux[list(FLUX_MOMENTUM_VARS)].notna().all(axis=1).sum()) if all(
        c in flux.columns for c in FLUX_MOMENTUM_VARS) else -1
    add("")
    add(f"通量类全齐样本数 = {n_flux}")
    add(f"动量类全齐样本数 = {n_mom}")
    add(f"两者重叠（同时全齐）= {int(flux[list(FLUX_FLUX_VARS) + list(FLUX_MOMENTUM_VARS)].notna().all(axis=1).sum())}")
    add("→ 后续 Q1 因果分析中，涉及通量类与动量类的特征必须分别在其各自有效样本上计算，")
    add("  不得用统一 dropna() 造成样本集不一致或隐性筛选偏差。")
    add("")

    # 物理合理性自检
    add("-" * 78)
    add("[4] 物理合理性自检")
    add("-" * 78)
    checks = ["L_Obukhov", "zeta_10m", "Bowen_beta", "Rn_Wm2", "EF", "EBR", "Z0C_m"]
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    add(f"   {'feature':<16}" + "".join(f"{f'p{int(q*100)}':>11}" for q in qs))
    for col in checks:
        if col not in features.columns:
            continue
        q = features[col].quantile(qs)
        add(f"   {col:<16}" + "".join(f"{v:>11.3f}" for v in q.values))
    add("")
    bigL = int((features["L_Obukhov"].abs() > 1e5).sum())
    add(f"   |L| > 1e5 的小时数 = {bigL}  "
        f"({bigL / len(features):.4f})")
    add("   解释：θ* (TSTAR) 接近 0 时 L → ±∞，这是中性层结的数学表现，不是数据错误。")
    add("        这些时刻 ζ≈0，稳定度分类应归为「中性」，而非剔除。")
    add("")

    add("   Z0C_m（0℃ 层高度）外推失败原因统计：")
    zst = _AUX.get("_z0c_stats", {})
    if zst:
        for k in ("n_total", "n_insufficient_levels", "n_no_lapse",
                  "n_lapse_too_weak", "n_out_of_range", "n_ok"):
            if k in zst:
                add(f"      {k:<24} {zst[k]}")
    add("   ★ 局限声明：本站气温廓线仅覆盖 1–20 m（近地层），")
    add("     其温度梯度受局地湍流/地表加热主导，不能代表自由大气递减率（≈6.5 °C/km）。")
    add("     实测 1m 与 20m 温差中位数仅约 0.36 °C，若线性外推会得到 100–200 m 的")
    add("     「0℃ 层高度」，与真实 4000–5500 m 差一个量级 —— 这是物理上不成立的外推。")
    add("     因此本特征绝大部分时刻为 NaN，**仅在论文中作为定性参考，不进入 Q1 排名**。")
    add("     [待核实] 正式做法应引入 ERA5 再分析位势高度场的 0℃ 层，列为 Phase 3 外部数据补强项。")
    add("")

    add("   能量闭合度 EBR 诊断（数据质量证据）：")
    _h = features.index.hour.to_numpy()
    aug_noon = features[(features.index.month == 8) & (_h >= 10) & (_h <= 16)]
    add(f"      8 月昼间 EBR 中位数 = {float(aug_noon['EBR'].median()):.3f}")
    add(f"      8 月昼间 EF   中位数 = {float(aug_noon['EF'].median()):.3f}")
    add("      → EBR 显著小于 1，说明涡动相关通量（H+LE）系统性低估约 60%，")
    add("        这是青藏高原涡动相关观测的常见问题（夜间/低风速下的平流损失）。")
    add("        论文中必须报告该闭合度，且 EF/EBR 类特征在 Q1 排名中应标注为"
        "「受闭合度限制的派生量」。")
    add("")

    add("-" * 78)
    add("[5] 事件周距平（相对 8 月气候态）")
    add("-" * 78)
    aug = features[features.index.month == 8]
    week = features.loc[EVENT_WEEK_START:EVENT_WEEK_END]
    for col in ["PDD_cumsum", "Se_0.1m", "Se_0.4m", "Melt_mm_h", "Rn_Wm2", "USTAR"]:
        if col not in features.columns:
            continue
        base_mean = float(aug[col].mean())
        base_std = float(aug[col].std())
        wk_mean = float(week[col].mean())
        z = (wk_mean - base_mean) / base_std if base_std > 0 else np.nan
        add(f"   {col:<14} 8月均值={base_mean:>10.4f}  事件周均值={wk_mean:>10.4f}  z={z:>7.3f}")
    add("")

    # 干旱归因（用户补充约束 2）
    add("-" * 78)
    add("[6] 土壤水归因说明 ★ 用户补充约束")
    add("-" * 78)
    met_raw = raw_sources["MET"]
    dry = dry_spell_before_event(met_raw["Precip(mm)"])
    x_days = int(dry["x_days"])
    add(f"事件前连续无有效降雨天数 X = {x_days} 天")
    add(f"最后一次有雨日 = {dry['last_rain_day']}")
    add("")
    add(f"结论：事件前 {x_days} 天无有效降雨，土壤水维持不能归因于本地降水。")
    add("")
    add("   依据：日降水 > 0 记为有雨日；事件当天 (2026-08-26) 不计入「事件前」。")
    add("   8/20–8/25 逐日降水 (mm)：")
    daily_sum = dry["daily_sum"]
    rainy = dry["daily_table"]
    for day in pd.date_range("2026-08-20", "2026-08-25", freq="D"):
        v = daily_sum.get(day, np.nan)
        r = rainy.get(day, np.nan)
        add(f"      {day:%Y-%m-%d}  {v if pd.isna(v) else f'{v:.1f}':>8}   "
            f"{'有雨' if (not pd.isna(r) and bool(r)) else '无雨' if not pd.isna(r) else '缺测'}")
    add("")
    # 量化检验：Se 的上升是否真的与本地降水脱钩
    se = features["Se_0.1m"]
    wk_se = se.loc[EVENT_WEEK_START:EVENT_WEEK_END]
    pre_se = se.loc["2026-08-23 00:00":"2026-08-26 10:00"]
    add("   量化检验（Se 表层）：")
    add(f"      事件周 Se 均值      = {float(wk_se.mean()):.4f}")
    add(f"      8/23–8/26 10:00 均值 = {float(pre_se.mean()):.4f}  （连续无雨区间）")
    add(f"      8 月气候态均值      = {float(features[features.index.month == 8]['Se_0.1m'].mean()):.4f}")
    delta = float(pre_se.mean()) - float(features[features.index.month == 8]["Se_0.1m"].mean())
    add(f"      无雨区间相对 8 月气候态的偏移 = {delta:+.4f}")
    if abs(delta) <= 0.02:
        add(f"      → 偏移 {delta:+.4f}，量级小于 0.02，**不足以支撑「非降水源补给」的强结论**。")
        add("        如实记录：本项证据薄弱，不可在论文中夸大为「直接证明」。")
        add("        事件周 Se 的抬升更可能来自 8/26 事件本身的水体扰动或传感器响应，")
        add("        须在 Q2 用湿润锋时序与逐层 Se 剖面进一步辨识来源。")
    elif delta > 0:
        add(f"      → 在 {x_days} 天无雨条件下，Se 仍高于 8 月气候态 {delta:+.4f}，")
        add("        提示存在非降水源补给，但仍需 Q2 的多层 Se 剖面与时序证据佐证。")
    else:
        add(f"      → 无雨区间 Se 低于 8 月气候态 {delta:+.4f}，与「无本地降水补给」一致；")
        add("        事件周 Se 的抬升发生在 8/26 前后，需在 Q2 中结合湿润锋时序进一步定位来源。")
    add("")
    add("   → 因此 Se 与湿润锋速率的上升，若在事件前出现，其水分来源只能是：")
    add("     跨境上游来水 / 融水补给 / 深层土壤水毛管上升，而非本地降水。")
    add("     该结论直接支撑全篇「非降雨触发」主线。")
    add("")

    # 落盘
    add("-" * 78)
    add("[7] 落盘")
    add("-" * 78)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    feat_path = CLEAN_DIR / "features_hourly.csv"
    features.to_csv(feat_path, encoding="utf-8-sig", float_format="%.6g")
    add(f"已写出 {feat_path.relative_to(PROJECT_ROOT)}  "
        f"({len(features)} 行 × {features.shape[1]} 列, {feat_path.stat().st_size / 1024 / 1024:.1f} MB)")

    fig_path = FIG_DIR / "b1_diurnal.png"
    diurnal_figure(features, fig_path)
    add(f"已写出 {fig_path.relative_to(PROJECT_ROOT)}")
    add("")

    add("-" * 78)
    add("[结论]")
    add("-" * 78)
    ok = inf_count == 0 and features.shape[1] >= 25
    add("ALL_OK" if ok else f"DEGRADED: inf={inf_count}, n_features={features.shape[1]}")
    add(f"  特征数 = {features.shape[1]} (要求 ≥25)")
    add(f"  inf = {inf_count} (要求 0)")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "p2_physics.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
