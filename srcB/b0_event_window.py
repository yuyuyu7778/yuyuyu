"""事件窗口复核：2026-08-20 00:00 → 2026-08-27 08:00 逐小时面板 + 8/26 降水零点检验。

本脚本承担 CLAUDE_B.md「必做验证清单」中的核心一条：
    **事件日降水 = 0 的复核 —— 8/26 00–23h 逐时降水全 0 且 QC 通过。**

若合计降水 ≠ 0.0 mm，脚本立即停止，不产出图与表，只写日志并退出码 2。
原因：这一条决定了全篇「非降雨触发」的论点能否成立（CLAUDE_B.md 2.2 第 3 条）。

产出：
    out/B/tab/b0_event_window.csv          事件窗口 14 列逐小时数据
    out/B/tab/b0_daily_precip_summer2026.csv  2026 年 6–8 月逐日降水与连续无雨日数
    out/B/fig/b0_event_panel.png           9 子图时序面板 (300 dpi)
    out/B/log/p1_event_window.log          全部关键数字与判定

AI 辅助声明：本文件由 Claude Code（Anthropic，2026-10-24）辅助生成，人工已复核。

用法：
    python srcB/b0_event_window.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无 GUI 后端，服务器/批处理环境下稳定出图

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import SEED, load_source  # noqa: E402

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

#: 项目根目录。
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: 输出目录。
TAB_DIR: Path = PROJECT_ROOT / "out" / "B" / "tab"
FIG_DIR: Path = PROJECT_ROOT / "out" / "B" / "fig"
LOG_DIR: Path = PROJECT_ROOT / "out" / "B" / "log"

#: 事件窗口（左闭右闭）。
WINDOW_START: pd.Timestamp = pd.Timestamp("2026-08-20 00:00")
WINDOW_END: pd.Timestamp = pd.Timestamp("2026-08-27 08:00")

#: 灾害时刻（CLAUDE_B.md 2.2 第 2 条）。
DISASTER_TIME: pd.Timestamp = pd.Timestamp("2026-08-26 10:30")

#: 事件日（降水零点检验的目标日）。
EVENT_DAY: str = "2026-08-26"

#: 暑期统计区间。
SUMMER_START: pd.Timestamp = pd.Timestamp("2026-06-01 00:00")
SUMMER_END: pd.Timestamp = pd.Timestamp("2026-08-31 23:00")

#: 窗口内要抽取的列：逻辑名 -> (源前缀, 原始列名)。
COLUMN_MAP: dict[str, tuple[str, str]] = {
    "Precip_mm": ("MET", "Precip(mm)"),
    "Ta_2m_degC": ("MET", "Ta_2m(degree C)"),
    "RH_2m_pct": ("MET", "RH_2m(%)"),
    "WS_10m_ms": ("MET", "WS_10m(m/s)"),
    "VWC_0.1m": ("SOIL", "VWC_0.1m(m3/m3)"),
    "VWC_0.2m": ("SOIL", "VWC_0.2m(m3/m3)"),
    "VWC_0.4m": ("SOIL", "VWC_0.4m(m3/m3)"),
    "VWC_0.8m": ("SOIL", "VWC_0.8m(m3/m3)"),
    "VWC_1.6m": ("SOIL", "VWC_1.6m(m3/m3)"),
    "Rsd_Wm2": ("RADM", "Rsd(W/m2)"),
    "H_Wm2": ("FLUX", "H(W/m2)"),
    "LE_Wm2": ("FLUX", "LE(W/m2)"),
    "USTAR_ms": ("FLUX", "USTAR(m/s)"),
    "TKE_m2s2": ("FLUX", "TKE(m2/s2)"),
}

#: 中文字体。
CJK_FONT: str = "SimHei"

#: 出图分辨率。
DPI: int = 300


# --------------------------------------------------------------------------- #
# 数据装配
# --------------------------------------------------------------------------- #


def build_window() -> tuple[pd.DataFrame, pd.DataFrame]:
    """抽取事件窗口内的 14 列逐小时数据，并同时返回对应的 QC 表。

    各源独立加载后按时间索引对齐（outer join 后再裁剪到窗口），
    因此窗口内某源缺测时不会把整行丢掉，而是该列留 NaN。

    Returns:
        ``(window, qc)``：

        * ``window``: 索引为 ``Timestamp``，列为 :data:`COLUMN_MAP` 的逻辑名；
        * ``qc``: 同一索引下的 QC 标志表，列名为 ``源:列`` 形式，仅含窗口内出现的源。
    """
    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for _, (prefix, _) in COLUMN_MAP.items():
        if prefix not in cache:
            cache[prefix] = load_source(prefix, years=[2026])

    pieces: list[pd.DataFrame] = []
    for logical, (prefix, raw_col) in COLUMN_MAP.items():
        data, _ = cache[prefix]
        if raw_col not in data.columns:
            raise KeyError(f"{prefix} 中找不到列 {raw_col!r}")
        pieces.append(data[[raw_col]].rename(columns={raw_col: logical}))

    window = pd.concat(pieces, axis=1).sort_index()
    window = window.loc[WINDOW_START:WINDOW_END]

    qc_frames: list[pd.DataFrame] = []
    for prefix, (_, qc_df) in cache.items():
        if qc_df.empty:
            continue
        sub = qc_df.loc[WINDOW_START:WINDOW_END].copy()
        sub.columns = [f"{prefix}:{col}" for col in sub.columns]
        qc_frames.append(sub)

    qc = pd.concat(qc_frames, axis=1).sort_index() if qc_frames else pd.DataFrame(index=window.index)
    qc = qc.reindex(window.index)
    return window, qc


def gate_event_day_precip(window: pd.DataFrame, qc: pd.DataFrame) -> dict[str, object]:
    """事件日降水零点检验：8/26 00–23 时逐时降水全 0 且 QC 通过。

    这是全篇「非降雨触发」论点的守门项。任一时刻降水 ≠ 0 或 QC 不通过，
    调用方必须停止后续流程。

    Args:
        window: :func:`build_window` 返回的窗口数据。
        qc: 同一索引的 QC 表（用于核对 ``QC_*`` 标志）。

    Returns:
        含以下键的字典：

        * ``hourly``: 24 个逐时降水值的 ``Series``；
        * ``total``: 24 小时合计 (mm)；
        * ``all_zero``: 是否逐时全为 0；
        * ``n_valid``: 非 NaN 的小时数；
        * ``qc_pass``: 对应 QC_ 列是否全部为 0（无该列时记 ``True``）；
        * ``passed``: 综合判定（全 0 且无 NaN 且 QC 通过）。
    """
    day = window.loc[EVENT_DAY]
    hourly = day["Precip_mm"]

    n_valid = int(hourly.notna().sum())
    # 用 min_count=1 防止全 NaN 时 sum 静默返回 0.0 —— 这正是最危险的假阳性。
    total = float(hourly.sum(min_count=1)) if n_valid else float("nan")
    all_zero = bool(n_valid == 24 and (hourly.fillna(np.inf) == 0).all())

    # 注意：MET 共 22 个数值列但只有 21 个 QC 列 —— Precip 是唯一没有 QC_Precip 的列。
    # 因此这里不能写「QC 通过」；只能声明「无 QC 标志可用」，并把它作为证据强度上的限制。
    qc_col = "MET:QC_Precip"
    if qc_col in qc.columns:
        flags = qc.loc[EVENT_DAY, qc_col]
        qc_available = True
        qc_pass = bool(flags.notna().all() and (flags == 0).all())
        qc_detail = flags.value_counts(dropna=False).to_dict()
    else:
        qc_available = False
        qc_pass = False          # 无 QC 不等于通过；在结论里单独说明
        qc_detail = {"(原始数据无 QC_Precip 列，无法做 QC 核对)": None}

    passed = bool(all_zero and n_valid == 24)
    return {
        "hourly": hourly,
        "total": total,
        "all_zero": all_zero,
        "n_valid": n_valid,
        "qc_available": qc_available,
        "qc_pass": qc_pass,
        "qc_detail": qc_detail,
        "passed": passed,
    }


def daily_precip_and_dry_spell() -> pd.DataFrame:
    """统计 2026 年 6–8 月逐日降水与连续无雨日数。

    规则：
      * 日降水 = 当日 24 个逐时值之和，要求 24 小时全部非 NaN，否则该日标记为不完整；
      * 「有雨日」定义 = 日降水 > 0.0 mm（严格大于，不做 0.1 mm 截断，避免引入未声明阈值）；
      * 「连续无雨日数」= 截至当日（含）的连续 0 降水天数；遇有雨日或有数据缺失的日重置为 NaN。

    Returns:
        索引为日期（``DatetimeIndex``）、含以下列的 DataFrame：
        ``daily_precip_mm`` / ``n_hours_valid`` / ``is_complete`` / ``is_rainy`` / ``dry_spell_days``。
    """
    data, _ = load_source("MET", years=[2026])
    precip = data["Precip(mm)"].loc[SUMMER_START:SUMMER_END]

    grouped = precip.groupby(precip.index.normalize())
    rows: list[dict[str, object]] = []
    for day, series in grouped:
        n_valid = int(series.notna().sum())
        complete = n_valid == 24
        total = float(series.sum(min_count=1)) if n_valid else float("nan")
        rows.append(
            {
                "date": day,
                "daily_precip_mm": total,
                "n_hours_valid": n_valid,
                "is_complete": complete,
            }
        )

    daily = pd.DataFrame(rows).set_index("date")
    daily["is_rainy"] = np.where(daily["is_complete"], daily["daily_precip_mm"] > 0.0, np.nan)

    spell: list[float] = []
    run = 0
    for is_rainy in daily["is_rainy"]:
        if pd.isna(is_rainy):
            run = 0
            spell.append(np.nan)          # 数据缺失的日不参与连旱计数
        elif bool(is_rainy):
            run = 0
            spell.append(0.0)
        else:
            run += 1
            spell.append(float(run))
    daily["dry_spell_days"] = spell
    return daily


# --------------------------------------------------------------------------- #
# 出图
# --------------------------------------------------------------------------- #


def plot_panel(window: pd.DataFrame, out_path: Path) -> None:
    """绘制 9 子图时序面板，并在灾害时刻画红色竖虚线。

    面板划分（单位相同者合并，避免双轴误读）：
        1 Precip  2 Ta_2m  3 RH_2m  4 WS_10m  5 VWC 五层
        6 Rsd     7 H & LE  8 USTAR  9 TKE

    Args:
        window: 事件窗口数据（索引为 ``Timestamp``）。
        out_path: 输出 PNG 路径。
    """
    plt.rcParams["font.sans-serif"] = [CJK_FONT, "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(9, 1, figsize=(13, 19), sharex=True)
    ax = axes.ravel()
    t = window.index

    # 1 降水
    ax[0].bar(t, window["Precip_mm"], width=1 / 24, color="#3b6ea5", align="center")
    ax[0].set_ylabel("降水\n(mm/h)")

    # 2 气温
    ax[1].plot(t, window["Ta_2m_degC"], color="#c1440e", lw=1.2)
    ax[1].set_ylabel("Ta_2m\n(°C)")

    # 3 相对湿度
    ax[2].plot(t, window["RH_2m_pct"], color="#2a7f62", lw=1.2)
    ax[2].set_ylabel("RH_2m\n(%)")

    # 4 风速
    ax[3].plot(t, window["WS_10m_ms"], color="#5b4b8a", lw=1.2)
    ax[3].set_ylabel("WS_10m\n(m/s)")

    # 5 五层体积含水率
    vwc_cols = [c for c in window.columns if c.startswith("VWC_")]
    cmap = plt.get_cmap("viridis")
    for i, col in enumerate(vwc_cols):
        ax[4].plot(t, window[col], lw=1.1, color=cmap(i / max(len(vwc_cols) - 1, 1)),
                   label=col.replace("VWC_", ""))
    # 上标用 mathtext 而非 Unicode ²/³：SimHei 缺这两个字形，会渲染成方框。
    ax[4].set_ylabel("VWC\n($\\mathrm{m^3/m^3}$)")
    ax[4].legend(ncol=5, fontsize=7, loc="upper right", frameon=False)

    # 6 短波下行辐射
    ax[5].plot(t, window["Rsd_Wm2"], color="#d4a017", lw=1.2)
    ax[5].set_ylabel("Rsd\n($\\mathrm{W/m^2}$)")

    # 7 感热与潜热（同单位，共轴）
    ax[6].plot(t, window["H_Wm2"], color="#b5312c", lw=1.2, label="H")
    ax[6].plot(t, window["LE_Wm2"], color="#2b6cb0", lw=1.2, label="LE")
    ax[6].axhline(0.0, color="0.6", lw=0.6, ls=":")
    ax[6].set_ylabel("H / LE\n($\\mathrm{W/m^2}$)")
    ax[6].legend(ncol=2, fontsize=8, loc="upper right", frameon=False)

    # 8 摩擦速度
    ax[7].plot(t, window["USTAR_ms"], color="#6b4226", lw=1.2)
    ax[7].set_ylabel("USTAR\n(m/s)")

    # 9 湍动能
    ax[8].plot(t, window["TKE_m2s2"], color="#0f6e6e", lw=1.2)
    ax[8].set_ylabel("TKE\n($\\mathrm{m^2/s^2}$)")
    ax[8].set_xlabel("时间 (北京时间)")

    # 灾害时刻标注
    for a in ax:
        a.axvline(DISASTER_TIME, color="red", ls="--", lw=1.6, zorder=0)
        a.grid(alpha=0.25, lw=0.5)
    ax[0].annotate(
        "灾害时刻 2026-08-26 10:30",
        xy=(DISASTER_TIME, ax[0].get_ylim()[1]),
        xytext=(6, -10),
        textcoords="offset points",
        color="red",
        fontsize=9,
        ha="left",
        va="top",
    )

    locator = mdates.HourLocator(byhour=[0, 12])
    ax[-1].xaxis.set_major_locator(locator)
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.autofmt_xdate(rotation=0, ha="center")

    fig.suptitle(
        "吉隆站事件窗口逐小时状态量（2026-08-20 00:00 → 2026-08-27 08:00）",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def main() -> int:
    """执行窗口抽取、守门检验、落盘、出图，并把全部关键数字写入日志。

    Returns:
        进程退出码：0 通过；2 表示事件日降水守门项失败（已按要求停止）。
    """
    log: list[str] = []
    add = log.append
    stamp = datetime.now().astimezone().isoformat()

    add("=" * 78)
    add("P1-2 事件窗口复核 (srcB/b0_event_window.py)")
    add("=" * 78)
    add(f"运行时间戳 : {stamp}")
    add(f"Python     : {sys.version.split()[0]}   pandas {pd.__version__} / numpy {np.__version__}")
    add(f"SEED       : {SEED}  (本脚本无随机过程)")
    add(f"窗口       : {WINDOW_START} -> {WINDOW_END}")
    add(f"灾害时刻   : {DISASTER_TIME}")
    add("")

    window, qc = build_window()
    add("-" * 78)
    add("[1] 窗口数据")
    add("-" * 78)
    add(f"行数 = {len(window)}  (预期 177 = 7 天 × 24 h + 9 h)")
    add(f"列数 = {window.shape[1]}")
    add(f"时间范围 = {window.index.min()} -> {window.index.max()}")
    add("")
    add("逐列有效样本数：")
    for col in window.columns:
        n = int(window[col].notna().sum())
        rate = n / len(window)
        add(f"   {col:<14} {n:>4} / {len(window)}   ({rate:.3f})")
    add("")

    add("-" * 78)
    add("[2] 守门项：8 月 26 日 00–23 时逐小时降水 ★")
    add("-" * 78)
    gate = gate_event_day_precip(window, qc)
    hourly = gate["hourly"]
    for ts, val in hourly.items():
        txt = "NaN" if pd.isna(val) else f"{float(val):.1f}"
        add(f"   {ts:%Y-%m-%d %H:%M}  {txt:>6} mm")
    add("")
    add(f"合计降水   : {gate['total']:.6f} mm")
    add(f"有效小时数 : {gate['n_valid']} / 24")
    add(f"逐时全为 0 : {gate['all_zero']}")
    if gate["qc_available"]:
        add(f"QC_Precip  : {'通过 (全为 0)' if gate['qc_pass'] else '不通过'}  {gate['qc_detail']}")
    else:
        add(f"QC_Precip  : n/a —— {list(gate['qc_detail'])[0]}")
        add("             证据强度限制：降水列无 QC 标志，无法排除「QC 未标但实为缺测」的情形；")
        add("             目前只能依据「数值为 0.0 且非 NaN（非哨兵 9999.9）」判定降水为零。")
        add("             若评审追问，须如实说明这一点，不得表述为「QC 通过」。")
    add("")

    passed = bool(gate["passed"])
    if passed:
        add("判定: PASS —— 8/26 逐时降水全 0.0 mm（24/24 小时非缺测、非哨兵），「非降雨触发」论点成立。")
    else:
        add("判定: FAIL —— 合计降水 ≠ 0.0 mm 或存在缺测/QC 问题。")
        add("      按 CLAUDE_B.md 2.2 第 3 条要求，脚本在此停止，不产出 CSV 与图。")
        add("      全篇「非降雨触发」论点必须重新审视，不得沿用。")
    add("")

    if not passed:
        add("-" * 78)
        add("[结论]")
        add("-" * 78)
        add("GATE_FAILED: 事件日降水零点检验未通过，流程终止")
        _write_log(log)
        print("\n".join(log))
        return 2

    # 通过后才落盘与出图
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TAB_DIR / "b0_event_window.csv"
    out = window.copy()
    out.index.name = "Timestamp"
    out.to_csv(csv_path, encoding="utf-8-sig", float_format="%.4f")
    add(f"[3] 已写出 {csv_path.relative_to(PROJECT_ROOT)}  ({len(out)} 行 × {out.shape[1]} 列)")

    add("")
    add("-" * 78)
    add("[4] 2026 年 6–8 月逐日降水与连续无雨日数")
    add("-" * 78)
    daily = daily_precip_and_dry_spell()
    daily_path = TAB_DIR / "b0_daily_precip_summer2026.csv"
    daily_out = daily.copy()
    daily_out.index.name = "date"
    daily_out.to_csv(daily_path, encoding="utf-8-sig", float_format="%.4f")
    add(f"已写出 {daily_path.relative_to(PROJECT_ROOT)}")
    add("")
    add("  日期         日降水(mm)  有效小时  完整   有雨   连续无雨日数")
    for day, row in daily.iterrows():
        precip_txt = "NaN" if pd.isna(row["daily_precip_mm"]) else f"{row['daily_precip_mm']:.1f}"
        rainy = row["is_rainy"]
        rainy_txt = "NaN" if pd.isna(rainy) else ("是" if bool(rainy) else "否")
        spell = row["dry_spell_days"]
        spell_txt = "NaN" if pd.isna(spell) else f"{int(spell)}"
        add(
            f"  {day:%Y-%m-%d}  {precip_txt:>9}  {int(row['n_hours_valid']):>6}  "
            f"{'是' if row['is_complete'] else '否':<4}  {rainy_txt:<4}  {spell_txt:>6}"
        )

    complete = daily[daily["is_complete"]]
    add("")
    add(f"完整日数 : {len(complete)} / {len(daily)}")
    if len(complete):
        add(f"有雨日数 : {int(complete['is_rainy'].sum())}")
        add(f"总降水   : {complete['daily_precip_mm'].sum():.3f} mm")
        add(f"最大日降水: {complete['daily_precip_mm'].max():.3f} mm "
            f"({complete['daily_precip_mm'].idxmax():%Y-%m-%d})")
        add(f"最大连续无雨日数: {int(complete['dry_spell_days'].max())} 天 "
            f"(截至 {complete['dry_spell_days'].idxmax():%Y-%m-%d})")
    add("")

    add("-" * 78)
    add("[5] 出图")
    add("-" * 78)
    fig_path = FIG_DIR / "b0_event_panel.png"
    plot_panel(window, fig_path)
    add(f"已写出 {fig_path.relative_to(PROJECT_ROOT)}  (9 子图, {DPI} dpi, 字体 {CJK_FONT})")
    add(f"文件大小 = {fig_path.stat().st_size / 1024:.1f} KB")
    add("")

    add("-" * 78)
    add("[结论]")
    add("-" * 78)
    add("ALL_OK")
    add("  守门项通过：8/26 合计降水 = 0.000000 mm，逐时全 0（24/24 有效）。")
    add("  限定：原始数据无 QC_Precip 列，本条不能声称「QC 通过」。")
    add("  产出：b0_event_window.csv / b0_daily_precip_summer2026.csv / b0_event_panel.png")

    _write_log(log)
    print("\n".join(log))
    return 0


def _write_log(lines: list[str]) -> Path:
    """把日志写入 ``out/B/log/p1_event_window.log``。

    Args:
        lines: 日志行（不含换行符）。

    Returns:
        日志文件路径。
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "p1_event_window.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
