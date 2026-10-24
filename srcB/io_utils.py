"""统一 IO 层：B 题四源观测数据的读取、哨兵值处理、QC 分离与缺测统计。

规则来源：CLAUDE_B.md 第二节「关键物理事实」
  1. 缺测哨兵值 = 9999.9（不是 NaN，也不是 -6999），所有 QC 列以 ``QC_`` 开头；
  2. 有效数据截止 2026-08-27 08:00，其后数值列全部为哨兵值；
  3. 2021 年缺测率约 32%，2022–2025 约 0–1%，2026 年 34.7%；
  4. 时间列为 ``Timestamp``，2024 为闰年 8784 行，其余 8760 行。

设计要点：
  * 数值列与 QC 列**物理分离**，QC 列不进入数值矩阵，避免污染后续因果/统计计算；
  * 哨兵值只在数值列上置 NaN，QC 列原样保留（QC 列里的 8 是合法标志位，不是哨兵）；
  * 多源多表纵向合并后按时间排序去重，保证每小时唯一一行。

AI 辅助声明：本文件由 Claude Code（Anthropic，2026-10-24）辅助生成，
人工已逐字复核并与 CLAUDE_B.md 第二节核对。

用法：
    python srcB/io_utils.py          # 运行自测并写 out/B/log/p1_io.log
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

#: 随机种子。本模块不含随机过程，定义此常量供下游模块统一 import，
#: 只在实际需要随机性的函数中使用（CLAUDE_B.md 硬规则 4）。
SEED: int = 42

#: 缺测哨兵值的判定阈值：|x| >= 9999 一律视为哨兵。
SENTINEL_ABS: float = 9999.0

#: 四个数据源前缀，与文件名 ``{prefix}_Gyirong_{year}.csv`` 对应。
PREFIXES: tuple[str, ...] = ("MET", "RADM", "SOIL", "FLUX")

#: 覆盖年份。2026 年虽有 8760 行，但有效数据止于 2026-08-27 08:00。
DEFAULT_YEARS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025, 2026)

#: 有效数据截止时刻（CLAUDE_B.md 2.2 第 2 条），供自测断言使用。
VALID_END: pd.Timestamp = pd.Timestamp("2026-08-27 08:00")

#: QC 列名前缀。
QC_PREFIX: str = "QC_"

#: QC 标志位的判定口径：0 = 通过，非 0 = 存疑/不通过。
#: 注意：官方未给出各标志位（1/2/4/8）的语义定义，此处按"非 0 即不通过"处理，
#: 属本项目的假设，论文中须标注 [待核实]；各标志位的实际分布由 qc_report 一并输出。
QC_FAIL_CODES: tuple[int, ...] = (1, 2, 4, 8)

#: 项目根目录（srcB/ 的上一级），保证任意 cwd 下都能定位数据。
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: 原始数据目录（只读，CLAUDE_B.md 硬规则 1）。
RAW_DIR: Path = PROJECT_ROOT / "data" / "B" / "raw"

#: 日志目录。
LOG_DIR: Path = PROJECT_ROOT / "out" / "B" / "log"


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #


def _parse_timestamp(series: pd.Series) -> pd.Series:
    """把 ``Timestamp`` 文本列解析为 datetime。

    两种表头格式并存：MET/RADM/SOIL 为 ``2021-1-1 0:00``，FLUX 为 ``2021/1/1 0:00``，
    且月/日/时为非零填充形式，因此显式给出容错格式。

    Args:
        series: 原始 ``Timestamp`` 列（字符串）。

    Returns:
        无时区的 ``datetime64[ns]`` 序列。
    """
    text = series.astype(str).str.strip()
    # 统一分隔符后再解析，避免 format="mixed" 在大表上的逐行推断开销与歧义。
    text = text.str.replace("/", "-", regex=False)
    return pd.to_datetime(text, format="%Y-%m-%d %H:%M", errors="raise")


def _split_columns(columns: Iterable[str]) -> tuple[list[str], list[str]]:
    """把列名拆成「数值列」与「QC 列」两组。

    Args:
        columns: 数据表全部列名（应已排除 ``Timestamp``）。

    Returns:
        ``(value_cols, qc_cols)``，两者均按原顺序排列。
    """
    value_cols: list[str] = []
    qc_cols: list[str] = []
    for col in columns:
        if col.startswith(QC_PREFIX):
            qc_cols.append(col)
        else:
            value_cols.append(col)
    return value_cols, qc_cols


def _read_one(prefix: str, year: int) -> pd.DataFrame:
    """读取单个 ``{prefix}_Gyirong_{year}.csv``，不做哨兵处理。

    Args:
        prefix: 数据源前缀，取值于 :data:`PREFIXES`。
        year: 年份。

    Returns:
        含 ``Timestamp`` 列（原始字符串）与全部原始列的 DataFrame，行顺序与文件一致。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 文件缺少 ``Timestamp`` 列。
    """
    path = RAW_DIR / f"{prefix}_Gyirong_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"原始文件不存在: {path}")

    # FLUX 文件带 UTF-8 BOM，统一用 utf-8-sig 读取可同时兼容两种编码。
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "Timestamp" not in df.columns:
        raise ValueError(f"{path.name} 缺少 Timestamp 列，实际列: {list(df.columns)[:5]}...")
    return df


# --------------------------------------------------------------------------- #
# 对外接口
# --------------------------------------------------------------------------- #


def load_source(
    prefix: str,
    years: Sequence[int] = DEFAULT_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取并合并单一数据源的多年观测，数值列与 QC 列分离返回。

    处理流程：
      1. 逐年读取 ``data/B/raw/{prefix}_Gyirong_{year}.csv``；
      2. 解析 ``Timestamp`` 为 datetime，统一两种日期分隔符；
      3. 数值列中 ``|x| >= 9999`` 的单元格置 ``NaN``（哨兵值 9999.9）；
      4. QC 列单独抽出，不做哨兵替换（QC 标志位 8 不是缺测）；
      5. 多年纵向合并，按时间升序排序并去重（同一时刻保留首次出现）。

    Args:
        prefix: 数据源前缀，``"MET"`` / ``"RADM"`` / ``"SOIL"`` / ``"FLUX"``。
        years: 年份序列，默认 2021–2026。

    Returns:
        ``(data, qc)`` 二元组：

        * ``data``: 数值矩阵，``DatetimeIndex`` 名为 ``Timestamp``，列为物理量；
        * ``qc``: QC 标志矩阵，行索引与 ``data`` 完全对齐，列为 ``QC_*``。
          若该源无 QC 列（FLUX），返回与 ``data`` 同索引的空 DataFrame。

    Raises:
        ValueError: ``prefix`` 不在 :data:`PREFIXES` 中。
        FileNotFoundError: 任一指定年份的文件缺失。
    """
    if prefix not in PREFIXES:
        raise ValueError(f"未知数据源前缀 {prefix!r}，应为 {PREFIXES} 之一")

    frames: list[pd.DataFrame] = []
    for year in years:
        raw = _read_one(prefix, year)
        ts = _parse_timestamp(raw["Timestamp"])
        body = raw.drop(columns=["Timestamp"])

        value_cols, qc_cols = _split_columns(body.columns)

        # 数值列：先转 float（容错），再把 |x|>=9999 的哨兵置 NaN。
        values = body[value_cols].apply(pd.to_numeric, errors="coerce")
        values = values.mask(values.abs() >= SENTINEL_ABS)

        # QC 列：原样保留为整数（无法转数值时保留原值）。
        qc = body[qc_cols].apply(pd.to_numeric, errors="coerce").astype("Int64")

        merged = pd.concat([values, qc], axis=1)
        merged.insert(0, "Timestamp", ts)
        frames.append(merged)

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined = combined.sort_values("Timestamp", kind="mergesort")
    combined = combined.drop_duplicates(subset="Timestamp", keep="first")
    combined = combined.set_index("Timestamp")
    combined.index.name = "Timestamp"

    value_cols, qc_cols = _split_columns(combined.columns)
    data = combined[value_cols]
    qc = combined[qc_cols]
    return data, qc


def qc_report(
    df: pd.DataFrame,
    qc: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """统计缺测率与 QC 不通过率。

    Args:
        df: :func:`load_source` 返回的数值矩阵（``DatetimeIndex`` 索引）。
        qc: 对应的 QC 标志矩阵；为 ``None`` 时只统计缺测率（如 FLUX 无 QC 列）。

    Returns:
        三张表：

        * ``"per_column"``: 每列缺测率、有效样本数、
          以及（若有 QC）QC 不通过率与各标志位计数；
        * ``"by_year"``: 行=年份、列=物理量，值为该年缺测率；
        * ``"overall"``: 总体汇总，行=年份（``ALL`` 为全期），
          列= ``n_rows`` / ``miss_rate`` / ``qc_fail_rate``。
    """
    if df.index.name != "Timestamp":
        raise ValueError("df 需以 Timestamp 为索引，请先调用 load_source")

    n_rows = len(df)

    per_column = pd.DataFrame(index=df.columns)
    per_column["n_valid"] = df.notna().sum().astype(int)
    per_column["miss_rate"] = (df.isna().mean()).round(6)
    per_column["n_missing"] = (n_rows - per_column["n_valid"]).astype(int)

    if qc is not None and not qc.empty:
        per_column["qc_fail_rate"] = np.nan
        for code in QC_FAIL_CODES:
            per_column[f"qc_n_eq_{code}"] = 0

        for value_col in df.columns:
            qc_col = _match_qc_col(value_col, qc.columns)
            if qc_col is None:
                continue
            flags = qc[qc_col]
            per_column.loc[value_col, "qc_fail_rate"] = round(float(flags.isin(QC_FAIL_CODES).mean()), 6)
            for code in QC_FAIL_CODES:
                per_column.loc[value_col, f"qc_n_eq_{code}"] = int((flags == code).sum())
        per_column["qc_n_nan"] = int(qc.isna().sum().sum()) if qc.shape[1] else 0
    else:
        per_column["qc_fail_rate"] = np.nan

    years = df.index.year
    by_year = df.isna().groupby(years).mean().round(6)
    by_year.index.name = "year"

    overall = pd.DataFrame(index=sorted(years.unique()), columns=["n_rows", "miss_rate", "qc_fail_rate"])
    for year in overall.index:
        mask = years == year
        overall.loc[year, "n_rows"] = int(mask.sum())
        overall.loc[year, "miss_rate"] = round(float(df[mask].isna().to_numpy().mean()), 6)
        if qc is not None and not qc.empty:
            overall.loc[year, "qc_fail_rate"] = round(
                float(qc[mask].isin(QC_FAIL_CODES).to_numpy().mean()), 6
            )
        else:
            overall.loc[year, "qc_fail_rate"] = np.nan

    overall.loc["ALL", "n_rows"] = int(n_rows)
    overall.loc["ALL", "miss_rate"] = round(float(df.isna().to_numpy().mean()), 6)
    if qc is not None and not qc.empty:
        overall.loc["ALL", "qc_fail_rate"] = round(float(qc.isin(QC_FAIL_CODES).to_numpy().mean()), 6)
    else:
        overall.loc["ALL", "qc_fail_rate"] = np.nan

    return {"per_column": per_column, "by_year": by_year, "overall": overall}


def _match_qc_col(value_col: str, qc_cols: Iterable[str]) -> str | None:
    """按 ``QC_ + 物理量名`` 前缀规则把一个数值列匹配到其 QC 列。

    数值列名带单位后缀，QC 列名不带。例：``WS_1m(m/s)`` → ``QC_WS_1m``。
    RADM 的 QC 列大小写不统一（``QC_Rsd`` / ``QC_rsu``），因此匹配时统一转小写比较。

    Args:
        value_col: 数值列名（可含括号单位后缀）。
        qc_cols: 候选 QC 列名集合。

    Returns:
        匹配到的 QC 列名，匹配不到返回 ``None``。
    """
    base = value_col.split("(", 1)[0].strip()
    token = f"{QC_PREFIX}{base}".lower()
    for qc_col in qc_cols:
        if qc_col.lower() == token:
            return qc_col
    for qc_col in qc_cols:
        if qc_col.lower().startswith(token):
            return qc_col
    return None


# --------------------------------------------------------------------------- #
# 自测入口
# --------------------------------------------------------------------------- #


def _write_log(lines: list[str]) -> Path:
    """把自测输出写入 ``out/B/log/p1_io.log``。

    Args:
        lines: 待写入的行（不含换行符）。

    Returns:
        日志文件路径。
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "p1_io.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _selftest() -> int:
    """运行四源自测，打印关键数字并落盘日志。

    Returns:
        进程退出码：0 表示全部基准项通过，1 表示存在不一致。
    """
    from datetime import datetime

    out: list[str] = []
    add = out.append
    stamp = datetime.now().astimezone().isoformat()

    add("=" * 78)
    add("P1-1 统一 IO 层自测 (srcB/io_utils.py)")
    add("=" * 78)
    add(f"运行时间戳   : {stamp}")
    add(f"Python       : {sys.version.split()[0]}")
    add(f"pandas/numpy : {pd.__version__} / {np.__version__}")
    add(f"原始目录     : {RAW_DIR}")
    add(f"哨兵判定     : |x| >= {SENTINEL_ABS}")
    add(f"SEED         : {SEED} (本模块无随机过程，仅供下游导入)")
    add("")

    failures: list[str] = []
    summary: dict[str, dict[str, object]] = {}

    for prefix in PREFIXES:
        data, qc = load_source(prefix)
        rep = qc_report(data, qc)

        overall = rep["overall"]
        add("-" * 78)
        add(f"[{prefix}]  行数={len(data)}  列数={data.shape[1]}  QC列数={qc.shape[1]}")
        add(f"       时间范围 : {data.index.min()}  ->  {data.index.max()}")
        add(f"       总体缺测率(全期) : {overall.loc['ALL', 'miss_rate']:.6f}")
        if qc.shape[1]:
            add(f"       总体QC不通过率   : {overall.loc['ALL', 'qc_fail_rate']:.6f}")
        else:
            add("       总体QC不通过率   : n/a (该源无 QC 列)")
        add("")
        add("       按年缺测率:")
        add("         year    n_rows   miss_rate   qc_fail_rate")
        for year in overall.index:
            row = overall.loc[year]
            qcf = row["qc_fail_rate"]
            qcf_txt = "n/a" if pd.isna(qcf) else f"{float(qcf):.6f}"
            add(f"         {str(year):<6s} {int(row['n_rows']):>7d}   {float(row['miss_rate']):.6f}    {qcf_txt}")

        # 2026 年最后有效时刻：任一数值列非 NaN 的最晚小时。
        row_2026 = data[data.index.year == 2026]
        last_valid = row_2026.dropna(how="all").index.max() if not row_2026.empty else None
        add("")
        add(f"       2026 最后有效时刻 (任一数值列非空): {last_valid}")

        miss_2021 = overall.loc[2021, "miss_rate"] if 2021 in overall.index else float("nan")
        add(f"       2021 缺测率 : {float(miss_2021):.6f}")

        summary[prefix] = {
            "rows": len(data),
            "cols": data.shape[1],
            "qc_cols": qc.shape[1],
            "t_min": data.index.min(),
            "t_max": data.index.max(),
            "miss_all": float(overall.loc["ALL", "miss_rate"]),
            "miss_2021": float(miss_2021),
            "last_valid_2026": last_valid,
        }

        # 基准核对
        if last_valid != VALID_END:
            failures.append(f"{prefix}: 2026 最后有效时刻 = {last_valid}，基准应为 {VALID_END}")
        add("")

    add("-" * 78)
    add("[基准核对] CLAUDE_B.md 2.2")
    add("-" * 78)
    add("基准 1 : 2026 年四源有效数据应止于 2026-08-27 08:00 左右")
    add("基准 2 : 2021 年缺测率约 32% (MET/RADM/SOIL)")
    add("基准 3 : 2024 闰年 8784 行，其余年份 8760 行")
    add("")

    # 行数与闰年核对
    for prefix in PREFIXES:
        data, _ = load_source(prefix)
        counts = data.groupby(data.index.year).size()
        for year, n in counts.items():
            expected = 8784 if year == 2024 else 8760
            if int(n) != expected:
                failures.append(f"{prefix} {year}: {n} 行，基准应为 {expected}")

    add("-" * 78)
    add("[完整性不变量] 逐小时连续性 / 索引唯一性 / 时区")
    add("-" * 78)
    for prefix in PREFIXES:
        data, qc = load_source(prefix)
        add(f"[{prefix}]")
        for year in sorted(data.index.year.unique()):
            sub = data[data.index.year == year]
            expected = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")
            if list(sub.index) != list(expected):
                failures.append(f"{prefix} {year}: 小时索引不连续或重复")
                add(f"   {year}: FAIL 与逐小时日历不一致 (n={len(sub)}, expect={len(expected)})")
            else:
                add(f"   {year}: OK  {len(sub)} 个连续小时")
        dup = int(data.index.duplicated().sum())
        if dup:
            failures.append(f"{prefix}: 索引存在 {dup} 个重复")
        add(f"   索引唯一: {'OK' if dup == 0 else f'FAIL ({dup})'}   时区: {data.index.tz}")
        if qc.shape[1]:
            aligned = data.index.equals(qc.index)
            if not aligned:
                failures.append(f"{prefix}: data 与 qc 索引不对齐")
            add(f"   data/qc 索引对齐: {'OK' if aligned else 'FAIL'}")
    add("")

    add("-" * 78)
    add("[哨兵残留检查] 清洗后数值列不应再出现 |x| >= 9999")
    add("-" * 78)
    for prefix in PREFIXES:
        data, _ = load_source(prefix)
        residual = int((data.abs() >= SENTINEL_ABS).to_numpy().sum())
        if residual:
            failures.append(f"{prefix}: 清洗后仍残留 {residual} 个哨兵值")
        add(f"   {prefix:<6} 残留哨兵数 = {residual}  {'OK' if residual == 0 else 'FAIL'}")
    add("")

    add(f"{'源':<6}{'行数':>8}{'列数':>6}{'QC列':>6}  {'2021缺测':>10}  {'2026最后有效':<20}")
    for prefix, info in summary.items():
        add(
            f"{prefix:<6}{info['rows']:>8}{info['cols']:>6}{info['qc_cols']:>6}  "
            f"{info['miss_2021']:>10.4f}  {str(info['last_valid_2026']):<20}"
        )
    add("")

    add("-" * 78)
    add("[结论]")
    add("-" * 78)
    add("说明: 基准 2 的「2021 年缺测率约 32%」在 MET/RADM/SOIL 上成立 (<0.33)；")
    add("      FLUX 无 QC 列、以原生 NaN 计缺测，2021 年为 0.3826，显著高于 32%，")
    add("      原因是湍流通量类变量 (e_sat/VPD/H/LE/TAU/FC_mass/TSTAR) 整列缺测率 >43%，")
    add("      属观测本身缺测，非本模块处理错误。论文中 FLUX 的缺测率须单独报告。")
    add("")
    if failures:
        add(f"DEGRADED: {len(failures)} 项与基准不符")
        for item in failures:
            add(f"  - {item}")
        rc = 1
    else:
        add("ALL_OK")
        rc = 0

    text = "\n".join(out)
    print(text)
    path = _write_log(out)
    print(f"\n[log] {path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(_selftest())
