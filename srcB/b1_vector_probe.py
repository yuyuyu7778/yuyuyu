"""矢量数据盘点：用 pyshp 读取历史灾点与人口普查 shapefile 的结构信息。

本脚本只读 ``data/B/raw``，不修改任何原始文件（CLAUDE_B.md 硬规则 1）。

产出：
    out/B/log/p1_vector.log    字段名、记录数、前 3 条记录、坐标系

AI 辅助声明：本文件由 Claude Code（Anthropic，2026-10-24）辅助生成，人工已复核。

用法：
    python srcB/b1_vector_probe.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import shapefile  # pyshp

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DIR: Path = PROJECT_ROOT / "data" / "B" / "raw"
LOG_DIR: Path = PROJECT_ROOT / "out" / "B" / "log"

#: 待检查的 shapefile（不含扩展名）。目录名即文件名。
#: 第二项为该 DBF 的文本编码：
#:   历史灾点 DBF 为中文 ArcGIS 导出，字段名用 GBK 编码，pyshp 默认 UTF-8 会抛 dbfFileException；
#:   人口普查 DBF 为 UTF-8，默认即可。
TARGETS: dict[str, tuple[Path, str]] = {
    "历史灾点": (
        RAW_DIR / "西藏日喀则市吉隆县泥石流灾害区历史灾点数据" / "西藏日喀则市吉隆县泥石流灾害区历史灾点数据",
        "gbk",
    ),
    "人口普查(七普)": (
        RAW_DIR / "西藏日喀则市吉隆县第七次人口普查数据" / "西藏日喀则市吉隆县第七次人口普查数据",
        "utf-8",
    ),
}

#: 每个文件展示的前 N 条记录。
HEAD_N: int = 3

#: 日志中每个几何体最多展示的坐标点数量（多边形可达上千点，全量打印会撑爆日志）。
MAX_COORD_PREVIEW: int = 4


def _read_prj(shp_base: Path) -> str:
    """读取同名 ``.prj`` 文件并返回坐标系 WKT。

    Args:
        shp_base: shapefile 路径（不含扩展名）。

    Returns:
        WKT 文本；文件缺失时返回 ``"(缺失 .prj)"``。
    """
    prj = shp_base.with_suffix(".prj")
    if not prj.exists():
        return "(缺失 .prj)"
    # ESRI 的 .prj 常见 UTF-8 或 ANSI 编码，容错读取。
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return prj.read_text(encoding=enc).strip()
        except UnicodeDecodeError:
            continue
    return "(无法解码 .prj)"


def _preview_geometry(geom: object) -> str:
    """把 GeoJSON 风格几何体压缩成一行摘要，避免多边形坐标灌满日志。

    Args:
        geom: ``__geo_interface__`` 返回的字典（或任意对象）。

    Returns:
        形如 ``type=Polygon n_points=145 first=[(x, y), ...]`` 的字符串。
    """
    if not isinstance(geom, dict):
        return f"<{type(geom).__name__}>"

    gtype = geom.get("type")
    coords = geom.get("coordinates")

    def _n_points(obj: object) -> int:
        """递归统计坐标点数。"""
        if not isinstance(obj, (list, tuple)):
            return 0
        if obj and not isinstance(obj[0], (list, tuple)):
            return 1
        return sum(_n_points(item) for item in obj)

    def _first_points(obj: object, depth: int = 0) -> list[tuple[float, float]]:
        """取第一个环的前若干个点。"""
        if not isinstance(obj, (list, tuple)) or not obj:
            return []
        if len(obj) >= 2 and all(isinstance(v, (int, float)) for v in obj[:2]):
            return [(float(obj[0]), float(obj[1]))]
        if depth > 6:
            return []
        out: list[tuple[float, float]] = []
        for item in obj:
            out.extend(_first_points(item, depth + 1))
            if len(out) >= MAX_COORD_PREVIEW:
                break
        return out[:MAX_COORD_PREVIEW]

    pts = _first_points(coords)
    pts_txt = ", ".join(f"({x:.6f}, {y:.6f})" for x, y in pts)
    return f"type={gtype}  n_points={_n_points(coords)}  first=[{pts_txt}{' ...' if pts else ''}]"


def _summarize(shp_base: Path, encoding: str) -> dict[str, object]:
    """读取单个 shapefile 的结构与样本记录。

    Args:
        shp_base: shapefile 路径（不含扩展名）。
        encoding: DBF 文本编码，中文 ArcGIS 导出常见 ``gbk``。

    Returns:
        含 ``n_records`` / ``fields`` / ``head`` / ``shapetype`` / ``crs`` /
        ``bbox`` / ``n_parts`` 的字典。
    """
    with shapefile.Reader(str(shp_base), encoding=encoding) as sf:
        fields = [(name, typ, size, dec) for name, typ, size, dec in sf.fields if name != "DeletionFlag"]
        n_records = len(sf)
        head: list[dict[str, object]] = []
        for i in range(min(HEAD_N, n_records)):
            rec = sf.record(i)
            head.append({"record": dict(zip([f[0] for f in fields], list(rec))),
                         "shape": sf.shape(i).__geo_interface__})
        crs = _read_prj(shp_base)
        return {
            "n_records": n_records,
            "fields": fields,
            "head": head,
            "shapetype": sf.shapeTypeName,
            "crs": crs,
            "bbox": sf.bbox,
            "n_parts": sum(len(getattr(s, "parts", []) or []) for s in sf.shapes()),
        }


def main() -> int:
    """遍历目标 shapefile，打印并写日志。

    Returns:
        进程退出码：0 全部成功；1 有文件缺失或读取失败。
    """
    log: list[str] = []
    add = log.append
    add("=" * 78)
    add("P1-2 矢量数据盘点 (srcB/b1_vector_probe.py)")
    add("=" * 78)
    add(f"运行时间戳 : {datetime.now().astimezone().isoformat()}")
    add(f"pyshp 版本 : {shapefile.__version__}")
    add("")

    failures: list[str] = []
    for label, (base, encoding) in TARGETS.items():
        add("-" * 78)
        add(f"[{label}] {base.name}.shp")
        add("-" * 78)
        add(f"路径 : {base}")
        add(f"DBF 编码 : {encoding}")

        if not base.with_suffix(".shp").exists():
            add("状态 : 文件不存在")
            failures.append(label)
            add("")
            continue

        try:
            info = _summarize(base, encoding)
        except Exception as exc:  # noqa: BLE001 - 需要在日志里保留完整原因
            add(f"状态 : 读取失败 {exc!r}")
            failures.append(label)
            add("")
            continue

        add(f"几何类型 : {info['shapetype']}")
        add(f"记录数   : {info['n_records']}")
        add(f"部件总数 : {info['n_parts']}")
        add(f"范围 bbox: {info['bbox']}")
        add(f"坐标系   : {info['crs']}")
        add("")
        add(f"字段 ({len(info['fields'])} 个): 字段名 | 类型 | 长度 | 小数位")
        for name, typ, size, dec in info["fields"]:
            add(f"   {name:<24} {typ:<3} {size:>4} {dec:>3}")
        add("")
        add(f"前 {HEAD_N} 条记录:")
        for i, item in enumerate(info["head"], start=1):
            add(f"   [{i}] {item['record']}")
            add(f"       几何: {_preview_geometry(item['shape'])}")
        add("")

    add("-" * 78)
    add("[结论]")
    add("-" * 78)
    if failures:
        add(f"DEGRADED: {', '.join(failures)}")
        rc = 1
    else:
        add("ALL_OK")
        rc = 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "p1_vector.log"
    path.write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    print(f"\n[log] {path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
