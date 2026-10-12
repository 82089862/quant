"""冻结快照的机器保护：核对面板与冻结锚点、拒绝越界数据进冻结判据。

背景（`doc/研究方法与验证协议.md` §9.1）：冻结锚点 = 提交 `3f66f46` / 2026-10-12T14:01:11+08:00
+ 面板 sha256。但那段此前**只是文档承诺** —— 面板产物名只到年份（`panel_2015_2026.parquet`），
重新拉数再合成会同名覆盖冻结快照，判据复算读到的就是未来数据，且不会报错。

本模块只做两件事，都读**实际文件内容**，不看文件名（同名路径证明不了任何东西）：

  `verify()`        实际文件 vs 冻结锚点（sha256 / 字节数 / 行数 / 最大日期 / 交易日数 / 标的数）
  `check_bounds()`  数据日期是否越过冻结边界

**哈希通过 ≠ 独立验证**：它只证明「与被冻结的那份快照逐位一致」。本仓 2015-2026 全段都
参与过候选筛选（§9.1 那张表），所以这里的任何数字都只是**同段确认**，不是独立样本外证据。

用法:
    uv run stocks/freeze_guard.py verify     # 不符则非零退出并逐项打印差异
    uv run stocks/freeze_guard.py stats      # 只看实际文件的规模
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# 冻结锚点（协议 §9.1，2026-10-12 实测）。改这里 = 重新冻结，必须同步改 §9.1 与本节注释。
FREEZE_COMMIT = "3f66f46702dcc4da8241c3adf8d4d7fbbbfc9398"
FREEZE_TS = "2026-10-12T14:01:11+08:00"
FREEZE_END = "2026-10-09"
FREEZE_SHA256 = "e9ba071607dbd4448cf201ac44cd7b9dc471fb62da34b50e4c9ccab3473e10ca"
FREEZE_BYTES = 334_916_252
FREEZE_ROWS = 11_450_179
FREEZE_DATES = 2858
FREEZE_SYMBOLS = 5476


def frozen_panel_path() -> pathlib.Path:
    """冻结快照路径（唯一允许出冻结判据结论的输入）。"""
    return DATA_DIR / "panel_2015_2026.parquet"


def research_panel_path(start: str, end: str) -> pathlib.Path:
    """越界数据的落盘名：带完整 end 日期，永不与冻结快照同名。"""
    return DATA_DIR / f"panel_{start[:4]}_{end}.parquet"


def anchors() -> dict:
    return {"sha256": FREEZE_SHA256, "bytes": FREEZE_BYTES, "rows": FREEZE_ROWS,
            "max_date": FREEZE_END, "dates": FREEZE_DATES, "symbols": FREEZE_SYMBOLS}


def verdict_level(research: bool) -> str:
    """判据结论的证据等级。研究模式不得冒充独立验证（协议 §9.1）。"""
    if research:
        return "研究数据（非独立验证）"
    return "冻结快照（同段确认，非独立验证）"


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def panel_stats(panel: pathlib.Path | str) -> dict:
    """读实际文件：sha256 + 字节数 + 行数 / 最大日期 / 交易日数 / 标的数。"""
    path = pathlib.Path(panel)
    t0 = time.time()
    print(f"[freeze] 读 {path}")
    got = {
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    import duckdb

    n, mx, d, s = duckdb.sql(
        f"select count(*), max(date), count(distinct date), count(distinct symbol) "
        f"from read_parquet('{path.as_posix()}')").fetchone()
    # date 列可能是 DATE 或 datetime64，统一裁成年-月-日再比（否则同一现实会当不一致）
    got.update({"rows": int(n), "max_date": str(mx)[:10], "dates": int(d), "symbols": int(s)})
    print(f"[freeze] stats {path.name}: sha256 {got['sha256'][:12]}… / {got['bytes']:,} 字节 / "
          f"{got['rows']:,} 行 / max {got['max_date']} / {got['dates']} 日 / {got['symbols']} 只 "
          f"（{time.time() - t0:.1f}s）")
    return got


def max_date(panel: pathlib.Path | str) -> str:
    """只要最大日期时走这条（不哈希，便宜）。"""
    import duckdb

    path = pathlib.Path(panel)
    return str(duckdb.sql(f"select max(date) from read_parquet('{path.as_posix()}')"
                          ).fetchone()[0])[:10]


def diff(got: dict, want: dict) -> list[str]:
    """逐项比，返回人类可读的差异串（空 = 一致）。"""
    return [f"{k}: 实际 {got.get(k)} != 冻结 {want[k]}" for k in want if got.get(k) != want[k]]


def verify(panel: pathlib.Path | str | None = None, want: dict | None = None) -> list[str]:
    """实际文件 vs 冻结锚点。返回差异清单，**空列表才算通过**。"""
    path = pathlib.Path(panel) if panel is not None else frozen_panel_path()
    print(f"[freeze] verify({path.name})")
    if not path.exists():
        return [f"面板不存在: {path}"]
    problems = diff(panel_stats(path), want if want is not None else anchors())
    print(f"[freeze] verify -> {'一致' if not problems else f'{len(problems)} 项不符'}")
    return problems


def check_bounds(panel: pathlib.Path | str | None = None,
                 freeze_end: str = FREEZE_END) -> list[str]:
    """数据日期是否越过冻结边界（严格大于才算越界）。"""
    path = pathlib.Path(panel) if panel is not None else frozen_panel_path()
    print(f"[freeze] check_bounds({path.name}, <= {freeze_end})")
    if not path.exists():
        return [f"面板不存在: {path}"]
    mx = max_date(path)
    if mx > freeze_end:
        return [f"越界: max(date) {mx} > 冻结边界 {freeze_end}"]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description="冻结快照校验（协议 §9.1）")
    ap.add_argument("cmd", choices=["verify", "stats"])
    ap.add_argument("--panel", default="", help="面板路径；默认 = 冻结快照")
    args = ap.parse_args()

    path = pathlib.Path(args.panel) if args.panel else frozen_panel_path()
    if args.cmd == "stats":
        try:
            panel_stats(path)
        except Exception as exc:                    # noqa: BLE001
            print(f"[freeze] 读不了 {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        return 0

    problems = verify(path)
    if problems:
        for m in problems:
            print(f"[freeze] ✗ {m}", file=sys.stderr)
        print(f"[freeze] 快照与冻结锚点不符 —— 结论不能再当冻结口径用（协议 §9.1，"
              f"冻结于 {FREEZE_COMMIT[:7]} / {FREEZE_TS}）", file=sys.stderr)
        return 1
    print(f"[freeze] ✓ {path.name} 与冻结锚点逐项一致（{FREEZE_COMMIT[:7]} / {FREEZE_TS}）")
    print("[freeze] 注意：一致 != 独立验证，只是「与冻结快照同段」")
    return 0


if __name__ == "__main__":
    sys.exit(main())
