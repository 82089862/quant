"""冻结快照的机器保护：核对面板与冻结产物、拒绝越界数据进冻结判据。

**事实源只有一个**：`stocks/freeze_anchor.json`（面板 sha256/字节/行数/最大日期/交易日数/标的数
+ 冻结提交与时间戳 + 冻结口径消费的 `stocks/out/layers_*.csv` 哈希）。本模块与
`doc/研究方法与验证协议.md` §9.1 都从它取数，`test_freeze_guard.py` 断言「§9.1 表格与 JSON 一致」——
改一处不改另一处会被测出来（此前是两处手工同步，有漂移风险）。

背景（§9.1）：面板产物名只到年份（`panel_2015_2026.parquet`），重新拉数再合成会同名覆盖
冻结快照，判据复算读到的就是未来数据，且不报错。

本模块只做三件事，都读**实际文件内容**，不看文件名（同名路径证明不了任何东西）：

  `verify()`           面板 vs 锚点（sha256 / 字节数 / 行数 / 最大日期 / 交易日数 / 标的数）
  `verify_artifacts()` 冻结口径消费的 layers CSV vs 锚点（含「锚点外的产物」检测）
  `check_bounds()`     数据日期是否越过冻结边界

**哈希通过 != 独立验证**：它只证明「与被冻结的那份快照逐位一致」。本仓 2015-2026 全段都
参与过候选筛选（§9.1 那张表），所以这里的任何数字都只是**同段确认**，不是独立样本外证据。

用法:
    uv run stocks/freeze_guard.py verify            # 面板：不符则非零退出并逐项打印差异
    uv run stocks/freeze_guard.py artifacts         # 冻结口径消费的产物
    uv run stocks/freeze_guard.py stats             # 只看实际文件的规模
    uv run stocks/freeze_guard.py seal --commit <SHA> --ts <ISO>   # 重新冻结（写 JSON，慎用）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parent
DATA_DIR = ROOT / "data"
ANCHOR_PATH = ROOT / "freeze_anchor.json"

# 结构性的、不属于锚点数值的部分：不进 JSON，免得同一件事两处写
ARTIFACT_EXCLUDE = ("layers_momentum20_5g.csv",)   # 动量基线：冻结口径的判据不读它
ARTIFACT_GLOB = "layers_*_5g.csv"


def load_anchor(path: pathlib.Path | str = ANCHOR_PATH) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        raise SystemExit(f"[freeze] 缺锚点文件 {p} —— 协议 §9.1 的机器可读源丢了，拒绝继续")
    return json.loads(p.read_text(encoding="utf-8"))


ANCHOR = load_anchor()
FREEZE_COMMIT: str = ANCHOR["commit"]
FREEZE_TS: str = ANCHOR["ts"]
FREEZE_END: str = ANCHOR["panel"]["max_date"]
_ARTIFACT_DIR_REL: str = ANCHOR["artifacts"]["dir"]
_ARTIFACT_FILES: dict = ANCHOR["artifacts"]["files"]


def frozen_panel_path() -> pathlib.Path:
    """冻结快照路径（唯一允许出冻结判据结论的输入）。"""
    return REPO / ANCHOR["panel"]["path"]


def artifact_dir() -> pathlib.Path:
    """冻结口径消费的产物目录（默认 `stocks/out/`）。"""
    return REPO / _ARTIFACT_DIR_REL


def artifact_files() -> dict:
    return dict(_ARTIFACT_FILES)


def research_panel_path(start: str, end: str) -> pathlib.Path:
    """越界数据的落盘名：带完整 end 日期，永不与冻结快照同名。"""
    return DATA_DIR / f"panel_{start[:4]}_{end}.parquet"


def research_name(path: pathlib.Path | str) -> pathlib.Path:
    """研究模式的产物名：`x.csv` -> `x.research.csv`，永不与冻结产物同名。"""
    p = pathlib.Path(path)
    return p.with_name(f"{p.stem}.research{p.suffix}")


def anchors() -> dict:
    return {k: ANCHOR["panel"][k] for k in ("sha256", "bytes", "rows", "max_date", "dates", "symbols")}


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
    got = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
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
    """面板 vs 锚点。返回差异清单，**空列表才算通过**。"""
    path = pathlib.Path(panel) if panel is not None else frozen_panel_path()
    print(f"[freeze] verify({path.name})")
    if not path.exists():
        return [f"面板不存在: {path}"]
    problems = diff(panel_stats(path), want if want is not None else anchors())
    print(f"[freeze] verify -> {'一致' if not problems else f'{len(problems)} 项不符'}")
    return problems


def verify_artifacts(directory: pathlib.Path | str | None = None) -> list[str]:
    """冻结口径消费的 layers CSV vs 锚点。

    两个方向都查：① 锚点里的文件缺失/被改；② 目录里多出锚点没有的 `layers_*_5g.csv`
    （多出来的产物说明有人在这个目录里跑过别的口径，此时「冻结结论」已经不对应同一批输入）。
    """
    d = pathlib.Path(directory) if directory is not None else artifact_dir()
    print(f"[freeze] verify_artifacts({d})")
    problems: list[str] = []
    for name, want in _ARTIFACT_FILES.items():
        p = d / name
        if not p.exists():
            problems.append(f"{name}: 缺文件")
            continue
        got = file_sha256(p)
        if got != want:
            problems.append(f"{name}: sha256 实际 {got[:12]}… != 冻结 {want[:12]}…")
    if d.exists():
        extra = sorted(p.name for p in d.glob(ARTIFACT_GLOB)
                       if p.name not in _ARTIFACT_FILES and p.name not in ARTIFACT_EXCLUDE)
        if extra:
            problems.append(f"锚点外的产物（不属于冻结口径的输入）: {extra}")
    print(f"[freeze] verify_artifacts -> {'一致' if not problems else f'{len(problems)} 项不符'}")
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


def seal(panel: pathlib.Path | str | None = None, commit: str | None = None,
         ts: str | None = None, out: pathlib.Path | str | None = None) -> pathlib.Path:
    """重新冻结：重算面板锚点与产物哈希写回 JSON。改口径/换面板时才用。"""
    p = pathlib.Path(panel) if panel is not None else frozen_panel_path()
    got = panel_stats(p)
    files = {f.name: file_sha256(f) for f in sorted(artifact_dir().glob(ARTIFACT_GLOB))
             if f.name not in ARTIFACT_EXCLUDE}
    doc = {
        "commit": commit or FREEZE_COMMIT,
        "ts": ts or FREEZE_TS,
        "panel": {"path": p.relative_to(REPO).as_posix(), **got},
        "artifacts": {"dir": _ARTIFACT_DIR_REL, "files": files},
    }
    outp = pathlib.Path(out) if out is not None else ANCHOR_PATH
    outp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[freeze] seal -> {outp}（面板 {got['rows']:,} 行 + 产物 {len(files)} 个）")
    return outp


def main() -> int:
    ap = argparse.ArgumentParser(description="冻结快照校验（协议 §9.1）")
    ap.add_argument("cmd", choices=["verify", "artifacts", "stats", "seal"])
    ap.add_argument("--panel", default="", help="面板路径；默认 = 冻结快照")
    ap.add_argument("--dir", default="", help="产物目录；默认 = 锚点里的目录")
    ap.add_argument("--commit", default=None, help="seal 用：新的冻结提交 SHA")
    ap.add_argument("--ts", default=None, help="seal 用：新的冻结时间戳（ISO）")
    args = ap.parse_args()

    path = pathlib.Path(args.panel) if args.panel else frozen_panel_path()
    if args.cmd == "stats":
        try:
            panel_stats(path)
        except Exception as exc:                    # noqa: BLE001
            print(f"[freeze] 读不了 {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.cmd == "seal":
        seal(panel=path, commit=args.commit, ts=args.ts)
        return 0
    if args.cmd == "artifacts":
        problems = verify_artifacts(args.dir or None)
        for m in problems:
            print(f"[freeze] [X] {m}", file=sys.stderr)
        return 1 if problems else 0

    problems = verify(path)
    if problems:
        for m in problems:
            print(f"[freeze] [X] {m}", file=sys.stderr)
        print(f"[freeze] 快照与冻结锚点不符 —— 结论不能再当冻结口径用（协议 §9.1，"
              f"冻结于 {FREEZE_COMMIT[:7]} / {FREEZE_TS}）", file=sys.stderr)
        return 1
    print(f"[freeze] [OK] {path.name} 与冻结锚点逐项一致（{FREEZE_COMMIT[:7]} / {FREEZE_TS}）")
    print("[freeze] 注意：一致 != 独立验证，只是「与冻结快照同段」")
    return 0


if __name__ == "__main__":
    sys.exit(main())
