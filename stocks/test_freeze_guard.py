"""冻结窗口保护的断言测试（与 test_backtest.py 同风格：assert + 自打印，不用 pytest）。

覆盖（协议 §9.1 的机器保护）：
  ① 真实冻结快照与锚点一致（sha256/字节/行数/最大日期/交易日数/标的数）
  ② 面板被同名覆盖 → 能检出
  ③ 形状不变、内容被静默改写（复权因子重算）→ 只有哈希能检出
  ④ 日期越界 → 能检出
  ⑤ 研究模式产物与冻结产物分流、且标注「非独立验证」
  ⑥ 端到端：越界 --end 被拒；被改过的面板在判据入口被拒
  ⑦ fetch 默认不越界、store 不覆盖冻结快照
  ⑧ 端到端：冻结快照上的原有判据结果与已存 out/layer_judge.csv 保持一致

用法:
    uv run stocks/test_freeze_guard.py          # 全部（含 ~40s 的端到端复算）
    uv run stocks/test_freeze_guard.py --fast   # 跳过端到端复算（⑧）
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import freeze_guard as fg  # noqa: E402

CASES: list = []
SLOW_CASES: list = []


def case(fn):
    CASES.append(fn)
    return fn


def slow_case(fn):
    """需要真跑判据/真拉数入口的，--fast 时跳过。"""
    SLOW_CASES.append(fn)
    return fn


def write_panel(path: pathlib.Path, dates: list[str], close: float = 1.0,
                symbol: str = "sh600000") -> pathlib.Path:
    rows = [{"date": pd.Timestamp(d), "symbol": symbol, "close": close} for d in dates]
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def run(script: str, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / script), *argv],
                          cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


# ---------------------------------------------------------------- ①②③④ 判据层
@case
def t_real_snapshot_matches_anchors() -> None:
    problems = fg.verify()
    assert problems == [], f"冻结快照与锚点不符: {problems}"


@case
def t_real_snapshot_not_beyond_freeze() -> None:
    assert fg.check_bounds() == [], "冻结快照本身越过了冻结边界"


@case
def t_overwrite_detected(tmp: pathlib.Path) -> None:
    p = write_panel(tmp / "panel_overwrite.parquet", ["2026-10-08", "2026-10-09"])
    want = fg.panel_stats(p)
    write_panel(p, ["2026-10-08", "2026-10-09", "2026-10-09"], close=2.0)   # 同名覆盖
    problems = fg.verify(p, want)
    assert problems, "同名覆盖没被检出"
    for key in ("sha256", "rows"):
        assert any(m.startswith(f"{key}:") for m in problems), f"没报出 {key} 差异: {problems}"


@case
def t_silent_revaluation_detected(tmp: pathlib.Path) -> None:
    """形状/日期/只数全不变，只有数值被改写（后复权因子重算）→ 只有哈希能抓。"""
    dates = ["2026-10-08", "2026-10-09"]
    p = write_panel(tmp / "panel_revalue.parquet", dates, close=1.0)
    want = fg.panel_stats(p)
    write_panel(p, dates, close=1.0000001)
    problems = fg.verify(p, want)
    assert any(m.startswith("sha256:") for m in problems), f"哈希没抓到静默改写: {problems}"
    for key in ("rows", "dates", "symbols", "max_date"):
        assert not any(m.startswith(f"{key}:") for m in problems), \
            f"{key} 不该报差异（说明这条只能靠哈希）: {problems}"


@case
def t_bounds_detected(tmp: pathlib.Path) -> None:
    ok = write_panel(tmp / "panel_in.parquet", ["2026-10-08", "2026-10-09"])
    out = write_panel(tmp / "panel_out.parquet", ["2026-10-09", "2026-10-30"])
    assert fg.check_bounds(ok) == [], "边界内的面板被判成越界"
    problems = fg.check_bounds(out)
    assert problems and "越界" in problems[0], f"越界没被检出: {problems}"


@case
def t_research_path_and_label_disjoint() -> None:
    frozen = fg.frozen_panel_path()
    research = fg.research_panel_path("2015-01-01", "2026-10-30")
    assert research != frozen, "研究面板与冻结快照同名，迟早覆盖"
    assert research.name == "panel_2015_2026-10-30.parquet", research.name
    assert fg.verdict_level(True) != fg.verdict_level(False)
    assert "非独立验证" in fg.verdict_level(True)
    assert "非独立验证" in fg.verdict_level(False)   # 冻结口径也只是同段确认


# ---------------------------------------------------------------- ⑥⑦ 入口层
@slow_case
def t_layer_judge_rejects_beyond_freeze(tmp: pathlib.Path) -> None:
    r = run("layer_judge.py", "--end", "2026-10-30", "--out", str(tmp / "x.csv"))
    assert r.returncode == 2, f"越界 --end 没被拒（rc={r.returncode}）:\n{r.stdout}\n{r.stderr}"
    assert "越过冻结边界" in (r.stdout + r.stderr)
    assert not (tmp / "x.csv").exists(), "被拒的调用仍然写了产物"


@slow_case
def t_layer_judge_rejects_tampered_panel(tmp: pathlib.Path) -> None:
    """端到端验收①：冻结面板被覆盖时，判据入口立即拦截。"""
    tampered = write_panel(tmp / "panel_tampered.parquet", ["2026-10-08", "2026-10-09"],
                           close=99.0)
    r = run("layer_judge.py", "--panel", str(tampered), "--out", str(tmp / "y.csv"))
    assert r.returncode == 1, f"被改过的面板没被拦（rc={r.returncode}）:\n{r.stdout}\n{r.stderr}"
    assert "冻结校验不过" in (r.stdout + r.stderr)
    assert not (tmp / "y.csv").exists(), "校验不过却仍然写了产物"


@slow_case
def t_layer_judge_research_labeled(tmp: pathlib.Path) -> None:
    """端到端验收②：研究模式产物与冻结产物分流且写明非独立验证。"""
    frozen_out = ROOT / "out" / "layer_judge.csv"
    before = frozen_out.read_bytes() if frozen_out.exists() else None
    dst = tmp / "layer_judge_research.csv"
    r = run("layer_judge.py", "--research", "--no-rank", "--end", "2026-10-30",
            "--out", str(dst))
    assert r.returncode == 0, f"研究模式跑不动:\n{r.stdout}\n{r.stderr}"
    assert "研究模式" in r.stderr
    df = pd.read_csv(dst)
    assert "证据等级" in df.columns
    assert (df["证据等级"] == fg.verdict_level(True)).all(), df["证据等级"].unique()
    if before is not None:
        assert frozen_out.read_bytes() == before, "研究模式动了冻结产物 out/layer_judge.csv"


@slow_case
def t_fetch_rejects_beyond_freeze() -> None:
    r = run("fetch.py", "--end", "2026-12-31")
    assert r.returncode == 2, f"fetch 越界没被拒（rc={r.returncode}）:\n{r.stdout}\n{r.stderr}"
    assert "--allow-future" in (r.stdout + r.stderr)
    assert "待拉取" not in r.stdout, "被拒的调用仍然进了拉取流程"


@case
def t_store_refuses_overwriting_frozen() -> None:
    import store

    frozen = fg.frozen_panel_path()
    before = frozen.stat().st_mtime_ns
    try:
        store.build_panel("2015-01-01", "2026-12-31", out_path=frozen)
    except ValueError as exc:
        assert "不得写冻结快照" in str(exc), str(exc)
    else:
        raise AssertionError("越界数据写冻结快照没被拒")
    assert frozen.stat().st_mtime_ns == before, "冻结快照被动了"


@case
def t_store_research_path_is_not_frozen() -> None:
    import store  # noqa: F401  仅确认 store 能 import freeze_guard（无循环依赖）

    assert fg.research_panel_path("2015-01-01", "2026-12-31") != fg.frozen_panel_path()


# ---------------------------------------------------------------- ⑧ 一致性
@slow_case
def t_frozen_recompute_consistent(tmp: pathlib.Path) -> None:
    """端到端验收③：冻结快照上的判据结果与已存 out/layer_judge.csv 一致。"""
    dst = tmp / "layer_judge_check.csv"
    r = run("layer_judge.py", "--out", str(dst))
    assert r.returncode == 0, f"冻结口径跑不动:\n{r.stdout}\n{r.stderr}"
    new = pd.read_csv(dst)
    assert "证据等级" in new.columns and (new["证据等级"] == fg.verdict_level(False)).all()
    assert int((new["判定"] == "通过").sum()) == 5, new[["factor", "判定"]].to_dict("records")
    a144 = new[new["factor"] == "alpha144"].iloc[0]
    assert abs(a144["去均值相关"] - 0.350) < 5e-4, a144["去均值相关"]
    assert abs(a144["样本外年化"] - 0.2042) < 5e-4, a144["样本外年化"]

    old_path = ROOT / "out" / "layer_judge.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        cols = ["factor", "全期年化", "样本外年化", "去均值相关", "秩相关均值", "判定"]
        merged = old[cols].merge(new[cols], on="factor", suffixes=("_old", "_new"))
        assert len(merged) == len(old) == len(new), (len(old), len(new))
        for c in cols[1:-1]:
            d = (merged[f"{c}_old"] - merged[f"{c}_new"]).abs().max()
            assert d < 1e-9, f"{c} 与已存产物不一致，最大差 {d}"
        assert (merged["判定_old"] == merged["判定_new"]).all(), "判定列与已存产物不一致"


def main() -> int:
    fast = "--fast" in sys.argv
    cases = CASES + ([] if fast else SLOW_CASES)
    print(f"[test] 冻结窗口保护：{len(cases)} 个用例，{len(CASES)} 个快用例 + "
          f"{len(SLOW_CASES)} 个端到端用例{'（--fast，端到端已跳过）' if fast else ''}\n")
    failed: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="freeze_guard_test_") as td:
        tmp = pathlib.Path(td)
        for fn in cases:
            name = fn.__name__
            try:
                if fn.__code__.co_argcount:
                    fn(tmp)
                else:
                    fn()
            except Exception as exc:                       # noqa: BLE001
                failed.append((name, f"{type(exc).__name__}: {exc}"))
                print(f"[test] FAIL {name} -- {type(exc).__name__}: {exc}")
            else:
                print(f"[test] PASS {name}")
    print(f"\n[test] 通过 {len(cases) - len(failed)} / 失败 {len(failed)}"
          f"（用例总数 {len(cases)}）")
    for name, err in failed:
        print(f"[test] FAIL {name}: {err}")
    print("[test] 未覆盖（专项测试不演练）：真实 334MB 快照被物理覆盖/删掉（会毁证据，"
          "用合成面板 + 端到端入口 t_layer_judge_rejects_tampered_panel 替代）；"
          "多进程并发写面板；磁盘满/权限不足等 I/O 故障路径")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
