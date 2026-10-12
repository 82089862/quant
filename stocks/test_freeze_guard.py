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
  ⑨ 锚点事实源：`freeze_anchor.json` 与协议 §9.1 表格逐项一致（此前两处手工同步）
  ⑩ 冻结口径消费的 layers CSV：被改 / 缺文件 / 锚点外多出产物 → 能检出
  (11) 端到端：产物被改时判据入口非零退出；factors.py / alpha191.py 日期闸（只比日期，不验内容）

用法:
    uv run stocks/test_freeze_guard.py          # 全部（含 ~40s 的端到端复算）
    uv run stocks/test_freeze_guard.py --fast   # 跳过端到端复算（⑧）
"""
from __future__ import annotations

import os
import pathlib
import shutil
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
    # 子进程的输出编码必须和这里的 encoding="utf-8" 对齐：Windows 上默认是 GBK，
    # 不钉住的话断言里比中文会全部落空（实测 4 个 layer_judge 用例假失败）
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, str(ROOT / script), *argv],
                          cwd=str(ROOT), env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def copy_artifacts(dst: pathlib.Path) -> dict:
    """把冻结口径消费的产物 + 排除文件拷到临时目录（不动 out/）。"""
    src = fg.artifact_dir()
    for name in list(fg.artifact_files()) + list(fg.ARTIFACT_EXCLUDE):
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)
    return fg.artifact_files()


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


# ---------------------------------------------------------------- ⑨⑩ 锚点与产物
@case
def t_anchor_json_matches_doc() -> None:
    """协议 §9.1 表格里的数字必须与锚点 JSON 一致（此前两处手工同步）。"""
    doc = (ROOT.parent / "doc" / "研究方法与验证协议.md").read_text(encoding="utf-8")
    start = doc.index("### 9.1")
    section = doc[start:doc.index("## 10.", start)]
    a = fg.anchors()
    want = [fg.FREEZE_COMMIT, fg.FREEZE_TS, a["sha256"], f"{a['bytes']:,}",
            f"{a['rows']:,}", a["max_date"], str(a["dates"]), str(a["symbols"])]
    missing = [w for w in want if w not in section]
    assert not missing, f"§9.1 里找不到这些锚点值（文档与 JSON 漂了）: {missing}"
    assert "freeze_anchor.json" in section, "§9.1 没写锚点的事实源文件"


@case
def t_anchor_fields_selfconsistent() -> None:
    a = fg.anchors()
    assert set(a) == {"sha256", "bytes", "rows", "max_date", "dates", "symbols"}
    assert len(a["sha256"]) == 64 and len(fg.FREEZE_COMMIT) == 40
    assert "+" in fg.FREEZE_TS and fg.FREEZE_END == a["max_date"]
    assert fg.frozen_panel_path().exists()


@case
def t_real_artifacts_match(tmp: pathlib.Path) -> None:
    assert fg.verify_artifacts() == [], "out/ 里的冻结产物与锚点不符"


@case
def t_artifact_tampered_detected(tmp: pathlib.Path) -> None:
    assert copy_artifacts(tmp), "锚点里没有产物，测试无意义"
    assert fg.verify_artifacts(tmp) == [], "原样拷贝就该通过"
    victim = tmp / "layers_alpha144_5g.csv"
    df = pd.read_csv(victim)
    df["G5"] = df["G5"] * 2.0            # 曲线被换掉，行数/列名不变
    df.to_csv(victim, index=False)
    problems = fg.verify_artifacts(tmp)
    assert problems and any("layers_alpha144_5g.csv" in m for m in problems), problems


@case
def t_artifact_missing_and_extra_detected(tmp: pathlib.Path) -> None:
    copy_artifacts(tmp)
    (tmp / "layers_alpha042_5g.csv").unlink()
    (tmp / "layers_zzz999_5g.csv").write_text("date,G1\n2026-10-09,1\n", encoding="utf-8")
    problems = fg.verify_artifacts(tmp)
    assert any("缺文件" in m for m in problems), problems
    assert any("锚点外的产物" in m and "layers_zzz999_5g.csv" in m for m in problems), problems
    assert not any("锚点外" in m and "layers_momentum20_5g.csv" in m for m in problems), \
        "排除清单（动量基线）被当成锚点外产物"


@case
def t_research_names_not_consumable() -> None:
    """研究产物名必须落在冻结口径的 glob 之外，否则会被当成冻结候选读进去。"""
    research = fg.research_name(pathlib.Path("out") / "layers_alpha144_5g.csv")
    assert research.name == "layers_alpha144_5g.research.csv", research.name
    assert not pathlib.Path(research.name).match(fg.ARTIFACT_GLOB), \
        "研究产物名仍然匹配判据的 glob（会被当成冻结候选）"
    assert fg.research_panel_path("2015-01-01", "2026-12-31").name != fg.frozen_panel_path().name


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


@slow_case
def t_layer_judge_rejects_tampered_artifact(tmp: pathlib.Path) -> None:
    """端到端：产物被改过时，判据入口立即拦截（只验面板挡不住这条）。"""
    copy_artifacts(tmp)
    victim = tmp / "layers_alpha164_5g.csv"
    df = pd.read_csv(victim)
    df["G5"] = df["G5"] * 3.0
    df.to_csv(victim, index=False)
    dst = tmp / "lj_should_not_exist.csv"
    r = run("layer_judge.py", "--out-dir", str(tmp), "--out", str(dst))
    assert r.returncode == 1, f"产物被改却没拦（rc={r.returncode}）:\n{r.stdout}\n{r.stderr}"
    assert "冻结校验不过" in (r.stdout + r.stderr)
    assert not dst.exists(), "校验不过却仍然写了产物"


@slow_case
def t_layer_judge_passes_on_clean_copy(tmp: pathlib.Path) -> None:
    """正常冻结读取：产物原样拷贝到别处，校验应通过（拦截来自内容而非路径）。"""
    copy_artifacts(tmp)
    dst = tmp / "lj_clean.csv"
    r = run("layer_judge.py", "--out-dir", str(tmp), "--out", str(dst))
    assert r.returncode == 0, f"干净的冻结产物被误杀:\n{r.stdout}\n{r.stderr}"
    df = pd.read_csv(dst)
    assert (df["证据等级"] == fg.verdict_level(False)).all()
    assert int((df["判定"] == "通过").sum()) == 5, df[["factor", "判定"]].to_dict("records")


@slow_case
def t_factors_and_alpha191_share_the_gate() -> None:
    """第 2 步审计的调用链：日期闸（只比日期、不验文件内容；内容闸只有 layer_judge 一处）
    + 研究模式必须显式 --panel。"""
    r = run("factors.py", "--end", "2026-12-31")
    assert r.returncode == 2 and "越过冻结边界" in (r.stdout + r.stderr), r.stdout + r.stderr
    r = run("factors.py", "--research", "--end", "2026-12-31")
    assert r.returncode == 2 and "--panel" in (r.stdout + r.stderr), r.stdout + r.stderr
    r = run("alpha191.py", "--end", "2026-12-31")
    assert r.returncode == 2 and "越过冻结边界" in (r.stdout + r.stderr), r.stdout + r.stderr


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
                    sub = tmp / name          # 每个用例一个独立目录，互不污染
                    sub.mkdir()
                    fn(sub)
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
    print("[test] 未覆盖（专项测试不演练）：真实 334MB 快照/产物被物理删掉后重跑（会毁证据，"
          "用合成面板 + 端到端入口替代）；多进程并发写面板或产物；磁盘满/权限不足等 I/O 故障路径；"
          "越界数据的真实拉取与合成（需 --allow-future 联网 + 重建 334MB 面板）；"
          "研究模式的完整回测链路（需先有研究面板，本轮只测了参数闸与产物命名）")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
