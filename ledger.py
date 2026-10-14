#!/usr/bin/env python3
"""agent 意图台账：记录用户消息的两类条目——他提的目标、他拍板的断点。

记的不是用户原话，是 agent 对用户消息的理解。契约见 doc/agent-ledger.md。
事实源 = doc/agent-ledger.db（SQLite，本机，不进 git）。

用法::

    uv run ledger.py add --type 断点 --domain stocks --title "..." --understood "..."
    uv run ledger.py list [--type X] [--domain X] [--code X] [--status X] [--since ISO]
    uv run ledger.py show L12
    uv run ledger.py update L12 --outcome "..." --status 已回填
    uv run ledger.py codes
    uv run ledger.py export [--csv out.csv]
    uv run ledger.py selftest
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "doc" / "agent-ledger.db"

TYPES = ("目标", "断点")
DOMAINS = ("stocks", "power", "global")
STATUSES = ("已定", "已回填", "已更正")

# 决策类型码：只给「断点」类用。只增不改；删码不删行，把说明改成「(已废弃)」。
CODES = [
    ("D1", "口径分歧", "同一事实两套算法或口径，必须选一套"),
    ("D2", "数据源取舍", "接哪个源，或降级、砍掉哪个源"),
    ("D3", "资源投入", "要不要继续投、投多少、做到哪一步停手"),
    ("D4", "优先级", "多条线先做哪条、排队顺序"),
    ("D5", "停手", "死因码同码即停，或不再加防线"),
    ("D6", "不可逆动作授权", "删库、强推、动生产数据前的显式授权"),
    ("D7", "实现形态选型", "存储、工具、接口长什么样"),
    ("D8", "语义与契约", "记什么、什么时候记、字段怎么定义"),
]
CODE_SET = {c for c, _, _ in CODES}

FIELDS = ("ts_msg", "ts_record", "type", "domain", "title", "understood",
          "options", "chosen", "rationale", "line", "impact", "code",
          "outcome", "status", "corrected_by")

SCHEMA = """CREATE TABLE IF NOT EXISTS ledger (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_msg TEXT NOT NULL, ts_record TEXT NOT NULL,
  type TEXT NOT NULL, domain TEXT NOT NULL,
  title TEXT NOT NULL, understood TEXT NOT NULL,
  options TEXT, chosen TEXT, rationale TEXT,
  line TEXT, impact TEXT, code TEXT,
  outcome TEXT, status TEXT NOT NULL DEFAULT '已定', corrected_by INTEGER)"""


# ---------------------------------------------------------------- 基础

def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def die(msg: str, code: int = 2) -> None:
    print("错误: " + msg, file=sys.stderr)
    raise SystemExit(code)


def con(create: bool = False) -> sqlite3.Connection:
    if create:
        DB.parent.mkdir(parents=True, exist_ok=True)
    elif not DB.exists():
        die(f"台账不存在: {DB}")
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    if create:
        c.executescript(SCHEMA)
        c.commit()
    return c


def dw(s) -> int:
    """显示宽度：东亚全角按 2 列算。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(s))


def pad(s, n: int) -> str:
    s = str(s)
    return s + " " * max(0, n - dw(s))


def val(v):
    """字段取值：`@stdin` 表示从标准输入读（长文本）。"""
    if v == "@stdin":
        return sys.stdin.buffer.read().decode("utf-8", "replace").strip()
    return v


def parse_id(s: str) -> int:
    t = str(s).strip()
    if t[:1] in ("L", "l"):
        t = t[1:]
    if not t.isdigit():
        die(f"条目号要写成 L12 或 12，收到 {s!r}")
    return int(t)


def print_table(head, rows) -> None:
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(dw(h), *(dw(r[i]) for r in rows)) for i, h in enumerate(head)]
    print("  ".join(pad(h, w) for h, w in zip(head, widths)).rstrip())
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(pad(c, w) for c, w in zip(r, widths)).rstrip())


# ---------------------------------------------------------------- 子命令

def cmd_add(a) -> None:
    rec = {
        "ts_msg": a.ts_msg or now(),
        "ts_record": now(),
        "type": a.type,
        "domain": a.domain,
        "title": val(a.title),
        "understood": val(a.understood),
        "options": val(a.options),
        "chosen": val(a.chosen),
        "rationale": val(a.rationale),
        "line": val(a.line),
        "impact": val(a.impact),
        "code": a.code,
        "outcome": val(a.outcome),
        "status": "已定",
        "corrected_by": None,
    }
    if rec["type"] not in TYPES:
        die(f"--type 必须取 {TYPES}")
    if rec["domain"] not in DOMAINS:
        die(f"--domain 必须取 {DOMAINS}")
    if not rec["title"]:
        die("--title 必填")
    if not rec["understood"]:
        die("--understood 必填（agent 对这个条目的理解）")
    if rec["code"]:
        if rec["code"] not in CODE_SET:
            die(f"--code 必须取 {sorted(CODE_SET)}")
        if rec["type"] != "断点":
            die("D 码只给「断点」条目用")
    if rec["type"] == "目标" and (rec["options"] or rec["chosen"]):
        die("「目标」条目不该有 --options / --chosen（那是断点的事）")
    cols = ",".join(FIELDS)
    sql = f"INSERT INTO ledger ({cols}) VALUES ({','.join('?' * len(FIELDS))})"
    c = con(create=True)
    cur = c.execute(sql, [rec[k] for k in FIELDS])
    c.commit()
    print(f"L{cur.lastrowid} 已记  ts_msg={rec['ts_msg']}  {rec['type']}/{rec['domain']}")


def cmd_list(a) -> None:
    q = "SELECT id,ts_msg,type,domain,title,code,status FROM ledger WHERE 1=1"
    args: list = []
    for col, v in (("type", a.type), ("domain", a.domain),
                   ("code", a.code), ("status", a.status)):
        if v:
            q += f" AND {col}=?"
            args.append(v)
    if a.since:
        q += " AND ts_msg>=?"
        args.append(a.since)
    q += " ORDER BY id"
    if a.limit:
        q += " LIMIT ?"
        args.append(int(a.limit))
    rows = con().execute(q, args).fetchall()
    if not rows:
        print("（无）")
        return
    out = [[f"L{r['id']}", r["ts_msg"][:16].replace("T", " "), r["type"],
            r["domain"], r["code"] or "-", r["status"], r["title"]] for r in rows]
    print_table(("号", "消息时间", "类型", "域", "码", "状态", "标题"), out)


def cmd_show(a) -> None:
    rid = parse_id(a.ref)
    r = con().execute("SELECT * FROM ledger WHERE id=?", (rid,)).fetchone()
    if not r:
        die(f"没有 L{rid}")
    for k in FIELDS:
        v = r[k]
        print(pad(k, 14) + (str(v) if v is not None else "-"))


def cmd_update(a) -> None:
    rid = parse_id(a.ref)
    sets, args = [], []
    if a.outcome is not None:
        sets.append("outcome=?")
        args.append(val(a.outcome))
    if a.status:
        if a.status not in STATUSES:
            die(f"--status 必须取 {STATUSES}")
        sets.append("status=?")
        args.append(a.status)
    if a.corrected_by is not None:
        sets.append("corrected_by=?")
        args.append(parse_id(a.corrected_by))
    if not sets:
        die("update 至少给一个 --outcome / --status / --corrected-by")
    args.append(rid)
    c = con()
    cur = c.execute(f"UPDATE ledger SET {','.join(sets)} WHERE id=?", args)
    c.commit()
    if not cur.rowcount:
        die(f"没有 L{rid}")
    print(f"L{rid} 已更新")


def cmd_codes(_a) -> None:
    print_table(("码", "决策类型", "什么时候用"), [list(c) for c in CODES])


def cmd_export(a) -> None:
    rows = con().execute(f"SELECT {','.join(FIELDS)} FROM ledger ORDER BY id").fetchall()
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            w.writerows([tuple(r) for r in rows])
        print(f"{len(rows)} 行 -> {a.csv}")
    else:
        w = csv.writer(sys.stdout)
        w.writerow(FIELDS)
        w.writerows([tuple(r) for r in rows])


def cmd_selftest(_a) -> None:
    import io
    import tempfile
    from contextlib import redirect_stderr, redirect_stdout

    global DB
    old = DB
    DB = Path(tempfile.mkdtemp()) / "selftest.db"
    n = 0
    sink = io.StringIO()

    def ns(**kw):
        base = dict(type="断点", domain="stocks", title="t", understood="u",
                    options="A/B", chosen="A", rationale="r", line="l",
                    impact="i", code="D1", outcome=None, ts_msg=None)
        base.update(kw)
        return argparse.Namespace(**base)

    def ok(fn, *a):
        nonlocal n
        with redirect_stdout(sink), redirect_stderr(sink):
            fn(*a)
        n += 1

    def rejected(label, *a):
        nonlocal n
        try:
            with redirect_stdout(sink), redirect_stderr(sink):
                cmd_add(*a)
        except SystemExit:
            n += 1
            return
        raise AssertionError(f"{label} 未被拒")

    try:
        ok(cmd_add, ns())
        ok(cmd_add, ns(type="目标", domain="power", code=None, options=None, chosen=None))
        assert con().execute("SELECT COUNT(*) FROM ledger").fetchone()[0] == 2
        n += 1
        rejected("非法码", ns(code="D99"))
        rejected("目标带 D 码", ns(type="目标", code="D1"))
        rejected("非法类型", ns(type="随便"))
        rejected("非法域", ns(domain="x"))
        rejected("空 understood", ns(understood=""))
        rejected("目标带 options", ns(type="目标", code=None, options="A"))
        ok(cmd_update, argparse.Namespace(ref="L1", outcome="事后", status="已回填",
                                          corrected_by=None))
        row = con().execute("SELECT outcome,status FROM ledger WHERE id=1").fetchone()
        assert (row["outcome"], row["status"]) == ("事后", "已回填"), dict(row)
        n += 1
        seen = io.StringIO()
        with redirect_stdout(seen), redirect_stderr(seen):
            cmd_list(argparse.Namespace(type="断点", domain=None, code=None,
                                        status=None, since=None, limit=None))
            cmd_show(argparse.Namespace(ref="L1"))
            cmd_codes(None)
        listed = seen.getvalue()
        assert "L1" in listed and "L2" not in listed, listed
        assert "D8" in listed, listed  # codes 也打出来了
        n += 2
        print(f"selftest 通过 ({n} 项)")
    finally:
        DB = old


# ---------------------------------------------------------------- 入口

def main() -> None:
    # 管道/重定向时钉住 utf-8：默认 gbk 会让下游（agent、tee、日志）读到乱码；
    # 真终端保持平台编码，Windows cmd.exe 下中文才不会花。
    for s in (sys.stdout, sys.stderr):
        if not s.isatty():
            try:
                s.reconfigure(encoding="utf-8")
            except Exception:
                pass

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_flags(q, required=True):
        q.add_argument("--type", required=required, choices=TYPES)
        q.add_argument("--domain", required=required, choices=DOMAINS)
        q.add_argument("--title", required=required, help="@stdin 表示从标准输入读")
        q.add_argument("--understood", required=required, help="agent 的理解；@stdin 可读标准输入")
        q.add_argument("--options", help="断点：A/B/C，每个带代价与后果")
        q.add_argument("--chosen", help="断点：用户选的")
        q.add_argument("--rationale", help="agent 理解的用户为什么这么选")
        q.add_argument("--line", help="断点：执行线路一句话方向")
        q.add_argument("--impact", help="可逆/不可逆 + 影响面")
        q.add_argument("--code", help=f"D 码（仅断点）：{'/'.join(CODE_SET)}")
        q.add_argument("--outcome", help="事后实况（也可之后 update 回填）")
        q.add_argument("--ts-msg", help="用户消息时刻（ISO），默认取落盘时刻")

    q = sub.add_parser("add", help="追加一条")
    add_flags(q)
    q.set_defaults(func=cmd_add)

    q = sub.add_parser("list", help="列表")
    q.add_argument("--type", choices=TYPES)
    q.add_argument("--domain", choices=DOMAINS)
    q.add_argument("--code")
    q.add_argument("--status", choices=STATUSES)
    q.add_argument("--since", help="只列 ts_msg >= 这个 ISO 时间")
    q.add_argument("--limit", type=int)
    q.set_defaults(func=cmd_list)

    q = sub.add_parser("show", help="看一条的全文")
    q.add_argument("ref", help="L12 或 12")
    q.set_defaults(func=cmd_show)

    q = sub.add_parser("update", help="回填事后 / 改状态 / 挂更正")
    q.add_argument("ref")
    q.add_argument("--outcome", help="@stdin 可读标准输入")
    q.add_argument("--status", choices=STATUSES)
    q.add_argument("--corrected-by", help="更正条的 id")
    q.set_defaults(func=cmd_update)

    sub.add_parser("codes", help="打印 D 码表").set_defaults(func=cmd_codes)

    q = sub.add_parser("export", help="导出 CSV")
    q.add_argument("--csv", help="输出文件；不给则打到标准输出")
    q.set_defaults(func=cmd_export)

    sub.add_parser("selftest", help="自检").set_defaults(func=cmd_selftest)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
