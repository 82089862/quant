"""把 doc/ 的假设登记表与死因码表镜像进本机 MySQL 的 quant 库。

**单向只读镜像**：事实源永远是 doc/stocks/假设登记表.md 与 doc/死因码表.md，
在库里手改的数据下次 load 会被 TRUNCATE 冲掉。回测链路不读这两张表。

用法:
    QUANT_MYSQL_PASSWORD=xxx uv run stocks/import_hypothesis.py ddl      # 建表 + 建视图（可重复执行）
    QUANT_MYSQL_PASSWORD=xxx uv run stocks/import_hypothesis.py load     # 重导（先 TRUNCATE）
    QUANT_MYSQL_PASSWORD=xxx uv run stocks/import_hypothesis.py verify   # 行数 + 死因码分布 + 抽样逐值
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re

import duckdb

ROOT = pathlib.Path(__file__).resolve().parent
DOC = ROOT.parent / "doc"
REGISTRY = DOC / "stocks" / "假设登记表.md"
CAUSE_TABLE = DOC / "死因码表.md"

HYP_COLS = ("id", "section", "title", "script", "claim", "mechanism", "trace",
            "falsify", "budget", "status", "created_date", "result", "evidence",
            "death_codes")
CAUSE_COLS = ("code", "reason", "first_seen", "blocked_by", "status")

REGISTRY_KEYS = ("假设", "机制", "痕迹", "证伪判据", "预算", "状态", "日期", "结果", "死因码", "证据")
REQUIRED_KEYS = ("假设", "机制", "痕迹", "证伪判据", "预算", "状态", "日期")
STATUSES = ("待验证", "验证中", "通过", "失败", "作废")
SECTION_BY_HEADING = {"## 2.": "在册", "## 4.": "归档"}

DDL_HYPOTHESIS = """
CREATE TABLE IF NOT EXISTS hypothesis (
  id VARCHAR(8) NOT NULL COMMENT '假设号 H2..H9，删除不复用；P 号留给电力域',
  section VARCHAR(8) NOT NULL COMMENT '来源小节：在册 / 归档',
  title VARCHAR(128) NOT NULL COMMENT '块标题（去掉 ID）',
  script VARCHAR(64) COMMENT '验证脚本名，标题里抽不到则为 NULL',
  claim TEXT COMMENT '假设：一句话，可证伪',
  mechanism TEXT COMMENT '机制：谁在被迫以错误价格交易；归档块为 -',
  trace TEXT COMMENT '痕迹：应该留痕的统计量，归档块为 -',
  falsify TEXT COMMENT '证伪判据：事先写死的量级线 + 控制集；归档块为 -',
  budget TEXT COMMENT '预算；归档块为 -',
  status VARCHAR(8) NOT NULL COMMENT '待验证/验证中/通过/失败/作废',
  created_date DATE COMMENT '建卡日',
  result TEXT COMMENT '验证结论，量级与方向',
  evidence TEXT COMMENT '脚本名 + 关键数字',
  death_codes VARCHAR(32) COMMENT '死因码，逗号连接 F1,F2（FIND_IN_SET 可直接用）；无则 NULL',
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

DDL_CAUSE = """
CREATE TABLE IF NOT EXISTS cause_code (
  code VARCHAR(4) NOT NULL COMMENT 'F1..F5 / FX，全域共享，不分域',
  reason VARCHAR(128) NOT NULL COMMENT '死因',
  first_seen VARCHAR(128) COMMENT '首次出现：脚本 + 检验次数',
  blocked_by VARCHAR(128) COMMENT '被哪道防线拦住',
  status VARCHAR(8) NOT NULL COMMENT '在用 / 已废弃',
  PRIMARY KEY (code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

DDL_VIEW = """
CREATE OR REPLACE VIEW v_cause_stat AS
SELECT c.code, c.reason, c.status AS code_status,
       COUNT(h.id) AS n_failed,
       GROUP_CONCAT(h.id ORDER BY h.id) AS hypotheses
FROM cause_code c
LEFT JOIN hypothesis h ON FIND_IN_SET(c.code, h.death_codes)
GROUP BY c.code, c.reason, c.status
"""


def log(msg: str) -> None:
    print(f"[hyp] {msg}", flush=True)


def parse_registry(path: pathlib.Path, errors: list[str]) -> list[dict]:
    """解析登记表里的记录块。fenced code 块（§1 的 H0 示例）整段跳过。"""
    log(f"读假设登记表 {path.name}")
    blocks: list[dict] = []
    section: str | None = None
    in_fence = False
    cur: dict | None = None

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line.startswith("## "):
            section = next((v for k, v in SECTION_BY_HEADING.items() if line.startswith(k)), None)
            cur = None
            continue
        if line.startswith("### "):
            m = re.match(r"^### (H\d+) (.+)$", line)
            if section is None or m is None:
                cur = None
                continue
            cur = {"id": m.group(1), "section": section, "title": m.group(2).strip(),
                   "lineno": lineno, "death_codes": [], "_keys": set()}
            blocks.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"^([^:]+): (.*)$", line)
        if m is None or m.group(1) not in REGISTRY_KEYS:
            continue
        key, val = m.group(1), m.group(2).strip()
        if key == "死因码":
            if val != "-":
                cur["death_codes"].append(val)
            cur["_keys"].add(key)
            continue
        cur[key] = val
        cur["_keys"].add(key)

    rows: list[dict] = []
    for b in blocks:
        missing = [k for k in REQUIRED_KEYS if k not in b["_keys"]]
        if missing:
            errors.append(f"{b['id']}: 缺必填键 {missing}（第 {b['lineno']} 行）")
            continue
        if b["状态"] not in STATUSES:
            errors.append(f"{b['id']}: 状态非法 {b['状态']}")
            continue
        if b["状态"] == "失败" and not b["death_codes"]:
            errors.append(f"{b['id']}: 失败但没填死因码")
        if b["状态"] != "失败" and b["death_codes"]:
            errors.append(f"{b['id']}: 非失败却填了死因码 {b['death_codes']}")
        script = re.search(r"([A-Za-z0-9_]+\.py)", b["title"])
        rows.append({
            "id": b["id"], "section": b["section"], "title": b["title"],
            "script": script.group(1) if script else None,
            "claim": b["假设"], "mechanism": b["机制"], "trace": b["痕迹"],
            "falsify": b["证伪判据"], "budget": b["预算"], "status": b["状态"],
            "created_date": b["日期"] if re.match(r"^\d{4}-\d{2}-\d{2}$", b["日期"]) else None,
            "result": b.get("结果", "-"), "evidence": b.get("证据", "-"),
            "death_codes": ",".join(b["death_codes"]) or None,
        })
    log(f"记录块 {len(rows)} 条（失败 {sum(1 for r in rows if r['status'] == '失败')}）")
    return rows


def parse_cause_table(path: pathlib.Path, errors: list[str]) -> list[dict]:
    """只解析 `## 码表` 那一节的表格；`## 跨域同源` 表整段跳过。"""
    log(f"读死因码表 {path.name}")
    rows: list[dict] = []
    in_table_section = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            in_table_section = line.startswith("## 码表")
            continue
        if not in_table_section or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[0] in ("码",) or set(cells[0]) <= set("- "):
            continue
        code, reason, first_seen, blocked_by, status = cells
        if not re.match(r"^F[A-Z0-9]+$", code):
            errors.append(f"死因码表异常行: {line[:60]}")
            continue
        # md 表格里的 `**已废弃**` 去掉强调符，否则 status 不可比对
        rows.append({"code": code, "reason": reason, "first_seen": first_seen,
                     "blocked_by": blocked_by, "status": status.replace("**", "")})
    log(f"死因码 {len(rows)} 条")
    return rows


def connect(host: str, port: int, user: str, password: str, database: str, errors: list[str]):
    log(f"连 {user}@{host}:{port}/{database}")
    try:
        conn = duckdb.connect()
        conn.execute("INSTALL mysql")
        conn.execute("LOAD mysql")
        conn.execute(f"ATTACH 'host={host} port={port} user={user} password={password} "
                     f"database={database}' AS mq (TYPE mysql)")
    except Exception as exc:
        errors.append(f"连不上 MySQL {host}:{port}/{database}: {exc}")
        log(f"失败: {exc}")
        return None
    log("连上了")
    return conn


def tmp_ddl(name: str, cols: tuple[str, ...], date_cols: tuple[str, ...] = ()) -> str:
    parts = [f"{c} DATE" if c in date_cols else f"{c} VARCHAR" for c in cols]
    return f"CREATE OR REPLACE TEMP TABLE {name} ({', '.join(parts)})"


def run_ddl(host: str, port: int, user: str, password: str, errors: list[str]) -> None:
    log("建表 + 建视图")
    conn = connect(host, port, user, password, "quant", errors)
    if conn is None:
        return
    for ddl in (DDL_HYPOTHESIS, DDL_CAUSE, DDL_VIEW):
        conn.execute("SELECT * FROM mysql_execute('mq', ?)", [ddl]).fetchall()
    log("建表：hypothesis + cause_code，视图：v_cause_stat")
    conn.close()


def load(host: str, port: int, user: str, password: str, errors: list[str]) -> None:
    log("导入（先 TRUNCATE，再整表覆盖）")
    hyps = parse_registry(REGISTRY, errors)
    causes = parse_cause_table(CAUSE_TABLE, errors)
    if errors:
        log("解析有错，不写库")
        return
    conn = connect(host, port, user, password, "quant", errors)
    if conn is None:
        return

    conn.execute(tmp_ddl("h_tmp", HYP_COLS, ("created_date",)))
    conn.executemany(f"INSERT INTO h_tmp ({', '.join(HYP_COLS)}) "
                     f"VALUES ({', '.join('?' * len(HYP_COLS))})",
                     [[r[c] for c in HYP_COLS] for r in hyps])
    conn.execute(tmp_ddl("c_tmp", CAUSE_COLS))
    conn.executemany(f"INSERT INTO c_tmp ({', '.join(CAUSE_COLS)}) "
                     f"VALUES ({', '.join('?' * len(CAUSE_COLS))})",
                     [[r[c] for c in CAUSE_COLS] for r in causes])

    conn.execute("SELECT * FROM mysql_execute('mq', 'TRUNCATE TABLE hypothesis')").fetchall()
    conn.execute("SELECT * FROM mysql_execute('mq', 'TRUNCATE TABLE cause_code')").fetchall()
    conn.execute(f"INSERT INTO mq.hypothesis ({', '.join(HYP_COLS)}) SELECT {', '.join(HYP_COLS)} FROM h_tmp")
    conn.execute(f"INSERT INTO mq.cause_code ({', '.join(CAUSE_COLS)}) SELECT {', '.join(CAUSE_COLS)} FROM c_tmp")
    n_h = conn.execute("SELECT count(*) FROM mq.hypothesis").fetchone()[0]
    n_c = conn.execute("SELECT count(*) FROM mq.cause_code").fetchone()[0]
    log(f"写入 hypothesis {n_h} 行 / cause_code {n_c} 行")
    if n_h != len(hyps) or n_c != len(causes):
        errors.append(f"写入行数不符: {n_h}/{len(hyps)}, {n_c}/{len(causes)}")
    conn.close()


def verify(host: str, port: int, user: str, password: str, errors: list[str]) -> None:
    """行数一致 + 死因码可归因 + 状态分布 + H1 抽样逐值。"""
    log("校验")
    hyps = parse_registry(REGISTRY, errors)
    causes = parse_cause_table(CAUSE_TABLE, errors)
    conn = connect(host, port, user, password, "quant", errors)
    if conn is None:
        return

    got = conn.execute("SELECT count(*) FROM mq.hypothesis").fetchone()[0]
    log(f"hypothesis 行数: mysql {got} / md {len(hyps)}")
    if got != len(hyps):
        errors.append(f"hypothesis 行数不符: {got} != {len(hyps)}")

    got_c = conn.execute("SELECT count(*) FROM mq.cause_code").fetchone()[0]
    log(f"cause_code 行数: mysql {got_c} / md {len(causes)}")
    if got_c != len(causes):
        errors.append(f"cause_code 行数不符: {got_c} != {len(causes)}")

    # 每个入库的死因码都必须能在 cause_code 里找到（LIKE 匹配的假阳性会在这里暴露）
    codes = [c["code"] for c in causes]
    for row in conn.execute("SELECT id, death_codes FROM mq.hypothesis WHERE death_codes IS NOT NULL").fetchall():
        for code in str(row[1]).split(","):
            if code not in codes:
                errors.append(f"{row[0]}: 死因码 {code} 不在 cause_code 表里")

    dist = conn.execute("SELECT status, count(*) FROM mq.hypothesis GROUP BY status ORDER BY 2 DESC").fetchall()
    log(f"状态分布: {[tuple(r) for r in dist]}")

    cause_dist = conn.execute("SELECT code, n_failed FROM mq.v_cause_stat ORDER BY n_failed DESC").fetchall()
    log(f"死因分布: {[tuple(r) for r in cause_dist]}")

    ref = next((h for h in hyps if h["id"] == "H1"), None)
    if ref is None:
        errors.append("md 里没有 H1，无法抽样")
    else:
        row = conn.execute("SELECT id, status, claim, death_codes, created_date FROM mq.hypothesis "
                           "WHERE id = 'H1'").fetchone()
        want = (ref["id"], ref["status"], ref["claim"], ref["death_codes"], ref["created_date"])
        log(f"H1 抽样: {tuple(str(v) for v in row)}")
        if tuple(str(v) for v in row) != tuple(str(v) for v in want):
            errors.append(f"H1 抽样不一致: {row} != {want}")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="把 doc/ 的假设登记表与死因码表镜像进 MySQL quant 库")
    parser.add_argument("action", choices=["ddl", "load", "verify"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--user", default="wfy")
    parser.add_argument("--password", default=os.environ.get("QUANT_MYSQL_PASSWORD", ""))
    args = parser.parse_args()

    errors: list[str] = []
    if not args.password:
        errors.append("没给密码：用 --password 或环境变量 QUANT_MYSQL_PASSWORD")
    elif args.action == "ddl":
        run_ddl(args.host, args.port, args.user, args.password, errors)
    elif args.action == "load":
        load(args.host, args.port, args.user, args.password, errors)
    else:
        verify(args.host, args.port, args.user, args.password, errors)

    for err in errors:
        log(f"错误: {err}")
    log("结束" if not errors else f"结束，{len(errors)} 个错误")


if __name__ == "__main__":
    main()
