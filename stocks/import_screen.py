"""把外部策略池的筛选结果镜像进本机 MySQL 的 quant 库。

**单向只读镜像**：事实源是 `stocks/out/timing_pool_summary.csv`（由 `stocks/timing.py` 生成），
在库里手改的数据下次 load 会被 TRUNCATE 冲掉。回测链路不读这张表。

表只有一张 `timing_screen`：一行 = 一个「指数 × 策略」组合的筛选结论。
passed 在导入时按 Bonferroni 门槛 1 - 0.05/n_tests 现算，n_tests 从不含「买入持有」的行数来，
所以策略池加人会自动收紧门槛，不用手改常量。

Alpha191 因子库的筛选结果**不进这张表** —— 它是截面因子、指标口径是 IC/ICIR 而不是累计收益，
到时候另立 `factor_screen`，不把两种口径塞进同一张表。

用法:
    QUANT_MYSQL_PASSWORD=xxx uv run stocks/import_screen.py ddl
    QUANT_MYSQL_PASSWORD=xxx uv run stocks/import_screen.py load
    QUANT_MYSQL_PASSWORD=xxx uv run stocks/import_screen.py verify
"""
from __future__ import annotations

import argparse
import os
import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parent
SUMMARY = ROOT / "out" / "timing_pool_summary.csv"
BENCHMARK = "买入持有"

# CSV 中文列 -> 表里英文列（列名即 API：英文 key，中文名进 COMMENT）
COL_MAP = {
    "指数": "symbol", "策略": "strategy",
    "全累计": "full_cum", "全夏普": "full_sharpe", "全回撤": "full_mdd",
    "训练累计": "train_cum", "训练夏普": "train_sharpe",
    "测试累计": "test_cum", "测试夏普": "test_sharpe",
    "交易笔数": "n_trades", "涨跌停被挡": "limit_blocked",
    "测试分位": "test_pct",
}
COLUMNS = tuple(COL_MAP.values())

DDL_TIMING_SCREEN = f"""
CREATE TABLE IF NOT EXISTS timing_screen (
  symbol VARCHAR(16) NOT NULL COMMENT '指数代码，形如 sh000300',
  strategy VARCHAR(16) NOT NULL COMMENT '策略名 rsrs/rsrs_r2/icu/icu2/llt/alligator/tsharpe；{BENCHMARK} 是对照',
  full_cum DOUBLE COMMENT '全样本累计收益，2015-01~2026-10',
  full_sharpe DOUBLE COMMENT '全样本夏普，日频年化',
  full_mdd DOUBLE COMMENT '全样本最大回撤',
  train_cum DOUBLE COMMENT '训练段累计收益，<= 2020-12-31',
  train_sharpe DOUBLE COMMENT '训练段夏普',
  test_cum DOUBLE COMMENT '测试段累计收益，> 2020-12-31',
  test_sharpe DOUBLE COMMENT '测试段夏普',
  n_trades INT COMMENT '交易笔数（含费用，lot_size=1）',
  limit_blocked INT COMMENT '涨跌停被挡次数（指数上实测为 0）',
  test_pct DOUBLE COMMENT '循环平移 surrogate 分位，300 次；策略持仓循环平移后看实际测试收益排第几',
  passed TINYINT COMMENT '1 = 过 Bonferroni 门槛 1-0.05/检验次数；NULL = 对照行不参与检验',
  PRIMARY KEY (symbol, strategy)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def log(msg: str) -> None:
    print(f"[screen] {msg}", flush=True)


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


def source_sql(conn, errors: list[str]) -> str | None:
    """返回读 CSV 事实源的 SELECT 语句。列名对不上就报错停手，不猜。"""
    if not SUMMARY.exists():
        errors.append(f"{SUMMARY} 不存在，先跑 stocks/timing.py --out stocks/out/timing_pool.csv")
        return None
    sql = (f"SELECT {', '.join(f'\"{k}\" AS {v}' for k, v in COL_MAP.items())} "
           f"FROM read_csv_auto('{SUMMARY.as_posix()}')")
    try:
        n = conn.execute(f"SELECT count(*) FROM ({sql})").fetchone()[0]
    except Exception as exc:
        errors.append(f"读 CSV 失败（列名对不上？期望 {list(COL_MAP)}）: {exc}")
        log(f"失败: {exc}")
        return None
    log(f"读 {SUMMARY.name}: {n} 行")
    return sql


def count_tests(conn, sql: str, errors: list[str]) -> int:
    """检验次数 = 不含对照行的行数，用它算门槛。"""
    n = conn.execute(f"SELECT count(*) FROM ({sql}) WHERE strategy <> '{BENCHMARK}'").fetchone()[0]
    log(f"检验次数 {n}，Bonferroni 门槛 1-0.05/{n} = {1 - 0.05 / n:.4f}")
    return int(n)


def run_ddl(host: str, port: int, user: str, password: str, errors: list[str]) -> None:
    log("建表")
    conn = connect(host, port, user, password, "quant", errors)
    if conn is None:
        return
    conn.execute("SELECT * FROM mysql_execute('mq', ?)", [DDL_TIMING_SCREEN]).fetchall()
    log("建表：timing_screen")
    conn.close()


def load(host: str, port: int, user: str, password: str, errors: list[str]) -> None:
    log("导入（先 TRUNCATE，再整表覆盖）")
    conn = connect(host, port, user, password, "quant", errors)
    if conn is None:
        return
    sql = source_sql(conn, errors)
    if sql is None or errors:
        log("没有可用事实源，不写库")
        conn.close()
        return
    threshold = 1 - 0.05 / count_tests(conn, sql, errors)
    cols = ", ".join(COLUMNS)
    conn.execute("SELECT * FROM mysql_execute('mq', 'TRUNCATE TABLE timing_screen')").fetchall()
    conn.execute(
        f"INSERT INTO mq.timing_screen ({cols}, passed) "
        f"SELECT {', '.join(s for s in [f'src.{c}' for c in COLUMNS])}, "
        f"CASE WHEN src.test_pct IS NULL THEN NULL "
        f"ELSE CAST(src.test_pct >= {threshold} AS TINYINT) END "
        f"FROM ({sql}) AS src"
    )
    n = conn.execute("SELECT count(*) FROM mq.timing_screen").fetchone()[0]
    n_pass = conn.execute("SELECT count(*) FROM mq.timing_screen WHERE passed = 1").fetchone()[0]
    log(f"写入 timing_screen {n} 行，其中通过 {n_pass} 行")
    conn.close()


def verify(host: str, port: int, user: str, password: str, errors: list[str]) -> None:
    """行数 + 门槛 + 通过数 + 最好的一行逐值比对。"""
    log("校验")
    conn = connect(host, port, user, password, "quant", errors)
    if conn is None:
        return
    sql = source_sql(conn, errors)
    if sql is None or errors:
        conn.close()
        return
    threshold = 1 - 0.05 / count_tests(conn, sql, errors)

    want = conn.execute(f"SELECT count(*) FROM ({sql})").fetchone()[0]
    got = conn.execute("SELECT count(*) FROM mq.timing_screen").fetchone()[0]
    log(f"行数: mysql {got} / csv {want}")
    if got != want:
        errors.append(f"行数不符: {got} != {want}")

    # 门槛一致性：库里的 passed 必须能由 test_pct 与门槛重算出来
    bad = conn.execute(
        f"SELECT count(*) FROM mq.timing_screen WHERE test_pct IS NOT NULL "
        f"AND passed <> CAST(test_pct >= {threshold} AS TINYINT)").fetchone()[0]
    if bad:
        errors.append(f"{bad} 行的 passed 与门槛 {threshold:.4f} 不自洽")

    n_pass = conn.execute("SELECT count(*) FROM mq.timing_screen WHERE passed = 1").fetchone()[0]
    log(f"通过 Bonferroni 门槛 {threshold:.4f} 的：{n_pass} 行")

    best = conn.execute("SELECT symbol, strategy, test_cum, test_sharpe, test_pct FROM mq.timing_screen "
                        "WHERE test_pct IS NOT NULL ORDER BY test_pct DESC LIMIT 1").fetchone()
    log(f"测试分位最高: {best}")
    ref = conn.execute(f"SELECT \"指数\", \"策略\", \"测试累计\", \"测试夏普\", \"测试分位\" "
                       f"FROM read_csv_auto('{SUMMARY.as_posix()}') "
                       f"ORDER BY \"测试分位\" DESC NULLS LAST LIMIT 1").fetchone()
    if best[0] != ref[0] or best[1] != ref[1]:
        errors.append(f"最高分位那行与 CSV 不一致: {best} vs {ref}")
    elif abs(best[2] - ref[2]) > 1e-12 or abs(best[4] - ref[4]) > 1e-12:
        errors.append(f"最高分位那行数值不一致: {best} vs {ref}")
    else:
        log("最高分位那行与 CSV 逐值一致")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="把择时策略池筛选结果镜像进 MySQL quant 库")
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
