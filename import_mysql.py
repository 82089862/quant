"""把 data/ 的 parquet 导入本机 MySQL 的 quant 库。

MySQL 只是旁路副本，回测链路（store.py / factors.py / backtest.py）不依赖它。
走 DuckDB 的 mysql 扩展直连，无中间文件；失败再退到 CSV + LOAD DATA。

用法:
    QUANT_MYSQL_PASSWORD=xxx uv run import_mysql.py ddl      # 建库 + 建表（可重复执行）
    QUANT_MYSQL_PASSWORD=xxx uv run import_mysql.py probe    # 写路径小样（1 万行，自动删探针表）
    QUANT_MYSQL_PASSWORD=xxx uv run import_mysql.py load --table all
    QUANT_MYSQL_PASSWORD=xxx uv run import_mysql.py verify
"""
from __future__ import annotations

import argparse
import os
import pathlib
import time

import duckdb

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PANEL_PATH = DATA_DIR / "panel_2015_2026.parquet"
UNIVERSE_PATH = DATA_DIR / "universe.parquet"

DAILY_COLS = ("symbol", "date", "open", "high", "low", "close",
              "volume", "amount", "outstanding_share", "turnover")
UNIVERSE_COLS = ("code", "code_name", "ipoDate", "outDate", "type", "status", "symbol")

YEAR_FIRST = 2015
YEAR_LAST = 2026

DDL_DATABASE = "CREATE DATABASE IF NOT EXISTS quant DEFAULT CHARACTER SET utf8mb4"

# 分区按年，不做 MAXVALUE 兜底：超范围的新数据会直接报错（1526），
# 比静默落进兜底分区好排查。加年份 = ADD PARTITION。
DDL_DAILY = f"""
CREATE TABLE IF NOT EXISTS daily (
  symbol VARCHAR(8) NOT NULL COMMENT '股票代码，形如 sh600519 / sz000001',
  `date` DATE NOT NULL COMMENT '交易日',
  `open` DOUBLE COMMENT '开盘价，后复权（新浪 hfq）',
  high DOUBLE COMMENT '最高价，后复权',
  low DOUBLE COMMENT '最低价，后复权',
  `close` DOUBLE COMMENT '收盘价，后复权',
  volume DOUBLE COMMENT '成交量（股），真实成交口径，不复权',
  amount DOUBLE COMMENT '成交额（元），真实成交口径，不复权',
  outstanding_share DOUBLE COMMENT '流通股本（股）',
  turnover DOUBLE COMMENT '换手率，小数（0.0028 = 0.28%）',
  PRIMARY KEY (symbol, `date`),
  KEY idx_date (`date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
PARTITION BY RANGE COLUMNS(`date`) (
{",\n".join(f"  PARTITION p{y} VALUES LESS THAN ('{y + 1}-01-01')" for y in range(YEAR_FIRST, YEAR_LAST + 1))}
)
"""

DDL_UNIVERSE = """
CREATE TABLE IF NOT EXISTS universe (
  code VARCHAR(16) NOT NULL COMMENT 'baostock 证券代码，形如 sh.600519',
  code_name VARCHAR(64) COMMENT '证券简称',
  ipoDate DATE COMMENT '上市日',
  outDate DATE COMMENT '退市日，未退市为 NULL',
  `type` VARCHAR(8) COMMENT 'baostock 证券类型：1=股票 2=指数 4=可转债 5=ETF（本表实测只有这 4 种）',
  `status` VARCHAR(8) COMMENT 'baostock 上市状态：1=上市 0=退市',
  symbol VARCHAR(16) COMMENT '本项目代码，形如 sh600519',
  PRIMARY KEY (code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def log(msg: str) -> None:
    print(f"[mysql] {msg}", flush=True)


def connect(host: str, port: int, user: str, password: str, database: str, errors: list[str]):
    """连 MySQL 并挂成 duckdb 的 mq 别名（可写）。失败原因进 errors，返回 None。"""
    log(f"连 {user}@{host}:{port}/{database}")
    try:
        conn = duckdb.connect()
        conn.execute("INSTALL mysql")
        conn.execute("LOAD mysql")
        conn.execute(
            f"ATTACH 'host={host} port={port} user={user} password={password} "
            f"database={database}' AS mq (TYPE mysql)"
        )
    except Exception as exc:
        errors.append(f"连不上 MySQL {host}:{port}/{database}: {exc}")
        log(f"失败: {exc}")
        return None
    log("连上了")
    return conn


def cols_sql(columns: tuple[str, ...]) -> str:
    """列名加双引号给 duckdb 用（date / open / close 是关键字）。"""
    return ", ".join(f'"{c}"' for c in columns)


def run_ddl(host: str, port: int, user: str, password: str, errors: list[str]) -> None:
    """建库 + 建表。attach 到 mysql 系统库只用来执行 CREATE DATABASE，不往里写。"""
    log("建库 + 建表")
    admin = connect(host, port, user, password, "mysql", errors)
    if admin is None:
        return
    admin.execute("SELECT * FROM mysql_execute('mq', ?)", [DDL_DATABASE]).fetchall()
    admin.close()

    conn = connect(host, port, user, password, "quant", errors)
    if conn is None:
        return
    conn.execute("SELECT * FROM mysql_execute('mq', ?)", [DDL_DAILY]).fetchall()
    conn.execute("SELECT * FROM mysql_execute('mq', ?)", [DDL_UNIVERSE]).fetchall()
    log("建表：daily（按年分区）+ universe")
    conn.close()


def probe(host: str, port: int, user: str, password: str, errors: list[str]) -> None:
    """用 1 万行实测 duckdb -> mysql 的写路径与速率，测完删掉探针表。"""
    log("写路径小样测试（1 万行）")
    conn = connect(host, port, user, password, "quant", errors)
    if conn is None:
        return
    conn.execute("SELECT * FROM mysql_execute('mq', 'DROP TABLE IF EXISTS daily_probe')").fetchall()
    conn.execute("SELECT * FROM mysql_execute('mq', 'CREATE TABLE daily_probe LIKE daily')").fetchall()
    t0 = time.time()
    inserted = conn.execute(
        f'INSERT INTO mq.daily_probe ({cols_sql(DAILY_COLS)}) '
        f'SELECT {cols_sql(DAILY_COLS)} FROM \'{PANEL_PATH.as_posix()}\' LIMIT 10000'
    ).fetchall()
    elapsed = time.time() - t0
    n = conn.execute("SELECT count(*) FROM mq.daily_probe").fetchone()[0]
    log(f"写入 {n} 行，{elapsed:.1f}s，{n / max(elapsed, 0.01):.0f} 行/s "
        f"-> 1102 万行外推约 {11022483 / max(n / max(elapsed, 0.01), 1) / 60:.0f} 分钟")
    conn.execute("SELECT * FROM mysql_execute('mq', 'DROP TABLE daily_probe')").fetchall()
    log("探针表已删")
    if n != 10000:
        errors.append(f"探针写入行数不对: {n} != 10000")
    conn.close()


def load_daily(conn, errors: list[str]) -> None:
    """按年分批插，一年一个 INSERT（单语句 = 单事务），已有数据的年份跳过。"""
    for year in range(YEAR_FIRST, YEAR_LAST + 1):
        has = conn.execute(
            f"SELECT count(*) FROM mq.daily WHERE \"date\" >= DATE '{year}-01-01' "
            f"AND \"date\" < DATE '{year + 1}-01-01'"
        ).fetchone()[0]
        if has:
            log(f"{year}: 已有 {has} 行，跳过")
            continue
        t0 = time.time()
        conn.execute(
            f'INSERT INTO mq.daily ({cols_sql(DAILY_COLS)}) '
            f'SELECT {cols_sql(DAILY_COLS)} FROM \'{PANEL_PATH.as_posix()}\' '
            f"WHERE \"date\" >= DATE '{year}-01-01' AND \"date\" < DATE '{year + 1}-01-01'"
        )
        n = conn.execute(
            f"SELECT count(*) FROM mq.daily WHERE \"date\" >= DATE '{year}-01-01' "
            f"AND \"date\" < DATE '{year + 1}-01-01'"
        ).fetchone()[0]
        log(f"{year}: {n} 行，{time.time() - t0:.1f}s")


def load_universe(conn, errors: list[str]) -> None:
    existing = conn.execute("SELECT count(*) FROM mq.universe").fetchone()[0]
    if existing:
        log(f"universe 已有 {existing} 行，跳过")
        return
    t0 = time.time()
    conn.execute(
        f'INSERT INTO mq.universe ({cols_sql(UNIVERSE_COLS)}) '
        f'SELECT {cols_sql(UNIVERSE_COLS)} FROM \'{UNIVERSE_PATH.as_posix()}\''
    )
    n = conn.execute("SELECT count(*) FROM mq.universe").fetchone()[0]
    log(f"universe: {n} 行，{time.time() - t0:.1f}s")


def verify(host: str, port: int, user: str, password: str, errors: list[str]) -> None:
    """行数 + 分区 + 抽样逐值比对 parquet。"""
    log("校验")
    conn = connect(host, port, user, password, "quant", errors)
    if conn is None:
        return

    want_daily = conn.execute(f"SELECT count(*) FROM '{PANEL_PATH.as_posix()}'").fetchone()[0]
    got_daily = conn.execute("SELECT count(*) FROM mq.daily").fetchone()[0]
    log(f"daily 行数: mysql {got_daily} / parquet {want_daily}")
    if got_daily != want_daily:
        errors.append(f"daily 行数不符: {got_daily} != {want_daily}")

    want_uni = conn.execute(f"SELECT count(*) FROM '{UNIVERSE_PATH.as_posix()}'").fetchone()[0]
    got_uni = conn.execute("SELECT count(*) FROM mq.universe").fetchone()[0]
    log(f"universe 行数: mysql {got_uni} / parquet {want_uni}")
    if got_uni != want_uni:
        errors.append(f"universe 行数不符: {got_uni} != {want_uni}")

    parts = conn.execute(
        "SELECT * FROM mysql_query('mq', 'SELECT partition_name FROM information_schema.partitions "
        "WHERE table_schema = ''quant'' AND table_name = ''daily'' "
        "AND partition_name IS NOT NULL ORDER BY partition_ordinal_position')"
    ).fetchall()
    log(f"分区数: {len(parts)}（{parts[0][0]} .. {parts[-1][0]}）")
    if len(parts) != YEAR_LAST - YEAR_FIRST + 1:
        errors.append(f"分区数不对: {len(parts)}")

    for symbol in ("sh600519", "sz000001", "sz300750"):
        rows = conn.execute(
            f'SELECT {cols_sql(DAILY_COLS)} FROM mq.daily WHERE symbol = ? '
            f'ORDER BY "date" LIMIT 5', [symbol]
        ).fetchall()
        ref = conn.execute(
            f'SELECT {cols_sql(DAILY_COLS)} FROM \'{PANEL_PATH.as_posix()}\' '
            f"WHERE symbol = ? ORDER BY \"date\" LIMIT 5", [symbol]
        ).fetchall()
        if len(rows) != 5:
            errors.append(f"{symbol} 抽样只有 {len(rows)} 行")
            continue
        max_diff = max((abs(float(a) - float(b)) for r, q in zip(rows, ref)
                        for a, b in zip(r[2:], q[2:]) if a is not None and b is not None),
                       default=0.0)
        same_date = [str(r[1])[:10] for r in rows] == [str(q[1])[:10] for q in ref]
        log(f"{symbol} 抽样 5 行: 日期一致 {same_date}，数值最大差 {max_diff:.3e}")
        if not same_date or max_diff > 1e-9:
            errors.append(f"{symbol} 抽样与 parquet 不一致（日期 {same_date}，最大差 {max_diff}）")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="把 quant 的 parquet 导入本机 MySQL")
    parser.add_argument("action", choices=["ddl", "probe", "load", "verify"])
    parser.add_argument("--table", choices=["daily", "universe", "all"], default="all")
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
    elif args.action == "probe":
        probe(args.host, args.port, args.user, args.password, errors)
    elif args.action == "verify":
        verify(args.host, args.port, args.user, args.password, errors)
    else:
        conn = connect(args.host, args.port, args.user, args.password, "quant", errors)
        if conn is not None:
            if args.table in ("daily", "all"):
                load_daily(conn, errors)
            if args.table in ("universe", "all"):
                load_universe(conn, errors)
            conn.close()

    for err in errors:
        log(f"错误: {err}")
    log("结束" if not errors else f"结束，{len(errors)} 个错误")


if __name__ == "__main__":
    main()
