"""把外部策略池 / 因子库的筛选结果镜像进本机 MySQL 的 quant 库。

**单向只读镜像**：事实源是两个 CSV，库里手改的数据下次 load 会被 TRUNCATE 冲掉。
回测链路不读这两张表。

| 表 | 事实源 | 一行是什么 |
|---|---|---|
| `timing_screen` | `stocks/out/timing_pool_summary.csv`（`stocks/timing.py` 生成） | 一个「指数 × 择时策略」的累计收益/夏普/surrogate 分位 |
| `factor_screen` | `stocks/out/alpha191_ic.csv`（`stocks/alpha191.py` 生成） | 一个 Alpha191 因子的 IC/ICIR/训练测试/正交化判读 |

两张表分家是有意的：择时的口径是累计收益，截面因子的口径是 IC/ICIR，塞进同一张表
以后没人分得清哪个指标该拿来比。

`timing_screen.passed` 与 `factor_screen.bonferroni` 都不是源里的列，是导入时按
Bonferroni 门槛 1-0.05/检验次数**现算**的；检验次数从行数推导，池子加人会自动收紧门槛。

用法:
    QUANT_MYSQL_PASSWORD=xxx uv run stocks/import_screen.py ddl    [--table timing|factor|all]
    QUANT_MYSQL_PASSWORD=xxx uv run stocks/import_screen.py load   [--table ...]
    QUANT_MYSQL_PASSWORD=xxx uv run stocks/import_screen.py verify [--table ...]
"""
from __future__ import annotations

import argparse
import os
import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parent
SUMMARY = ROOT / "out" / "timing_pool_summary.csv"
FACTOR_CSV = ROOT / "out" / "alpha191_ic.csv"
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

# alpha191_ic.csv 的列顺序，显式列出；对不上就停手（位置错一列会静默串数据）
FACTOR_COLUMNS = ("factor", "status", "reason", "n", "ic_mean", "ic_std", "icir", "t", "p",
                  "bonferroni", "fdr", "ic_train", "ic_test", "t_train", "t_test", "sign_same",
                  "ic_nocontrol", "ic_single_min", "ic_allctrl", "r2_allctrl", "orth_n",
                  "verdict", "verdict_why")

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
  `passed` TINYINT COMMENT '1 = 过 Bonferroni 门槛 1-0.05/检验次数；NULL = 对照行不参与检验',
  PRIMARY KEY (symbol, strategy)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

DDL_FACTOR_SCREEN = """
CREATE TABLE IF NOT EXISTS factor_screen (
  factor VARCHAR(16) NOT NULL COMMENT '因子名 alpha001..alpha191',
  status VARCHAR(8) NOT NULL COMMENT '已算 / 跳过 / 报错',
  reason VARCHAR(255) COMMENT '跳过或报错的原因；已算为空',
  n INT COMMENT 'IC 期数（月频）',
  ic_mean DOUBLE COMMENT '全样本月频 Spearman IC 均值',
  ic_std DOUBLE COMMENT 'IC 标准差',
  icir DOUBLE COMMENT 'IC 均值/标准差',
  t DOUBLE COMMENT 't = 均值/标准差*sqrt(期数)，未做自相关修正',
  p DOUBLE COMMENT '双侧正态近似 p 值',
  bonferroni TINYINT COMMENT '1 = 过 p < 0.05/检验次数',
  fdr TINYINT COMMENT '1 = 过 Benjamini-Hochberg 5%',
  ic_train DOUBLE COMMENT '训练段（<=2020-12-31）IC 均值',
  ic_test DOUBLE COMMENT '测试段（>2020-12-31）IC 均值',
  t_train DOUBLE COMMENT '训练段 t',
  t_test DOUBLE COMMENT '测试段 t',
  sign_same TINYINT COMMENT '训练/测试 IC 符号是否一致',
  ic_nocontrol DOUBLE COMMENT '测试段无控制 IC（对账列：必须逐位等于 ic_test）',
  ic_single_min DOUBLE COMMENT '测试段被单个控制变量打回后的最弱 |IC|',
  ic_allctrl DOUBLE COMMENT '测试段对 11 个控制变量正交化后的残差 IC',
  r2_allctrl DOUBLE COMMENT '被 11 个控制变量解释的 R²',
  orth_n INT COMMENT '正交化用到的截面期数',
  verdict VARCHAR(24) COMMENT 'F3 确凿换皮 / 被多变量联手吃掉 / 部分独立 / F5 本来就无信号',
  verdict_why VARCHAR(255) COMMENT '判读依据的量化说明',
  PRIMARY KEY (factor)
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


def timing_sql(conn, errors: list[str]) -> str | None:
    """返回读择时汇总表的 SELECT。列名对不上就报错停手，不猜。"""
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
    """择时的检验次数 = 不含对照行的行数，用它算门槛。"""
    n = conn.execute(f"SELECT count(*) FROM ({sql}) WHERE strategy <> '{BENCHMARK}'").fetchone()[0]
    log(f"择时检验次数 {n}，Bonferroni 门槛 1-0.05/{n} = {1 - 0.05 / n:.4f}")
    return int(n)


def factor_sql(conn, errors: list[str]) -> str | None:
    """返回读 Alpha191 筛选结果的 SELECT，并校验列名列序。"""
    if not FACTOR_CSV.exists():
        errors.append(f"{FACTOR_CSV} 不存在，先跑 stocks/alpha191.py")
        return None
    cols = tuple(conn.execute(f"SELECT * FROM read_csv_auto('{FACTOR_CSV.as_posix()}') LIMIT 0")
                 .df().columns)
    if cols != FACTOR_COLUMNS:
        errors.append(f"{FACTOR_CSV.name} 列不符：期望 {FACTOR_COLUMNS}，实际 {cols}")
        return None
    n = conn.execute(f"SELECT count(*) FROM read_csv_auto('{FACTOR_CSV.as_posix()}')").fetchone()[0]
    m = conn.execute(f"SELECT count(*) FROM read_csv_auto('{FACTOR_CSV.as_posix()}') "
                     f"WHERE status = '已算'").fetchone()[0]
    log(f"读 {FACTOR_CSV.name}: {n} 行（已算 {m}），Bonferroni 阈值 p < {0.05 / m:.2e}")
    return f"SELECT * FROM read_csv_auto('{FACTOR_CSV.as_posix()}')"


def run_ddl_timing(conn, errors: list[str]) -> None:
    conn.execute("SELECT * FROM mysql_execute('mq', ?)", [DDL_TIMING_SCREEN]).fetchall()
    log("建表：timing_screen")


def run_ddl_factor(conn, errors: list[str]) -> None:
    conn.execute("SELECT * FROM mysql_execute('mq', ?)", [DDL_FACTOR_SCREEN]).fetchall()
    log("建表：factor_screen")


def load_timing(conn, errors: list[str]) -> None:
    log("导入 timing_screen（先 TRUNCATE，再整表覆盖）")
    sql = timing_sql(conn, errors)
    if sql is None or errors:
        log("没有可用事实源，不写库")
        return
    threshold = 1 - 0.05 / count_tests(conn, sql, errors)
    cols = ", ".join(COLUMNS)
    conn.execute("SELECT * FROM mysql_execute('mq', 'TRUNCATE TABLE timing_screen')").fetchall()
    conn.execute(
        f"INSERT INTO mq.timing_screen ({cols}, passed) "
        f"SELECT {', '.join(f'src.{c}' for c in COLUMNS)}, "
        f"CASE WHEN src.test_pct IS NULL THEN NULL "
        f"ELSE CAST(src.test_pct >= {threshold} AS TINYINT) END "
        f"FROM ({sql}) AS src"
    )
    n = conn.execute("SELECT count(*) FROM mq.timing_screen").fetchone()[0]
    n_pass = conn.execute("SELECT count(*) FROM mq.timing_screen WHERE passed = 1").fetchone()[0]
    log(f"写入 timing_screen {n} 行，其中通过 {n_pass} 行")


def load_factor(conn, errors: list[str]) -> None:
    log("导入 factor_screen（先 TRUNCATE，再整表覆盖）")
    sql = factor_sql(conn, errors)
    if sql is None or errors:
        log("没有可用事实源，不写库")
        return
    conn.execute("SELECT * FROM mysql_execute('mq', 'TRUNCATE TABLE factor_screen')").fetchall()
    conn.execute(f"INSERT INTO mq.factor_screen ({', '.join(FACTOR_COLUMNS)}) "
                 f"SELECT {', '.join(FACTOR_COLUMNS)} FROM ({sql}) AS src")
    n = conn.execute("SELECT count(*) FROM mq.factor_screen").fetchone()[0]
    ok = conn.execute("SELECT count(*) FROM mq.factor_screen WHERE status = '已算'").fetchone()[0]
    log(f"写入 factor_screen {n} 行（已算 {ok}）")


def verify_timing(conn, errors: list[str]) -> None:
    """行数 + 门槛 + 通过数 + 最好的一行逐值比对。"""
    log("校验 timing_screen")
    sql = timing_sql(conn, errors)
    if sql is None or errors:
        return
    threshold = 1 - 0.05 / count_tests(conn, sql, errors)

    want = conn.execute(f"SELECT count(*) FROM ({sql})").fetchone()[0]
    got = conn.execute("SELECT count(*) FROM mq.timing_screen").fetchone()[0]
    log(f"行数: mysql {got} / csv {want}")
    if got != want:
        errors.append(f"行数不符: {got} != {want}")

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


def verify_factor(conn, errors: list[str]) -> None:
    """行数 + 正交化对账列 + 判读分布 + 候选清单。"""
    log("校验 factor_screen")
    sql = factor_sql(conn, errors)
    if sql is None or errors:
        return
    want = conn.execute(f"SELECT count(*) FROM ({sql})").fetchone()[0]
    got = conn.execute("SELECT count(*) FROM mq.factor_screen").fetchone()[0]
    log(f"行数: mysql {got} / csv {want}")
    if got != want:
        errors.append(f"行数不符: {got} != {want}")

    # 对账列：无控制 IC 必须等于测试段 IC（接错行集/秩时这里会先炸）
    bad = conn.execute("SELECT count(*) FROM mq.factor_screen WHERE status = '已算' "
                       "AND abs(ic_nocontrol - ic_test) > 1e-12").fetchone()[0]
    if bad:
        errors.append(f"{bad} 行 ic_nocontrol != ic_test，正交化对账列被污染")
    else:
        log("对账列 ic_nocontrol == ic_test 逐位一致")

    dist = conn.execute("SELECT verdict, count(*) FROM mq.factor_screen WHERE status = '已算' "
                        "GROUP BY verdict ORDER BY 2 DESC").fetchall()
    log(f"判读分布: {[tuple(r) for r in dist]}")
    n_b = conn.execute("SELECT count(*) FROM mq.factor_screen WHERE bonferroni = 1").fetchone()[0]
    n_f = conn.execute("SELECT count(*) FROM mq.factor_screen WHERE fdr = 1").fetchone()[0]
    log(f"过 Bonferroni {n_b} 个 / 过 FDR {n_f} 个")

    keep = conn.execute("SELECT factor, ic_allctrl, r2_allctrl, t FROM mq.factor_screen "
                        "WHERE status = '已算' AND bonferroni = 1 AND sign_same = 1 "
                        "AND verdict = '部分独立' ORDER BY abs(ic_allctrl) DESC").fetchall()
    log(f"过 Bonferroni + 训练测试同号 + 部分独立: {len(keep)} 个")
    for r in keep:
        print(f"      {r[0]}  残差 IC {r[1]:+.4f}  R2 {r[2]:.3f}  t {r[3]:+.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="把筛选结果镜像进 MySQL quant 库")
    parser.add_argument("action", choices=["ddl", "load", "verify"])
    parser.add_argument("--table", choices=["timing", "factor", "all"], default="all")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--user", default="wfy")
    parser.add_argument("--password", default=os.environ.get("QUANT_MYSQL_PASSWORD", ""))
    args = parser.parse_args()

    errors: list[str] = []
    if not args.password:
        errors.append("没给密码：用 --password 或环境变量 QUANT_MYSQL_PASSWORD")
    else:
        conn = connect(args.host, args.port, args.user, args.password, "quant", errors)
        if conn is not None:
            want_timing = args.table in ("timing", "all")
            want_factor = args.table in ("factor", "all")
            if args.action == "ddl":
                if want_timing:
                    run_ddl_timing(conn, errors)
                if want_factor:
                    run_ddl_factor(conn, errors)
            elif args.action == "load":
                if want_timing:
                    load_timing(conn, errors)
                if want_factor:
                    load_factor(conn, errors)
            else:
                if want_timing:
                    verify_timing(conn, errors)
                if want_factor:
                    verify_factor(conn, errors)
            conn.close()

    for err in errors:
        log(f"错误: {err}")
    log("结束" if not errors else f"结束，{len(errors)} 个错误")


if __name__ == "__main__":
    main()
