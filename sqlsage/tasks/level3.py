"""Level 3 TPC-H task definitions."""

from sqlsage.tasks import Task

task1 = Task(
    query="""
SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr,
       s_address, s_phone, s_comment
FROM part, supplier, partsupp, nation, region
WHERE p_partkey = ps_partkey
  AND s_suppkey = ps_suppkey
  AND p_size = 15
  AND p_type LIKE '%BRASS'
  AND s_nationkey = n_nationkey
  AND n_regionkey = r_regionkey
  AND r_name = 'EUROPE'
  AND ps_supplycost = (
      SELECT MIN(ps_supplycost)
      FROM partsupp, supplier, nation, region
      WHERE p_partkey = ps_partkey
        AND s_suppkey = ps_suppkey
        AND s_nationkey = n_nationkey
        AND n_regionkey = r_regionkey
        AND r_name = 'EUROPE'
  )
ORDER BY s_acctbal DESC, n_name, s_name, p_partkey
LIMIT 100
""".strip(),
    target_ms=2000.0,
    level=3,
    description="Correlated subquery with MIN — convert to CTE for massive speedup",
)

task2 = Task(
    query="""
SELECT nation, o_year, SUM(amount) AS sum_profit
FROM (
    SELECT n_name AS nation,
           EXTRACT(YEAR FROM o_orderdate) AS o_year,
           l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
    FROM part, supplier, lineitem, partsupp, orders, nation
    WHERE s_suppkey = l_suppkey
      AND ps_suppkey = l_suppkey
      AND ps_partkey = l_partkey
      AND p_partkey = l_partkey
      AND o_orderkey = l_orderkey
      AND s_nationkey = n_nationkey
      AND p_name LIKE '%green%'
) AS profit
GROUP BY nation, o_year
ORDER BY nation, o_year DESC
""".strip(),
    target_ms=2000.0,
    level=3,
    description="Six-table join inside subquery — extract CTE, push p_name filter early",
)

task3 = Task(
    query="""
SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly
FROM lineitem, part
WHERE p_partkey = l_partkey
  AND p_brand = 'Brand#23'
  AND p_container = 'MED BOX'
  AND l_quantity < (
      SELECT 0.2 * AVG(l_quantity)
      FROM lineitem
      WHERE l_partkey = p_partkey
  )
""".strip(),
    target_ms=2000.0,
    level=3,
    description="Correlated subquery executed per row — rewrite as window function or CTE",
)

LEVEL3_TASKS = [task1, task2, task3]
