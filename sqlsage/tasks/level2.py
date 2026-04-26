"""Level 2 TPC-H task definitions."""

from sqlsage.tasks import Task

task1 = Task(
    query="""
SELECT
    l_orderkey,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue,
    o_orderdate,
    o_shippriority
FROM customer, orders, lineitem
WHERE c_mktsegment = 'BUILDING'
  AND c_custkey = o_custkey
  AND l_orderkey = o_orderkey
  AND o_orderdate < DATE '1995-03-15'
  AND l_shipdate > DATE '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY revenue DESC, o_orderdate
LIMIT 10
""".strip(),
    target_ms=1000.0,
    level=2,
    description="Three-table implicit join with nested loop — needs hash join + filter pushdown",
)

task2 = Task(
    query="""
SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM customer, orders, lineitem, supplier, nation, region
WHERE c_custkey = o_custkey
  AND l_orderkey = o_orderkey
  AND l_suppkey = s_suppkey
  AND c_nationkey = s_nationkey
  AND s_nationkey = n_nationkey
  AND n_regionkey = r_regionkey
  AND r_name = 'ASIA'
  AND o_orderdate >= DATE '1994-01-01'
  AND o_orderdate < DATE '1994-01-01' + INTERVAL '1 year'
GROUP BY n_name
ORDER BY revenue DESC
""".strip(),
    target_ms=1000.0,
    level=2,
    description="Six-table join — nested loop catastrophic at this cardinality, needs hash joins",
)

task3 = Task(
    query="""
SELECT
    c_custkey, c_name,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue,
    c_acctbal, n_name, c_address, c_phone, c_comment
FROM customer, orders, lineitem, nation
WHERE c_custkey = o_custkey
  AND l_orderkey = o_orderkey
  AND o_orderdate >= DATE '1993-10-01'
  AND o_orderdate < DATE '1993-10-01' + INTERVAL '3 months'
  AND l_returnflag = 'R'
  AND c_nationkey = n_nationkey
GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
ORDER BY revenue DESC
LIMIT 20
""".strip(),
    target_ms=1000.0,
    level=2,
    description="Four-table join with date range and returnflag filter — CTE pushdown opportunity",
)

task4 = Task(
    query="""
SELECT
    l_shipmode,
    SUM(CASE WHEN o_orderpriority = '1-URGENT'
             OR o_orderpriority = '2-HIGH'
             THEN 1 ELSE 0 END) AS high_line_count,
    SUM(CASE WHEN o_orderpriority <> '1-URGENT'
             AND o_orderpriority <> '2-HIGH'
             THEN 1 ELSE 0 END) AS low_line_count
FROM orders, lineitem
WHERE o_orderkey = l_orderkey
  AND l_shipmode IN ('MAIL', 'SHIP')
  AND l_commitdate < l_receiptdate
  AND l_shipdate < l_commitdate
  AND l_receiptdate >= DATE '1994-01-01'
  AND l_receiptdate < DATE '1994-01-01' + INTERVAL '1 year'
GROUP BY l_shipmode
ORDER BY l_shipmode
""".strip(),
    target_ms=1000.0,
    level=2,
    description="Two-table join with IN filter on shipmode — index on receiptdate needed",
)

LEVEL2_TASKS = [task1, task2, task3, task4]
