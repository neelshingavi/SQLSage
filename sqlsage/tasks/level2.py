"""Level 2 TPC-H tasks: two-table join optimization."""

from __future__ import annotations

from .level1 import Task


LEVEL2_TASKS = [
    Task(
        description="TPC-H Q3 style customer-orders-lineitem join",
        level=2,
        target_ms=1000.0,
        query="""
SELECT
    l_orderkey,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue,
    o_orderdate,
    o_shippriority
FROM customer
JOIN orders ON c_custkey = o_custkey
JOIN lineitem ON l_orderkey = o_orderkey
WHERE c_mktsegment = 'BUILDING'
  AND o_orderdate < DATE '1995-03-15'
  AND l_shipdate > DATE '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY revenue DESC, o_orderdate
LIMIT 10;
""".strip(),
    ),
    Task(
        description="TPC-H Q5 style region revenue join chain",
        level=2,
        target_ms=1000.0,
        query="""
SELECT
    n_name,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM customer
JOIN orders ON c_custkey = o_custkey
JOIN lineitem ON l_orderkey = o_orderkey
JOIN supplier ON l_suppkey = s_suppkey
JOIN nation ON s_nationkey = n_nationkey AND c_nationkey = s_nationkey
JOIN region ON n_regionkey = r_regionkey
WHERE r_name = 'ASIA'
  AND o_orderdate >= DATE '1994-01-01'
  AND o_orderdate < DATE '1995-01-01'
GROUP BY n_name
ORDER BY revenue DESC;
""".strip(),
    ),
    Task(
        description="TPC-H Q10 style returned item revenue",
        level=2,
        target_ms=1000.0,
        query="""
SELECT
    c_custkey,
    c_name,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue,
    c_acctbal,
    n_name,
    c_address,
    c_phone,
    c_comment
FROM customer
JOIN orders ON c_custkey = o_custkey
JOIN lineitem ON l_orderkey = o_orderkey
JOIN nation ON c_nationkey = n_nationkey
WHERE o_orderdate >= DATE '1993-10-01'
  AND o_orderdate < DATE '1994-01-01'
  AND l_returnflag = 'R'
GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
ORDER BY revenue DESC
LIMIT 20;
""".strip(),
    ),
    Task(
        description="TPC-H Q12 style shipping mode analysis",
        level=2,
        target_ms=1000.0,
        query="""
SELECT
    l_shipmode,
    SUM(CASE WHEN o_orderpriority IN ('1-URGENT', '2-HIGH') THEN 1 ELSE 0 END) AS high_line_count,
    SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT', '2-HIGH') THEN 1 ELSE 0 END) AS low_line_count
FROM orders
JOIN lineitem ON o_orderkey = l_orderkey
WHERE l_shipmode IN ('MAIL', 'SHIP')
  AND l_commitdate < l_receiptdate
  AND l_shipdate < l_commitdate
  AND l_receiptdate >= DATE '1994-01-01'
  AND l_receiptdate < DATE '1995-01-01'
GROUP BY l_shipmode
ORDER BY l_shipmode;
""".strip(),
    ),
]
