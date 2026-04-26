"""Level 1 TPC-H task definitions."""

from sqlsage.tasks import Task

task1 = Task(
    query="""
SELECT
    l_returnflag,
    l_linestatus,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    AVG(l_quantity) AS avg_qty,
    AVG(l_extendedprice) AS avg_price,
    AVG(l_discount) AS avg_disc,
    COUNT(*) AS count_order
FROM lineitem
WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90 days'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus
""".strip(),
    target_ms=500.0,
    level=1,
    description="Full table scan on 6M row lineitem — needs index on l_shipdate",
)

task2 = Task(
    query="""
SELECT SUM(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= DATE '1994-01-01'
  AND l_shipdate < DATE '1994-01-01' + INTERVAL '1 year'
  AND l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01
  AND l_quantity < 24
""".strip(),
    target_ms=500.0,
    level=1,
    description="Seq scan with multiple filters on lineitem — filter pushdown",
)

task3 = Task(
    query="""
SELECT
    100.00 * SUM(
        CASE WHEN p_type LIKE 'PROMO%'
             THEN l_extendedprice * (1 - l_discount)
             ELSE 0
        END
    ) / SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue
FROM lineitem, part
WHERE l_partkey = p_partkey
  AND l_shipdate >= DATE '1995-09-01'
  AND l_shipdate < DATE '1995-09-01' + INTERVAL '1 month'
""".strip(),
    target_ms=500.0,
    level=1,
    description="Implicit cross join — needs explicit JOIN and index on l_shipdate",
)

LEVEL1_TASKS = [task1, task2, task3]
