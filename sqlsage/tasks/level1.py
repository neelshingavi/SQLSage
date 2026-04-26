"""Level 1 TPC-H tasks: single-table optimization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    query: str
    target_ms: float
    level: int
    description: str


LEVEL1_TASKS = [
    Task(
        description="TPC-H Q1 style aggregation with date filter",
        level=1,
        target_ms=500.0,
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
WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90 day'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;
""".strip(),
    ),
    Task(
        description="TPC-H Q6 style revenue query",
        level=1,
        target_ms=500.0,
        query="""
SELECT
    SUM(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= DATE '1994-01-01'
  AND l_shipdate < DATE '1995-01-01'
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24;
""".strip(),
    ),
    Task(
        description="TPC-H Q14 style promotional revenue",
        level=1,
        target_ms=500.0,
        query="""
SELECT
    100.00 * SUM(
        CASE
            WHEN p_type LIKE 'PROMO%%' THEN l_extendedprice * (1 - l_discount)
            ELSE 0
        END
    ) / SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue
FROM lineitem
JOIN part ON l_partkey = p_partkey
WHERE l_shipdate >= DATE '1995-09-01'
  AND l_shipdate < DATE '1995-10-01';
""".strip(),
    ),
]
