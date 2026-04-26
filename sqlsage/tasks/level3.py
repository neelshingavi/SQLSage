"""Level 3 TPC-H tasks: multi-table complex optimization."""

from __future__ import annotations

from .level1 import Task


LEVEL3_TASKS = [
    Task(
        description="TPC-H Q2 style minimum cost supplier query",
        level=3,
        target_ms=2000.0,
        query="""
SELECT
    s_acctbal,
    s_name,
    n_name,
    p_partkey,
    p_mfgr,
    s_address,
    s_phone,
    s_comment
FROM part
JOIN partsupp ON p_partkey = ps_partkey
JOIN supplier ON s_suppkey = ps_suppkey
JOIN nation ON s_nationkey = n_nationkey
JOIN region ON n_regionkey = r_regionkey
WHERE p_size = 15
  AND p_type LIKE '%BRASS'
  AND r_name = 'EUROPE'
  AND ps_supplycost = (
    SELECT MIN(ps_supplycost)
    FROM partsupp
    JOIN supplier ON s_suppkey = ps_suppkey
    JOIN nation ON s_nationkey = n_nationkey
    JOIN region ON n_regionkey = r_regionkey
    WHERE p_partkey = ps_partkey
      AND r_name = 'EUROPE'
  )
ORDER BY s_acctbal DESC, n_name, s_name, p_partkey
LIMIT 100;
""".strip(),
    ),
    Task(
        description="TPC-H Q7 style volume shipping query",
        level=3,
        target_ms=2000.0,
        query="""
SELECT
    supp_nation,
    cust_nation,
    l_year,
    SUM(volume) AS revenue
FROM (
    SELECT
        n1.n_name AS supp_nation,
        n2.n_name AS cust_nation,
        EXTRACT(YEAR FROM l_shipdate) AS l_year,
        l_extendedprice * (1 - l_discount) AS volume
    FROM supplier
    JOIN lineitem ON s_suppkey = l_suppkey
    JOIN orders ON o_orderkey = l_orderkey
    JOIN customer ON c_custkey = o_custkey
    JOIN nation n1 ON s_nationkey = n1.n_nationkey
    JOIN nation n2 ON c_nationkey = n2.n_nationkey
    WHERE ((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY')
       OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE'))
      AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
) shipping
GROUP BY supp_nation, cust_nation, l_year
ORDER BY supp_nation, cust_nation, l_year;
""".strip(),
    ),
    Task(
        description="TPC-H Q8 style market share query",
        level=3,
        target_ms=2000.0,
        query="""
SELECT
    o_year,
    SUM(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) / SUM(volume) AS mkt_share
FROM (
    SELECT
        EXTRACT(YEAR FROM o_orderdate) AS o_year,
        l_extendedprice * (1 - l_discount) AS volume,
        n2.n_name AS nation
    FROM part
    JOIN lineitem ON p_partkey = l_partkey
    JOIN supplier ON s_suppkey = l_suppkey
    JOIN orders ON o_orderkey = l_orderkey
    JOIN customer ON c_custkey = o_custkey
    JOIN nation n1 ON c_nationkey = n1.n_nationkey
    JOIN region ON n1.n_regionkey = r_regionkey
    JOIN nation n2 ON s_nationkey = n2.n_nationkey
    WHERE r_name = 'AMERICA'
      AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
      AND p_type = 'ECONOMY ANODIZED STEEL'
) all_nations
GROUP BY o_year
ORDER BY o_year;
""".strip(),
    ),
    Task(
        description="TPC-H Q9 style profit by nation and year",
        level=3,
        target_ms=2000.0,
        query="""
SELECT
    nation,
    o_year,
    SUM(amount) AS sum_profit
FROM (
    SELECT
        n_name AS nation,
        EXTRACT(YEAR FROM o_orderdate) AS o_year,
        l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
    FROM lineitem
    JOIN part ON p_partkey = l_partkey
    JOIN supplier ON s_suppkey = l_suppkey
    JOIN partsupp ON ps_partkey = l_partkey AND ps_suppkey = l_suppkey
    JOIN orders ON o_orderkey = l_orderkey
    JOIN nation ON s_nationkey = n_nationkey
    WHERE p_name LIKE '%green%'
) profit
GROUP BY nation, o_year
ORDER BY nation, o_year DESC;
""".strip(),
    ),
    Task(
        description="TPC-H Q17 style average yearly revenue",
        level=3,
        target_ms=2000.0,
        query="""
SELECT
    SUM(l_extendedprice) / 7.0 AS avg_yearly
FROM lineitem
JOIN part ON p_partkey = l_partkey
WHERE p_brand = 'Brand#23'
  AND p_container = 'MED BOX'
  AND l_quantity < (
      SELECT 0.2 * AVG(l_quantity)
      FROM lineitem
      WHERE l_partkey = p_partkey
  );
""".strip(),
    ),
    Task(
        description="TPC-H Q18 style large volume customer orders",
        level=3,
        target_ms=2000.0,
        query="""
SELECT
    c_name,
    c_custkey,
    o_orderkey,
    o_orderdate,
    o_totalprice,
    SUM(l_quantity) AS quantity
FROM customer
JOIN orders ON c_custkey = o_custkey
JOIN lineitem ON o_orderkey = l_orderkey
WHERE o_orderkey IN (
    SELECT l_orderkey
    FROM lineitem
    GROUP BY l_orderkey
    HAVING SUM(l_quantity) > 300
)
GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
ORDER BY o_totalprice DESC, o_orderdate
LIMIT 100;
""".strip(),
    ),
]
