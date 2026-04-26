"""TPC-H schema and curriculum metadata used by SQLSage."""

from __future__ import annotations

SCHEMA_OVERVIEW_SF1 = {
    "lineitem": {"rows": 6_001_215, "description": "Largest table - order line items"},
    "orders": {"rows": 1_500_000, "description": "Order records - status, date, customer"},
    "customer": {"rows": 150_000, "description": "Customer records - region, balance"},
    "part": {"rows": 200_000, "description": "Part catalog"},
    "supplier": {"rows": 10_000, "description": "Supplier records"},
    "partsupp": {"rows": 800_000, "description": "Part-supplier relationships"},
    "nation": {"rows": 25, "description": "Country reference"},
    "region": {"rows": 5, "description": "Region reference"},
}

CURRICULUM = {
    1: {
        "queries": ["Q1", "Q6", "Q14"],
        "target": "Single table Seq Scan -> Index Scan, simple filter pushdown",
        "hours": "0-8",
    },
    2: {
        "queries": ["Q3", "Q5", "Q10", "Q12"],
        "target": "Two-table nested loop -> hash join, subquery -> CTE",
        "hours": "8-16",
    },
    3: {
        "queries": ["Q2", "Q7", "Q8", "Q9", "Q17", "Q18"],
        "target": "Multi-join, aggregation, correlated subqueries",
        "hours": "16-24",
    },
}
