"""
Master rewrite pattern library for SQLSage RL: canonical before/after SQL, signals, few-shot text.

All patterns are instances of :class:`RewritePattern`. Detection expects either:

- The normalized plan dict from :func:`sqlsage.explain_parser.extract_key_fields` /
  :func:`sqlsage.explain_parser.get_explain_dict` (``seq_scans``, ``rows``,
  ``nested_loops``, ``node_type``, ``children``, etc.), or
- A raw PostgreSQL EXPLAIN root ``{"Plan": {...}}``; it is normalized automatically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

# --- data structure ---------------------------------------------------------------------------


@dataclass
class RewritePattern:
    """
    One rewrite strategy: TPC-H-oriented problem, signals, and canonical SQL pair.

    ``signal_in_plan`` uses a small DSL (all keys **AND**'d; omitting a key skips it):

    - ``"seq_scans"``, ``"rows"``, ``"total_cost"``, ``"nested_loops"`` …:
      ``{"ge": 1}`` / ``{"le": n}`` / ``{"gt"`` / ``{"lt"`` / ``{"eq"``
    - ``"node_type"``: ``{"eq": "Limit"}`` or ``{"in": ("Nested Loop", "Hash Join")}``
    - ``"child0_rows"`` / ``"child0_node_type"``: first entry in ``children[]``
    - ``"filter_contains_or"``: ``True`` — see :func:`pattern_signal_matches`
    - ``"p5_outer_child_rows"``: ``{"ge": 1_000_000}`` — first child row estimate (P5)
    - ``"p6_limit_with_big_sort"``: ``True`` — Limit over Sort with large child
    """

    pattern_id: str
    name: str
    problem_description: str
    signal_in_plan: dict[str, Any]
    before_sql: str
    after_sql: str
    expected_plan_change: dict[str, Any]
    tpch_queries: list[str] = field(default_factory=list)
    target_speedup_ratio: float = 0.0
    few_shot_example: str = ""


# --- plan normalization -----------------------------------------------------------------------


def _unwrap_explain_json(explain_dict: dict) -> dict:
    """Get PostgreSQL top ``Plan`` node from various EXPLAIN JSON shapes."""
    if not explain_dict or not isinstance(explain_dict, dict):
        return {}
    if "seq_scans" in explain_dict and "hash_joins" in explain_dict:
        return explain_dict
    if "Plan" in explain_dict and isinstance(explain_dict["Plan"], dict):
        return explain_dict["Plan"]
    p = explain_dict.get("Plan", explain_dict)
    if isinstance(p, list) and p:
        p0 = p[0]
        if isinstance(p0, dict) and "Plan" in p0:
            return p0.get("Plan", {})
    if isinstance(p, dict) and "Plan" in p:
        return p.get("Plan", {})
    return p if isinstance(p, dict) else {}


def normalize_explain_dict(explain_dict: dict) -> dict:
    """
    Return the normalized key-field dict used in SQLSage (``extract_key_fields``).

    If the input is already from :func:`sqlsage.explain_parser.extract_key_fields`,
    it is returned unchanged. Otherwise a raw plan tree is unwrapped and normalized.
    """
    if not explain_dict or not isinstance(explain_dict, dict):
        from sqlsage.explain_parser import extract_key_fields

        return extract_key_fields({})
    if "seq_scans" in explain_dict and "hash_joins" in explain_dict:
        return explain_dict
    from sqlsage.explain_parser import extract_key_fields

    root = _unwrap_explain_json(explain_dict)
    return extract_key_fields(root if isinstance(root, dict) else {})


# --- signal matching --------------------------------------------------------------------------


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _as_int(x: Any) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def _node_type_matches(d: dict[str, Any], spec: dict[str, Any]) -> bool:
    if not d:
        return False
    ntpg = d.get("node_type", "")
    ntraw = d.get("Node Type", "")
    s = f"{ntpg} {ntraw}"
    s_low = s.lower()
    if "eq" in spec:
        t = str(spec["eq"])
        return t.lower() in s_low or t == ntraw
    if "in" in spec and isinstance(spec["in"], (list, tuple)):
        return any(str(v) and str(v).lower() in s_low for v in spec["in"])
    return True


def _cmp(v: float, op: str, ref: float) -> bool:
    if op == "ge":
        return v >= ref
    if op == "le":
        return v <= ref
    if op == "gt":
        return v > ref
    if op == "lt":
        return v < ref
    if op == "eq":
        return v == ref
    return False


def _or_in_plan_json(d: Any) -> bool:
    """
    Heuristic: ``\" OR \"`` in serialized plan. Set ``query_has_or: True`` on
    the explain dict to force a match.
    """
    if isinstance(d, dict) and d.get("query_has_or") is True:
        return True
    blob = json.dumps(d, default=str).lower()
    if " or " not in blob:
        return False
    return "filter" in blob


def _p6_limit_big_sort(p: dict[str, Any]) -> bool:
    """True if top node is Limit, first child is Sort, child rows >= 100k."""
    n = str(p.get("node_type", "")).lower()
    if "limit" not in n:
        return False
    ch = p.get("children", [])
    if not isinstance(ch, list) or not ch or not isinstance(ch[0], dict):
        return False
    c0 = ch[0]
    c0n = str(c0.get("node_type", "")).lower()
    if "sort" not in c0n:
        return False
    return _as_int(c0.get("rows", 0)) >= 100_000


def pattern_signal_matches(plan: dict[str, Any], signal: dict[str, Any]) -> bool:
    """
    Return whether normalized ``plan`` matches ``signal`` (conjunction of predicates).

    ``filter_contains_or``: pass if ``query_has_or`` is set, or the JSON heuristics
    see `` OR `` in filter text; otherwise this predicate fails.
    """
    if not signal:
        return True
    p: dict[str, Any] = plan if (isinstance(plan, dict) and "seq_scans" in plan) else normalize_explain_dict(plan)

    if signal.get("p6_limit_with_big_sort") is True:
        return _p6_limit_big_sort(p)

    for key, v in signal.items():
        if key in ("p6_limit_with_big_sort",):
            continue
        if key == "p5_outer_child_rows" and isinstance(v, dict):
            ch = p.get("children", [])
            if not isinstance(ch, list) or not ch or not isinstance(ch[0], dict):
                return False
            r = _as_float(ch[0].get("rows", 0))
            if "ge" in v and not _cmp(r, "ge", _as_float(v["ge"])):
                return False
            if "le" in v and not _cmp(r, "le", _as_float(v["le"])):
                return False
            continue
        if key == "node_type" and isinstance(v, dict) and p:
            if not _node_type_matches(p, v):
                return False
            continue
        if key == "child0_node_type" and isinstance(v, dict):
            ch = p.get("children", [])
            c0 = ch[0] if isinstance(ch, list) and ch and isinstance(ch[0], dict) else {}
            if not _node_type_matches(c0, v):
                return False
            continue
        if key == "child0_rows" and isinstance(v, dict):
            ch = p.get("children", [])
            c0 = ch[0] if isinstance(ch, list) and ch and isinstance(ch[0], dict) else {}
            r = _as_float(c0.get("rows", 0))
            if "ge" in v and not _cmp(r, "ge", _as_float(v["ge"])):
                return False
            if "le" in v and not _cmp(r, "le", _as_float(v["le"])):
                return False
            continue
        if key == "filter_contains_or" and v is True:
            if not _or_in_plan_json(p):
                return False
            continue
        if not isinstance(v, dict):
            continue
        if not any(x in v for x in ("ge", "le", "gt", "lt", "eq")):
            continue
        if key in (
            "seq_scans",
            "rows",
            "total_cost",
            "nested_loops",
            "index_scans",
            "hash_joins",
            "actual_time_ms",
        ):
            val = _as_float(p.get(key, 0))
        else:
            continue
        for opn in ("ge", "le", "gt", "lt", "eq"):
            if opn in v and not _cmp(val, opn, _as_float(v[opn])):
                return False
    return True


# --- the seven patterns -----------------------------------------------------------------------

P1_FILTER_PUSHDOWN = RewritePattern(
    pattern_id="P1_FILTER_PUSHDOWN",
    name="Filter pushdown (join before filter)",
    problem_description=(
        "A large table is fully joined with dimension rows before a selective WHERE "
        "reduces the row set — the optimizer may scan and join more than needed."
    ),
    signal_in_plan={
        "seq_scans": {"ge": 1},
        "rows": {"ge": 500_000},
        "nested_loops": {"ge": 1},
    },
    before_sql="""\
SELECT o.orderkey, c.name, o.totalprice
FROM orders o
JOIN customer c ON o.custkey = c.custkey
WHERE o.orderstatus = 'F'
  AND o.orderdate BETWEEN DATE '1993-01-01' AND DATE '1993-12-31'
""",
    after_sql="""\
WITH f_orders AS (
  SELECT *
  FROM orders
  WHERE orderstatus = 'F'
    AND orderdate BETWEEN DATE '1993-01-01' AND DATE '1993-12-31'
)
SELECT fo.orderkey, c.name, fo.totalprice
FROM f_orders fo
JOIN customer c ON fo.custkey = c.custkey
""",
    expected_plan_change={
        "seq_scans": "at least one fewer on large fact path after rewrite (compare plans)",
        "rows": "intermediate join rows well below unfiltered fact cardinality",
    },
    tpch_queries=["Q3", "Q5", "Q10"],
    target_speedup_ratio=0.60,
    few_shot_example="",
)

P2_NESTED_LOOP_ELIMINATION = RewritePattern(
    pattern_id="P2_NESTED_LOOP_ELIMINATION",
    name="Nested-loop elimination (pre-aggregate, hash join friendly)",
    problem_description=(
        "Default nested loop join at high row counts is O(n×m); pre-aggregating the "
        "lineitem side or using hash join reduces work."
    ),
    signal_in_plan={"nested_loops": {"ge": 1}, "rows": {"ge": 100_000}},
    before_sql="""\
SELECT o.orderkey, SUM(l.extendedprice * (1 - l.discount)) AS revenue
FROM lineitem l
JOIN orders o ON l.orderkey = o.orderkey
WHERE o.orderdate < DATE '1994-12-25'
GROUP BY o.orderkey
""",
    after_sql="""\
WITH l_agg AS (
  SELECT orderkey, SUM(extendedprice * (1 - discount)) AS revenue
  FROM lineitem
  GROUP BY orderkey
)
SELECT o.orderkey, l_agg.revenue
FROM orders o
JOIN l_agg ON o.orderkey = l_agg.orderkey
WHERE o.orderdate < DATE '1994-12-25'
""",
    expected_plan_change={
        "nested_loops": 0,
        "hash_joins": {"ge": 1},
    },
    tpch_queries=["Q3", "Q5", "Q12"],
    target_speedup_ratio=0.70,
    few_shot_example="",
)

P3_SUBQUERY_TO_CTE = RewritePattern(
    pattern_id="P3_SUBQUERY_TO_CTE",
    name="Correlated subquery to CTE (scalar precompute)",
    problem_description=(
        "A correlated subquery re-executes for every outer row; materializing the "
        "aggregate once in a CTE can be cheaper when decorrelation is possible."
    ),
    signal_in_plan={"nested_loops": {"ge": 1}, "total_cost": {"ge": 50_000}},
    before_sql="""\
SELECT p.partkey, p.retailprice
FROM part p
WHERE p.retailprice > (
  SELECT 0.2 * AVG(ps.supplycost) FROM partsupp ps WHERE ps.partkey = p.partkey
)
""",
    after_sql="""\
WITH part_avg AS (
  SELECT partkey, 0.2 * AVG(supplycost) AS thr
  FROM partsupp
  GROUP BY partkey
)
SELECT p.partkey, p.retailprice
FROM part p
JOIN part_avg pa ON p.partkey = pa.partkey
WHERE p.retailprice > pa.thr
""",
    expected_plan_change={"total_cost": "material drop in top node vs correlated plan"},
    tpch_queries=["Q2", "Q17"],
    target_speedup_ratio=0.80,
    few_shot_example="",
)

P4_EARLY_AGGREGATION = RewritePattern(
    pattern_id="P4_EARLY_AGGREGATION",
    name="Early aggregation before join",
    problem_description=(
        "Joining full lineitem+orders then GROUP BY; aggregating lineitem first "
        "shrinks the join."
    ),
    signal_in_plan={"seq_scans": {"ge": 2}, "rows": {"ge": 1_000_000}},
    before_sql="""\
SELECT o.orderkey, l.partkey, SUM(l.extendedprice * (1 - l.discount)) AS s
FROM lineitem l
JOIN orders o ON l.orderkey = o.orderkey
GROUP BY o.orderkey, l.partkey
""",
    after_sql="""\
WITH lsum AS (
  SELECT orderkey, partkey, SUM(extendedprice * (1 - discount)) AS s
  FROM lineitem
  GROUP BY orderkey, partkey
)
SELECT o.orderkey, lsum.partkey, lsum.s
FROM orders o
JOIN lsum ON o.orderkey = lsum.orderkey
""",
    expected_plan_change={"rows_before_top_join": "large reduction vs pre-rewrite"},
    tpch_queries=["Q1", "Q10", "Q12"],
    target_speedup_ratio=0.65,
    few_shot_example="",
)

P5_JOIN_REORDER = RewritePattern(
    pattern_id="P5_JOIN_REORDER",
    name="Join order (selective / smaller build first)",
    problem_description=(
        "When the first child of a nest-loop has a huge row estimate, rewrite so a "
        "selective or smaller side drives the join (or a hash join appears)."
    ),
    signal_in_plan={"nested_loops": {"ge": 1}, "p5_outer_child_rows": {"ge": 1_000_000}},
    before_sql="""\
SELECT o.orderkey, c.name
FROM lineitem l
JOIN orders o ON l.orderkey = o.orderkey
JOIN customer c ON o.custkey = c.custkey
""",
    after_sql="""\
WITH o_small AS (SELECT * FROM orders WHERE orderkey < 2000000)
SELECT o.orderkey, c.name
FROM o_small o
JOIN lineitem l ON o.orderkey = l.orderkey
JOIN customer c ON o.custkey = c.custkey
""",
    expected_plan_change={"outer_child_rows": "reduced at driving join"},
    tpch_queries=["Q7", "Q8", "Q9"],
    target_speedup_ratio=0.40,
    few_shot_example="",
)

P6_LIMIT_PUSHDOWN = RewritePattern(
    pattern_id="P6_LIMIT_PUSHDOWN",
    name="Limit with large sort (reduce sort input)",
    problem_description=(
        "LIMIT is applied after sorting a large set; pre-filtering or a bounded "
        "subquery shrinks the Sort input."
    ),
    signal_in_plan={"p6_limit_with_big_sort": True},
    before_sql="""\
SELECT l.orderkey, l.extendedprice
FROM lineitem l
ORDER BY l.shipdate DESC, l.extendedprice DESC
LIMIT 100
""",
    after_sql="""\
WITH recent AS (
  SELECT orderkey, extendedprice, shipdate
  FROM lineitem
  WHERE shipdate >= DATE '1993-01-01'
  ORDER BY shipdate DESC, extendedprice DESC
  LIMIT 5000
)
SELECT orderkey, extendedprice
FROM recent
ORDER BY shipdate DESC, extendedprice DESC
LIMIT 100
""",
    expected_plan_change={"sort_input_rows": "closer to LIMIT scale vs full table"},
    tpch_queries=["Q18"],
    target_speedup_ratio=0.85,
    few_shot_example="",
)

P7_PREDICATE_ISOLATION = RewritePattern(
    pattern_id="P7_PREDICATE_ISOLATION",
    name="OR split (UNION ALL branches)",
    problem_description=(
        "OR in WHERE on the same table can block good index use; two selective paths "
        "UNION ALL can beat one wide scan."
    ),
    signal_in_plan={"seq_scans": {"ge": 1}, "filter_contains_or": True},
    before_sql="""\
SELECT o.orderkey, o.totalprice
FROM orders o
WHERE o.orderstatus = 'F' OR o.orderstatus = 'O'
""",
    after_sql="""\
SELECT o.orderkey, o.totalprice FROM orders o WHERE o.orderstatus = 'F'
UNION ALL
SELECT o.orderkey, o.totalprice FROM orders o WHERE o.orderstatus = 'O'
""",
    expected_plan_change={"index_scans": {"ge": 2}, "seq_scans": {"le": 0}},
    tpch_queries=["Q9", "Q14"],
    target_speedup_ratio=0.55,
    few_shot_example="",
)


def format_few_shot_for_prompt(patterns: list[RewritePattern], max_patterns: int = 2) -> str:
    """
    Format up to ``max_patterns`` patterns as text blocks for model prompts.

    Each block: pattern name, problem, SLOW / FAST SQL, and a short *why* line including
    ``target_speedup_ratio`` (interpreted as minimum expected (t_old - t_new) / t_old
    for comparable workloads where measurable).
    """
    if not patterns:
        return ""
    out: list[str] = []
    for pat in patterns[: max(0, int(max_patterns))]:
        block = f"""--- EXAMPLE OPTIMIZATION (Pattern: {pat.name}) ---
PROBLEM: {pat.problem_description}

SLOW QUERY:
{pat.before_sql.strip()}

FAST REWRITE:
{pat.after_sql.strip()}

WHY THIS IS FASTER: Expected plan shift: {pat.expected_plan_change!r}. Target minimum relative time reduction ≈ {pat.target_speedup_ratio:.0%} (same result set).
---"""
        out.append(block)
    return "\n\n".join(out)


def get_pattern_by_id(pattern_id: str) -> RewritePattern:
    """
    Return the :class:`RewritePattern` for ``pattern_id`` (e.g. ``"P1_FILTER_PUSHDOWN"``).

    Raises:
        KeyError: if the id is unknown.
    """
    return _PATTERN_BY_ID[pattern_id]


def get_patterns_for_query(tpch_query_id: str) -> list[RewritePattern]:
    """
    Return all patterns that list ``tpch_query_id`` in :attr:`RewritePattern.tpch_queries`.

    ``tpch_query_id`` is case-insensitive (``"q5"`` and ``"Q5"`` both work).
    """
    q = tpch_query_id.strip().upper()
    if not q.startswith("Q"):
        q = "Q" + q.lstrip("Qq")
    return [p for p in ALL_PATTERNS if any(x.upper() == q for x in p.tpch_queries)]


def detect_applicable_patterns(explain_dict: dict) -> list[RewritePattern]:
    """
    Return patterns whose :attr:`RewritePattern.signal_in_plan` matches the EXPLAIN output.

    ``explain_dict`` is normalized via :func:`normalize_explain_dict` inside matchers.
    Results are sorted by :attr:`RewritePattern.target_speedup_ratio` **descending**
    (stronger expected gains first) for few-shot ordering.

    Used by the prompt builder to inject only relevant patterns.
    """
    matched: list[RewritePattern] = []
    for p in ALL_PATTERNS:
        if pattern_signal_matches(explain_dict, p.signal_in_plan):
            matched.append(p)
    return sorted(
        matched,
        key=lambda x: x.target_speedup_ratio,
        reverse=True,
    )


_RAW: tuple[RewritePattern, ...] = (
    P1_FILTER_PUSHDOWN,
    P2_NESTED_LOOP_ELIMINATION,
    P3_SUBQUERY_TO_CTE,
    P4_EARLY_AGGREGATION,
    P5_JOIN_REORDER,
    P6_LIMIT_PUSHDOWN,
    P7_PREDICATE_ISOLATION,
)
ALL_PATTERNS: list[RewritePattern] = [
    replace(p, few_shot_example=format_few_shot_for_prompt([p], max_patterns=1))
    for p in _RAW
]
P1_FILTER_PUSHDOWN = ALL_PATTERNS[0]
P2_NESTED_LOOP_ELIMINATION = ALL_PATTERNS[1]
P3_SUBQUERY_TO_CTE = ALL_PATTERNS[2]
P4_EARLY_AGGREGATION = ALL_PATTERNS[3]
P5_JOIN_REORDER = ALL_PATTERNS[4]
P6_LIMIT_PUSHDOWN = ALL_PATTERNS[5]
P7_PREDICATE_ISOLATION = ALL_PATTERNS[6]
_PATTERN_BY_ID: dict[str, RewritePattern] = {p.pattern_id: p for p in ALL_PATTERNS}
