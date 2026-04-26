#!/usr/bin/env python3
"""
Phase 14 end-to-end integration: pattern library, prompt builder, training verifier, DB smoke.

Run from repo root (sqlsage-env on PYTHONPATH), e.g.:
  pip install -e . && python test_phase14_integration.py

Target: completes in < 2 minutes; prints PASS/FAIL per test.
"""

from __future__ import annotations

import os
import re
import sys
import unittest.mock as mock
from typing import Any

# --- TEST 1 -----------------------------------------------------------------


def test_pattern_library() -> tuple[str, str]:
    from sqlsage.rewrite_patterns import (
        ALL_PATTERNS,
        P1_FILTER_PUSHDOWN,
        P2_NESTED_LOOP_ELIMINATION,
        detect_applicable_patterns,
    )

    if len(ALL_PATTERNS) != 7:
        return "FAIL", f"expected 7 patterns, got {len(ALL_PATTERNS)}"
    for p in ALL_PATTERNS:
        if p.before_sql.strip() == p.after_sql.strip():
            return "FAIL", f"{p.pattern_id}: before_sql == after_sql"
        fs = p.few_shot_example
        if "SLOW QUERY" not in fs or "FAST REWRITE" not in fs:
            return "FAIL", f"{p.pattern_id}: few_shot missing SLOW/FAST labels"
    plan = {"seq_scans": 2, "rows": 600_000, "nested_loops": 1}
    got = detect_applicable_patterns(plan)
    ids = {p.pattern_id for p in got}
    if P1_FILTER_PUSHDOWN.pattern_id not in ids or P2_NESTED_LOOP_ELIMINATION.pattern_id not in ids:
        return "FAIL", f"expected P1 and P2 in {ids}"
    return "PASS", "7 patterns, before≠after, few-shots, P1+P2 detected"


# --- TEST 2 -----------------------------------------------------------------


def test_prompt_builder() -> tuple[str, str]:
    from sqlsage.prompt_builder import build_baseline_prompt, build_optimized_prompt
    from sqlsage.rewrite_patterns import ALL_PATTERNS

    obs: dict[str, Any] = {
        "original_query": "SELECT 1 AS x",
        "schema_context": "TPC-H",
        "explain_plan": {
            "node_type": "Nested Loop",
            "seq_scans": 1,
            "rows": 800_000,
            "nested_loops": 1,
            "hash_joins": 0,
            "total_cost": 1_200_000.0,
            "highest_cost_node": {"Node Type": "Seq Scan", "Total Cost": 9e5},
        },
        "execution_ms": 12_000.0,
        "result_hash": "x",
    }
    p = build_optimized_prompt(obs, ALL_PATTERNS)
    if "OUTPUT FORMAT" not in p:
        return "FAIL", "optimized prompt missing OUTPUT FORMAT"
    if "PROVEN OPTIMIZATION PATTERNS" not in p:
        return "FAIL", "no few-shot section"
    with_matches = re.findall(
        r"(?i)\bWITH\s+[A-Za-z0-9_`\"\[\].]+\s+AS\s",
        p,
    )
    if not with_matches:
        return "FAIL", "no CTE-style WITH in optimized prompt (expected pattern few-shots)"
    if len(p) / 4.0 >= 1800.0:
        return "FAIL", f"prompt too long: {len(p)}/4 >= 1800"
    b = build_baseline_prompt(obs)
    if "PROVEN OPTIMIZATION" in b:
        return "FAIL", "baseline should not include proven-patterns block"
    if re.search(
        r"(?i)===\s*QUERY TO OPTIMIZE[\s\S]*\bWITH\s+[A-Za-z0-9_`\"\[\].]+\s+AS\s",
        b,
    ):
        return "FAIL", "baseline query/examples area should not have CTE WITH"
    return "PASS", "optimized has OUTPUT+WITH+few-shots, within token budget, baseline minimal"


# --- TEST 3 -----------------------------------------------------------------


def _synthetic_reward_series() -> list[float]:
    s = []
    for i in range(101):
        if i < 37:
            s.append(0.5)
        elif i == 37:
            s.append(1.2)
        else:
            s.append(1.2)
    return s


def test_training_verifier_inflection() -> tuple[str, str]:
    import sqlsage.training_verifier as tv

    ser = _synthetic_reward_series()
    assert len(ser) == 101

    class _FakeRun:
        def history(
            self,
            keys: list | None = None,
            samples: int | int = 50_000,
            pandas: bool = True,
            sort: str | None = "step",
        ) -> Any:
            import pandas as pd

            return pd.DataFrame(
                [
                    {"reward/mean": ser[i], "_step": i}
                    for i in range(len(ser))
                ]
            )

        def scan_history(self):
            for i, v in enumerate(ser):
                yield {"reward/mean": v, "_step": i}

    def _fake_wandb(path: str) -> tuple[Any, Any] | None:
        return (None, _FakeRun())

    with mock.patch.object(tv, "_wandb_api_run", _fake_wandb):
        r = tv.verify_reward_inflection("entity/project/phase14-mock")
    if r.get("pass") is not True:
        return "FAIL", f"expected pass, got {r!r}"
    ep = r.get("inflection_episode")
    if not isinstance(ep, int) or not (28 <= ep <= 35):
        return "FAIL", f"expected inflection_episode in [28,35], got {ep!r} ({r.get('diagnosis')})"
    return "PASS", f"pass=True, inflection_episode={ep}"


# --- TEST 4 -----------------------------------------------------------------


def test_few_shot_injection_delta() -> tuple[str, str]:
    from sqlsage.prompt_builder import build_baseline_prompt, build_optimized_prompt
    from sqlsage.rewrite_patterns import ALL_PATTERNS

    obs: dict[str, Any] = {
        "original_query": "SELECT 1",
        "schema_context": "schema",
        "explain_plan": {
            "node_type": "Seq Scan",
            "seq_scans": 1,
            "rows": 9e5,
            "nested_loops": 1,
            "hash_joins": 0,
            "total_cost": 50_000.0,
            "highest_cost_node": {"Node Type": "Seq Scan", "Total Cost": 4e4},
        },
        "execution_ms": 5_000.0,
        "result_hash": "h",
    }
    full = build_optimized_prompt(obs, ALL_PATTERNS)
    if "PROVEN OPTIMIZATION PATTERNS" not in full:
        return "FAIL", "injected prompt should include proven-patterns section"
    if "=== PROVEN OPTIMIZATION PATTERNS (apply if relevant) ===" not in full:
        return "FAIL", "injected prompt missing few-shot header"
    _h = "=== PROVEN OPTIMIZATION PATTERNS (apply if relevant) ===\n"
    after = full.split(_h, 1)[1]
    few_region = after.split("=== PREVIOUS ATTEMPTS", 1)[0]
    cte_w = len(
        re.findall(
            r"(?is)\bWITH\s+[A-Za-z0-9_`\"\[\].]+\s+AS\s",
            few_region,
        )
    )
    if cte_w < 1:
        return "FAIL", f"expected CTE WITH in few-shot block, got len={len(few_region)}"
    base = build_baseline_prompt(obs)
    if "PROVEN OPTIMIZATION PATTERNS" in base:
        return "FAIL", "baseline must not have PROVEN block"
    if len(full) < len(base) + 200:
        return "FAIL", f"injection delta {len(full) - len(base)} < 200 chars"
    return "PASS", f"+{len(full) - len(base)} chars, WITH hits in few-shot block"


# --- TEST 5 -----------------------------------------------------------------


def test_full_pipeline_db() -> tuple[str, str]:
    try:
        import psycopg2
    except ImportError as e:  # pragma: no cover
        return "SKIP", f"psycopg2: {e}"

    try:
        from sqlsage.explain_parser import get_explain_dict
        from sqlsage.prompt_builder import build_optimized_prompt
        from sqlsage.rewrite_patterns import ALL_PATTERNS, detect_applicable_patterns
        from sqlsage.tasks.level1 import LEVEL1_TASKS
    except Exception as e:  # noqa: BLE001
        return "FAIL", str(e)

    q1 = LEVEL1_TASKS[0].query
    try:
        conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "127.0.0.1"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "sqlsage"),
            dbname=os.environ.get("POSTGRES_DB", "sqlsage"),
            connect_timeout=5,
        )
    except Exception as e:  # noqa: BLE001
        return "SKIP", f"No DB ({e})"

    try:
        ex = get_explain_dict(conn, q1, timeout_ms=60_000)
    except Exception as e:  # noqa: BLE001
        err = str(e).lower()
        if "lineitem" in err or "does not exist" in err or "relation" in err:
            return "SKIP", f"No TPC-H tables ({e})"
        return "FAIL", str(e)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    pat = detect_applicable_patterns(ex)
    if not pat:
        return "FAIL", "expected ≥1 pattern for TPC-H Q1 shape plan on live DB"
    obs = {
        "original_query": q1,
        "schema_context": "TPC-H (live)",
        "explain_plan": ex,
        "execution_ms": 0.0,
        "result_hash": "",
    }
    prompt = build_optimized_prompt(obs, ALL_PATTERNS)
    required = (
        "=== DATABASE SCHEMA ===",
        "=== CURRENT QUERY PERFORMANCE ===",
        "=== QUERY TO OPTIMIZE ===",
        "=== YOUR TASK ===",
        "OUTPUT FORMAT",
    )
    miss = [s for s in required if s not in prompt]
    if miss:
        return "FAIL", f"prompt missing sections: {miss}"
    print("  FULL PIPELINE: PASS (DB + Q1 -> patterns -> valid prompt)")
    return "PASS", "Q1 EXPLAIN, patterns, prompt sections ok"


# --- main ------------------------------------------------------------------


def main() -> int:
    results: list[tuple[str, str, str]] = []
    for name, fn in [
        ("Pattern Library", test_pattern_library),
        ("Prompt Builder", test_prompt_builder),
        ("Training Verifier (mock wandb)", test_training_verifier_inflection),
        ("Few-Shot Injection Effect", test_few_shot_injection_delta),
        ("Full Pipeline (PostgreSQL)", test_full_pipeline_db),
    ]:
        try:
            status, msg = fn()
        except Exception as e:  # noqa: BLE001
            status, msg = "FAIL", f"{type(e).__name__}: {e}"
        mark = "PASS" if status == "PASS" else ("SKIP" if status == "SKIP" else "FAIL")
        results.append((name, mark, msg))
        print(f"  [{mark:4}] {name}: {msg}")

    print()
    print("  Phase 14 Integration Test Results:")
    pl, pb, tv, inj, full = (r[1] for r in results)
    print("  ✓ Pattern Library (7 patterns, detection working)" if pl == "PASS" else f"  ✗ Pattern Library ({pl})")
    print("  ✓ Prompt Builder (injection + length guard working)" if pb == "PASS" else f"  ✗ Prompt Builder ({pb})")
    print("  ✓ Training Verifier (inflection detection working)" if tv == "PASS" else f"  ✗ Training Verifier ({tv})")
    print("  ✓ Few-Shot Injection (measurable prompt difference)" if inj == "PASS" else f"  ✗ Few-Shot Injection ({inj})")
    if full == "PASS":
        print("  ✓ Full Pipeline (DB connected, end-to-end PASS)")
    elif full == "SKIP":
        print("  ○ Full Pipeline (SKIP: no TPC-H tables or DB — load data / set POSTGRES_*)")
    else:
        print(f"  ✗ Full Pipeline ({full})")
    print()
    if any(m == "FAIL" for _, m, _ in results):
        print("  Phase 14: INTEGRATION ISSUES — fix FAIL rows above", file=sys.stderr)
        return 1
    print("  Phase 14: READY FOR TRAINING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
