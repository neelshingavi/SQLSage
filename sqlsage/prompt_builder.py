"""
Model prompt construction: inject only the few-shot rewrite patterns that match the plan.

Pair with :func:`truncate_prompt_if_needed` so long schemas or histories stay within budget.
"""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any, TypeVar, Union, cast

from sqlsage.env import Observation
from sqlsage.rewrite_patterns import (
    RewritePattern,
    detect_applicable_patterns,
    format_few_shot_for_prompt,
)

# --- observation normalization -------------------------------------------------


T = TypeVar("T", bound=Any)


def _as_dict(obs: Union[Observation, dict[str, Any]]) -> dict[str, Any]:
    if isinstance(obs, dict):
        return obs
    if is_dataclass(obs):
        return asdict(obs)
    if hasattr(obs, "to_dict"):
        return cast(dict[str, Any], obs.to_dict())  # type: ignore[no-any-return]
    raise TypeError(f"expected Observation or dict, got {type(obs)!r}")


def _get(d: dict[str, Any], key: str, default: T | None = None) -> Any:
    return d.get(key, default)


def _plan(obs_dict: dict[str, Any]) -> dict[str, Any]:
    p = _get(obs_dict, "explain_plan", {})
    return p if isinstance(p, dict) else {}


def _highest_node_display(plan: dict[str, Any]) -> str:
    h = plan.get("highest_cost_node")
    if isinstance(h, dict):
        nt = h.get("Node Type", h.get("node_type", "unknown"))
        try:
            tc = float(h.get("Total Cost", h.get("total_cost", 0.0) or 0.0))
        except (TypeError, ValueError):
            tc = 0.0
        return f"{nt} (cost={tc:.0f})"
    return str(h) if h is not None else "unknown"


# --- few-shot and history ------------------------------------------------------


def format_few_shot_section(patterns: list[RewritePattern] | list[Any]) -> str:
    """
    If ``patterns`` is empty, return ``""``.

    Otherwise return the proven-patterns block wrapping
    :func:`sqlsage.rewrite_patterns.format_few_shot_for_prompt`.
    """
    if not patterns:
        return ""
    typed = [p for p in patterns if isinstance(p, RewritePattern)]
    if not typed:
        return ""
    body = format_few_shot_for_prompt(typed, max_patterns=min(2, len(typed)))
    if not body.strip():
        return ""
    return (
        "=== PROVEN OPTIMIZATION PATTERNS (apply if relevant) ===\n"
        f"{body}\n"
        "===\n"
    )


def format_previous_attempts(
    obs: Union[Observation, dict[str, Any]],
    *,
    max_listed: int | None = None,
) -> str:
    """
    If there are no previous rewrites, return **"No previous attempts."**

    Otherwise list numbered attempts with optional rewards from
    ``previous_rewards`` (aligned by index). Per-step execution times are
    not stored on :class:`sqlsage.env.Observation`; reward lines are always shown when present.
    """
    d = _as_dict(obs)
    prev_w = [str(x) for x in (d.get("previous_rewrites") or [])]
    rews: list[Any] = list(d.get("previous_rewards") or [])
    if not prev_w:
        return "No previous attempts."
    if max_listed is not None and max_listed > 0:
        k = int(max_listed)
        prev_w = prev_w[-k:]
        if rews:
            rews = rews[-len(prev_w) :]
    lines: list[str] = []
    for i, q in enumerate(prev_w, start=1):
        rtxt = "no per-step reward logged"
        if i - 1 < len(rews):
            r = rews[i - 1]
            try:
                rv = float(r)
            except (TypeError, ValueError):
                rtxt = str(r)
            else:
                rtxt = f"{rv:+.2f} reward" if rv == rv else "n/a"
                if rv < 0:
                    rtxt = f"{rtxt} (penalty)"
        lines.append(f"Attempt {i} (result: {rtxt}):\n{q}\n")
    return "\n".join(lines).rstrip() + "\n"


# --- truncation ----------------------------------------------------------------

_MARK_FS = re.compile(
    r"=== PROVEN OPTIMIZATION PATTERNS \(apply if relevant\) ===\n[\s\S]*?\n===\n",
    re.MULTILINE,
)
_MARK_PREV = "=== PREVIOUS ATTEMPTS THIS EPISODE ===\n"
_MARK_YOUR = "\n=== YOUR TASK ===\n"


def truncate_prompt_if_needed(
    prompt: str,
    max_tokens: int = 1800,
    *,
    schema_max_chars: int = 300,
) -> str:
    """
    Rough size bound: ``max_tokens * 4`` characters (≈1 token / 4 chars).

    1. Remove the few-shot (proven patterns) block.
    2. Collapse the previous-attempts body to a short note, or keep only the last
       two ``Attempt n`` stanzas when the section is long.
    3. Truncate the schema block to ``schema_max_chars``.

    The **OUTPUT FORMAT** / JSON tail is not intentionally removed; a final hard
    trim only runs if the body is still too long after the steps above.
    """
    max_chars = int(max(1, max_tokens)) * 4
    s = prompt
    if len(s) <= max_chars:
        return s

    s = _MARK_FS.sub("", s, count=1)
    if len(s) <= max_chars:
        return s

    if _MARK_PREV in s and _MARK_YOUR in s:
        pre, rest = s.split(_MARK_PREV, 1)
        mid, post = rest.split(_MARK_YOUR, 1)
        attempts = [m.group(0) for m in re.finditer(r"(?ms)^Attempt \d+ .*?(?=^Attempt \d+ |\Z)", mid)]
        if len(attempts) > 2:
            mid = "\n".join(attempts[-2:]) + "\n"
        elif len(mid) > max_chars // 2:
            mid = mid[: max_chars // 4] + "\n[previous attempts truncated…]\n"
        s = pre + _MARK_PREV + mid + _MARK_YOUR + post
    if len(s) <= max_chars:
        return s

    sm = re.search(
        r"(=== DATABASE SCHEMA ===\n)([\s\S]*?)(\n=== CURRENT QUERY PERFORMANCE ===\n)",
        s,
    )
    if sm and len(sm.group(2)) > schema_max_chars:
        new_body = sm.group(2)[:schema_max_chars] + "\n[schema truncated…]"
        s = s[: sm.start()] + sm.group(1) + new_body + sm.group(3) + s[sm.end() :]
    if len(s) <= max_chars:
        return s

    outi = s.rfind("OUTPUT FORMAT")
    if outi >= 0:
        tail = s[outi:]
        if len(tail) <= max_chars:
            return s[outi - (max_chars - len(tail)) :]
    if len(s) > max_chars:
        return s[:max_chars]
    return s


def _filter_library(
    matches: list[RewritePattern], pattern_library: list[RewritePattern] | None
) -> list[RewritePattern]:
    if pattern_library is None:
        return matches
    if not pattern_library:
        return []
    lib_ids = {p.pattern_id for p in pattern_library if isinstance(p, RewritePattern)}
    return [p for p in matches if p.pattern_id in lib_ids]


# --- main template --------------------------------------------------------------


def build_optimized_prompt(
    obs: Union[Observation, dict[str, Any]],
    pattern_library: list[RewritePattern] | list[Any] | None = None,
) -> str:
    """
    Build a single user prompt: schema, plan summary, query, up to 2 few-shots, history, JSON.

    Steps: run :func:`sqlsage.rewrite_patterns.detect_applicable_patterns` on
    ``explain_plan``, take the top two by ``target_speedup_ratio`` (already sorted
    in that function), format few-shots, then apply the project template.
    If ``pattern_library`` is a non-empty list, only patterns in that set (by
    ``pattern_id``) are kept. ``None`` means use all applicable patterns; ``[]``
    means no few-shots.
    """
    d = _as_dict(obs)
    plan = _plan(d)
    matches = detect_applicable_patterns(plan)
    if pattern_library is None:
        pl: list[RewritePattern] | None = None
    else:
        pl = [p for p in pattern_library if isinstance(p, RewritePattern)]
    matches = _filter_library(matches, pl)
    top2 = matches[:2]
    few_shot_section = format_few_shot_section(top2)
    previous_attempts_section = format_previous_attempts(d)

    node_type = str(plan.get("node_type", "unknown"))
    try:
        seq_scans = int(plan.get("seq_scans", 0) or 0)
    except (TypeError, ValueError):
        seq_scans = 0
    try:
        nested_loops = int(plan.get("nested_loops", 0) or 0)
    except (TypeError, ValueError):
        nested_loops = 0
    try:
        hash_joins = int(plan.get("hash_joins", 0) or 0)
    except (TypeError, ValueError):
        hash_joins = 0
    try:
        total_cost = float(plan.get("total_cost", 0.0) or 0.0)
    except (TypeError, ValueError):
        total_cost = 0.0
    h_str = _highest_node_display(plan)
    ex_ms = float(d.get("execution_ms", 0.0) or 0.0)
    sch = str(d.get("schema_context", ""))
    oq = str(d.get("original_query", ""))

    prompt = f'''You are SQLSage, an expert database query optimizer. Your job is to rewrite a slow
PostgreSQL SQL query to run faster, without changing its results.

=== DATABASE SCHEMA ===
{sch}

=== CURRENT QUERY PERFORMANCE ===
Execution Time: {ex_ms:.1f} ms
Query Plan Summary:
  - Top node: {node_type}
  - Sequential scans: {seq_scans}
  - Nested loops: {nested_loops}
  - Hash joins: {hash_joins}
  - Highest cost node: {h_str}
  - Total estimated cost: {total_cost:.0f}

=== QUERY TO OPTIMIZE ===
{oq}

{few_shot_section}=== PREVIOUS ATTEMPTS THIS EPISODE ===
{previous_attempts_section}

=== YOUR TASK ===
Analyze the query plan above. Identify the bottleneck. Rewrite the query to eliminate it.

RULES (violation = -20 penalty):
1. The result set must be IDENTICAL — same rows, same columns, same values
2. Do NOT use CREATE INDEX, CREATE TABLE, or any DDL
3. Do NOT use LIMIT 0 or WHERE 1=0
4. Do NOT remove WHERE conditions, GROUP BY columns, or JOIN keys

OUTPUT FORMAT — respond with ONLY this JSON, nothing else:
{{
  "action": "<one of: rewrite_join | add_cte | push_filter | reorder_joins | suggest_index | limit_early | revert>",
  "reasoning": "<one sentence: what you saw in the plan and why this rewrite helps>",
  "rewritten_query": "<your complete rewritten SQL>"
}}'''
    return truncate_prompt_if_needed(prompt, max_tokens=1800)


def build_baseline_prompt(obs: Union[Observation, dict[str, Any]]) -> str:
    """
    Minimal prompt: schema, query, JSON output. No few-shots, no plan summary.
    """
    d = _as_dict(obs)
    oq = str(d.get("original_query", ""))
    sch = str(d.get("schema_context", ""))
    return (
        f"You are SQLSage, an expert database query optimizer.\n\n"
        f"=== DATABASE SCHEMA ===\n{sch}\n\n"
        f"=== QUERY TO OPTIMIZE ===\n{oq}\n\n"
        f"OUTPUT FORMAT — respond with ONLY this JSON, nothing else:\n"
        f'{{"action": "push_filter", "reasoning": "one sentence", "rewritten_query": "..."}}'
    )


def build_plan_only_prompt(obs: Union[Observation, dict[str, Any]]) -> str:
    """
    Plan summary + query + JSON output, without few-shot pattern injection.
    """
    d = _as_dict(obs)
    plan = _plan(d)
    node_type = str(plan.get("node_type", "unknown"))
    try:
        seq_scans = int(plan.get("seq_scans", 0) or 0)
    except (TypeError, ValueError):
        seq_scans = 0
    try:
        nested_loops = int(plan.get("nested_loops", 0) or 0)
    except (TypeError, ValueError):
        nested_loops = 0
    try:
        hash_joins = int(plan.get("hash_joins", 0) or 0)
    except (TypeError, ValueError):
        hash_joins = 0
    try:
        total_cost = float(plan.get("total_cost", 0.0) or 0.0)
    except (TypeError, ValueError):
        total_cost = 0.0
    h_str = _highest_node_display(plan)
    ex_ms = float(d.get("execution_ms", 0.0) or 0.0)
    sch = str(d.get("schema_context", ""))
    oq = str(d.get("original_query", ""))
    p = f'''You are SQLSage, an expert database query optimizer.

=== DATABASE SCHEMA ===
{sch}

=== CURRENT QUERY PERFORMANCE ===
Execution Time: {ex_ms:.1f} ms
Query Plan Summary:
  - Top node: {node_type}
  - Sequential scans: {seq_scans}
  - Nested loops: {nested_loops}
  - Hash joins: {hash_joins}
  - Highest cost node: {h_str}
  - Total estimated cost: {total_cost:.0f}

=== QUERY TO OPTIMIZE ===
{oq}

OUTPUT FORMAT — respond with ONLY this JSON, nothing else:
{{
  "action": "push_filter",
  "reasoning": "<one sentence>",
  "rewritten_query": "<your complete rewritten SQL>"
}}'''
    return truncate_prompt_if_needed(p, max_tokens=1800)


__all__ = [
    "build_baseline_prompt",
    "build_optimized_prompt",
    "build_plan_only_prompt",
    "format_few_shot_section",
    "format_previous_attempts",
    "truncate_prompt_if_needed",
]
