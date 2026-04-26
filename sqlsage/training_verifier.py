"""
Live training verification (Section 14 / “is training working?” checks).

Runs five checks against the model, env, DB, and optionally Weights & Biases history.
Intended to be called periodically from a TRL/GRPO callback or a manual script.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# --- TPC-H Q5 (project curriculum) — same six-table shape as `run_benchmark` Q5 ---

Q5_TPC_H_SQL: str
try:
    from sqlsage.tasks.level2 import LEVEL2_TASKS

    Q5_TPC_H_SQL = LEVEL2_TASKS[1].query.strip()
except Exception:  # pragma: no cover
    Q5_TPC_H_SQL = """
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
""".strip()


# --- W&B ---------------------------------------------------------------------------


def _wandb_api_run(wandb_run_id: str) -> tuple[Any, Any] | None:
    """Return ``(api, run)`` or None on failure."""
    rid = (wandb_run_id or "").strip()
    if not rid:
        return None
    try:
        import wandb
    except ImportError:  # pragma: no cover
        return None
    try:
        api = wandb.Api(timeout=int(os.environ.get("SQLSAGE_WANDB_API_TIMEOUT", "60")))
        run = api.run(rid)
    except Exception:  # noqa: BLE001
        return None
    return (api, run)


def _resolve_wandb_path(run_id: str) -> str:
    p = (run_id or "").strip()
    if p.count("/") >= 2:
        return p
    wid = os.environ.get("WANDB_RUN_ID", "").strip()
    if wid:
        p = f"{os.environ.get('WANDB_ENTITY', '')}/{os.environ.get('WANDB_PROJECT', 'sqlsage-grpo')}/{wid}"
    return p


def _history_rows(
    run: Any, keys: list[str], cap: int = 50_000
) -> list[dict[str, Any]]:
    """Pull tabular run history, falling back to ``scan_history``."""
    out: list[dict[str, Any]] = []
    try:
        import pandas  # noqa: F401
    except Exception:  # pragma: no cover
        df = None
    else:
        try:
            df = run.history(keys=keys, samples=cap, pandas=True, sort="step")
        except TypeError:
            try:
                df = run.history(keys=keys, samples=cap, pandas=True)
            except Exception:  # noqa: BLE001
                df = None
        if df is not None and len(df) > 0:
            return df.to_dict(orient="records")
    for i, h in enumerate(run.scan_history()):
        if i >= cap:
            break
        if isinstance(h, dict):
            out.append(h)
    return out


def _reward_mean_series(rows: list[dict[str, Any]]) -> list[float]:
    s: list[float] = []
    for r in rows:
        v = r.get("reward/mean", r.get("reward_mean", r.get("env/reward", None)))
        if v is not None and isinstance(v, (int, float)) and v == v:
            s.append(float(v))
    return s


# --- Check 1: CTE production ---------------------------------------------------------


def _has_cte(sql_text: str) -> bool:
    """
    Heuristic: a WITH clause (common table expression), not the word in ``WITHOUT``.
    """
    t = (sql_text or "").upper()
    if "WITHOUT" in t and "WITH" not in t:
        return False
    return bool(
        re.search(
            r"(?is)(?<![A-Z0-9_])\bWITH\s+[A-Z0-9_`\"\[\]\.]+\s+AS\s*\(",
            t,
        )
        or re.search(r"(?is)\bWITH\s+[A-Z0-9_`\"\[\]\.]+\s+AS\s+MATERIALIZED", t)
    )


def _sample_level1_seqscan_obs(
    env: Any, n_samples: int, max_tries: int
) -> list[dict[str, Any]]:
    """Resets ``env`` with increasing seeds; keeps Level-1 obs with at least one seq scan."""
    out: list[dict[str, Any]] = []
    for seed in range(max_tries):
        if len(out) >= n_samples:
            break
        if hasattr(env, "reset"):
            obs = env.reset(seed=seed)  # type: ignore[union-attr]
        else:  # pragma: no cover
            break
        if not isinstance(obs, dict):
            continue
        if int(obs.get("task_level", 0) or 0) != 1:
            continue
        ex = obs.get("explain_plan", {})
        if not isinstance(ex, dict):
            ex = {}
        if int(ex.get("seq_scans", 0) or 0) < 1:
            continue
        out.append(dict(obs))
    return out


def _obs_from_level1_query(conn: Any, query: str, schema_context: str) -> dict[str, Any]:
    from sqlsage.explain_parser import get_explain_dict
    from sqlsage.anti_cheat import execute_read_only

    ms, res_hash, _n = execute_read_only(conn, query, timeout_ms=120_000)
    plan = get_explain_dict(conn, query, timeout_ms=120_000)
    return {
        "original_query": query,
        "explain_plan": plan,
        "execution_ms": float(ms),
        "result_hash": res_hash,
        "schema_context": schema_context,
        "task_level": 1,
    }


def _fill_obs_from_conn(
    env: Any, conn: Any, n_samples: int, need: int
) -> list[dict[str, Any]]:
    if conn is None or need <= 0:
        return []
    from sqlsage.env import SCHEMA_SUMMARY
    from sqlsage.tasks.level1 import LEVEL1_TASKS

    extra: list[dict[str, Any]] = []
    for task in LEVEL1_TASKS:
        if len(extra) >= need:
            break
        o = _obs_from_level1_query(conn, task.query, SCHEMA_SUMMARY)
        ex = o.get("explain_plan") or {}
        if not isinstance(ex, dict):
            continue
        if int(ex.get("seq_scans", 0) or 0) < 1:
            continue
        extra.append(o)
    return extra


def _generate_for_obs(
    model: Any,
    tokenizer: Any,
    observation: dict[str, Any],
    max_new_tokens: int = 256,
    *,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """Build ``make_prompt`` from observation; generate text (chat template if present)."""
    from sqlsage.dataset import make_prompt

    prompt = make_prompt(observation)
    if hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    ) is not None:
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001
            text = prompt
    else:
        text = prompt
    import torch

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    dev = next(model.parameters()).device
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            pad_token_id=getattr(tokenizer, "pad_token_id", None)
            or getattr(tokenizer, "eos_token_id", 0),
        )
    n_in = int(enc["input_ids"].shape[1])  # type: ignore[union-attr]
    gen = out[0, n_in:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def verify_cte_production(
    model: Any,
    env: Any,
    pattern_library: Any,
    n_samples: int = 20,
    *,
    tokenizer: Any = None,
    conn: Any = None,
    max_env_tries: int = 4000,
) -> dict[str, Any]:
    """
    Confirm the model often emits CTEs (``WITH``) when the plan shows a seq scan + filters.

    ``pattern_library`` is reserved for future filtering (e.g. P1 only); pass
    :data:`sqlsage.rewrite_patterns.ALL_PATTERNS` to satisfy the project API.

    Keyword args ``tokenizer`` and ``conn`` are optional: without ``tokenizer`` the check fails
    with a clear diagnosis. ``conn`` helps build observations if env sampling is sparse.
    """
    del pattern_library
    if model is None or tokenizer is None:
        return {
            "check": "CTE_PRODUCTION",
            "cte_rate": 0.0,
            "pass": False,
            "threshold": 0.40,
            "samples_checked": 0,
            "diagnosis": "Model or tokenizer missing — cannot run CTE check.",
        }
    need = int(max(1, n_samples))
    obs_l = _sample_level1_seqscan_obs(env, need, max_env_tries)
    if len(obs_l) < need and conn is not None:
        extra = _fill_obs_from_conn(env, conn, n_samples, need - len(obs_l))
        for e in extra:
            if len(obs_l) >= need:
                break
            obs_l.append(e)
    if not obs_l:
        return {
            "check": "CTE_PRODUCTION",
            "cte_rate": 0.0,
            "pass": False,
            "threshold": 0.40,
            "samples_checked": 0,
            "diagnosis": (
                "No Level-1 seq-scan observations (try more seeds, TPC-H loaded DB, or pass conn)."
            ),
        }

    cte = 0
    for o in obs_l:
        try:
            out = _generate_for_obs(model, tokenizer, o)
            if _has_cte(out):
                cte += 1
        except Exception:  # noqa: BLE001
            continue
    n = len(obs_l)
    rate = cte / max(n, 1)
    th = 0.40
    ok = rate >= th
    return {
        "check": "CTE_PRODUCTION",
        "cte_rate": round(rate, 4),
        "pass": bool(ok),
        "threshold": th,
        "samples_checked": n,
        "diagnosis": (
            f"Model produced WITH/CTE in {100 * rate:.0f}% of seq-scan L1 samples — GOOD"
            if ok
            else f"Only {100 * rate:.0f}% CTE rate — model not learning filter pushdown / CTE pattern"
        ),
    }


# --- Check 2: no-op elimination --------------------------------------------------------


def verify_no_op_elimination(
    wandb_run_id: str, episode_window: int = 50
) -> dict[str, Any]:
    """
    Confirm no-op rewrites (unchanged query or no_op action) are rare in recent W&B data.

    Expects optional keys: ``train/action``, ``action``, or ``train/rewrite_unchanged`` (0/1).
    If W&B has no such columns, the check is **skipped** (not a hard fail); log these from
    the training loop for a definitive signal.
    """
    path = _resolve_wandb_path(wandb_run_id)
    got = _wandb_api_run(path)
    if not got:
        return {
            "check": "NOOP_ELIMINATION",
            "noop_rate": None,
            "pass": None,
            "threshold": 0.10,
            "skipped": True,
            "diagnosis": "No valid wandb run id; set WANDB_PATH or entity/project/run_id.",
        }
    _api, run = got
    rows = _history_rows(
        run, keys=[], cap=max(5_000, int(episode_window) * 20)
    )
    if not rows:
        return {
            "check": "NOOP_ELIMINATION",
            "noop_rate": None,
            "pass": None,
            "threshold": 0.10,
            "skipped": True,
            "diagnosis": "W&B run has no history rows.",
        }
    w = int(max(3, min(len(rows), int(episode_window))))
    last = rows[-w:]
    noop = 0
    for r in last:
        if r.get("train/rewrite_unchanged") in (1, 1.0, True) or r.get("noop") in (1, 1.0, True):
            noop += 1
            continue
        a = r.get("train/action", r.get("action", r.get("step/action", "")))
        if str(a).strip().lower() in ("no_op", "noop", "no-op", "identity"):
            noop += 1
            continue
    _keys = (
        "train/rewrite_unchanged",
        "train/action",
        "action",
        "noop",
        "step/action",
    )
    has_signal = any(
        k in (r or {}) and (r or {}).get(k) is not None for r in last for k in _keys
    )
    if not has_signal and noop == 0:
        return {
            "check": "NOOP_ELIMINATION",
            "noop_rate": None,
            "pass": None,
            "threshold": 0.10,
            "skipped": True,
            "diagnosis": (
                "W&B has no action/noop columns — log `train/action` and `train/rewrite_unchanged` from training, "
                "or accept this as inconclusive."
            ),
        }
    rate = noop / max(w, 1)
    th = 0.10
    ok = rate <= th
    return {
        "check": "NOOP_ELIMINATION",
        "noop_rate": round(rate, 4),
        "pass": bool(ok),
        "threshold": th,
        "diagnosis": (
            f"No-op rate {100 * rate:.0f}% — model learned to attempt rewrites"
            if ok
            else f"No-op rate {100 * rate:.0f}% — model still echoing or no_op; check prompt / action space"
        ),
    }


# --- Check 3: reward inflection --------------------------------------------------------


def _rolling_mean(xs: list[float], w: int) -> list[float]:
    if w <= 0 or len(xs) < w:
        return []
    out: list[float] = []
    for i in range(w - 1, len(xs)):
        out.append(float(sum(xs[i - w + 1 : i + 1]) / w))
    return out


def verify_reward_inflection(wandb_run_id: str) -> dict[str, Any]:
    """
    Find the first step where a 10-step rolling mean of ``reward/mean`` rises by >10% vs
    the previous rolling value; pass if that happens before the 100th rolling index (roughly
    W&B step < 100+window).
    """
    path = _resolve_wandb_path(wandb_run_id)
    got = _wandb_api_run(path)
    if not got:
        return {
            "check": "REWARD_INFLECTION",
            "inflection_episode": None,
            "pass": None,
            "reward_before": None,
            "reward_after": None,
            "skipped": True,
            "diagnosis": "No valid W&B id — cannot read reward/mean time series.",
        }
    _api, run = got
    rows = _history_rows(
        run, keys=["reward/mean", "_step"], cap=50_000
    )
    s = _reward_mean_series(rows)
    if len(s) < 20:
        return {
            "check": "REWARD_INFLECTION",
            "inflection_episode": None,
            "pass": False,
            "reward_before": None,
            "reward_after": None,
            "diagnosis": f"Not enough reward/mean points ({len(s)}); need longer run.",
        }
    w = 10
    roll = _rolling_mean(s, w)
    if len(roll) < 2:
        return {
            "check": "REWARD_INFLECTION",
            "inflection_episode": None,
            "pass": False,
            "diagnosis": "Could not form rolling means.",
        }
    infl: Optional[int] = None
    rb, ra = 0.0, 0.0
    for j in range(1, min(len(roll), 500)):
        prev, cur = roll[j - 1], roll[j]
        if prev > 0 and cur > 1.1 * prev:
            infl, rb, ra = j, prev, cur
            break
    if infl is None:
        return {
            "check": "REWARD_INFLECTION",
            "inflection_episode": None,
            "pass": False,
            "reward_before": roll[0] if roll else 0.0,
            "reward_after": roll[-1] if roll else 0.0,
            "episodes_until_inflection": None,
            "diagnosis": "No inflection found in rolling reward — curve still flat or noisy.",
        }
    pass_ok = infl < 100
    return {
        "check": "REWARD_INFLECTION",
        "inflection_episode": int(infl),
        "episodes_until_inflection": int(infl),
        "pass": bool(pass_ok),
        "reward_before": round(float(rb), 4),
        "reward_after": round(float(ra), 4),
        "reward_at_end": round(float(s[-1] if s else 0.0), 4),
        "diagnosis": (
            f"Inflection at rolling window index {infl} — learning signal picked up (before ep 100 target)"
            if pass_ok
            else f"Inflection only at {infl} (>= 100) — run longer or improve prompt/reward"
        ),
    }


# --- Check 4: plan improvements --------------------------------------------------------


def verify_plan_improvements(
    wandb_run_id: str, episode_window: int = 50
) -> dict[str, Any]:
    """
    Check that ``plan/seq_scans_removed`` (or best-effort) is > 0 in a good fraction
    of recent W&B points.
    """
    path = _resolve_wandb_path(wandb_run_id)
    got = _wandb_api_run(path)
    if not got:
        return {
            "check": "PLAN_IMPROVEMENTS",
            "improvement_rate": None,
            "pass": None,
            "threshold": 0.30,
            "skipped": True,
            "diagnosis": "No valid W&B id.",
        }
    _api, run = got
    rows = _history_rows(
        run, keys=["plan/seq_scans_removed", "reward/mean", "_step"], cap=20_000
    )
    if not rows:
        return {
            "check": "PLAN_IMPROVEMENTS",
            "improvement_rate": None,
            "pass": None,
            "skipped": True,
            "diagnosis": "Empty W&B history.",
        }
    w = int(max(3, min(len(rows), int(episode_window))))
    last = rows[-w:]
    key = "plan/seq_scans_removed"
    seen_key = any(key in (r or {}) and (r or {}).get(key) is not None for r in last)
    n_ok = 0
    for r in last:
        v = (r or {}).get(key, None)
        if v is not None and isinstance(v, (int, float)) and float(v) > 0:
            n_ok += 1
    if not seen_key:
        return {
            "check": "PLAN_IMPROVEMENTS",
            "improvement_rate": None,
            "pass": None,
            "threshold": 0.30,
            "skipped": True,
            "diagnosis": (
                f"W&B has no `{key}`; log it from the env (see monitor_training) for this check."
            ),
        }
    rate = n_ok / max(w, 1)
    th = 0.30
    ok = rate >= th
    return {
        "check": "PLAN_IMPROVEMENTS",
        "improvement_rate": round(rate, 4),
        "pass": bool(ok),
        "threshold": th,
        "diagnosis": (
            f"{100 * rate:.0f}% of last {w} steps show seq scan removal — training signal is real"
            if ok
            else f"Only {100 * rate:.0f}% of steps show `plan/seq_scans_removed`>0"
        ),
    }


# --- Check 5: Q5 performance ----------------------------------------------------------


def _median_runs(
    conn: Any, query: str, n_runs: int = 3, timeout_s: int = 120
) -> tuple[float, str]:
    from sqlsage.anti_cheat import execute_read_only

    times: list[float] = []
    h_last = ""
    for _ in range(n_runs):
        ms, h, _ = execute_read_only(conn, query, timeout_ms=timeout_s * 1000)
        times.append(float(ms))
        h_last = h
    return float(statistics.median(times)) if times else 0.0, h_last


def _generate_q5_rewrite(
    model: Any,
    tokenizer: Any,
    q5_sql: str,
    conn: Any,
    temperature: float = 0.1,
) -> str:
    """Prompt the model with ``make_prompt``-style fields and parse JSON or SQL."""
    from sqlsage.env import SCHEMA_SUMMARY
    from sqlsage.explain_parser import get_explain_dict

    if model is None or tokenizer is None:
        return ""
    obs = {
        "original_query": q5_sql,
        "explain_plan": {},
        "execution_ms": 0.0,
        "result_hash": "",
        "schema_context": SCHEMA_SUMMARY,
        "disable_rewrite_pattern_few_shot": True,
    }
    if conn is not None:
        try:
            obs["explain_plan"] = get_explain_dict(
                conn, q5_sql, timeout_ms=120_000
            )
        except Exception:  # noqa: BLE001
            pass
    text = _generate_for_obs(
        model, tokenizer, obs, max_new_tokens=512, temperature=temperature, top_p=0.9
    )
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
    blob = m.group(1).strip() if m else text.strip()
    start = blob.find("{")
    if start >= 0:
        try:
            from json import JSONDecoder

            data, _ = JSONDecoder().raw_decode(blob, start)
            if isinstance(data, dict):
                rq = str(data.get("rewritten_query", "")).strip()
                if rq:
                    return rq
        except Exception:  # noqa: BLE001
            pass
    m2 = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if m2 and m2.group(1).strip():
        return m2.group(1).strip()
    return text.strip() or Q5_TPC_H_SQL


def verify_q5_performance(
    conn: Any,
    trained_model: Any,
    tokenizer: Any,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """
    Run curriculum TPC-H Q5 shape: median timing (3 runs), model rewrite at low temperature,
    hash match vs original result.
    """
    target_ms = 500.0
    stretch_ms = 310.0
    hard_cap_ms = 1000.0
    if conn is None or trained_model is None or tokenizer is None:
        return {
            "check": "Q5_PERFORMANCE",
            "original_ms": None,
            "rewritten_ms": None,
            "speedup": None,
            "pass": False,
            "hash_verified": False,
            "target_ms": target_ms,
            "stretch_target_ms": stretch_ms,
            "diagnosis": "conn, model, or tokenizer missing — cannot run Q5 check.",
        }
    q5 = Q5_TPC_H_SQL
    try:
        orig_ms, orig_hash = _median_runs(conn, q5, n_runs=3)
    except Exception as e:  # noqa: BLE001
        return {
            "check": "Q5_PERFORMANCE",
            "original_ms": None,
            "pass": False,
            "hash_verified": False,
            "target_ms": target_ms,
            "stretch_target_ms": stretch_ms,
            "diagnosis": f"Original Q5 execution failed: {e}",
        }
    try:
        rewritten = _generate_q5_rewrite(
            trained_model, tokenizer, q5, conn, temperature=temperature
        )
        rw_ms, rw_hash = _median_runs(conn, rewritten, n_runs=3)
    except Exception as e:  # noqa: BLE001
        return {
            "check": "Q5_PERFORMANCE",
            "original_ms": round(orig_ms, 2),
            "rewritten_ms": None,
            "pass": False,
            "hash_verified": False,
            "target_ms": target_ms,
            "diagnosis": f"Rewritten Q5 failed: {e}",
        }
    hash_ok = bool(orig_hash and rw_hash and orig_hash == rw_hash)
    sp = (orig_ms / rw_ms) if rw_ms and rw_ms > 0 else 0.0
    pass_ok = bool(hash_ok and rw_ms <= hard_cap_ms)
    strong = hash_ok and rw_ms <= target_ms
    diag = (
        f"Q5 rewrite at {rw_ms:.0f}ms — PASS (under {hard_cap_ms:.0f}ms cap; target {target_ms:.0f}ms; stretch {stretch_ms:.0f}ms)."
        if pass_ok and strong
        else (
            f"Q5 at {rw_ms:.0f}ms — pass under {hard_cap_ms:.0f}ms but above {target_ms:.0f}ms target."
            if pass_ok and not strong
            else f"Q5 rewrite slow or hash mismatch (orig hash match: {hash_ok})."
        )
    )
    return {
        "check": "Q5_PERFORMANCE",
        "original_ms": round(orig_ms, 2),
        "rewritten_ms": round(rw_ms, 2),
        "speedup": round(sp, 2),
        "pass": bool(pass_ok),
        "hash_verified": bool(hash_ok),
        "target_ms": target_ms,
        "stretch_target_ms": stretch_ms,
        "hard_cap_ms": hard_cap_ms,
        "meets_target_500": bool(hash_ok and rw_ms <= target_ms),
        "meets_stretch_310": bool(hash_ok and rw_ms <= stretch_ms),
        "diagnosis": diag,
    }


# --- Master report ---------------------------------------------------------------------


def _recommendation(report: dict[str, Any]) -> str:
    if report.get("overall") == "PASS":
        return "Training is working. Advance to Level 2 curriculum."
    ch = report.get("checks", {})
    c1 = ch.get("CTE_PRODUCTION", {})
    c2 = ch.get("NOOP_ELIMINATION", {})
    c3 = ch.get("REWARD_INFLECTION", {})
    c4 = ch.get("PLAN_IMPROVEMENTS", {})
    if c1.get("pass") is False and c1.get("skipped") is not True:
        return "CTE production low — inject Pattern P1 few-shot example into prompt and retrain."
    if c3.get("pass") is False and c3.get("skipped") is not True:
        return "Reward flat — see fix_training.py --issue flat_reward"
    if c2.get("pass") is False and c2.get("skipped") is not True:
        return "No-op rate high — verify prompt asks for rewrites and action space matches training."
    if c4.get("pass") is False and c4.get("skipped") is not True:
        return "Plan improvements rare — log plan/seq_scans_removed from the env; confirm reward decomposes scans."
    if ch.get("Q5_PERFORMANCE", {}).get("pass") is False and ch.get("Q5_PERFORMANCE", {}).get("skipped") is not True:
        return "Q5 end-to-end target missed — re-check curriculum data and model outputs."
    return (
        "Review failed checks; add W&B metrics (train/action, plan/seq_scans_removed) if checks are skipped, "
        "or see fix_training for reward/prompt triage."
    )


def run_all_checks(
    model: Any,
    tokenizer: Any,
    env: Any,
    conn: Any,
    wandb_run_id: str,
    pattern_library: Any = None,
) -> dict[str, Any]:
    """
    Run all five checks and return a combined report with ``overall`` status.

    ``pattern_library`` defaults to :data:`sqlsage.rewrite_patterns.ALL_PATTERNS` for the
    CTE check API (filtering reserved for future use).
    """
    if pattern_library is None:
        try:
            from sqlsage.rewrite_patterns import ALL_PATTERNS as pattern_library
        except Exception:  # pragma: no cover
            pattern_library = []

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    c1 = verify_cte_production(
        model, env, pattern_library, 20, tokenizer=tokenizer, conn=conn
    )
    c2 = verify_no_op_elimination(wandb_run_id, episode_window=50)
    c3 = verify_reward_inflection(wandb_run_id)
    c4 = verify_plan_improvements(wandb_run_id, episode_window=50)
    c5 = verify_q5_performance(conn, model, tokenizer)

    checks: dict[str, Any] = {
        "CTE_PRODUCTION": c1,
        "NOOP_ELIMINATION": c2,
        "REWARD_INFLECTION": c3,
        "PLAN_IMPROVEMENTS": c4,
        "Q5_PERFORMANCE": c5,
    }
    n_skip = sum(1 for c in checks.values() if c.get("skipped"))
    n_true_pass = sum(1 for c in checks.values() if c.get("pass") is True)
    n_fail = sum(
        1
        for c in checks.values()
        if c.get("skipped") is not True and c.get("pass") is False
    )
    n_active = 5 - n_skip
    if n_fail == 0 and n_true_pass == 5 and n_skip == 0:
        overall = "PASS"
    elif n_fail == 0 and (n_true_pass + n_skip) == 5 and n_skip > 0:
        overall = "PARTIAL"
    elif n_fail == 0 and n_active > 0 and n_true_pass == n_active:
        overall = "PASS"
    elif n_true_pass >= 3 and n_fail <= 2 and (n_true_pass + n_fail + n_skip) == 5:
        overall = "PARTIAL"
    else:
        overall = "FAIL"

    rep = {
        "timestamp": ts,
        "overall": overall,
        "checks_passed": int(n_true_pass),
        "checks_total": 5,
        "checks_skipped": n_skip,
        "checks": checks,
        "recommendation": "",
    }
    rep["recommendation"] = _recommendation(rep)
    return rep


# --- Reporting & scheduler -------------------------------------------------------------


def print_verification_report(report: dict[str, Any]) -> None:
    """Render a :mod:`rich` table of per-check result and an overall line."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:  # pragma: no cover
        print(json.dumps(report, indent=2, default=str))
        return
    c = Console()
    t = Table(title="SQLSage training verification")
    t.add_column("Check", style="bold")
    t.add_column("Result", justify="center")
    t.add_column("Diagnosis")
    label = {
        "CTE_PRODUCTION": "CTE production",
        "NOOP_ELIMINATION": "No-op elimination",
        "REWARD_INFLECTION": "Reward inflection",
        "PLAN_IMPROVEMENTS": "Plan improvements",
        "Q5_PERFORMANCE": "Q5 performance",
    }
    for key, ch in (report.get("checks") or {}).items():
        p = ch.get("pass")
        if ch.get("skipped"):
            rtxt = "SKIP"
        elif p is True:
            rtxt = "[green]PASS[/]"
        elif p is False:
            rtxt = "[red]FAIL[/]"
        else:
            rtxt = "N/A"
        t.add_row(label.get(key, key), rtxt, str(ch.get("diagnosis", ""))[:120])
    c.print(t)
    c.print(
        f"\n[bold]Overall:[/] {report.get('overall')} "
        f"({report.get('checks_passed',0)}/5) — {report.get('recommendation','')}"
    )


def _default_reports_dir() -> Path:
    return Path(os.environ.get("SQLSAGE_VERIFICATION_DIR", "verification_reports"))


def run_verification_every_n_episodes(
    n: int,
    callback_fn: Optional[Callable[[dict[str, Any]], None]] = None,
    *,
    env: Any,
    conn: Any,
    wandb_run_id: str = "",
    model: Any = None,
    tokenizer: Any = None,
) -> Any:
    """
    Build a :class:`transformers.TrainerCallback` that runs :func:`run_all_checks` every
    ``n`` training steps (``TrainerState.global_step``; map ``n`` to “episodes” in your
    GRPO config if you use one step per log).

    ``model`` and ``tokenizer`` are used when set; each ``on_step_end`` also looks for
    ``model`` and ``processing_class`` / ``tokenizer`` in the callback ``kwargs`` so new
    references from the trainer are picked up.
    """
    try:
        from transformers import (  # noqa: I001
            TrainerCallback,
            TrainerControl,
            TrainingArguments,
            TrainerState,
        )
    except ImportError:  # pragma: no cover
        raise RuntimeError("transformers is required for training callbacks")

    out_dir = _default_reports_dir()
    every = int(max(1, n))

    class _V(TrainerCallback):
        def __init__(self) -> None:
            self.model_ref: Any = model
            self.tok_ref: Any = tokenizer

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
        ) -> TrainerControl:  # type: ignore[override]
            if int(state.global_step) == 0 or int(state.global_step) % every != 0:
                return control
            mod = kwargs.get("model", self.model_ref)
            tok = kwargs.get("processing_class", kwargs.get("tokenizer", self.tok_ref))
            if mod is not None:
                self.model_ref = mod
            if tok is not None:
                self.tok_ref = tok
            rep = run_all_checks(
                self.model_ref,
                self.tok_ref,
                env,
                conn,
                wandb_run_id,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"episode_{int(state.global_step)}.json"
            path.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
            print_verification_report(rep)
            if callback_fn is not None:
                try:
                    callback_fn(rep)
                except Exception:  # noqa: BLE001
                    pass
            return control

    return _V()


__all__ = [
    "Q5_TPC_H_SQL",
    "print_verification_report",
    "run_all_checks",
    "run_verification_every_n_episodes",
    "verify_cte_production",
    "verify_no_op_elimination",
    "verify_plan_improvements",
    "verify_q5_performance",
    "verify_reward_inflection",
]
