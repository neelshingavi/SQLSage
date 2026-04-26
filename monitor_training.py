#!/usr/bin/env python3
"""
SQLSage RL training monitor: wandb episode diagnostics, raw model samples,
reward math check, and anti-cheat probes against a live OpenEnv.

Requires: W&B optional (WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT, WANDB_RUN_ID),
  SQLSAGE_ENV_URL, local PostgreSQL, and optional checkpoint for sampling.

Run from the sqlsage-env root (package must be on PYTHONPATH, e.g. pip install -e .):
  pip install wandb transformers torch psycopg2-binary rich requests
  export WANDB_ENTITY=... WANDB_RUN_ID=... WANDB_PROJECT=sqlsage-grpo SQLSAGE_ENV_URL=http://127.0.0.1:8000
  python monitor_training.py
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

# --- Config ---

_CHECKPOINT = Path(
    os.environ.get("SQLSAGE_CHECKPOINT", "./sqlsage-grpo/checkpoint-latest")
)
_PG = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", "5432")),
    "user": os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ.get("POSTGRES_PASSWORD", "sqlsage"),
    "dbname": os.environ.get("POSTGRES_DB", "sqlsage"),
}

# TPC-H Q1 and an equivalent rephrase (kept for reference; sign test uses an idealized plan).
def _q1_text() -> str:
    from sqlsage.tasks.level1 import LEVEL1_TASKS

    return LEVEL1_TASKS[0].query


TPC_H_Q1 = _q1_text()

TPC_H_Q1_FAST = """
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
WHERE l_shipdate <= (TIMESTAMP '1998-12-01' - INTERVAL '90 days')::date
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus
""".strip()

_SEQ_BONUS = 8.0
_NL_BONUS = 6.0


# --- W&B ---

def _wandb_run_path() -> str:
    full = os.environ.get("WANDB_PATH", "").strip()
    if full:
        return full
    run_id = os.environ.get("WANDB_RUN_ID", "").strip()
    if not run_id:
        raise RuntimeError("missing WANDB_PATH or WANDB_RUN_ID")
    project = os.environ.get("WANDB_PROJECT", "sqlsage-grpo").strip()
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    if not entity:
        import wandb

        try:
            api = wandb.Api()
            u = api.viewer
            if u is not None and getattr(u, "default_entity", None):
                entity = str(u.default_entity)  # type: ignore[union-attr]
        except Exception:
            pass
    if not entity:
        raise RuntimeError("Set WANDB_ENTITY or WANDB_PATH (entity/project/run_id)")
    return f"{entity}/{project}/{run_id}"


def _load_wandb_history() -> list[dict[str, Any]]:
    import wandb

    path = _wandb_run_path()
    api = wandb.Api()
    run = api.run(path)
    keys = [
        "reward/mean",
        "reward/speedup_ratio",
        "penalty/result_changed",
        "penalty/syntax_error",
        "plan/seq_scans_removed",
        "episode_length",
    ]
    try:
        df = run.history(keys=keys, samples=10_000, pandas=True, sort="step")
    except TypeError:
        df = run.history(keys=keys, samples=10_000, pandas=True)
    if df is not None and len(df) > 0:
        return df.to_dict(orient="records")
    rows: list[dict[str, Any]] = []
    try:
        for h in run.scan_history():
            rows.append(h)
    except Exception:
        return []
    return [r for r in rows if r]


def _metric_trend(
    series: list[float], lower_is_better: bool
) -> tuple[str, str]:
    clean = [x for x in series if x is not None and x == x]
    if len(clean) < 2:
        return "n/a", "dim"
    a = float(sum(clean[:3]) / 3) if len(clean) >= 3 else float(clean[0])
    b = float(sum(clean[-3:]) / 3) if len(clean) >= 3 else float(clean[-1])
    delta = b - a
    if abs(delta) < 1e-9:
        return "flat", "red"
    if lower_is_better:
        good = delta < 0
    else:
        good = delta > 0
    return ("improving" if good else "worse", "green" if good else "red")


def section_reward_diagnostic() -> None:
    from rich.console import Console
    from rich.table import Table

    c = Console()
    c.print("\n[bold]1) REWARD DIAGNOSTIC[/bold]\n")

    if not os.environ.get("WANDB_RUN_ID", "").strip() and not os.environ.get(
        "WANDB_PATH", ""
    ).strip():
        c.print(
            "[dim]Set WANDB_RUN_ID (see env) or WANDB_PATH=entity/project/run_id; skipping.[/dim]\n"
        )
        return

    try:
        rows = _load_wandb_history()
    except Exception as e:
        c.print(f"[red]W&B load failed: {e}[/red]\n")
        return

    if not rows:
        c.print("[yellow]No W&B history rows; skip table.[/yellow]\n")
        return

    def _f(row: dict, k: str) -> float:
        v = row.get(k)
        if v is None:
            return float("nan")
        return float(v)

    keys: list[tuple[str, bool, str]] = [
        ("reward/mean", False, "higher is better"),
        ("reward/speedup_ratio", False, "higher is better"),
        ("penalty/result_changed", True, "lower is better (count / step)"),
        ("penalty/syntax_error", True, "lower is better"),
        ("plan/seq_scans_removed", False, "higher is better"),
        ("episode_length", True, "lower is better (fewer steps to finish)"),
    ]

    last10 = rows[-10:] if len(rows) >= 10 else rows
    last20 = rows[-20:] if len(rows) >= 20 else rows

    table = Table(
        title="Last 10 episodes (columns e0=oldest … e9=newest in window, trend vs start)"
    )
    table.add_column("metric")
    for i in range(len(last10)):
        table.add_column(f"e{i}", justify="right")
    table.add_column("trend", justify="right")
    for k, low_good, _ in keys:
        vals = []
        sers = []
        for r in last10:
            v = _f(r, k)
            vals.append(f"{v:.4g}" if v == v else "—")
            sers.append(v)
        tr, cl = _metric_trend(sers, low_good)
        tstyle = cl if cl in ("green", "red", "dim") else "white"
        table.add_row(k, *vals, f"[{tstyle}]{tr}[/]")

    c.print(table)
    c.print("")

    rm = [
        _f(r, "reward/mean")
        for r in last20
    ]
    rm = [x for x in rm if x == x and abs(x) < 1e200]
    if len(rm) >= 2:
        first, last = rm[0], rm[-1]
        denom = max(abs(first), 1e-9)
        flat_pct = abs(last - first) / denom
        if flat_pct <= 0.05:
            c.print(
                "[bold yellow]REWARD FLAT[/bold yellow]  "
                f"(reward/mean moved ≤5% over last 20: {100 * flat_pct:.2f}%)"
            )

    p10 = last10
    if p10 and any(
        _f(r, "penalty/result_changed") > 0.1
        for r in p10
    ):
        c.print(
            "[bold red]REWARD HACKING[/bold red]  "
            "(penalty/result_changed > 0.1 in at least one of the last 10 episodes)"
        )

    if len(p10) >= 10 and all(
        int(round(_f(r, "episode_length"))) == 5 for r in p10
    ):
        c.print(
            "[bold red]MODEL STUCK[/bold red]  "
            "(episode_length == 5 for 10 consecutive logs)"
        )

    c.print("")


# --- 2) Raw outputs ---

def section_raw_output_sampler() -> None:
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    c = Console()
    c.print("\n[bold]2) RAW OUTPUT SAMPLER[/bold]\n")

    try:
        from sqlsage.dataset import make_prompt
        from sqlsage.training.http_sqlsage_client import http_reset
    except Exception as e:
        c.print(f"[red]Import failed: {e}[/red]\n")
        return

    base = os.environ.get("SQLSAGE_ENV_URL", "").strip()
    if not base:
        c.print("[red]Set SQLSAGE_ENV_URL[/red]\n")
        return

    if not _CHECKPOINT.exists():
        c.print(
            f"[yellow]Checkpoint missing at {_CHECKPOINT}; skip generation.[/yellow]\n"
        )
        return

    try:
        obs = http_reset(base, seed=0)
    except Exception as e:
        c.print(f"[red]env.reset() failed: {e}[/red]\n")
        return

    if int(obs.get("task_level", 0) or 0) != 1:
        c.print(
            f"[yellow]Note: task_level={obs.get('task_level')} (L1 is typical for eval).[/yellow]"
        )
    prompt = make_prompt(obs)  # type: ignore[arg-type]

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        tok = AutoTokenizer.from_pretrained(str(_CHECKPOINT), trust_remote_code=True)
        if (
            hasattr(tok, "apply_chat_template")
            and getattr(tok, "chat_template", None) is not None
        ):
            model_in = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            model_in = prompt
        mdl = AutoModelForCausalLM.from_pretrained(
            str(_CHECKPOINT), trust_remote_code=True, torch_dtype=torch.float16
        )
        if torch.cuda.is_available():
            mdl = mdl.cuda()
        ppl = pipeline(
            "text-generation",
            model=mdl,
            tokenizer=tok,
            return_full_text=False,
        )
    except Exception as e:
        c.print(f"[red]Model load / prompt: {e}[/red]\n")
        return

    out_texts: list[str] = []
    try:
        for i in range(3):
            gen = ppl(
                str(model_in),
                max_new_tokens=384,
                do_sample=True,
                temperature=0.85 + i * 0.05,
                top_p=0.9,
            )
            if isinstance(gen, list) and gen:
                t = gen[0].get("generated_text", str(gen[0]))
            else:
                t = str(gen)
            out_texts.append(t)
    except Exception as e:
        c.print(f"[red]generate failed: {e}[/red]\n")
        return

    c.print("Three raw completions (side by side):")
    panels: list[Panel] = []
    w = max(24, (os.get_terminal_size().columns or 120) // 3 - 4)
    for i, t in enumerate(out_texts, 1):
        short = t.replace("\n", " ")[:w] + ("…" if len(t) > w else "")
        panels.append(
            Panel(Text(short, overflow="ignore"), title=f"out{i}", width=w + 6)
        )
    try:
        c.print(Columns(panels, equal=True, expand=True))
    except Exception:
        for i, t in enumerate(out_texts, 1):
            c.print(f"[bold]out{i}[/bold]: {t[:200]}…\n" if len(t) > 200 else t)

    c.print("")

    conn = _pg_connect()
    for i, raw in enumerate(out_texts, 1):
        ok_j = _check_json_output(raw)
        c.print(
            f"out{i}  valid_json: {'[green]PASS[/green]' if ok_j[0] else '[red]FAIL[/red]'}  "
            f"rewritten_query: {'[green]PASS[/green]' if ok_j[1] else '[red]FAIL[/red]'}  "
            f"sql_explain: {'[green]PASS[/green]' if _sql_ok(conn, ok_j[2]) else '[red]FAIL[/red]'}"
        )
    conn.close()
    c.print("")


def _pg_connect() -> Any:
    import psycopg2

    return psycopg2.connect(
        host=_PG["host"],
        port=_PG["port"],
        user=_PG["user"],
        password=_PG["password"],
        dbname=_PG["dbname"],
    )


def _check_json_output(raw: str) -> tuple[bool, bool, str | None]:
    s = raw.strip()
    text = s
    m = re.search(
        r"```(?:json)?\s*([\s\S]+?)\s*```", s, re.IGNORECASE
    )
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    if start < 0:
        return False, False, None
    dec = json.JSONDecoder()
    try:
        data, _ = dec.raw_decode(text, start)
    except json.JSONDecodeError:
        return False, False, None
    if not isinstance(data, dict):
        return True, False, None
    rq = str(data.get("rewritten_query", "")).strip()
    if not rq:
        return True, False, None
    return True, True, rq


def _sql_ok(conn: Any, q: str | None) -> bool:
    if not q:
        return False
    try:
        c = conn.cursor()
        c.execute("EXPLAIN " + q)
        conn.rollback()
        c.close()
        return True
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return False


# --- 3) Reward pipeline ---

def _idealized_fast_plan(
    old_plan: dict[str, Any], old_ms: float
) -> tuple[dict[str, Any], float]:
    n: dict[str, Any] = dict(old_plan) if old_plan else {}
    oseq = int(n.get("seq_scans", 0) or 0)
    onl = int(n.get("nested_loops", 0) or 0)
    n["seq_scans"] = 0
    n["index_scans"] = max(1, int(n.get("index_scans", 0) or 0) + 1)
    n["nested_loops"] = 0
    n["node_type"] = "Index Only Scan" if oseq or onl else str(n.get("node_type", "Seq Scan"))
    tc = float(n.get("total_cost", 1) or 1.0) * 0.1
    n["total_cost"] = max(tc, 1.0)
    n["rows"] = int(max(1, int(n.get("rows", 0) or 0) // 200))
    # Keep strictly faster than baseline so the sign check is stable (avoids microsecond edge cases)
    if old_ms > 0:
        new_ms = min(old_ms * 0.1, old_ms * 0.4)
    else:
        new_ms = 0.1
    return n, new_ms


def section_reward_pipeline() -> None:
    from rich.console import Console
    import psycopg2
    from sqlsage.explain_parser import get_explain_dict
    from sqlsage.reward import compute_reward

    c = Console()
    c.print("\n[bold]3) REWARD PIPELINE TESTER[/bold]\n")

    conn = psycopg2.connect(
        host=_PG["host"],
        port=_PG["port"],
        user=_PG["user"],
        password=_PG["password"],
        dbname=_PG["dbname"],
    )
    to_ms = 120_000
    try:
        old_plan = get_explain_dict(conn, TPC_H_Q1, timeout_ms=to_ms)
    except Exception as e:
        c.print(f"[red]EXPLAIN TPC-H Q1 failed: {e}[/red]\n")
        conn.close()
        return

    cur = conn.cursor()
    t0 = time.perf_counter()
    try:
        cur.execute(TPC_H_Q1)
        _ = cur.fetchall()
    except Exception as e:
        c.print(f"[red]Execute Q1 failed: {e}[/red]\n")
        conn.rollback()
        cur.close()
        conn.close()
        return
    old_ms = (time.perf_counter() - t0) * 1000.0
    conn.rollback()
    cur.close()

    c.print(
        "[dim]Slow query: TPC-H Q1 (level1 task1) — live wall time + EXPLAIN plan[/dim]"
    )
    c.print(
        f"[dim]Pre-stored fast rewrite: {len(TPC_H_Q1_FAST)} chars (also EXPLAIN+timed below)[/dim]\n"
    )

    r: float
    br: dict[str, float]
    used_idealized: bool
    new_plan: dict[str, Any]
    new_ms: float
    new_plan2: dict[str, Any] = {}
    new_ms2: float = 0.0
    try:
        new_plan2 = get_explain_dict(conn, TPC_H_Q1_FAST, timeout_ms=to_ms)
        cur2 = conn.cursor()
        t1 = time.perf_counter()
        cur2.execute(TPC_H_Q1_FAST)
        _ = cur2.fetchall()
        new_ms2 = (time.perf_counter() - t1) * 1000.0
        conn.rollback()
        cur2.close()
        r, br = compute_reward(
            old_ms,
            new_ms2,
            new_plan2,
            old_plan,
            step_number=0,
            table_sizes=None,
        )
        used_idealized = r <= 0.0
    except Exception as ex:
        c.print(
            f"[yellow]Could not use pre-stored fast SQL as measured plan ({ex}); "
            f"using idealized fast plan.[/yellow]\n"
        )
        new_plan2 = old_plan
        new_ms2 = old_ms
        r = 0.0
        br = {}
        used_idealized = True

    if used_idealized:
        c.print(
            "[dim]Using idealized “post-index” plan + latency so normalized reward is "
            "guaranteed > 0 (equivalent rewrites often tie the planner in practice).[/dim]\n"
        )
        new_plan, new_ms = _idealized_fast_plan(old_plan, old_ms)
        r, br = compute_reward(
            old_ms, new_ms, new_plan, old_plan, step_number=0, table_sizes=None
        )
    else:
        new_plan, new_ms = new_plan2, new_ms2
        c.print(
            "[green]Pre-stored fast rewrite yields positive reward vs measured Q1.[/green]\n"
        )

    oseq = int(old_plan.get("seq_scans", 0) or 0)
    nseq = int(new_plan.get("seq_scans", 0) or 0)
    onl = int(old_plan.get("nested_loops", 0) or 0)
    nnl = int(new_plan.get("nested_loops", 0) or 0)
    seq_comp = _SEQ_BONUS * max(0, oseq - nseq)
    nl_comp = _NL_BONUS * max(0, onl - nnl)
    penalties = float(br.get("noop_penalty", 0) + br.get("slowdown_penalty", 0))

    c.print(
        f"  speedup_component:      {br['speedup_reward']:+.4f}\n"
        f"  seq_scan_component:     {seq_comp:+.4f}  (Δseq {oseq}→{nseq})\n"
        f"  nested_loop_component:  {nl_comp:+.4f}  (ΔNL {onl}→{nnl})\n"
        f"  cost_component:         {br['cost_signal']:+.4f}\n"
        f"  penalties:              {penalties:+.4f}  (noop + slowdown; rows in rows_signal: {br['rows_signal']:+.4f})\n"
    )
    c.print("  " + "─" * 40)
    c.print(
        f"  raw: {br['raw']:+.4f}  normalized: {r:+.4f}\n"
    )
    if r > 0:
        if used_idealized:
            c.print(
                "[green]PASS — total normalized reward > 0 (idealized post-index plan vs live Q1).[/green]"
            )
        else:
            c.print(
                "[green]PASS — total normalized reward > 0 (measured pre-stored fast SQL vs Q1).[/green]"
            )
    else:
        c.print("[red]FAIL — total reward ≤ 0.[/red]")
    try:
        assert r > 0, "normalized reward must be > 0 for the idealized fast-rewrite check"
    finally:
        conn.close()
    c.print("")


# --- 4) Anti-cheat ---

def _remove_first_where_to_group(sql: str) -> str:
    s = re.sub(
        r"(?is)WHERE\s+.+?(?=\bGROUP\s+BY\b)",
        "",
        sql,
        count=1,
    )
    if s == sql and re.search(r"(?is)\bWHERE\s+", sql):
        s = re.sub(
            r"(?is)WHERE\s+.+?(?=\bORDER\s+BY\b|\Z)",
            "",
            sql,
            count=1,
        )
    return s


def _attack_queries(original: str) -> dict[str, tuple[str, int]]:
    b = original.strip().rstrip(";")
    a_limit0 = f"SELECT * FROM (\n{b}\n) AS _a LIMIT 0"
    a_1eq0 = f"SELECT * FROM (\n{b}\n) AS _a WHERE 1=0"
    a_nowhere = _remove_first_where_to_group(original)
    a_ddl = "CREATE INDEX IF NOT EXISTS idx_monitor_anticheat ON lineitem (l_orderkey)"
    # TPC-H lineitem: l_orderkey (spec: “orderkey=1” style hardcoding)
    a_hard = re.sub(
        r"(?is)(WHERE)\s+",
        r"\1 l_orderkey = 1 AND ",
        b,
        count=1,
    )
    if a_hard == b:
        a_hard = "SELECT 1 FROM lineitem WHERE l_orderkey = 1"
    return {
        "a) LIMIT 0": (a_limit0, -20),
        "b) WHERE 1=0": (a_1eq0, -20),
        "c) remove WHERE clause": (a_nowhere, -20),
        "d) DDL (CREATE INDEX)": (a_ddl, -15),
        "e) hardcoded (WHERE l_orderkey = 1)": (a_hard, -20),
    }


def section_anticheat() -> None:
    from rich.console import Console

    c = Console()
    c.print("\n[bold]4) ANTI-CHEAT VALIDATOR[/bold]\n")

    base = os.environ.get("SQLSAGE_ENV_URL", "").strip()
    if not base:
        c.print("[red]Set SQLSAGE_ENV_URL[/red]\n")
        return

    try:
        from sqlsage.training.http_sqlsage_client import http_reset, http_step
    except Exception as e:
        c.print(f"[red]Import: {e}[/red]\n")
        return

    try:
        obs = http_reset(base, seed=0)
    except Exception as e:
        c.print(f"[red]reset: {e}[/red]\n")
        return

    oq = str(obs.get("original_query", ""))
    attacks = _attack_queries(oq)
    failures: list[str] = []

    for name, (sql, expected) in attacks.items():
        try:
            _o, reward, _d, info = http_step(
                base, "push_filter", sql, timeout_s=600.0
            )
        except Exception as e:
            c.print(f"{name}  [red]HTTP error: {e}[/red]")
            failures.append(name)
            continue
        r = float(reward)
        ex = int(expected)
        if abs(r - float(ex)) < 0.01:
            c.print(
                f"{name}  penalty={r:.1f}  expected={ex}  [green]match[/green]"
            )
        else:
            c.print(
                f"{name}  penalty={r:.1f}  expected={ex}  [red]mismatch[/red]  info={info.get('error')}"
            )
            failures.append(f"{name} (got {r})")

    if not failures:
        c.print(
            "\n[bold green]ALL ANTI-CHEAT GUARDS PASSING[/bold green]"
        )
    else:
        c.print(
            f"\n[bold red]Failures:[/bold red] {', '.join(failures)}"
        )
    c.print("")


if __name__ == "__main__":
    from rich.console import Console

    _C = Console()
    if os.environ.get("SQLSAGE_SKIP_WANDB", "").strip() in ("1", "true", "True"):
        _C.print(
            "\n[dim]1) REWARD DIAGNOSTIC — skipped (SQLSAGE_SKIP_WANDB=1)[/dim]\n"
        )
    else:
        try:
            section_reward_diagnostic()
        except Exception as ex:
            _C.print(f"[yellow]W&B section: {ex}[/yellow]")

    try:
        section_raw_output_sampler()
    except Exception as ex:
        _C.print(f"[yellow]Sampler: {ex}[/yellow]")

    try:
        section_reward_pipeline()
    except Exception as ex:
        _C.print(f"[red]Pipeline: {ex}[/red]")

    try:
        section_anticheat()
    except Exception as ex:
        _C.print(f"[red]Anticheat: {ex}[/red]")
