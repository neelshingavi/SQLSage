#!/usr/bin/env python3
"""
Run TPC-H benchmark queries: baseline times vs untrained (base) vs trained (LoRA) rewrites.

  cd sqlsage-env && python run_benchmark.py

Requires: PostgreSQL with TPC-H data, GPU recommended, transformers, torch, (peft for LoRA).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# --- Config ---

PG = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", "5432")),
    "user": os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ.get("POSTGRES_PASSWORD", "sqlsage"),
    "dbname": os.environ.get("POSTGRES_DB", "sqlsage"),
}
BASE_MODEL = os.environ.get("SQLSAGE_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
CHECKPOINT = Path(
    os.environ.get("SQLSAGE_CHECKPOINT", "sqlsage-grpo/checkpoint-latest")
)
RESULTS_DIR = Path(os.environ.get("SQLSAGE_RESULTS_DIR", "results"))
OUT_JSON = RESULTS_DIR / "benchmark_results.json"
EXPLAIN_TIMEOUT_MS = int(os.environ.get("SQLSAGE_TIMEOUT_MS", "120000"))

# Map standard TPC-H names to project tasks (Q1, Q3, Q5, Q6, Q14 shapes from curriculum).
def _tpch_query_map() -> dict[str, str]:
    from sqlsage.tasks.level1 import LEVEL1_TASKS
    from sqlsage.tasks.level2 import LEVEL2_TASKS

    return {
        "Q1": LEVEL1_TASKS[0].query,
        "Q3": LEVEL2_TASKS[0].query,
        "Q5": LEVEL2_TASKS[1].query,
        "Q6": LEVEL1_TASKS[1].query,
        "Q14": LEVEL1_TASKS[2].query,
    }


# --- DB ---

def _connect():
    import psycopg2

    return psycopg2.connect(
        host=PG["host"],
        port=PG["port"],
        user=PG["user"],
        password=PG["password"],
        dbname=PG["dbname"],
    )


def _result_hash_from_rows(rows: list[tuple[Any, ...]]) -> str:
    payload = json.dumps(rows, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _execute_time_ms(conn, sql: str, timeout_s: int = 120) -> tuple[float, str, int]:
    cur = conn.cursor()
    try:
        cur.execute(f"SET statement_timeout = {int(timeout_s * 1000)}")
        cur.execute("BEGIN READ ONLY")
        t0 = time.perf_counter()
        cur.execute(sql)
        rows = cur.fetchall()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        conn.rollback()
        h = _result_hash_from_rows(rows)
        n = len(rows)
        return float(elapsed_ms), h, n
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def _median_ms_of_three(conn, sql: str) -> float:
    times: list[float] = []
    for _ in range(3):
        ms, _, _ = _execute_time_ms(conn, sql)
        times.append(ms)
    return float(statistics.median(times))


# --- Model ---

def _load_tokenizer(name_or_path: str | Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(name_or_path), trust_remote_code=True)


def _load_causal(name_or_path: str | Path, torch_dtype: Any = None) -> Any:
    from transformers import AutoModelForCausalLM
    import torch

    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return AutoModelForCausalLM.from_pretrained(
        str(name_or_path),
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )


def _load_trained() -> Any:
    """Base model + LoRA from checkpoint, or full merged weights at CHECKPOINT."""
    import torch
    from transformers import AutoModelForCausalLM

    p = Path(CHECKPOINT)
    if not p.exists():
        raise FileNotFoundError(f"Missing trained checkpoint: {p}")

    if (p / "adapter_config.json").exists() or (p / "adapter_model.safetensors").exists():
        try:
            from peft import PeftModel
        except ImportError as e:
            raise RuntimeError(
                "Trained model is a LoRA adapter. Install: pip install peft"
            ) from e
        base = _load_causal(BASE_MODEL)
        return PeftModel.from_pretrained(base, str(p))

    return AutoModelForCausalLM.from_pretrained(
        str(p),
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        device_map="auto" if torch.cuda.is_available() else None,
    )


def _build_observation_for_prompt(conn, query: str) -> dict[str, Any]:
    from sqlsage.dataset import make_prompt
    from sqlsage.env import SCHEMA_SUMMARY
    from sqlsage.explain_parser import get_explain_dict

    plan = get_explain_dict(conn, query, timeout_ms=EXPLAIN_TIMEOUT_MS)
    ms, res_hash, _ = _execute_time_ms(conn, query)
    return {
        "original_query": query,
        "explain_plan": plan,
        "execution_ms": float(ms),
        "result_hash": res_hash,
        "schema_context": SCHEMA_SUMMARY,
    }, make_prompt(
        {
            "original_query": query,
            "explain_plan": plan,
            "execution_ms": float(ms),
            "result_hash": res_hash,
            "schema_context": SCHEMA_SUMMARY,
        }
    )


def _parse_generation_json(text: str) -> str:
    """Return rewritten_query or raise."""
    s = (text or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", s, re.IGNORECASE)
    blob = m.group(1).strip() if m else s
    start = blob.find("{")
    if start < 0:
        raise ValueError("no_json")
    import json

    from json import JSONDecoder

    dec = JSONDecoder()
    data, _ = dec.raw_decode(blob, start)
    if not isinstance(data, dict):
        raise ValueError("not_a_dict")
    rq = str(data.get("rewritten_query", "")).strip()
    if not rq:
        raise ValueError("no_rewritten_query")
    return rq


def _sql_fallback(text: str) -> str:
    m = re.search(
        r"```sql\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL
    )
    if m and m.group(1).strip():
        return m.group(1).strip()
    return text.strip() or "SELECT 1"


def _generate_rewrite(model: Any, tokenizer: Any, user_prompt: str) -> str:
    import torch

    if hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    ) is not None:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = user_prompt

    enc = tokenizer(text, return_tensors="pt")
    dev = next(model.parameters()).device
    enc = {k: v.to(dev) for k, v in enc.items()}

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=getattr(
                tokenizer, "pad_token_id", tokenizer.eos_token_id
            ),
        )
    in_len = enc["input_ids"].shape[-1]
    gen = out[0, in_len:]
    dec = tokenizer.decode(gen, skip_special_tokens=True)
    try:
        return _parse_generation_json(dec)
    except Exception:
        return _sql_fallback(dec)


# --- Record ---

@dataclass
class QueryBench:
    query_id: str
    baseline_ms: float
    untrained_ms: float
    untrained_rewrite: str
    trained_ms: float
    trained_rewrite: str
    hash_match_untrained: bool
    hash_match_trained: bool


def run() -> int:
    from rich.console import Console
    from rich.table import Table

    c = Console()
    c.print(
        f"\n[bold]SQLSage TPC-H benchmark[/bold]  base=[cyan]{BASE_MODEL}[/cyan]  "
        f"trained=[cyan]{CHECKPOINT}[/cyan]\n"
    )

    queries = _tpch_query_map()
    conn = _connect()
    # Baseline / observation uses one timing pass per query inside _build; benchmark baseline is separate 3x median

    c.print("Loading [bold]untrained[/bold] (base) model…")
    tok = _load_tokenizer(BASE_MODEL)
    model_u = _load_causal(BASE_MODEL)
    c.print("Loading [bold]trained[/bold] model…")
    try:
        model_t = _load_trained()
    except Exception as e:
        c.print(f"[red]Trained model load failed: {e}[/red]")
        return 1

    results: list[QueryBench] = []
    for qid, qsql in queries.items():
        c.print(f"  [dim]→ {qid}…[/dim]")
        try:
            baseline = _median_ms_of_three(conn, qsql)
        except Exception as e:
            c.print(f"[red]    baseline EX failed for {qid}: {e}[/red]")
            continue

        try:
            obs, user_prompt = _build_observation_for_prompt(conn, qsql)
        except Exception as e:
            c.print(f"[red]    EXPLAIN/prompt for {qid}: {e}[/red]")
            continue

        h_orig = str(obs.get("result_hash", ""))

        u_ms = t_ms = float("nan")
        u_ok = t_ok = False
        u_rewrite, t_rewrite = "", ""

        try:
            u_rewrite = _generate_rewrite(model_u, tok, user_prompt)
        except Exception as ex:
            c.print(f"    [yellow]untrained gen: {ex}[/yellow]")
        try:
            t_rewrite = _generate_rewrite(model_t, tok, user_prompt)
        except Exception as ex:
            c.print(f"    [yellow]trained gen: {ex}[/yellow]")

        if u_rewrite.strip():
            try:
                times_u: list[float] = []
                hus: list[str] = []
                for _ in range(3):
                    m, hu, _ = _execute_time_ms(conn, u_rewrite)
                    times_u.append(m)
                    hus.append(hu)
                u_ms = float(statistics.median(times_u))
                u_ok = all(h == h_orig for h in hus)
            except Exception as ex:
                c.print(f"    [yellow]untrained exec: {ex}[/yellow]")

        if t_rewrite.strip():
            try:
                times_t: list[float] = []
                hts: list[str] = []
                for _ in range(3):
                    m, ht, _ = _execute_time_ms(conn, t_rewrite)
                    times_t.append(m)
                    hts.append(ht)
                t_ms = float(statistics.median(times_t))
                t_ok = all(h == h_orig for h in hts)
            except Exception as ex:
                c.print(f"    [yellow]trained exec: {ex}[/yellow]")

        results.append(
            QueryBench(
                query_id=qid,
                baseline_ms=baseline,
                untrained_ms=u_ms,
                untrained_rewrite=u_rewrite[:2000],
                trained_ms=t_ms,
                trained_rewrite=t_rewrite[:2000],
                hash_match_untrained=u_ok,
                hash_match_trained=t_ok,
            )
        )
    try:
        conn.close()
    except Exception:
        pass

    if not results:
        c.print("[red]No benchmark rows — fix DB and queries.[/red]")
        return 1

    # Table
    table = Table(
        title="SQLSage benchmark: trained vs untrained rewrites (median of 3 runs, ms)"
    )
    table.add_column("Query")
    table.add_column("Baseline", justify="right")
    table.add_column("Untrained", justify="right")
    table.add_column("Trained", justify="right")
    table.add_column("Speedup vs untrained", justify="right")
    table.add_column("Hash U/T", justify="left")

    speedups: list[float] = []
    for r in results:
        su: float
        if (
            r.untrained_ms == r.untrained_ms
            and r.trained_ms == r.trained_ms
            and r.trained_ms > 0
        ):
            su = r.untrained_ms / max(r.trained_ms, 1e-6)
        else:
            su = float("nan")
        if su == su and su > 0:
            speedups.append(su)
        sp = f"{su:.1f}x" if su == su and su > 0 else "—"
        hmark = f"{'✓' if r.hash_match_untrained else '✗'}/{'✓' if r.hash_match_trained else '✗'}"
        table.add_row(
            r.query_id,
            f"{r.baseline_ms:.0f}ms",
            f"{r.untrained_ms:.0f}ms" if r.untrained_ms == r.untrained_ms else "—",
            f"{r.trained_ms:.0f}ms" if r.trained_ms == r.trained_ms else "—",
            sp,
            hmark,
        )

    pass_count = 0
    for r in results:
        if not (r.untrained_ms == r.untrained_ms and r.trained_ms == r.trained_ms):
            continue
        if r.trained_ms <= 0 or r.untrained_ms <= 0:
            continue
        ratio = r.untrained_ms / r.trained_ms
        if (
            ratio >= 2.0
            and r.hash_match_untrained
            and r.hash_match_trained
        ):
            pass_count += 1

    avg_su = (
        statistics.mean([s for s in speedups if s and s == s and s > 0])
        if speedups
        else 0.0
    )
    table.add_row(
        "AVERAGE speedup (trained vs untrained)",
        "—",
        "—",
        "—",
        f"[bold]{avg_su:.1f}x[/bold]" if avg_su and avg_su == avg_su else "—",
        "",
    )
    c.print(table)

    # Save JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": 1,
        "base_model": BASE_MODEL,
        "checkpoint": str(CHECKPOINT),
        "postgres": {k: v for k, v in PG.items() if k != "password"},
        "assert_rule": "trained >= 2x untrained (median ms), same hash, on >= 3/5 queries",
        "queries": [asdict(x) for x in results],
        "summary": {
            "avg_speedup_trained_vs_untrained": avg_su,
            "assert_pass_3of5_2x": pass_count >= 3,
            "count_passing_2x": pass_count,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    c.print(
        f"\n[green]Wrote {OUT_JSON}[/green]\n"
    )

    if pass_count >= 3:
        c.print(
            f"[bold green]BENCHMARK PASSED[/bold green]  ({pass_count}/5 queries with ≥2× speedup, matching hashes)\n"
        )
        return 0
    c.print(
        f"[bold red]BENCHMARK FAILED — check training[/bold red]  "
        f"({pass_count}/5 met ≥2× with hash match)\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(run())
