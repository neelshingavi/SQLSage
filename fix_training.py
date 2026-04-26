#!/usr/bin/env python3
"""
Auto-fix handlers for common SQLSage training failure modes (project doc Section 12).

  cd sqlsage-env && python fix_training.py --issue <mode>

  python fix_training.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "train.py"
DATASET = ROOT / "sqlsage" / "dataset.py"
ANTICHEAT = ROOT / "sqlsage" / "anti_cheat.py"
ENV_PY = ROOT / "sqlsage" / "env.py"
GRPO_DIR = ROOT / "sqlsage-grpo"
CURR_FILE = ROOT / "sqlsage-curriculum.json"

MARK_T = "SQLSage-FIX: flat-reward-temperature"
MARK_RC = "SQLSage-FIX: result-changed-guard"
MARK_SY = "SQLSage-FIX: syntax-fewshot"
MARK_OOM = "SQLSage-FIX: oom"


def _c():
    from rich.console import Console

    return Console()


def _wb_run() -> Any | None:
    run_id = os.environ.get("WANDB_RUN_ID", "").strip()
    if not run_id:
        return None
    import wandb

    project = os.environ.get("WANDB_PROJECT", "sqlsage-grpo").strip()
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    if not entity:
        try:
            api = wandb.Api()
            u = api.viewer
            if u is not None and getattr(u, "default_entity", None):
                entity = str(u.default_entity)  # type: ignore[union-attr]
        except Exception:
            return None
    if not entity:
        return None
    return wandb.Api().run(f"{entity}/{project}/{run_id}")


def _cp() -> Any | None:
    try:
        import psycopg2

        return psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "sqlsage"),
            dbname=os.environ.get("POSTGRES_DB", "sqlsage"),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# flat_reward
# ---------------------------------------------------------------------------


def _last_ckpt() -> Path | None:
    if not GRPO_DIR.is_dir():
        return None
    p = GRPO_DIR / "checkpoint-latest"
    if p.is_symlink() or p.is_file():
        return p
    cps = sorted(GRPO_DIR.glob("checkpoint-*"), key=lambda x: x.stat().st_mtime, reverse=True)
    return cps[0] if cps else None


def _lev(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    return dp[n][m]


def _avg_edit(five: list[str]) -> float:
    s, t = 0, 0
    for i in range(len(five)):
        for j in range(i + 1, len(five)):
            s += _lev(five[i], five[j])
            t += 1
    return s / max(t, 1)


def issue_flat_reward() -> int:
    C = _c()
    C.print(
        "\n[bold]Model generating same query — checking diversity[/bold]\n"
    )
    ck = _last_ckpt()
    prompt = os.environ.get(
        "SQLSAGE_FLAT_REWARD_TEST_PROMPT",
        "You are SQLSage. Return JSON: {\"action\":\"push_filter\",\"rewritten_query\":\"SELECT 1\"} only.",
    )
    if ck and ck.exists():
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch

            C.print(f"Checkpoint: [cyan]{ck}[/cyan]\n")
            tok = AutoTokenizer.from_pretrained(str(ck), trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                str(ck), trust_remote_code=True, torch_dtype=torch.float16
            )
            if torch.cuda.is_available():
                mdl = mdl.cuda()
            ppl = pipeline(
                "text-generation", model=mdl, tokenizer=tok, return_full_text=False
            )
            out = []
            for i in range(5):
                g = ppl(
                    prompt,
                    max_new_tokens=160,
                    do_sample=True,
                    temperature=0.9 + 0.03 * i,
                )
                t = g[0].get("generated_text", "") if g else ""
                out.append(t)
            avg = _avg_edit(out)
            C.print(
                f"Pairwise mean edit distance (5 samples): [bold]{avg:.1f}[/bold]\n"
            )
            if avg < 50:
                C.print(
                    "[red]LOW DIVERSITY — increase temperature in GRPOConfig.[/red]\n"
                )
            else:
                C.print(
                    "[green]Diversity OK — check reward pipeline / env.[/green]\n"
                )
        except Exception as e:
            C.print(f"[yellow]Skip generation ({e}). Patching train.py only.[/yellow]\n")
    else:
        C.print(f"[dim]No checkpoint in {GRPO_DIR}.[/dim]\n")

    if not TRAIN.exists():
        C.print(f"[red]Missing {TRAIN}[/red]")
        return 1
    t = TRAIN.read_text()
    if MARK_T in t:
        C.print(f"[dim]Already patched ({MARK_T}).[/dim]\n")
        return 0
    t2, n = re.subn(
        r"temperature=([\d.]+),",
        lambda m: f"temperature=1.2,  # {MARK_T} was {m.group(1)}",
        t,
        count=1,
    )
    if n:
        TRAIN.write_text(t2)
        C.print(
            f"[green]Patched GRPOConfig in train.py: temperature=1.2 ({MARK_T})[/green]\n"
        )
    else:
        C.print("[red]Could not find temperature= in train.py.[/red]")
    return 0


# ---------------------------------------------------------------------------
# result_changed
# ---------------------------------------------------------------------------


def _clause_hunt(o: str, n: str) -> str:
    a, b = o.upper()[:2000], n.upper()[:2000]
    for lab, w in (("GROUP BY", "GROUP BY"), ("WHERE", "WHERE"), ("JOIN", "JOIN")):
        if a.count(w) != b.count(w) or (w in a and a[a.find(w) : a.find(w) + 40] != b[b.find(w) : b.find(w) + 40] if w in b else True):
            return lab
    return "other"


def issue_result_changed() -> int:
    C = _c()
    C.print("\n[bold]result_changed: examples + prompt guard[/bold]\n")
    from sqlsage.tasks.level1 import LEVEL1_TASKS

    orig, rew = os.environ.get("SQLSAGE_RC_ORIG", ""), os.environ.get(
        "SQLSAGE_RC_REW", ""
    )
    if not orig:
        orig = LEVEL1_TASKS[0].query.strip()
    if not rew:
        rew = re.sub(
            r"(?is)WHERE\s+",
            "WHERE 1=0 AND ",
            orig,
            count=1,
        )

    conn = _cp()
    if conn:
        cur = conn.cursor()
        for label, a, b in (("orig", orig, rew),):
            C.print(
                f"[dim]{label}: first 5 rows of original / rewritten (if both run)[/dim]"
            )
            for q, name in ((a, "original"), (b, "rewritten")):
                try:
                    cur.execute(q)
                    C.print(
                        f"  {name}: {cur.fetchall()[:5]!r}"
                    )
                except Exception as e:
                    C.print(f"  {name}: [red]{e}[/red]")
                try:
                    conn.rollback()
                except Exception:
                    pass
        cur.close()
        conn.close()
    C.print(f"Clause heuristic: [cyan]{_clause_hunt(orig, rew)}[/cyan]\n")

    if not DATASET.exists():
        return 1
    s = DATASET.read_text()
    if MARK_RC in s:
        C.print(f"[dim]dataset.py already has {MARK_RC}[/dim]\n")
        return 0
    needle = (
        '        "Given a SQL query, explain plan summary, runtime, and schema context, "'
    )
    if needle not in s:
        C.print(
            f"[red]make_prompt() layout changed; add guard manually.[/red]"
        )
        return 1
    line = (
        f'        "CRITICAL: Do not modify GROUP BY columns, WHERE conditions that affect row identity, or JOIN keys. The result set must be byte-for-byte identical.\\n"  # {MARK_RC}\n'
        + needle
    )
    s2 = s.replace(needle, line, 1)
    DATASET.write_text(s2)
    C.print(
        f"[green]Patched {DATASET} make_prompt with CRITICAL line ({MARK_RC})[/green]\n"
    )
    return 0


# ---------------------------------------------------------------------------
# syntax_error
# ---------------------------------------------------------------------------


def issue_syntax_error() -> int:
    C = _c()
    C.print(
        "\n[bold]syntax_error: W&B + few-shot in train prompt[/bold]\n"
    )
    w = _wb_run()
    if w:
        c = 0
        for h in w.scan_history():
            if float(h.get("penalty/syntax_error", 0) or 0) > 0 and c < 5:
                C.print(
                    f"[dim]W&B step with syntax flag: {h!r}[/dim]"
                )
                c += 1
    if not TRAIN.exists():
        return 1
    t = TRAIN.read_text()
    if MARK_SY in t:
        C.print(f"[dim]train.py already has {MARK_SY}[/dim]\n")
        return 0
    old = """    prompt_rows = [
        {
            "prompt": (
                "You are SQLSage. Rewrite the SQL query for better performance while preserving result semantics. "
                "Return only SQL (or a fenced ```sql block)."
            )
        }
        for _ in range(256)
    ]"""
    new = f"""    prompt_rows = [
        {{
            "prompt": (
                'Valid JSON: {{"action": "push_filter", "rewritten_query": "SELECT 1"}}.  # {MARK_SY}
                "You are SQLSage. Rewrite the SQL query for better performance while preserving result semantics. "
                "Return only SQL (or a fenced ```sql block)."
            )
        }}
        for _ in range(256)
    ]"""
    if old not in t:
        C.print(
            f"[red]train.py prompt_rows block not found (layout changed).[/red]"
        )
        return 1
    TRAIN.write_text(t.replace(old, new, 1))
    C.print(
        f"[green]Patched train.py prompt_rows with JSON few-shot ({MARK_SY})[/green]\n"
    )
    return 0


# ---------------------------------------------------------------------------
# oom
# ---------------------------------------------------------------------------


def issue_oom() -> int:
    C = _c()
    C.print(
        "\n[bold]oom: per_device_train_batch_size, max_completion_length, grad accum[/bold]\n"
    )
    tpath = TRAIN if TRAIN.exists() else ROOT / "train_sqlsage.py"
    if not tpath.exists():
        C.print(
            f"[red]No train.py / train_sqlsage.py under {ROOT}[/red]"
        )
        return 1
    t = tpath.read_text()
    if MARK_OOM in t:
        C.print(
            f"[dim]Already OOM-tagged in {tpath.name}.[/dim]\n"
        )
        return 0
    t = re.sub(
        r"per_device_train_batch_size=\d+",
        f"per_device_train_batch_size=2  # {MARK_OOM} reduce from 4 if present",
        t,
        count=1,
    )
    t = re.sub(
        r"max_completion_length=\d+",
        f"max_completion_length=256  # {MARK_OOM} was 512",
        t,
        count=1,
    )
    t = re.sub(
        r"gradient_accumulation_steps=\d+",
        f"gradient_accumulation_steps=8  # {MARK_OOM} was 4 to keep effective batch",
        t,
        count=1,
    )
    tpath.write_text(t)
    C.print(
        "[green]OOM fix applied. Effective batch: 2×8 = 16 microbatches (match prior 4×4).[/green]\n"
    )
    C.print(
        f"[dim]Patched: {tpath}[/dim]\n"
    )
    return 0


# ---------------------------------------------------------------------------
# env_500
# ---------------------------------------------------------------------------


def issue_env_500() -> int:
    C = _c()
    C.print("\n[bold]env_500: docker + health[/bold]\n")
    for name in ("sqlsage-pg",):
        try:
            subs = subprocess.run(
                ["docker", "ps", "-a", "--format", "{{.Names}}"],
                text=True,
                capture_output=True,
            )
            if name in (subs.stdout or ""):
                subprocess.run(
                    ["docker", "start", name],
                    capture_output=True,
                )
        except FileNotFoundError:
            C.print(
                "[yellow]docker not found.[/yellow]\n"
            )
            break
    for u in (
        "http://localhost:8000/health",
        "http://127.0.0.1:8000/health",
        "http://localhost:8000/",
    ):
        try:
            p = subprocess.run(
                ["curl", "-sS", "-i", u],
                capture_output=True,
                text=True,
                timeout=8,
            )
            C.print(
                f"[cyan]{u}[/cyan] →\n[dim]{(p.stdout or p.stderr)[:500]}…[/dim]\n"
            )
        except (FileNotFoundError, OSError) as e:
            C.print(
                f"[red]curl: {e}[/red]\n"
            )
    C.print(
        "[dim]On HTTP 500, run: docker ps -q | head -1 | xargs -I%% docker logs --tail 20 %%[/dim]\n"
    )
    return 0


# ---------------------------------------------------------------------------
# slow_episodes
# ---------------------------------------------------------------------------


def issue_slow_episodes() -> int:
    C = _c()
    C.print(
        "\n[bold]slow_episodes: Q1 + optional dbgen -s 0.1[/bold]\n"
    )
    from sqlsage.explain_parser import get_explain_dict
    from sqlsage.tasks.level1 import LEVEL1_TASKS

    q1 = LEVEL1_TASKS[0].query
    conn = _cp()
    if not conn:
        C.print(
            f"[red]Postgres unavailable.[/red]"
        )
        return 1
    t0 = time.perf_counter()
    get_explain_dict(conn, q1, 120_000)
    t_ex = (time.perf_counter() - t0) * 1000.0
    cur = conn.cursor()
    t0 = time.perf_counter()
    try:
        cur.execute(q1)
        _ = cur.fetchall()
    finally:
        conn.rollback()
    t_w = (time.perf_counter() - t0) * 1000.0
    cur.close()
    conn.close()
    C.print(
        f"EXPLAIN path ~{t_ex:.0f} ms, execute ~{t_w:.0f} ms\n"
    )
    if max(t_ex, t_w) > 2000:
        C.print(
            "[red]DB slow — recreating with SF=0.1 (if TPC-H toolkit installed).[/red]\n"
        )
    else:
        C.print(
            "[green]Under 2s; no data rebuild.[/green]\n"
        )
        return 0
    for cmd in (["dbgen", "-s", "0.1"],):
        try:
            p = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=600,
            )
            C.print(
                f"dbgen exit {p.returncode} stderr={p.stderr[:200]}\n"
            )
        except FileNotFoundError:
            C.print(
                "[yellow]dbgen not in PATH.[/yellow]\n"
            )
    return 0


# ---------------------------------------------------------------------------
# reward_hacking (5 vectors like monitor_training)
# ---------------------------------------------------------------------------


def _remove_where(s: str) -> str:
    t = re.sub(
        r"(?is)WHERE\s+.+?(?=\bGROUP\s+BY\b)", "", s, count=1
    )
    return t if t != s else s


def _attacks(oq: str) -> list[tuple[str, str, int]]:
    b = oq.strip().rstrip(";")
    return [
        (
            "LIMIT 0",
            f"SELECT * FROM (\n{b}\n) AS _a LIMIT 0",
            -20,
        ),
        (
            "WHERE 1=0",
            f"SELECT * FROM (\n{b}\n) AS _a WHERE 1=0",
            -20,
        ),
        (
            "remove WHERE",
            _remove_where(oq),
            -20,
        ),
        (
            "DDL",
            "CREATE INDEX IF NOT EXISTS i_fix_tr ON lineitem (l_orderkey)",
            -15,
        ),
        (
            "hardcode l_orderkey=1",
            re.sub(
                r"(?is)(WHERE)\s+",
                r"\1 l_orderkey = 1 AND ",
                b,
                count=1,
            )
            or "SELECT 1 FROM lineitem WHERE l_orderkey=1",
            -20,
        ),
    ]


def issue_reward_hacking() -> int:
    C = _c()
    C.print(
        "\n[bold]reward_hacking: 5 anti-cheat probes[/bold]\n"
    )
    base = os.environ.get("SQLSAGE_ENV_URL", "").strip()
    if not base:
        C.print(
            f"[red]Set SQLSAGE_ENV_URL.[/red]"
        )
        return 1
    from sqlsage.training.http_sqlsage_client import http_reset, http_step

    try:
        obs = http_reset(base, seed=0)
    except Exception as e:
        C.print(
            f"[red]reset: {e}[/red]"
        )
        return 1
    oq = str(obs.get("original_query", ""))
    rows = _attacks(oq)
    for name, sql, ex in rows:
        try:
            _o, rw, _d, _ = http_step(
                base, "push_filter", sql, timeout_s=400.0
            )
        except Exception as e:
            C.print(
                f"{name} [red]{e}[/red]"
            )
            continue
        ok = abs(float(rw) - ex) < 0.01
        C.print(
            f"{name}: got {float(rw):.1f}  expect {ex}  "
            f"({'[green]OK[/green]' if ok else '[red]FAIL[/red]'})"
        )
        if not ok and ANTCHEAT.exists():
            C.print(
                f"  [dim]Relevant: {ANTICHEAT} — validate_read_only_sql, BLOCKED_KEYWORDS. "
                f"Env: {ENV_PY} result_hash compare.[/dim]"
            )
    return 0


# ---------------------------------------------------------------------------
# curriculum_stuck
# ---------------------------------------------------------------------------


def issue_curriculum_stuck() -> int:
    C = _c()
    C.print(
        "\n[bold]curriculum_stuck: W&B L1 + curriculum file[/bold]\n"
    )
    rate: float | None = None
    w = _wb_run()
    if w:
        try:
            d = w.history(
                keys=["task_level", "reward/mean"],
                samples=200,
                pandas=True,
            )
            if d is not None and len(d) and "task_level" in d.columns:
                m = d["task_level"] == 1.0
                if m.any() and "reward/mean" in d.columns:
                    sub = d.loc[m, "reward/mean"]
                    rate = float((sub > 0.0).mean())
        except Exception:
            pass
    if rate is not None:
        C.print(
            f"W&B: Level-1 'reward>0' rate ≈ [bold]{100*rate:.1f}%[/bold]\n"
        )
    else:
        C.print(
            "[yellow]W&B: could not read task_level + reward (set WANDB_*).[/yellow]\n"
        )

    gate = rate is not None and rate < 0.30
    if rate is None:
        gate = os.environ.get("SQLSAGE_FORCE_CURRICULUM_GATE", "") in (
            "1",
            "true",
        )
    st = {
        "gating": gate,
        "l1_min": 200,
        "l1_count": 0,
        "unlocked_max_level": 3,
    }
    CURR_FILE.write_text(json.dumps(st, indent=2) + "\n")
    C.print(
        f"Wrote [cyan]{CURR_FILE}[/cyan]:\n{json.dumps(st, indent=2)}"
    )
    C.print(
        "\n[green]If gating: env.reset() uses Level 1 for the first l1_count<=l1_min resets, then random 1..unlocked_max_level.[/green]\n"
    )
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="SQLSage: Section-12 style training failure fixes",
    )
    p.add_argument(
        "--issue",
        required=True,
        choices=[
            "flat_reward",
            "result_changed",
            "syntax_error",
            "oom",
            "env_500",
            "slow_episodes",
            "reward_hacking",
            "curriculum_stuck",
        ],
    )
    a = p.parse_args()
    return {
        "flat_reward": issue_flat_reward,
        "result_changed": issue_result_changed,
        "syntax_error": issue_syntax_error,
        "oom": issue_oom,
        "env_500": issue_env_500,
        "slow_episodes": issue_slow_episodes,
        "reward_hacking": issue_reward_hacking,
        "curriculum_stuck": issue_curriculum_stuck,
    }[a.issue]()


if __name__ == "__main__":
    sys.exit(main() or 0)
