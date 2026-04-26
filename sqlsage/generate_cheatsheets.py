"""
Generate one-page printable cheat sheets (Markdown + HTML) for each SQLSage role.

  cd sqlsage-env && python -m sqlsage.generate_cheatsheets

Output: ./cheatsheets/person{1,2,3}_*.{md,html} — no third-party dependencies.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cheatsheets"

# --- Markdown bodies (Section 13 style + commands from project phases) ---

PERSON1_MD = r"""# SQLSage — Person 1: Environment Engineer
**Deadline: Hour 8 = HF Space live**

## Setup Commands
| Command | What it does |
|---------|-------------|
| `openenv init sqlsage-env` | Creates the scaffold (run from parent of the repo dir) |
| `docker compose up -d` | Start Postgres (see `docker-compose.yml`; default **port 5433** in README) |
| `uvicorn sqlsage.app:app --reload --port 8000` | Runs environment locally (set `POSTGRES_HOST` / `POSTGRES_PORT`) |
| `openenv push your-username/sqlsage-env` | Deploys to HF Spaces |
| `docker run --name sqlsage-pg -e POSTGRES_PASSWORD=sqlsage -e POSTGRES_DB=sqlsage -p 5432:5432 -d postgres:16` | Single-container Postgres 16 (alt. to compose) |
| `curl http://localhost:8000/reset` | Tests reset endpoint (local) |
| `curl -sS "$SQLSAGE_ENV_URL/reset"` | Tests deployed Space (set URL from `.env`) |
| `psql "host=127.0.0.1 port=5433 user=postgres dbname=sqlsage password=sqlsage" -c 'SELECT version();'` | Verify DB (project default port **5433**) |
| `openenv validate` | OpenEnv project validation |
| `python -m sqlsage.run p1-serve` | **Alias** → uvicorn (from `sqlsage.run`) |
| `python -m sqlsage.run p1-test` | **Alias** → curl local `/reset` \| `json.tool` |
| `python -m sqlsage.run p1-push` / `p1-deploy` | **Alias** → push + optional HF curl |

## TPC-H Data Load
| Command | What it does |
|---------|-------------|
| `git clone https://github.com/electrum/tpch-dbgen.git && cd tpch-dbgen && make` | Clone TPC-H tools |
| `./dbgen -s 1` | Generate SF=1 data (~1GB, ~2 min) |
| `./dbgen -s 0.1` | Generate SF=0.1 (fast, for training) |
| `psql -U postgres -d sqlsage -f dss.ddl` | Create schema (paths per your load process) |
| TPC-H bootstrap in Docker / HF image | `sql/bootstrap_tpch_schema.sql` auto-runs in Space; load data separately for real rewards |

## Drop Indexes (Creates Optimization Headroom)
```bash
psql "host=127.0.0.1 port=5433 user=postgres dbname=sqlsage password=sqlsage" \
  -f sql/scripts/drop_curriculum_indexes.sql
psql -U postgres -d sqlsage -c "DROP INDEX IF EXISTS idx_orders_status;"
psql -U postgres -d sqlsage -c "DROP INDEX IF EXISTS idx_lineitem_shipdate;"
```

## Stress & Levels (phases: curriculum / env hardening)
| Command | What it does |
|---------|-------------|
| `POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=5433 python scripts/stress_env.py --episodes 50 --levels 2 --identity-step` | Stress `reset` / optional identity `step` (no GRPO) |
| `SQLSageEnv(tasks=tasks_for_levels(2))` | Level 2 only (see `sqlsage.tasks`) |
| `GET /health` | Health check (OpenEnv HTTP) |
| `GET /state`, `GET /schema`, `GET /metadata` | Session / schema metadata |

## Troubleshooting
| Symptom | Fix |
|---------|-----|
| `curl /reset` returns 500 | `docker ps` / `docker compose ps` — DB + API up; check logs |
| HF Space build failing | `Dockerfile` + entrypoint: Postgres as service, `uvicorn` on `$PORT` (7860 on HF) |
| TPC-H load too long | Use SF=0.1 for dev; SF=1 for final demo; raise `SQLSAGE_TIMEOUT_MS` if needed |
| Intermittent 500 under load | OpenEnv uses a **lock** for reset/step; avoid parallel `step` to same process |

## Hour Gates
- Hour 1: psql connects ✓
- Hour 4: `curl /reset` returns JSON ✓
- Hour 6: `step()` returns `{reward, done, state}` ✓
- Hour 8: **HF Space live** — **SHARE URL WITH TEAM** (`SQLSAGE_ENV_URL` in Colab) ✓
"""

PERSON2_MD = r"""# SQLSage — Person 2: Reward & Verifier Engineer
**Your job: reward pipeline works, no reward hacking, plots committed by Hour 16**

## Core Commands
| Command | What it does |
|---------|-------------|
| `pytest tests/ -v` | All unit tests |
| `pytest tests/test_reward.py -v` | Reward tests only |
| `pytest tests/test_explain_parser.py -v` | Parser tests only |
| `pytest tests/test_anti_cheat.py -v` | Anti-cheat tests only |
| `wandb login` | Authenticate W&B |
| `python monitor_training.py` | Live reward diagnostics (when configured) |
| `python plots/generate_plots.py` | **Actual path** — generate PNGs in `plots/` |
| `python -m sqlsage.run p2-reward` | **Alias** → quick `compute_reward` demo + breakdown |
| `python -m sqlsage.run p2-plots` | **Alias** → `plots/generate_plots.py` |
| `python -m sqlsage.run p2-anticheat` | **Alias** → `fix_training.py --issue reward_hacking` |
| `python -m sqlsage.status_checker --json` | Milestone + reward/plots/git checks (JSON) |

## Quick Reward Test
`compute_reward` returns **(normalized ∈ [-1,1], breakdown dict)** — use both:

```python
from sqlsage.reward import compute_reward, format_reward_breakdown
new_plan = {
    "seq_scans": 0, "nested_loops": 0, "hash_joins": 1, "total_cost": 1000,
    "rows": 1e5, "node_type": "x",
}
old_plan = {
    "seq_scans": 2, "nested_loops": 1, "hash_joins": 0, "total_cost": 50000,
    "rows": 1e6, "node_type": "y",
}
n, b = compute_reward(8420.0, 310.0, new_plan, old_plan, step_number=0)
print("normalized:", n)
print(format_reward_breakdown(b))
# Raw sum can be large; see breakdown["raw"] and breakdown["normalized"]
```

## Quick EXPLAIN Test
```bash
psql -h 127.0.0.1 -p 5433 -U postgres -d sqlsage -c \
"EXPLAIN (ANALYZE, FORMAT JSON) SELECT * FROM orders WHERE orderstatus='F' LIMIT 100;"
```

## Anti-Cheat Attack Vectors to Test Manually
1. `LIMIT 0` → strong negative / guard
2. `WHERE 1=0` → strong negative / guard
3. Remove `WHERE` clause (wrong result) → `result_changed` / penalty
4. `CREATE INDEX` / DDL → should be rejected / exception
5. Unchanged query → `noop_penalty` (escalates by step)

## Reward Component Reference (see `sqlsage/reward.py`)
| Component | Rule (simplified) |
|-----------|------------------|
| Speedup | Up to `MAX_SPEEDUP_REWARD` × speedup ratio + `BIG_WIN` / `EXCEPTIONAL` bonuses at high ratio |
| Plan | `+SEQ_SCAN_BONUS` per seq scan removed, `+NESTED_LOOP_BONUS` per nested loop removed |
| Rows / cost | `ROWS_SIGNAL_WEIGHT` × row reduction; `COST_SIGNAL_WEIGHT` × cost reduction |
| No-op | `NOOP_PENALTIES[step]` if plan+latency effectively unchanged |
| Slowdown | `-SLOWDOWN_WEIGHT` × slowdown ratio |
| Normalize | `raw / NORMALIZATION_DIVISOR` → **[-1, 1]** |

## If Reward Hacking Detected
```bash
python fix_training.py --issue reward_hacking
```

## Plots + Evidence (phases: Phase 8, rollout comparison)
| Command | What it does |
|---------|-------------|
| `export WANDB_ENTITY=...; export WANDB_PROJECT=sqlsage-grpo` | Match training project (see `docs/PHASE8_CLOSEOUT_CHECKLIST.md`) |
| `SQLSAGE_PLOTS_DEBUG=1 python plots/generate_plots.py` | Debug missing W&B data for plots |
| `python scripts/compare_rollouts.py --baseline results/baseline.jsonl --trained results/trained.jsonl` | Table when both JSONL exist |
"""

PERSON3_MD = r"""# SQLSage — Person 3: Training Lead
**Critical: Non-zero reward by Hour 12. 300+ episodes by Hour 18.**

## Colab Setup Sequence
1. Runtime → Change runtime type → **A100** GPU
2. `!pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' -q`
3. `!pip install trl openenv wandb psycopg2-binary -q` — add `torch` as needed
4. `import torch; print(torch.cuda.get_device_name(0))` — must be A100 for Colab template
5. `pip install -e '.[training]'` (local) per README for `scripts/rollout_wandb.py`

## Env URL for rollouts
| Variable | What it does |
|----------|---------------|
| `SQLSAGE_ENV_URL` | Person 1 Space or local `http://127.0.0.1:8000` (see `README` / `.env.example`) |
| `WANDB_ENTITY`, `WANDB_PROJECT` | W&B org + project (e.g. `sqlsage-grpo`) |

## One-liners (phases: rollout, benchmark, closeout)
| Command | What it does |
|---------|-------------|
| `python scripts/rollout_wandb.py --episodes 5` | Quick smoke (see README) |
| `python run_benchmark.py` | TPC-H style benchmark; writes `results/benchmark_results.json` |
| `python -m sqlsage.run p3-benchmark` | **Alias** → `run_benchmark.py` |
| `python -m sqlsage.run p3-gpu` | **Alias** → CUDA check (`torch`) |
| `python -m sqlsage.run p3-verify` | Quick 5 checks (URL, W&B, plots, benchmark file, README links) |
| `python -m sqlsage.status_checker --json` | Full hackathon gate JSON |
| `docs/PERSON3_PHASE8_MANUAL.md` | **§7–8 + §9** — Colab, Hub, demo video, submission |
| `docs/PHASE8_CLOSEOUT_CHECKLIST.md` | All-roles Phase 8 closeout |
| `notebooks/sqlsage_grpo_colab.ipynb` | GRPO training skeleton |

## Model Save — CRITICAL
**CORRECT**
```python
model.save_pretrained_merged(
    "sqlsage-trained", tokenizer, save_method="merged_16bit"
)
```

**WRONG — can corrupt 4-bit quality**
```python
model.merge_and_unload()  # DO NOT USE on 4-bit as sole export path
```

## Common Fixes
| Symptom | Fix |
|---------|-----|
| CUDA OOM | `batch_size=2`, `max_completion_length=256`, `grad_accum=8` |
| Invalid JSON from model | Few-shot JSON example in prompt (see `fix_training` / notebook) |
| Reward flat after ~50 ep | `python fix_training.py --issue flat_reward` |
| `result_changed` penalty | `python fix_training.py --issue result_changed` |
| Episodes very slow | Smaller TPC-H SF (0.1), or lighter tasks |
| Curriculum stuck | `python fix_training.py --issue curriculum_stuck` |
| Syntax errors in actions | `python fix_training.py --issue syntax_error` |

## Reward is working when
- `reward/mean` trending up in W&B
- `plan/seq_scans_removed` (or similar) > 0 when appropriate
- `penalty/result_changed` near **zero** when results match
- Episode length / latency moving in the right direction for your run
- Benchmark: example target **Q5** latency under your documented threshold (adjust per SF)

## wandb metrics to watch
`reward/mean`, speedup-related scalars, `penalty/result_changed`, `penalty/syntax_error`, plan/scan metrics, `episode_length` — names depend on your logging callback (see `monitor_training.py` / notebook).
"""


def _is_gfm_table_separator(cells: list[str]) -> bool:
    if not cells:
        return False
    for c in cells:
        t = c.strip()
        t = t.replace(" ", "").lstrip(":").rstrip(":")
        if not t or not all(ch == "-" for ch in t):
            return False
    return True


def _html_wrap(fragment: str, title: str) -> str:
    css = """
@page { size: A4; margin: 10mm; }
@media print {
  body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
}
* { box-sizing: border-box; }
body {
  background: #fff; color: #000;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  font-size: 8.5pt; line-height: 1.22;
  max-width: 195mm; margin: 0 auto; padding: 4mm 5mm 8mm 5mm;
}
h1 { font-size: 12pt; margin: 0.15em 0; font-weight: 700; }
h2 { font-size: 9.5pt; margin: 0.5em 0 0.15em; border-bottom: 1px solid #333; }
h3 { font-size: 8.7pt; margin: 0.35em 0 0.1em; }
p { margin: 0.2em 0; }
ul, ol { margin: 0.15em 0; padding-left: 1.2em; }
ul { list-style: disc; }
table.cheat { width: 100%; border-collapse: collapse; margin: 0.25em 0; font-size: 7.8pt; }
table.cheat th {
  background: #1a365d; color: #fff; text-align: left; padding: 2px 4px;
  font-weight: 600; font-size: 7.5pt;
}
table.cheat td { border: 1px solid #999; padding: 2px 3px; vertical-align: top; }
pre.cmd {
  font-family: ui-monospace, "Cascadia Code", "Consolas", monospace;
  font-size: 7.3pt; background: #f4f4f4; border: 1px solid #ccc; padding: 3px 4px;
  margin: 0.25em 0; white-space: pre-wrap; word-break: break-word;
}
code.ic {
  font-family: ui-monospace, "Cascadia Code", "Consolas", monospace;
  font-size: 7.5pt; background: #f0f0f0; padding: 0 2px;
}
br { line-height: 0.2em; }
""".strip()
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<style>
{css}
</style>
</head>
<body>
{fragment}
</body>
</html>
"""


def _md_to_html_v2(src: str, title: str) -> str:
    """Line-oriented Markdown → HTML (tables, code fences, headers, lists, paragraphs)."""
    lines = src.splitlines()
    out: list[str] = []
    i = 0
    in_fence = False
    code_lines: list[str] = []
    in_ol = False
    in_ul = False

    def close_lists() -> None:
        nonlocal in_ol, in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    while i < len(lines):
        line = lines[i]
        st = line.strip()
        if st.startswith("```"):
            if in_fence:
                body = "\n".join(code_lines)
                out.append(
                    "<pre class=\"cmd\">" + html.escape(body) + "</pre>"
                )
                code_lines = []
                in_fence = False
            else:
                close_lists()
                in_fence = True
            i += 1
            continue
        if in_fence:
            code_lines.append(line)
            i += 1
            continue

        if st.startswith("|") and st.count("|") >= 2:
            close_lists()
            tbl: list[list[str]] = []
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                tbl.append(
                    [c.strip() for c in lines[j].strip().strip("|").split("|")]
                )
                j += 1
            if len(tbl) >= 2 and _is_gfm_table_separator(tbl[1]):
                head, body = tbl[0], tbl[2:]
            else:
                head = tbl[0] if tbl else []
                body = tbl[1:]

            out.append('<table class="cheat">')
            out.append(
                "<thead><tr>"
                + "".join(
                    f"<th>{_inline_on(html.escape(c))}</th>" for c in head
                )
                + "</tr></thead><tbody>"
            )
            for row in body:
                if not row or (len(row) == 1 and not row[0].strip()):
                    continue
                out.append(
                    "<tr>"
                    + "".join(
                        f"<td>{_inline_on(html.escape(c))}</td>" for c in row
                    )
                    + "</tr>"
                )
            out.append("</tbody></table>")
            i = j
            continue

        if st.startswith("### "):
            close_lists()
            out.append(f"<h3>{_inline_on(html.escape(st[4:]))}</h3>")
        elif st.startswith("## "):
            close_lists()
            out.append(f"<h2>{_inline_on(html.escape(st[3:]))}</h2>")
        elif st.startswith("# "):
            close_lists()
            out.append(f"<h1>{_inline_on(html.escape(st[2:]))}</h1>")
        elif st.startswith(("- ", "* ")):
            if not in_ul:
                close_lists()
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{_inline_on(html.escape(st[2:]))}</li>")
        elif re.match(r"^\d+\.\s", st):
            if not in_ol:
                close_lists()
                out.append("<ol>")
                in_ol = True
            item = re.sub(r"^\d+\.\s+", "", st)
            out.append(f"<li>{_inline_on(html.escape(item))}</li>")
        elif st == "":
            # Blank line: keep <ol>/<ul> open (numbered Colab list spans blanks)
            pass
        else:
            close_lists()
            if st:
                out.append(f"<p>{_inline_on(html.escape(line))}</p>")

        i += 1

    close_lists()
    return _html_wrap("\n".join(out), title)


def _inline_on(escaped: str) -> str:
    """`escaped` is HTML-escaped; add ** and ` handling."""
    t = re.sub(
        r"\*\*([^*]+)\*\*",
        lambda m: f"<strong>{m.group(1)}</strong>",
        escaped,
    )
    t = re.sub(
        r"`([^`]+)`",
        lambda m: f'<code class="ic">{m.group(1)}</code>',
        t,
    )
    return t


def _write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    sheets = [
        (
            "person1_env_engineer",
            "SQLSage — Person 1: Environment Engineer",
            PERSON1_MD,
        ),
        (
            "person2_reward_engineer",
            "SQLSage — Person 2: Reward & Verifier Engineer",
            PERSON2_MD,
        ),
        (
            "person3_training_lead",
            "SQLSage — Person 3: Training Lead",
            PERSON3_MD,
        ),
    ]
    for stem, t, md in sheets:
        _write_file(OUT / f"{stem}.md", md)
        h = _md_to_html_v2(md, t)
        _write_file(OUT / f"{stem}.html", h)
    print("Generated 6 files in ./cheatsheets/")


if __name__ == "__main__":
    main()
