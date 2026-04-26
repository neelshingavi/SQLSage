"""
Live terminal dashboard for the SQLSage hackathon (Rich).

  cd sqlsage-env
  python dashboard.py --person 1
  python -m sqlsage.dashboard --team
  python -m sqlsage.dashboard --person 2 --live --refresh 30

Env: ``SQLSAGE_HF_RESET_URL`` (full ``.../reset`` or Space base URL) or, if unset,
``SQLSAGE_HF_SPACE_URL`` / ``SQLSAGE_ENV_URL`` — used to ``POST`` OpenEnv ``/reset`` for the hour-8 check (else NOT CHECKED).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from typing import Any, Callable, Optional

# --- Time constants (naive local time) ---

HACKATHON_START = datetime(2026, 4, 25, 17, 0, 0)  # Day 1, 5 PM
HACKATHON_END = datetime(2026, 4, 26, 17, 0, 0)  # Day 2, 5 PM
SUBMISSION_HARD_DEADLINE = datetime(2026, 4, 26, 16, 30, 0)  # 23.5h

# --- Rich (optional heavy imports in render) ---

PERSON_ROLES: dict[int, str] = {
    1: "Environment Engineer",
    2: "Reward & Verifier Engineer",
    3: "Training Lead",
}

MILESTONES: list[dict[str, Any]] = [
    {
        "hour": 8,
        "gate": "Environment deployed to HF Spaces",
        "owner": "Person 1",
        "check_command": 'curl -sS -X POST -H "Content-Type: application/json" -d \'{}\' https://your-hf-username-sqlsage-env.hf.space/reset',
        "if_missed": "CRISIS: Stop all other work. Fix deployment NOW.",
        "verify_fn": "check_hf_space_live()",
    },
    {
        "hour": 12,
        "gate": "First training run showing non-zero reward",
        "owner": "Person 3",
        "check_command": "python monitor_training.py",
        "if_missed": "DEBUG: Print every reward value. Check reward pipeline.",
        "verify_fn": "check_wandb_nonzero_reward()",
    },
    {
        "hour": 18,
        "gate": "Full training run complete (300+ episodes). Plots committed.",
        "owner": "Person 3",
        "check_command": "python generate_plots.py && git add plots/ && git commit -m 'plots'",
        "if_missed": "Reduce to 100 episodes. Plots > long run.",
        "verify_fn": "check_plots_committed()",
    },
    {
        "hour": 21,
        "gate": "Before/after comparison documented with real numbers",
        "owner": "All",
        "check_command": "python run_benchmark.py",
        "if_missed": "Use whatever numbers you have. Do not invent.",
        "verify_fn": "check_benchmark_results_exist()",
    },
    {
        "hour": 23,
        "gate": "README done. All links. Form ready.",
        "owner": "All",
        "check_command": "cat README.md | grep 'hf.space'",
        "if_missed": "Submit what you have. 80% submitted > 100% unsubmitted.",
        "verify_fn": "check_readme_has_links()",
    },
    {
        "hour": 24,
        "gate": "Google Form submitted",
        "owner": "All",
        "check_command": "# Open submission form link",
        "if_missed": "HARD DEADLINE. Submit at hour 23.5 — do NOT wait.",
        "verify_fn": None,
    },
]

# --- Per-person tasks (data from hackathon spec) ---

PERSON_TASKS: dict[int, list[dict[str, Any]]] = {
    1: [
        {
            "hours": "0-1",
            "task": "openenv init + PostgreSQL Docker + TPC-H schema load",
            "commands": [
                "openenv init sqlsage-env",
                "docker run --name sqlsage-pg -e POSTGRES_PASSWORD=sqlsage -e POSTGRES_DB=sqlsage -p 5432:5432 -d postgres:16",
                "psql -h localhost -U postgres -d sqlsage -c 'SELECT version();'",
            ],
            "done_when": "psql connects, tables exist, row counts correct",
        },
        {
            "hours": "1-4",
            "task": "Implement env.py: reset(), step(), state(). Implement explain_parser.py.",
            "commands": [
                "uvicorn sqlsage.app:app --reload --port 8000",
                'curl -sS -X POST -H "Content-Type: application/json" -d \'{}\' http://localhost:8000/reset',
            ],
            "done_when": "POST /reset returns valid JSON observation",
        },
        {
            "hours": "4-6",
            "task": "execute_and_measure() + get_result_hash() + anti_cheat.py",
            "commands": [
                "pytest tests/test_anti_cheat.py -v",
                'python -c "from sqlsage.env import SQLSageEnv; e=SQLSageEnv(); print(e.step(\'rewrite_join\', \'SELECT 1\'))"',
            ],
            "done_when": "step() with valid rewrite returns {reward, done, state}",
        },
        {
            "hours": "6-8",
            "task": "Level 1 tasks (Q1, Q6, Q14). Drop indexes. Deploy to HF Spaces.",
            "commands": [
                "psql -U postgres -d sqlsage -c 'DROP INDEX IF EXISTS idx_orders_status;'",
                "psql -U postgres -d sqlsage -c 'DROP INDEX IF EXISTS idx_lineitem_shipdate;'",
                "openenv push your-hf-username/sqlsage-env",
                'curl -sS -X POST -H "Content-Type: application/json" -d \'{}\' https://your-hf-username-sqlsage-env.hf.space/reset',
            ],
            "done_when": "Remote env.reset() works. URL shared with Person 3.",
        },
        {
            "hours": "8-14",
            "task": "Debug issues. Add Level 2 tasks. Test anti-cheat guards.",
            "commands": [
                "python fix_training.py --issue env_500",
                "pytest tests/ -v",
            ],
            "done_when": "Training run completes 50+ episodes",
        },
        {
            "hours": "14-18",
            "task": "Add Level 3 tasks. Stress test. Fix race conditions.",
            "commands": [
                "python monitor_training.py",
                "pytest tests/test_level3.py -v",
            ],
            "done_when": "All 3 curriculum levels work end-to-end",
        },
        {
            "hours": "18-22",
            "task": "Write README environment section. Test HF Space UI.",
            "commands": [
                "openenv push your-hf-username/sqlsage-env",
                'curl -sS -X POST -H "Content-Type: application/json" -d \'{}\' https://your-hf-username-sqlsage-env.hf.space/reset',
            ],
            "done_when": "README done. Judges can run env.reset()",
        },
        {
            "hours": "22-24",
            "task": "Buffer for submission issues.",
            "commands": ["# Monitor HF Space. Fix any last-minute issues."],
            "done_when": "Google Form submitted",
        },
    ],
    2: [
        {
            "hours": "0-3",
            "task": "Implement explain_parser.py. Write unit tests.",
            "commands": [
                'psql -h localhost -U postgres -d sqlsage -c "EXPLAIN (ANALYZE, FORMAT JSON) SELECT * FROM orders LIMIT 100;"',
                'python -c \'from sqlsage.explain_parser import get_explain_dict; import psycopg2; conn=psycopg2.connect(host="localhost",user="postgres",password="sqlsage",dbname="sqlsage"); print(get_explain_dict(conn, "SELECT * FROM orders LIMIT 100"))\'',
                "pytest tests/test_explain_parser.py -v",
            ],
            "done_when": "All parser unit tests pass",
        },
        {
            "hours": "3-7",
            "task": "Implement reward.py with all components. pytest all reward cases.",
            "commands": [
                "pytest tests/test_reward.py -v",
                'python -c \'from sqlsage.reward import compute_reward; print(compute_reward(8420, 310, {"seq_scans":0,"nested_loops":0,"hash_joins":1,"total_cost":1000}, {"seq_scans":2,"nested_loops":1,"hash_joins":0,"total_cost":50000}))\'',
            ],
            "done_when": "All 10+ reward unit tests pass",
        },
        {
            "hours": "7-10",
            "task": "Implement anti_cheat.py. Test all 5 attack vectors.",
            "commands": [
                "python fix_training.py --issue reward_hacking",
                "pytest tests/test_anti_cheat.py -v",
            ],
            "done_when": "All 5 attack vectors correctly penalized",
        },
        {
            "hours": "10-16",
            "task": "Monitor training. Watch wandb. Flag reward hacking.",
            "commands": [
                "wandb login",
                "python monitor_training.py",
                "python fix_training.py --issue flat_reward",
            ],
            "done_when": "100+ episodes. No reward hacking detected.",
        },
        {
            "hours": "16-20",
            "task": "Generate reward curve plots. Commit to repo.",
            "commands": [
                "python generate_plots.py",
                "git add plots/ && git commit -m 'Add training reward plots'",
                "git push",
            ],
            "done_when": "3 plots committed to repo",
        },
        {
            "hours": "20-22",
            "task": "Write HF blog post (< 2 min read).",
            "commands": [
                "# Go to huggingface.co/blog/new",
                "# Title: SQLSage: Teaching LLMs to Read Query Plans with RL",
            ],
            "done_when": "Blog post published on HF",
        },
        {
            "hours": "22-24",
            "task": "Buffer. Help with pitch rehearsal.",
            "commands": ["# Run demo script from Section 10"],
            "done_when": "Pitch rehearsed. Form ready.",
        },
    ],
    3: [
        {
            "hours": "0-3",
            "task": "Set up Colab. Install deps. Verify GPU. Load Qwen 2.5 1.5B.",
            "commands": [
                "# Runtime → Change runtime type → A100 GPU",
                "!pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' -q",
                "!pip install trl openenv wandb psycopg2-binary -q",
                "import torch; print(torch.cuda.get_device_name(0))",
            ],
            "done_when": "Model loads. GPU confirmed A100.",
        },
        {
            "hours": "3-6",
            "task": "Build dataset pipeline. Test prompt template.",
            "commands": [
                "python sqlsage/prompt_builder.py --test",
                "python -c 'from sqlsage.dataset import build_dataset; print(len(build_dataset()))'",
            ],
            "done_when": "Model generates parseable JSON",
        },
        {
            "hours": "6-10",
            "task": "Connect to P1's env. Run first small training: 20 episodes, Level 1.",
            "commands": [
                "# In Colab: env = Environment.from_hub('your-hf-username/sqlsage-env')",
                "# Run GRPOTrainer for 20 episodes",
            ],
            "done_when": "First reward curve visible in wandb",
        },
        {
            "hours": "10-14",
            "task": "Tune prompt. Run 100-episode Level 1 run.",
            "commands": [
                "python fix_training.py --issue syntax_error",
                "python fix_training.py --issue flat_reward",
            ],
            "done_when": "Reward trend upward. >30% pass rate.",
        },
        {
            "hours": "14-18",
            "task": "Add Level 2. Run full 300-episode training. Save checkpoints.",
            "commands": [
                "# In Colab: run full training cell",
                "model.save_pretrained_merged('sqlsage-trained', tokenizer, save_method='merged_16bit')",
            ],
            "done_when": "Full training run complete. Plots show improvement.",
        },
        {
            "hours": "18-20",
            "task": "Run baseline comparison. Record before/after numbers.",
            "commands": [
                "python run_benchmark.py",
            ],
            "done_when": "Specific ms values documented.",
        },
        {
            "hours": "20-22",
            "task": "Save final model. Push to HF Hub. Test inference.",
            "commands": [
                "huggingface-cli login",
                "model.save_pretrained_merged('sqlsage-trained', tokenizer, save_method='merged_16bit')",
                "# Upload to HF Hub via notebook",
            ],
            "done_when": "Model on HF Hub. Inference works.",
        },
        {
            "hours": "22-24",
            "task": "Record 2-minute demo video.",
            "commands": [
                "# Follow demo script from Section 10 exactly",
                "# Record screen. Upload to YouTube (unlisted). Copy URL.",
            ],
            "done_when": "Video uploaded. Link in README.",
        },
    ],
}


# --- Time helpers ---


def _now() -> datetime:
    return datetime.now().replace(microsecond=0)


def get_current_hour() -> float:
    """Returns elapsed hours since HACKATHON_START (0–24) as a float, e.g. 7.5."""
    n = _now()
    if n < HACKATHON_START:
        return 0.0
    if n >= HACKATHON_END:
        return 24.0
    delta = n - HACKATHON_START
    h = delta.total_seconds() / 3600.0
    return min(24.0, h)


def get_time_remaining() -> str:
    """e.g. '9h 23m remaining' or 'DEADLINE PASSED'."""
    n = _now()
    if n >= HACKATHON_END:
        return "DEADLINE PASSED"
    sec = (HACKATHON_END - n).total_seconds()
    if sec <= 0:
        return "DEADLINE PASSED"
    h, r = divmod(int(sec), 3600)
    m, _ = divmod(r, 60)
    return f"{h}h {m:02d}m remaining"


def get_submission_time_remaining() -> str:
    """Time until the hard form submission cut (23.5h)."""
    n = _now()
    if n >= SUBMISSION_HARD_DEADLINE:
        if n >= HACKATHON_END:
            return "PAST DEADLINES"
        return "PAST SUBMISSION — final 30m"
    sec = (SUBMISSION_HARD_DEADLINE - n).total_seconds()
    h, r = divmod(int(sec), 3600)
    m, _ = divmod(r, 60)
    return f"{h}h {m:02d}m to submission (23.5h)"


# --- Project root (sqlsage-env) ---


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# --- Verify fns: return True / False, or None = NOT CHECKED (no crash) ---


def check_hf_space_live() -> Optional[bool]:
    from sqlsage.status_checker import _reset_url

    raw = (os.environ.get("SQLSAGE_HF_RESET_URL") or "").strip()
    if not raw:
        raw = (os.environ.get("SQLSAGE_HF_SPACE_URL") or os.environ.get("SQLSAGE_ENV_URL") or "").strip()
    if not raw:
        return None
    endpoint = _reset_url(raw)
    try:
        import requests
    except ImportError:
        return None
    try:
        r = requests.post(
            endpoint,
            json={},
            timeout=float(os.environ.get("SQLSAGE_DASHBOARD_HTTP_TIMEOUT", "5")),
        )
        return r.status_code == 200
    except Exception:
        return None


def check_wandb_nonzero_reward() -> Optional[bool]:
    try:
        from pathlib import Path

        root = Path(_project_root())
        wdir = root / "wandb"
        if not wdir.is_dir():
            return None
        for run in sorted(wdir.glob("run-*"), reverse=True):
            summ = run / "files" / "wandb-summary.json"
            if not summ.is_file():
                continue
            try:
                data = json.loads(summ.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            found_reward_key = False
            for k, v in data.items():
                lk = k.lower()
                if "reward" in lk and isinstance(v, (int, float)):
                    found_reward_key = True
                    if v != 0:
                        return True
            if found_reward_key:
                return False
        return None
    except Exception:
        return None


def check_plots_committed() -> Optional[bool]:
    try:
        root = _project_root()
        r = subprocess.run(
            ["git", "log", "-1", "--oneline", "--", "plots/"],
            capture_output=True,
            text=True,
            timeout=8,
            cwd=root,
        )
        if r.returncode != 0:
            return None
        if not (r.stdout or "").strip():
            return False
        return True
    except Exception:
        return None


def check_benchmark_results_exist() -> Optional[bool]:
    try:
        from pathlib import Path

        p = Path(_project_root()) / "results" / "benchmark_results.json"
        if not p.is_file():
            return False
        if p.stat().st_size < 4:
            return False
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(data, (dict, list)) or (isinstance(data, dict) and not data):
            return False
        return True
    except Exception:
        return None


def check_readme_has_links() -> Optional[bool]:
    try:
        from pathlib import Path

        p = Path(_project_root()) / "README.md"
        if not p.is_file():
            return None
        t = p.read_text(encoding="utf-8", errors="replace")
        t_lower = t.lower()
        if "http" in t and (
            "hf." in t_lower or "huggingface" in t_lower or "hf.space" in t_lower
        ):
            return True
        if "http" in t and "space" in t_lower:
            return True
        return False
    except Exception:
        return None


def _parse_verify_name(verify_field: str | None) -> str:
    if not verify_field:
        return ""
    s = str(verify_field).strip()
    if s.endswith("()"):
        s = s[:-2]
    return s


_VERIFY_DISPATCH: dict[str, Callable[[], Optional[bool]]] = {
    "check_hf_space_live": check_hf_space_live,
    "check_wandb_nonzero_reward": check_wandb_nonzero_reward,
    "check_plots_committed": check_plots_committed,
    "check_benchmark_results_exist": check_benchmark_results_exist,
    "check_readme_has_links": check_readme_has_links,
}


def run_verify_for_milestone(
    milestone: dict[str, Any], cache: dict[str, Optional[bool]]
) -> Optional[bool]:
    name = _parse_verify_name(milestone.get("verify_fn"))
    if not name:
        return None
    if name in cache:
        return cache[name]
    fn = _VERIFY_DISPATCH.get(name)
    if not fn:
        cache[name] = None
        return None
    try:
        out = fn()
    except Exception:
        out = None
    cache[name] = out
    return out


def get_milestone_status(
    milestone: dict[str, Any],
    current_hour: float,
    cache: dict[str, Optional[bool]] | None = None,
) -> str:
    """
    Returns: "DONE" | "IN_PROGRESS" | "UPCOMING" | "MISSED"
    MISSED = current hour > hour + 0.5 and verify returns False
    IN_PROGRESS = within 1h before the gate, grace window, or late / NOT CHECKED
    """
    if cache is None:
        cache = {}
    h = float(milestone["hour"])
    vr = run_verify_for_milestone(milestone, cache)

    if vr is True:
        return "DONE"
    if current_hour > h + 0.5 and vr is False:
        return "MISSED"
    in_before = h - 1.0 <= current_hour < h
    in_grace = h <= current_hour <= h + 0.5
    if in_before or (in_grace and vr is not True):
        return "IN_PROGRESS"
    if current_hour < h - 1.0:
        return "UPCOMING"
    if current_hour < h:
        return "IN_PROGRESS"
    if current_hour > h + 0.5:
        return "IN_PROGRESS"
    return "UPCOMING"


# --- Task engine ---


def _parse_hour_range(s: str) -> tuple[float, float]:
    m = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$", s.strip())
    if not m:
        return (0.0, 0.0)
    return (float(m.group(1)), float(m.group(2)))


def get_current_task(person: int, current_hour: float) -> dict[str, Any]:
    """
    Return the task dict for this person at current_hour.
    Intervals are [lo, hi) in hackathon hours; the last block includes up to 24.0.
    """
    tasks = PERSON_TASKS.get(person, [])
    for i, t in enumerate(tasks):
        lo, hi = _parse_hour_range(str(t.get("hours", "0-0")))
        is_last = i == len(tasks) - 1
        if is_last:
            if lo <= current_hour <= 24.0:
                return t
        elif lo <= current_hour < hi:
            return t
    return tasks[-1] if tasks else {}


def get_next_task(
    person: int, current_hour: float, current: dict[str, Any] | None = None
) -> Optional[dict[str, Any]]:
    cur = current if current is not None else get_current_task(person, current_hour)
    tasks = PERSON_TASKS.get(person, [])
    for i, t in enumerate(tasks):
        if t is cur and i + 1 < len(tasks):
            return tasks[i + 1]
    return None


# --- Dashboard (Rich) ---


def _milestone_label(gate: str) -> str:
    short = {
        "Environment deployed to HF Spaces": "Environment on HF Spaces",
        "First training run showing non-zero reward": "Non-zero reward",
        "Full training run complete (300+ episodes). Plots committed.": "Full training + plots",
        "Before/after comparison documented with real numbers": "Before/after numbers",
        "README done. All links. Form ready.": "README + form ready",
        "Google Form submitted": "SUBMITTED",
    }
    return short.get(gate, gate[:48] + ("…" if len(gate) > 48 else ""))


def _format_status_pretty(code: str) -> str:
    m = {
        "DONE": "✓ DONE",
        "IN_PROGRESS": "⚡ IN PROGRESS",
        "UPCOMING": "○ UPCOMING",
        "MISSED": "✗ MISSED",
    }
    return m.get(code, code)


def _style_for_status_text(code: str) -> str:
    if code == "DONE":
        return "green"
    if code == "IN_PROGRESS":
        return "yellow"
    if code == "MISSED":
        return "bold red"
    return "dim"


def _milestone_away(hour: float, ch: float, st: str) -> str:
    if st != "IN_PROGRESS" or ch >= hour:
        return ""
    d = max(0.0, hour - ch)
    return f"  ←{d:.1f}h"


def _next_milestone_hours(ch: float) -> float | None:
    """Smallest m['hour'] with m['hour'] > ch, else None."""
    nxt: list[float] = []
    for m in MILESTONES:
        mh = float(m["hour"])
        if mh > ch + 1e-6:
            nxt.append(mh)
    if not nxt:
        return None
    return min(nxt) - ch


def _any_milestone_missed(ch: float, cache: dict[str, Optional[bool]]) -> bool:
    for m in MILESTONES:
        if get_milestone_status(m, ch, cache) == "MISSED":
            return True
    return False


def _bar_style(ch: float, cache: dict[str, Optional[bool]]) -> tuple[str, str]:
    """
    GREEN: on track (next > 2h)
    YELLOW: next < 2h
    RED: any MISSED or < 0.5h to next
    """
    if _any_milestone_missed(ch, cache):
        return (
            "bold red",
            "CRISIS: milestone missed (or run checks). See MISSED row(s).",
        )
    d = _next_milestone_hours(ch)
    if d is None:
        return (
            "green",
            "Hackathon window: no more gates ahead. Finish & submit on time.",
        )
    if d < 0.5:
        return ("bold red", f"CRISIS: next gate in {d*60:.0f} min.")
    if d < 2.0:
        return (
            "yellow",
            f"Watch out: next gate in {d:.1f}h. Prioritize that milestone.",
        )
    return ("green", f"On track. Next gate in {d:.1f}h.")


def _build_task_panel(
    person: int, ch: float, show_role_header: bool = True
) -> "Columns":
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.text import Text

    cur = get_current_task(person, ch)
    nxt = get_next_task(person, ch, cur)
    cur_block = str(cur.get("hours", "—")) if cur else "—"
    task_lines: list[Text] = []
    if show_role_header:
        task_lines.append(
            Text(
                f"Person {person} — {PERSON_ROLES.get(person, 'Member')}\n",
                style="bold cyan",
            )
        )
    task_lines.append(
        Text(f"NOW: Hour {cur_block}\n", style="bold")
        + Text(f"{(cur or {}).get('task', '—')}\n\n", style="default")
    )
    task_lines.append(Text("COMMANDS TO RUN:\n", style="bold"))
    for c in (cur or {}).get("commands", []):
        task_lines.append(Text(f"  $ {c}\n", style="white"))
    task_lines.append(
        Text(f"\nDONE WHEN: {(cur or {}).get('done_when', '—')}", style="green")
    )
    now_panel = Panel(
        Text.assemble(*task_lines),
        title="[bold]Current Task[/]",
        border_style="blue",
    )
    nxt_text = (
        f"Hour {nxt.get('hours', '—')}\n{nxt.get('task', '—')}" if nxt else "— (final block)"
    )
    next_panel = Panel(
        nxt_text,
        title="[bold]Next[/]",
        border_style="magenta",
    )
    return Columns([now_panel, next_panel], equal=True, expand=True)


def _build_milestone_table(
    ch: float, cache: dict[str, Optional[bool]]
) -> "Table":
    from rich import box
    from rich.table import Table
    from rich.text import Text

    t = Table(
        title="Team milestones",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    t.add_column("When", style="dim", width=6)
    t.add_column("Milestone", width=28, no_wrap=True)
    t.add_column("Owner", width=8, no_wrap=True)
    t.add_column("Status", no_wrap=True, width=28)
    t.add_column("Verify", style="dim", width=14, no_wrap=True)

    for m in MILESTONES:
        st = get_milestone_status(m, ch, cache)
        vr = run_verify_for_milestone(m, cache)
        vlabel = "NOT CHECKED" if vr is None else (
            "pass" if vr else "no"
        )
        away = _milestone_away(float(m["hour"]), ch, st)
        row = [
            f"H{int(m['hour']):d}",
            _milestone_label(m["gate"]),
            m.get("owner", "—"),
            Text(
                f"{_format_status_pretty(st)}{away}", style=_style_for_status_text(st)
            ),
            vlabel,
        ]
        t.add_row(*row)
    return t


def _top_bar_ch(person: int | None) -> "Text":
    from rich.text import Text

    ch = get_current_hour()
    rem = get_time_remaining()
    sub = get_submission_time_remaining()
    label = "Team (all 3)" if person is None else f"Person {person}: {PERSON_ROLES.get(person, '')}"
    line1 = f" SQLSage Dashboard — {label} "
    line2 = f" Hour {ch:.1f} / 24   |   {rem}   |   {sub} "
    return Text.assemble(
        (line1, "bold white on dark_blue"),
        "\n",
        (line2, "default"),
    )


def _footer_bar(ch: float, cache: dict[str, Optional[bool]]) -> "Text":
    from rich.text import Text

    style, msg = _bar_style(ch, cache)
    return Text(f" {msg} ", style=style)


def render_person_dashboard(person: int) -> "Group":
    from rich.console import Group

    ch = get_current_hour()
    cache: dict[str, Optional[bool]] = {}
    return Group(
        _top_bar_ch(person),
        _build_task_panel(person, ch),
        _build_milestone_table(ch, cache),
        _footer_bar(ch, cache),
    )


def render_team_dashboard() -> "Group":
    from rich import box
    from rich.columns import Columns
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    ch = get_current_hour()
    cache: dict[str, Optional[bool]] = {}
    per_cols: list[Panel] = []
    for p in (1, 2, 3):
        cur = get_current_task(p, ch)
        nxt = get_next_task(p, ch, cur)
        lines: list[Text] = [
            Text(
                f"{PERSON_ROLES.get(p, '')}\n",
                style="bold cyan",
            ),
            Text(
                f"NOW ({cur.get('hours', '—')}): {cur.get('task', '')}\n\n", style="default"
            ),
        ]
        for c in (cur.get("commands") or [])[:5]:
            lines.append(Text(f"  {c}\n", style="dim white"))
        if nxt:
            lines.append(
                Text(
                    f"\nNext ({nxt.get('hours', '')}): {nxt.get('task', '')}\n",
                    style="magenta",
                )
            )
        per_cols.append(
            Panel(
                Text.assemble(*lines) if lines else Text("—"),
                title=f"Person {p}",
                box=box.ROUNDED,
            )
        )
    return Group(
        _top_bar_ch(None),
        Columns(per_cols, equal=True, expand=True, padding=1),
        _build_milestone_table(ch, cache),
        _footer_bar(ch, cache),
    )


def _console() -> "Console":
    from rich.console import Console

    try:
        w = os.get_terminal_size().columns
    except OSError:
        w = 120
    return Console(width=max(100, w))


def run_live(
    person: int | None, refresh_seconds: int = 60, team: bool = False
) -> None:
    from rich.live import Live

    console = _console()
    with Live(
        render_team_dashboard() if team else render_person_dashboard(person or 1),
        console=console,
        refresh_per_second=4,
        screen=True,
    ) as live:
        try:
            while True:
                time.sleep(refresh_seconds)
                if team:
                    live.update(render_team_dashboard())
                else:
                    live.update(render_person_dashboard(int(person or 1)))
        except KeyboardInterrupt:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SQLSage hackathon live terminal dashboard (Rich).",
    )
    parser.add_argument(
        "--person",
        type=int,
        choices=(1, 2, 3),
        help="Teammate: 1=Environment, 2=Reward, 3=Training",
    )
    parser.add_argument(
        "--team",
        action="store_true",
        help="All three roles side by side (compact)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Auto-refresh (Ctrl+C to exit)",
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=60,
        metavar="S",
        help="Seconds between refreshes in --live mode (default: 60)",
    )
    args = parser.parse_args()
    if args.team and args.person is not None:
        parser.error("Use either --team or --person, not both.")
    if not args.team and args.person is None:
        parser.error("Specify --person 1, 2, or 3, or use --team.")

    console = _console()
    if args.live:
        run_live(
            person=args.person, refresh_seconds=max(1, args.refresh), team=args.team
        )
    elif args.team:
        console.print(render_team_dashboard())
    else:
        console.print(render_person_dashboard(int(args.person)))


if __name__ == "__main__":
    main()
