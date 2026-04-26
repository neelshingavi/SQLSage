"""
SQLSage command alias dispatcher — one short command per common task.

  cd sqlsage-env
  python -m sqlsage.run p1-serve
  python -m sqlsage.run dashboard 2
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

# sqlsage-env/
ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
DOTENV_OK = True

# --- .env -----------------------------------------------------------------

_DOTENV = False


def _load_dotenv() -> None:
    global _DOTENV
    if _DOTENV:
        return
    if DOTENV_OK:
        try:
            from dotenv import load_dotenv

            load_dotenv(ENV_PATH)
            load_dotenv(ROOT / ".env.local", override=True)
        except ImportError:
            pass
    _DOTENV = True


def _set_env_key(key: str, value: str) -> None:
    os.environ[key] = value
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
        return
    if not ENV_PATH.is_file():
        ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
        ENV_PATH.write_text(f"{key}={value}\n", encoding="utf-8")
        return
    text = ENV_PATH.read_text(encoding="utf-8", errors="replace")
    lines: list[str] = []
    pat = re.compile(rf"^{re.escape(key)}=")
    replaced = False
    for line in text.splitlines(keepends=True):
        if line.strip() and not line.lstrip().startswith("#") and pat.match(line):
            lines.append(f"{key}={value}\n")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        if lines and (not lines[-1].endswith("\n")):
            lines[-1] = lines[-1] + "\n"
        if lines and lines[-1].strip() != "":
            lines.append("\n")
        lines.append(f"{key}={value}\n")
    with ENV_PATH.open("w", encoding="utf-8") as f:
        f.write("".join(lines).rstrip() + "\n")


def _prompt_and_save(
    key: str, message: str, default: str | None = None
) -> str:
    _load_dotenv()
    v = (os.environ.get(key) or "").strip()
    if v:
        return v
    hint = f" [{default}]" if default else ""
    s = input(f"{message}{hint}: ").strip() or (default or "")
    if s:
        _set_env_key(key, s)
    return s


def _get_hf_space_url() -> str:
    _load_dotenv()
    for k in ("SQLSAGE_HF_SPACE_URL", "SQLSAGE_ENV_URL"):
        u = (os.environ.get(k) or "").strip()
        if u:
            return u.rstrip("/")
    s = _prompt_and_save(
        "SQLSAGE_HF_SPACE_URL", "Hugging Face Space base URL (https://...hf.space)", ""
    )
    return s.rstrip("/")


def _get_hf_username() -> str:
    _load_dotenv()
    for k in ("HUGGINGFACE_HUB_USER", "HF_USERNAME", "SQLSAGE_HF_USERNAME"):
        u = (os.environ.get(k) or "").strip()
        if u:
            return u
    s = _prompt_and_save("HUGGINGFACE_HUB_USER", "Hugging Face username (for openenv push)", "")
    return s


# --- console / run --------------------------------------------------------


def _c() -> Any:
    from rich.console import Console

    return Console(highlight=False)


def _announce(alias: str, command_desc: str) -> None:
    _c().print(f"[bold cyan]{alias}[/] → [yellow]{command_desc}[/]")


def _suggest_fix_training() -> None:
    _c().print(
        "[dim]If this looks like training/reward/RL: "
        "try [white]python -m sqlsage.run p2-anticheat[/] or [white]python fix_training.py --issue NAME[/][/]"
    )


def _run(
    label: str,
    cmd: list[str] | str,
    *,
    cwd: Path | None = None,
    shell: bool = False,
    env: dict[str, str] | None = None,
) -> int:
    if isinstance(cmd, str):
        _announce(label, cmd)
    else:
        _announce(label, " ".join(shlex.quote(p) for p in cmd))
    wdir = str(cwd or ROOT)
    e = {**os.environ, **(env or {})}
    if shell and isinstance(cmd, str):
        r = subprocess.run(cmd, shell=True, cwd=wdir, env=e, text=True)
    elif isinstance(cmd, list):
        r = subprocess.run(cmd, cwd=wdir, env=e, text=True)
    else:
        r = subprocess.run(
            shlex.split(cmd), cwd=wdir, env=e, text=True
        )
    if r.returncode != 0:
        _c().print(
            f"[red]Exited {r.returncode}[/] — [white]{label}[/] failed", style="red"
        )
        _suggest_fix_training()
    return r.returncode


# --- P1 -------------------------------------------------------------------


def cmd_p1_init() -> int:
    _load_dotenv()
    _announce("p1-init", f"openenv init sqlsage-env  (cwd: {ROOT.parent})")
    return _run(
        "p1-init", ["openenv", "init", "sqlsage-env"], cwd=ROOT.parent, shell=False
    )


def cmd_p1_db() -> int:
    return _run(
        "p1-db",
        [
            "docker",
            "run",
            "--name",
            "sqlsage-pg",
            "-e",
            "POSTGRES_PASSWORD=sqlsage",
            "-e",
            "POSTGRES_DB=sqlsage",
            "-p",
            "5432:5432",
            "-d",
            "postgres:16",
        ],
    )


def cmd_p1_serve() -> int:
    return _run("p1-serve", ["uvicorn", "sqlsage.app:app", "--reload", "--port", "8000"])


def cmd_p1_push() -> int:
    _announce(
        "p1-push",
        "openenv push <HUGGINGFACE_HUB_USER>/sqlsage-env  (.env or prompt)",
    )
    user = _get_hf_username()
    if not user:
        _c().print("[red]Set HUGGINGFACE_HUB_USER in .env or enter when prompted.[/]")
        return 1
    return _run("p1-push", ["openenv", "push", f"{user}/sqlsage-env"])


def cmd_p1_test() -> int:
    # OpenEnv exposes POST /reset only (JSON body may be {}).
    cmd = (
        'curl -sS -X POST -H "Content-Type: application/json" -d "{}" '
        f'"http://localhost:8000/reset" | {sys.executable} -m json.tool'
    )
    return _run("p1-test", cmd, shell=True)


def cmd_p1_deploy() -> int:
    r = cmd_p1_push()
    if r != 0:
        return r
    base = _get_hf_space_url()
    if not base:
        _c().print("[red]Set SQLSAGE_HF_SPACE_URL (or SQLSAGE_ENV_URL) in .env.[/]")
        return 1
    b = base.rstrip("/")
    u = f"{b}/reset" if not b.endswith("reset") else b
    if not u.startswith("http"):
        u = "https://" + u.lstrip("/")
    cmd = (
        "curl -sS -X POST -H \"Content-Type: application/json\" -d "
        f"\"{{}}\" {shlex.quote(u)} | {shlex.quote(sys.executable)} -m json.tool"
    )
    return _run("p1-deploy  (then curl POST HF /reset + json.tool)", cmd, shell=True)


# --- P2 -------------------------------------------------------------------


def _pg_dsn() -> dict[str, Any]:
    _load_dotenv()
    return {
        "host": os.environ.get("POSTGRES_HOST", "127.0.0.1"),
        "port": int(os.environ.get("POSTGRES_PORT", "5433")),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD", "sqlsage"),
        "dbname": os.environ.get("POSTGRES_DB", "sqlsage"),
    }


def cmd_p2_explain() -> int:
    _announce("p2-explain", "psycopg2: EXPLAIN (ANALYZE, FORMAT JSON) for your query")
    sql = input("SQL to EXPLAIN: ").strip().rstrip(";")
    if not sql:
        _c().print("[red]Empty query.[/]")
        return 1
    try:
        import psycopg2

        c = _pg_dsn()
        conn = psycopg2.connect(
            host=c["host"],
            port=c["port"],
            user=c["user"],
            password=c["password"],
            dbname=c["dbname"],
        )
        ex = f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}"
        cur = conn.cursor()
        cur.execute(ex)
        row = cur.fetchone()
        conn.close()
        if not row or not row[0]:
            _c().print("[red]No result[/]")
            return 1
        plan = row[0]
        if isinstance(plan, str):
            data = json.loads(plan)
        else:
            data = plan
        _c().print(
            json.dumps(data, indent=2),
            style="white",
        )
    except Exception as e:  # noqa: BLE001
        _c().print(f"[red]{e}[/]")
        _suggest_fix_training()
        return 1
    return 0


def cmd_p2_test() -> int:
    return _run("p2-test", [sys.executable, "-m", "pytest", "tests/", "-v"])


def cmd_p2_reward() -> int:
    _announce("p2-reward", "compute_reward() demo with sample before/after plans")
    try:
        from sqlsage.reward import compute_reward, format_reward_breakdown
    except Exception as e:  # noqa: BLE001
        _c().print(f"[red]{e}[/]")
        return 1
    old_p = {
        "seq_scans": 2,
        "nested_loops": 1,
        "hash_joins": 0,
        "total_cost": 50000,
        "rows": 1e6,
        "node_type": "foo",
    }
    new_p = {
        "seq_scans": 0,
        "nested_loops": 0,
        "hash_joins": 1,
        "total_cost": 1000,
        "rows": 5e4,
        "node_type": "bar",
    }
    total, b = compute_reward(8420.0, 310.0, new_p, old_p, step_number=0)
    _c().print(
        f"[white]Total normalized reward:[/] [white]{total:.4f}[/]\n{format_reward_breakdown(b)}"
    )
    return 0


def cmd_p2_anticheat() -> int:
    return _run("p2-anticheat", [sys.executable, "fix_training.py", "--issue", "reward_hacking"])


def cmd_p2_plots() -> int:
    return _run("p2-plots", [sys.executable, "plots/generate_plots.py"])


def cmd_p2_wandb() -> int:
    return _run("p2-wandb", ["wandb", "login"])


# --- P3 -------------------------------------------------------------------


def cmd_p3_gpu() -> int:
    _announce("p3-gpu", "torch: device name + total VRAM (if available)")
    try:
        import torch
    except ImportError as e:
        _c().print(f"[red]install torch: {e}[/]")
        return 1
    if not torch.cuda.is_available():
        _c().print(
            "[white]CUDA not available in this process (expect GPU Colab for training).[/]"
        )
        return 0
    _c().print(
        f"[white]{torch.cuda.get_device_name(0)}[/]",
        style="white",
    )
    try:
        mem = int(torch.cuda.get_device_properties(0).total_memory)
        _c().print(
            f"[white]device memory ≈ {mem / 1_073_741_824.0:.2f} GiB[/]",
            style="white",
        )
    except Exception:  # noqa: BLE001
        pass
    return 0


def cmd_p3_benchmark() -> int:
    return _run("p3-benchmark", [sys.executable, "run_benchmark.py"])


def cmd_p3_save() -> int:
    _announce("p3-save", "Unsloth/TRL merged export (read carefully)")
    _c().print(
        "[white]Use Unsloth’s merged 16-bit save in Colab (per notebook), e.g.:\n"
        "  [yellow]model.save_pretrained_merged("
        "'sqlsage-trained', tokenizer, save_method='merged_16bit')[/]\n"
        "Do not use raw [yellow]model.save_pretrained(...)[/] alone if your workflow expects a merged model for vLLM/HF.[/]\n"
        "Also push with [yellow]huggingface-cli upload ...[/] or the Hub notebook cell after [yellow]huggingface-cli login[/]."
    )
    return 0


def cmd_p3_verify() -> int:
    _announce("p3-verify", "training pipeline checks 1–5 (env URL, DB path, W&B, plots, benchmark JSON)")
    _load_dotenv()
    ok1 = ok2 = ok3 = ok4 = ok5 = False
    c = _c()
    u = (os.environ.get("SQLSAGE_HF_SPACE_URL") or os.environ.get("SQLSAGE_ENV_URL") or "").strip()
    if u:
        try:
            import requests

            b = u.rstrip("/")
            test = f"{b}/reset" if not b.lower().rstrip("/").endswith("reset") else b
            if not test.startswith("http"):
                test = "https://" + test.lstrip("/")
            r = requests.post(test, json={}, timeout=5)
            ok1 = r.status_code == 200
        except Exception:
            ok1 = False
    if not u:
        c.print(
            "[dim][1] No SQLSAGE_HF_SPACE_URL / SQLSAGE_ENV_URL — set for remote /reset check[/]"
        )
    if not u:
        t1 = "SKIP"
    elif ok1:
        t1 = "PASS"
    else:
        t1 = "FAIL"
    c.print(
        f"[1]  [white]{t1:4}  HF POST /reset (if URL in .env) — HTTP 200 = ok[/]"
    )
    p = ROOT / "results" / "benchmark_results.json"
    ok2 = p.is_file() and p.stat().st_size > 4
    c.print(
        f"[2]  [white]{'PASS' if ok2 else 'FAIL':4}  benchmark_results.json present[/]"
    )
    ok3 = bool((os.environ.get("WANDB_API_KEY") or "").strip())
    c.print(
        f"[3]  [white]{'PASS' if ok3 else 'FAIL':4}  WANDB_API_KEY in env[/] "
        f"(or run [yellow]python -m sqlsage.run p2-wandb[/])"
    )
    plots = list((ROOT / "plots").glob("*.png")) if (ROOT / "plots").is_dir() else []
    ok4 = len(plots) >= 3
    c.print(
        f"[4]  [white]{'PASS' if ok4 else 'FAIL':4}  ≥3 plot PNGs in plots/[/] "
        f"([dim]{len(plots)} found[/])"
    )
    readme = ROOT / "README.md"
    t = (readme.read_text(encoding="utf-8", errors="replace") if readme.is_file() else "").lower()
    ok5 = "hf.space" in t and "colab" in t and "wandb" in t and (
        "youtube" in t or "huggingface.co/blog" in t
    )
    c.print(
        f"[5]  [white]{'PASS' if ok5 else 'FAIL':4}  README: hf.space + colab + wandb + (youtube|blog)[/]"
    )
    check1 = (not u) or ok1
    if not (check1 and ok2 and ok3 and ok4 and ok5):
        c.print(
            "\n[dim]Full milestone JSON: [white]python -m sqlsage.status_checker --json[/][/]"
        )
    return 0 if (check1 and ok2 and ok3 and ok4 and ok5) else 1


def cmd_p3_video() -> int:
    cands = [
        ROOT / "docs" / "PERSON3_PHASE8_MANUAL.md",
        ROOT / "docs" / "PHASE8_CLOSEOUT_CHECKLIST.md",
    ]
    path = next((p for p in cands if p.is_file()), None)
    if not path:
        _c().print("[red]No docs found for demo rehearsal.[/]")
        return 1
    rel = str(path)
    if str(ROOT) in str(path) and str(path).startswith(str(ROOT) + os.sep):
        rel = str(path).replace(str(ROOT) + os.sep, "", 1)
    _announce("p3-video", f"less {rel} (Section §7/§8 style demo checklist)")
    return _run("p3-video (less — q to quit)", f'less {shlex.quote(str(path))}', shell=True)


# --- shared ---------------------------------------------------------------


def cmd_status() -> int:
    return _run("status", [sys.executable, "-m", "sqlsage.status_checker"])


def cmd_dashboard(n: int) -> int:
    dash = ROOT / "dashboard.py"
    if not dash.is_file():
        return _run(
            f"dashboard {n}", [sys.executable, "-m", "sqlsage.dashboard", "--person", str(n)]
        )
    return _run(f"dashboard {n}", [sys.executable, str(dash), "--person", str(n)])


def cmd_crisis() -> None:
    _announce("crisis", "CRISIS PROTOCOL (from runbook)")
    text = """
[bold cyan]Missed Hour 8  (HF Space not live):[/]
  1. Check Docker: [yellow]docker ps | grep sqlsage-pg[/]
  2. Check FastAPI logs: [yellow]docker logs sqlsage-pg[/]  (if API is in another container, use that name)
  3. Check HF Spaces [white]build logs in the UI[/]
  4. If still failing after 30 min: [white]run environment locally and use ngrok as fallback[/]

[bold cyan]Missed Hour 12 (reward still zero):[/]
  1. [yellow]python -m sqlsage.run p2-reward[/] — verify compute_reward returns > 0 for a fast rewrite
  2. [yellow]python fix_training.py --issue flat_reward[/]
  3. Print 5 raw model outputs — are they all the same?
  4. Check prompt: does the model see the EXPLAIN plan? Print the full prompt.

[bold cyan]Missed Hour 18 (training not done):[/]
  1. [white]Reduce to 100 episodes immediately[/]
  2. [yellow]python plots/generate_plots.py[/] — generate plots from whatever episodes you have
  3. [yellow]git add plots/ && git commit -m 'partial training plots'[/]
  4. Document in README: "100 episodes shown — full run in progress"

[bold cyan]Missed Hour 21 (no benchmark numbers):[/]
  1. [yellow]python run_benchmark.py[/] — runs in ~5 minutes
  2. If model not saved: [white]use untrained baseline with an honest note[/]
  3. [red]NEVER invent numbers.[/] Use real numbers from any run, even partial.
""".strip()
    _c().print(text)


def cmd_submit() -> None:
    _announce("submit", "§9-style submission checklist (PHASE8 closeout)")
    lines = [
        "[ ] Space URL works (open in browser, /reset returns JSON)",
        "[ ] Colab / notebook link opens and has GPU / deps instructions",
        "[ ] Public GitHub repo is up to date (latest plots, results/)",
        "[ ] W&B: entity/project and at least one run with reward history",
        "[ ] `results/benchmark_results.json` (or doc’d numbers from run_benchmark.py)",
        "[ ] `plots/*.png` committed with real W&B data (Person 2)",
        "[ ] README: problem, env, reward, anti-cheat, training, baseline vs trained table, **links** (HF, Colab, video/blog, repo)",
        "[ ] Demo video (≤2 min) or blog URL in README (§7 PERSON3 doc)",
        "[ ] Google Form filled before deadline (all required links)",
    ]
    for line in lines:
        _c().print(f"[white]{line}[/]")


def cmd_help() -> None:
    from rich import box
    from rich.table import Table

    t = Table(
        title="[bold]SQLSage [cyan]run.py[/] aliases[/]",
        box=box.ROUNDED,
        show_header=True,
    )
    t.add_column("Alias", style="bold cyan", no_wrap=True)
    t.add_column("What it does", style="white")
    rows: list[tuple[str, str]] = [
        ("p1-init", "openenv init sqlsage-env (parent directory)"),
        ("p1-db", "Start Postgres 16 in Docker (port 5432)"),
        ("p1-serve", "uvicorn sqlsage.app:app --reload :8000"),
        ("p1-push", "openenv push <user>/sqlsage-env (.env / prompt)"),
        ("p1-test", "curl /reset on localhost:8000 | json.tool"),
        ("p1-deploy", "p1-push, then test HF /reset with curl"),
        ("p2-explain", "Prompt for SQL → EXPLAIN (ANALYZE, FORMAT JSON) (pretty)"),
        ("p2-test", "pytest tests/ -v"),
        ("p2-reward", "Demo compute_reward() + per-component breakdown"),
        ("p2-anticheat", "fix_training.py --issue reward_hacking"),
        ("p2-plots", "plots/generate_plots.py"),
        ("p2-wandb", "wandb login"),
        ("p3-gpu", "torch: CUDA device name + memory (if any)"),
        ("p3-benchmark", "python run_benchmark.py"),
        ("p3-save", "Print correct merged save + upload guidance"),
        ("p3-verify", "5 quick environment/training/benchmark/README checks"),
        ("p3-video", "less docs (demo + submission rehearsal)"),
        ("status", "python -m sqlsage.status_checker (milestone JSON)"),
        ("dashboard 1|2|3", "python dashboard.py --person N (or -m sqlsage.dashboard)"),
        ("crisis", "CRISIS PROTOCOL for missed hours"),
        ("submit", "Submission checklist with checkboxes"),
        ("help", "This table"),
    ]
    for a, b in rows:
        t.add_row(a, b)
    _c().print(t)


# --- main -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        cmd_help()
        return 0
    a0 = argv[0]
    if a0 in ("-h", "--help", "help"):
        cmd_help()
        return 0
    if a0 == "dashboard" and len(argv) < 2:
        _c().print(
            "[red]Usage: python -m sqlsage.run dashboard 1|2|3[/]"
        )
        return 1
    if a0 == "dashboard" and len(argv) >= 2:
        n = argv[1]
        if n not in ("1", "2", "3"):
            _c().print(f"[red]Person must be 1, 2, or 3, got {n!r}[/]")
            return 1
        return cmd_dashboard(int(n))
    m: dict[str, Callable[[], int | None] | None] = {
        "p1-init": cmd_p1_init,
        "p1-db": cmd_p1_db,
        "p1-serve": cmd_p1_serve,
        "p1-push": cmd_p1_push,
        "p1-test": cmd_p1_test,
        "p1-deploy": cmd_p1_deploy,
        "p2-explain": cmd_p2_explain,
        "p2-test": cmd_p2_test,
        "p2-reward": cmd_p2_reward,
        "p2-anticheat": cmd_p2_anticheat,
        "p2-plots": cmd_p2_plots,
        "p2-wandb": cmd_p2_wandb,
        "p3-gpu": cmd_p3_gpu,
        "p3-benchmark": cmd_p3_benchmark,
        "p3-save": cmd_p3_save,
        "p3-verify": cmd_p3_verify,
        "p3-video": cmd_p3_video,
        "status": cmd_status,
        "crisis": None,
        "submit": None,
    }
    if a0 not in m:
        _c().print(
            f"[red]Unknown alias {a0!r}. Use [white]python -m sqlsage.run help[/][/]"
        )
        return 1
    if a0 == "crisis":
        cmd_crisis()
        return 0
    if a0 == "submit":
        cmd_submit()
        return 0
    if a0 == "help":
        cmd_help()
        return 0
    fn = m[a0]
    if fn is None:
        return 1
    r = fn()
    return 0 if r is None else int(r)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        _c().print("\n[dim]Interrupted.[/]")
