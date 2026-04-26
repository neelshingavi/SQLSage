"""
Automated hackathon milestone checks. Used by the dashboard; usable standalone.

  cd sqlsage-env
  python -m sqlsage.status_checker

Set ``SQLSAGE_HF_SPACE_URL`` and/or ``SQLSAGE_WANDB_RUN_ID`` when not passing args.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from typing import Any

# Same hackathon window as ``dashboard.py`` (keep in sync; avoids circular imports)
HACKATHON_START = datetime(2026, 4, 25, 17, 0, 0)
HACKATHON_END = datetime(2026, 4, 26, 17, 0, 0)


def _now() -> datetime:
    return datetime.now().replace(microsecond=0)


def get_current_hour() -> float:
    n = _now()
    if n < HACKATHON_START:
        return 0.0
    if n >= HACKATHON_END:
        return 24.0
    delta = n - HACKATHON_START
    h = delta.total_seconds() / 3600.0
    return min(24.0, h)


def get_time_remaining() -> str:
    n = _now()
    if n >= HACKATHON_END:
        return "DEADLINE PASSED"
    sec = (HACKATHON_END - n).total_seconds()
    if sec <= 0:
        return "DEADLINE PASSED"
    h, r = divmod(int(sec), 3600)
    m, _ = divmod(r, 60)
    return f"{h}h {m:02d}m remaining"


# --- project root: sqlsage-env ---


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _reset_url(hf_space_url: str) -> str:
    u = (hf_space_url or "").strip().rstrip("/")
    if not u:
        return u
    if re.search(r"/reset(\?|#|$)", u):
        return u
    return f"{u}/reset"


# --- per-check implementations ---


def check_hf_space_live(hf_space_url: str) -> dict[str, Any]:
    """
    POST OpenEnv ``/reset`` (JSON body ``{}``) at ``_reset_url(hf_space_url)`` with a short timeout.

    OpenEnv registers only ``POST /reset``; ``GET`` would return 405. Success requires HTTP 200.
    Returns: ``{"ok": bool, "latency_ms": int, "http_status": int|None, "error": None|str}``
    """
    t0 = time.perf_counter()
    try:
        u = (hf_space_url or "").strip()
        if not u:
            return {
                "ok": False,
                "latency_ms": 0,
                "http_status": None,
                "error": "empty hf_space_url",
            }
        try:
            import requests
        except ImportError as e:
            return {
                "ok": False,
                "latency_ms": 0,
                "http_status": None,
                "error": f"requests not available: {e}",
            }
        full = _reset_url(u)
        r = requests.post(
            full,
            json={},
            timeout=float(os.environ.get("SQLSAGE_DASHBOARD_HTTP_TIMEOUT", "5")),
        )
        ms = int((time.perf_counter() - t0) * 1000)
        ok = r.status_code == 200
        return {
            "ok": ok,
            "latency_ms": ms,
            "http_status": r.status_code,
            "error": (None if ok else f"HTTP {r.status_code}"),
        }
    except Exception as e:  # noqa: BLE001 — intentional graceful reporting
        ms = int((time.perf_counter() - t0) * 1000)
        return {
            "ok": False,
            "latency_ms": ms,
            "http_status": None,
            "error": str(e),
        }


def check_wandb_nonzero_reward(wandb_run_id: str) -> dict[str, Any]:
    """
    W&B API: reward/mean (or any *reward* series). Pass if any value > 0.5.
    ``wandb_run_id`` is the full run path, e.g. ``entity/project/run_id``.
    """
    rid = (wandb_run_id or "").strip()
    if not rid:
        return {
            "ok": False,
            "max_reward_seen": 0.0,
            "episodes_logged": 0,
            "error": "empty wandb_run_id",
        }
    try:
        import wandb
    except ImportError as e:
        return {
            "ok": False,
            "max_reward_seen": 0.0,
            "episodes_logged": 0,
            "error": f"wandb not installed: {e}",
        }
    try:
        api = wandb.Api(
            timeout=int(os.environ.get("SQLSAGE_WANDB_API_TIMEOUT", "30"))
        )
        run = api.run(rid)
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "max_reward_seen": 0.0,
            "episodes_logged": 0,
            "error": str(e),
        }

    max_r = 0.0
    eps = 0
    err: str | None = None
    _cap = 50_000
    try:
        for row in run.scan_history():
            if not isinstance(row, dict):
                continue
            for k, v in row.items():
                if "reward" in k.lower() and isinstance(v, (int, float)) and v == v:
                    max_r = max(max_r, float(v))
            eps += 1
            if eps >= _cap:
                break
    except Exception as e:  # noqa: BLE001
        err = str(e)
    if max_r <= 0.5:
        try:
            s = getattr(run, "summary", None)
            if s is not None and hasattr(s, "items"):
                for k, v in s.items():
                    if "reward" in k.lower() and isinstance(v, (int, float)) and v == v:
                        max_r = max(max_r, float(v))
                st = s.get("_step") if hasattr(s, "get") else None
                if st is not None:
                    try:
                        eps = max(eps, int(st or 0))
                    except (TypeError, ValueError):
                        pass
        except Exception as e2:  # noqa: BLE001
            if err is None:
                err = str(e2)

    ok = max_r > 0.5
    return {
        "ok": ok,
        "max_reward_seen": round(max_r, 4),
        "episodes_logged": eps,
        "error": err,
    }


def check_plots_committed() -> dict[str, Any]:
    """
    ≥3 PNGs under ``plots/``; recent git commit mentioning ``plots`` in the last 24h.
    """
    from pathlib import Path

    root = Path(_project_root())
    pdir = root / "plots"
    plots_found: list[str] = []
    try:
        if pdir.is_dir():
            for p in sorted(pdir.iterdir()):
                if p.suffix.lower() == ".png" and p.is_file():
                    plots_found.append(p.name)
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "plots_found": [],
            "committed": False,
            "error": str(e),
        }

    committed = False
    try:
        r = subprocess.run(
            [
                "git",
                "log",
                "--since=24.hours.ago",
                "-20",
                "--oneline",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(root),
        )
        logtxt = (r.stdout or "") if r.returncode == 0 else ""
        for line in logtxt.splitlines():
            if "plot" in line.lower():
                committed = True
                break
        if not committed:
            r2 = subprocess.run(
                [
                    "git",
                    "log",
                    "-1",
                    "--oneline",
                    "--",
                    "plots/",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(root),
            )
            if r2.returncode == 0 and (r2.stdout or "").strip():
                out = (r2.stdout or "").lower()
                committed = "plot" in out
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "plots_found": plots_found,
            "committed": False,
            "error": str(e),
        }

    ok = len(plots_found) >= 3 and committed
    return {
        "ok": ok,
        "plots_found": plots_found,
        "committed": committed,
        "error": None,
    }


def _best_speedup_from_payload(payload: dict[str, Any]) -> tuple[str, float, int]:
    """Return (summary string, best ratio, n queries)."""
    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        return ("", 0.0, 0)
    best = 0.0
    best_id = ""
    for q in queries:
        if not isinstance(q, dict):
            continue
        um = q.get("untrained_ms")
        tm = q.get("trained_ms")
        qid = str(q.get("query_id", "?"))
        try:
            if (
                um is not None
                and tm is not None
                and float(um) == float(um)
                and float(tm) == float(tm)
                and float(tm) > 0
            ):
                su = float(um) / float(tm)
                if su == su and su > best:
                    best, best_id = su, qid
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    s = f"{best:.1f}x on {best_id}" if best > 0 and best == best else ""
    return (s, best, len(queries))


def check_benchmark_results_exist() -> dict[str, Any]:
    from pathlib import Path

    p = Path(_project_root()) / "results" / "benchmark_results.json"
    if not p.is_file():
        return {
            "ok": False,
            "queries_benchmarked": 0,
            "best_speedup": "",
            "error": f"missing {p.name}",
        }
    try:
        payload = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "queries_benchmarked": 0,
            "best_speedup": "",
            "error": str(e),
        }
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "queries_benchmarked": 0,
            "best_speedup": "",
            "error": "invalid JSON structure",
        }
    qlist = payload.get("queries", [])
    n = len(qlist) if isinstance(qlist, list) else 0
    best_s, _ratio, nq = _best_speedup_from_payload(payload)
    if nq == 0 and isinstance(qlist, list):
        nq = n
    ok = n >= 3
    return {
        "ok": ok,
        "queries_benchmarked": nq or n,
        "best_speedup": best_s,
        "error": None if ok else "need at least 3 query entries",
    }


def check_readme_has_links() -> dict[str, Any]:
    from pathlib import Path

    p = Path(_project_root()) / "README.md"
    if not p.is_file():
        return {
            "ok": False,
            "missing_links": ["hf.space", "colab", "wandb", "youtube|huggingface blog"],
            "error": "README.md not found",
        }
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "missing_links": ["readme"],
            "error": str(e),
        }
    t = text.lower()
    has_hf = "hf.space" in t
    has_colab = "colab" in t
    has_wandb = "wandb" in t
    has_video = "youtube" in t or "huggingface.co/blog" in t
    missing: list[str] = []
    if not has_hf:
        missing.append("hf.space")
    if not has_colab:
        missing.append("colab")
    if not has_wandb:
        missing.append("wandb")
    if not has_video:
        missing.append("youtube or huggingface.co/blog")
    ok = not missing
    return {
        "ok": ok,
        "missing_links": missing,
        "error": None,
    }


def _time_remaining_short() -> str:
    s = get_time_remaining()
    if s == "DEADLINE PASSED":
        return s
    return s.replace(" remaining", "").strip()


def _milestone_time_status(
    gate_hour: float, ch: float, satisfied: bool
) -> tuple[str, str]:
    """
    Return (IN_PROGRESS|UPCOMING|DONE, empty detail for gate).
    """
    h = float(gate_hour)
    if satisfied:
        return "DONE", ""
    if ch < h - 1.0:
        return "UPCOMING", ""
    if h - 1.0 <= ch < h or (h <= ch <= h + 0.5):
        return "IN_PROGRESS", ""
    if ch > h + 0.5 and not satisfied:
        return "IN_PROGRESS", "missed"
    return "UPCOMING", ""


def check_all_milestones(hf_space_url: str, wandb_run_id: str) -> dict[str, Any]:
    """
    Run all checks, map to hours, set overall_status and optional crisis_message.
    Fills empty ``hf_space_url`` / ``wandb_run_id`` from the environment.
    """
    h_url = (hf_space_url or "").strip() or os.environ.get("SQLSAGE_HF_SPACE_URL", "")
    w_id = (wandb_run_id or "").strip() or os.environ.get("SQLSAGE_WANDB_RUN_ID", "")

    ch = get_current_hour()
    tr = _time_remaining_short()

    hf = check_hf_space_live(h_url)
    wb = check_wandb_nonzero_reward(w_id)
    pl = check_plots_committed()
    bm = check_benchmark_results_exist()
    rd = check_readme_has_links()

    def det_hf() -> str:
        if hf.get("ok"):
            return f"latency {hf.get('latency_ms', 0)}ms"
        return hf.get("error") or "not live"

    def det_wb() -> str:
        if wb.get("ok"):
            return f"max reward {wb.get('max_reward_seen', 0)} (threshold >0.5)"
        m = float(wb.get("max_reward_seen") or 0.0)
        err = wb.get("error")
        if err:
            return f"max {m} — {err}"
        return f"max reward {m} (need >0.5)"

    def det_pl() -> str:
        if pl.get("ok"):
            n = len(pl.get("plots_found", []) or [])
            c = "yes" if pl.get("committed") else "no"
            return f"{n} png(s), committed={c}"
        perr = pl.get("error")
        return (perr and str(perr)) or f"need ≥3 png + git; found {pl.get('plots_found')}"

    def det_bm() -> str:
        if bm.get("ok"):
            b = bm.get("best_speedup", "")
            n = bm.get("queries_benchmarked", 0)
            return f"{n} queries; best {b}" if b else f"{n} queries"
        return bm.get("error") or "benchmarks incomplete"

    def det_rd() -> str:
        if rd.get("ok"):
            return "all required link tokens present"
        miss = rd.get("missing_links", [])
        return f"missing: {', '.join(miss)}" if miss else "README check failed"

    # Per-gate check satisfaction (independent of clock)
    s8 = bool(hf.get("ok"))
    s12 = bool(wb.get("ok"))
    s18 = bool(pl.get("ok"))
    s21 = bool(bm.get("ok"))
    s23 = bool(rd.get("ok"))

    st8, _ = _milestone_time_status(8, ch, s8)
    st8 = "DONE" if s8 else st8
    st12, _ = _milestone_time_status(12, ch, s12)
    st12 = "DONE" if s12 else st12
    st18, _ = _milestone_time_status(18, ch, s18)
    st18 = "DONE" if s18 else st18
    st21, _ = _milestone_time_status(21, ch, s21)
    st21 = "DONE" if s21 else st21
    st23, _ = _milestone_time_status(23, ch, s23)
    st23 = "DONE" if s23 else st23
    st24 = "UPCOMING"
    d24 = "Manual — cannot auto-check"

    # Crisis: past grace + check failed (only if env was set so a real attempt was made)
    crisis: str | None = None
    if h_url and ch > 8.5 and not s8:
        crisis = "CRISIS: Hour 8 milestone missed — HF Space not live"
    elif w_id and ch > 12.5 and not s12:
        crisis = "CRISIS: Hour 12 milestone missed — reward still ≤0.5 (or W&B unreachable)"
    elif ch > 18.5 and not s18:
        crisis = "CRISIS: Hour 18 milestone — need ≥3 PNGs in plots/ + commit mentioning plots"
    elif ch > 21.5 and not s21:
        crisis = "CRISIS: Hour 21 milestone — benchmark results incomplete"
    elif ch > 23.5 and not s23:
        crisis = "CRISIS: Hour 23 milestone — README links incomplete"

    nxt: list[tuple[str, float]] = []
    for k, h in (("8", 8.0), ("12", 12.0), ("18", 18.0), ("21", 21.0), ("23", 23.0), ("24", 24.0)):
        nxt.append((k, h))
    hours_left: list[float] = [h - ch for _k, h in nxt if h > ch + 0.0001]
    d_next = min(hours_left) if hours_left else None

    overall = "GREEN"
    if crisis:
        overall = "RED"
    elif d_next is not None and d_next < 0.5:
        overall = "RED"
    elif d_next is not None and d_next < 2.0:
        overall = "YELLOW"

    return {
        "checked_at_hour": round(ch, 2),
        "time_remaining": tr,
        "checks_raw": {
            "hf": hf,
            "wandb": wb,
            "plots": pl,
            "benchmark": bm,
            "readme": rd,
        },
        "milestones": {
            "hour_8": {
                "gate": "HF Space live",
                "status": st8,
                "detail": det_hf(),
            },
            "hour_12": {
                "gate": "Non-zero reward",
                "status": st12,
                "detail": det_wb() if w_id else "set SQLSAGE_WANDB_RUN_ID (or pass wandb run path)",
            },
            "hour_18": {
                "gate": "Plots committed",
                "status": st18,
                "detail": det_pl(),
            },
            "hour_21": {
                "gate": "Benchmark results",
                "status": st21,
                "detail": det_bm(),
            },
            "hour_23": {
                "gate": "README links complete",
                "status": st23,
                "detail": det_rd(),
            },
            "hour_24": {
                "gate": "Form submitted",
                "status": st24,
                "detail": d24,
            },
        },
        "overall_status": overall,
        "crisis_message": crisis,
    }


def _cli() -> None:
    import argparse
    import pprint

    p = argparse.ArgumentParser(description="Run SQLSage hackathon status checks (JSON out).")
    p.add_argument("--hf", default=None, help="Override SQLSAGE_HF_SPACE_URL")
    p.add_argument("--wandb", default=None, help="Override SQLSAGE_WANDB_RUN_ID (entity/project/run)")
    p.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a pretty object dump",
    )
    a = p.parse_args()
    env_hf = a.hf or os.environ.get("SQLSAGE_HF_SPACE_URL", "")
    env_w = a.wandb or os.environ.get("SQLSAGE_WANDB_RUN_ID", "")
    out = check_all_milestones(env_hf, env_w)
    if a.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        pprint.pprint(out, width=100)


if __name__ == "__main__":
    _cli()
