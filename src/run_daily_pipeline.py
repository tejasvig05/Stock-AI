"""
run_daily_pipeline.py
Master orchestration script -- runs the full pipeline in sequence and
generates a plain-text "morning report" summarizing today's top picks.

Designed to run once daily via Windows Task Scheduler, before market
open. This is what makes the project's original goal real: "before the
market starts, analyze and suggest" -- rather than the user manually
running five separate scripts each morning.

Logs everything to a timestamped log file so failures are diagnosable
without watching it run live.

Run manually to test:
    python src/run_daily_pipeline.py

Set up as a scheduled task (see accompanying instructions) to run
automatically each morning.
"""

import subprocess
import sys
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOG_DIR, f"pipeline_{TIMESTAMP}.log")


def log(message: str):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(description: str, script_path: str) -> bool:
    """Runs a pipeline step as a subprocess, logs output, returns success."""
    log(f"START: {description}")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=3600,
        )
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(result.stdout)
            f.write(result.stderr)

        if result.returncode == 0:
            log(f"SUCCESS: {description}")
            return True
        else:
            log(f"FAILED: {description} (exit code {result.returncode}) -- see log for details")
            return False
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT: {description} exceeded 1 hour -- skipped")
        return False
    except Exception as e:
        log(f"ERROR: {description} -- {e}")
        return False


def generate_morning_report():
    """Builds a plain-text summary of today's top long-term picks and any
    strong short-term signals, saved as the daily 'report to check
    before market open'."""
    import pandas as pd

    report_lines = [
        f"{'='*60}",
        f"DAILY MARKET ANALYSIS REPORT -- {datetime.now().strftime('%A, %B %d, %Y')}",
        f"{'='*60}",
        "",
        "DISCLAIMER: Educational/informational only. Not financial advice.",
        "Stock market investments are subject to market risk.",
        "",
    ]

    try:
        scores_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/long_term_scores.csv"))
        top_picks = scores_df.nlargest(10, "total_score")[
            ["symbol", "sector", "cap_category", "total_score", "recommendation"]
        ]
        report_lines.append("TOP 10 LONG-TERM FUNDAMENTAL SCORES TODAY:")
        report_lines.append(top_picks.to_string(index=False))
        report_lines.append("")
    except FileNotFoundError:
        report_lines.append("(Long-term scores not available -- run long_term_scorer.py)")

    report_path = os.path.join(PROJECT_ROOT, "data", "processed",
                                f"morning_report_{datetime.now().strftime('%Y-%m-%d')}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    log(f"Morning report saved to {report_path}")
    return report_path


def archive_and_reset_cache():
    """Moves yesterday's universe cache into a dated archive folder, then
    clears the working checkpoint -- WITHOUT this, batch_fetch_universe.py's
    resume logic would skip every stock forever after the first run, since
    they'd all already be marked 'done'. This ensures each scheduled run
    does a genuine fresh refetch while keeping history for backtesting/audit."""
    import shutil

    cache_path = os.path.join(PROJECT_ROOT, "data/processed/universe_fundamentals.csv")
    if os.path.exists(cache_path):
        archive_dir = os.path.join(PROJECT_ROOT, "data", "archive")
        os.makedirs(archive_dir, exist_ok=True)
        archive_path = os.path.join(archive_dir, f"universe_fundamentals_{datetime.now().strftime('%Y-%m-%d')}.csv")
        shutil.copy(cache_path, archive_path)
        os.remove(cache_path)
        log(f"Archived previous cache to {archive_path} and cleared for fresh fetch")
    else:
        log("No previous cache found -- running first-time fetch")


if __name__ == "__main__":
    log("="*60)
    log("DAILY PIPELINE STARTED")
    log("="*60)

    archive_and_reset_cache()

    steps = [
        ("Refresh full NSE universe fundamentals", "src/batch_fetch_universe.py"),
        ("Recompute long-term fundamental scores", "src/long_term_scorer.py"),
    ]

    all_success = True
    for description, script in steps:
        success = run_step(description, script)
        all_success = all_success and success

    report_path = generate_morning_report()

    log("="*60)
    log(f"PIPELINE COMPLETE -- {'ALL STEPS SUCCEEDED' if all_success else 'SOME STEPS FAILED (check log)'}")
    log(f"Log file: {LOG_FILE}")
    log(f"Morning report: {report_path}")
    log("="*60)