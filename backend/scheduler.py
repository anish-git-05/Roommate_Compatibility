from __future__ import annotations

from typing import Any

from backend.database import SessionLocal
from backend.services.retraining import run_feedback_batch_job

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except ImportError:  # pragma: no cover - depends on optional runtime install
    BackgroundScheduler = None


scheduler: Any = None


def _run_nightly_batch_job() -> None:
    db = SessionLocal()
    try:
        result = run_feedback_batch_job(db)
        print(f"Nightly retraining job finished: {result}")
    except Exception as exc:  # pragma: no cover - background logging path
        db.rollback()
        print(f"Nightly retraining job failed: {exc}")
    finally:
        db.close()


def start_scheduler() -> None:
    global scheduler

    if BackgroundScheduler is None:
        print("APScheduler is not installed. Nightly batch retraining scheduler is disabled.")
        return

    if scheduler is not None and scheduler.running:
        return

    scheduler = BackgroundScheduler(timezone="Asia/Calcutta")
    scheduler.add_job(
        _run_nightly_batch_job,
        trigger="cron",
        hour=0,
        minute=0,
        id="nightly-roommate-retraining",
        replace_existing=True,
    )
    scheduler.start()
    print("Nightly retraining scheduler started. Next run is at 00:00 Asia/Calcutta.")


def stop_scheduler() -> None:
    global scheduler

    if scheduler is None:
        return

    scheduler.shutdown(wait=False)
    scheduler = None
