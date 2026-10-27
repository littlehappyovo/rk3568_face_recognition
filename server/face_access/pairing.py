"""Derive visits from immutable, time-ordered allowed recognition events."""

from datetime import datetime, timedelta, timezone


def utc_timestamp(value: str) -> str:
    """Validate an aware ISO timestamp and normalize for lexical SQLite order."""
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError("时间格式错误")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("时间必须包含时区")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def derive_visits(events: list[dict], timeout_seconds: int,
                  now_utc: str | None) -> tuple[list[dict], list[dict]]:
    """Call separately for each person; input must exclude denied/unknown events."""
    visits, anomalies = [], []
    pending = None

    def anomaly(event, kind, active=True):
        anomalies.append(dict(id=f"{event['event_id']}:{kind}",
                              event_id=event["event_id"], person_id=event["person_id"],
                              kind=kind, active=active))

    def overdue(entry, until):
        if not timeout_seconds or until is None:
            return False
        start = datetime.fromisoformat(entry["observed_at"].replace("Z", "+00:00"))
        finish = datetime.fromisoformat(until.replace("Z", "+00:00"))
        return finish >= start + timedelta(seconds=timeout_seconds)

    ordered = sorted(events, key=lambda e: (e["observed_at"], e["direction"] != "IN", e["event_id"]))
    for event in ordered:
        if event["direction"] == "IN":
            if pending is None:
                pending = event
            else:
                anomaly(event, "duplicate_entry")
        elif pending is None:
            anomaly(event, "exit_without_entry")
        else:
            if overdue(pending, event["observed_at"]):
                anomaly(pending, "missing_exit", active=False)
            visits.append(dict(id=pending["event_id"], person_id=pending["person_id"],
                               entry_event_id=pending["event_id"], exit_event_id=event["event_id"],
                               entered_at=pending["observed_at"], exited_at=event["observed_at"],
                               display_name=pending["display_name"], status="matched"))
            pending = None
    if pending is not None:
        expired = overdue(pending, now_utc)
        if expired:
            anomaly(pending, "missing_exit")
        visits.append(dict(id=pending["event_id"], person_id=pending["person_id"],
                           entry_event_id=pending["event_id"], exit_event_id=None,
                           entered_at=pending["observed_at"], exited_at=None,
                           display_name=pending["display_name"], status="overdue" if expired else "pending"))
    return visits, anomalies
