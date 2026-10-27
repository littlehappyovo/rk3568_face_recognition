import concurrent.futures
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from face_access.auth import AuthManager
from face_access.store import AccessStore, StoreConflict, StoreValidationError


class AccessStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "access" / "access.sqlite3"
        self.store = AccessStore(self.path)

    def event(self, at="2026-10-26T01:00:00.000Z", direction="IN",
              person="p1", decision="allowed", name="张三", version=None):
        if version is None and direction == "OUT":
            with self.store.connection() as db:
                row = db.execute("SELECT version FROM station_versions WHERE station_id='board-in' AND direction='OUT' ORDER BY version LIMIT 1").fetchone()
            if row:
                version = row[0]
            else:
                with self.store.connection(write=True) as db:
                    version = db.execute("SELECT coalesce(max(version),0)+1 FROM station_versions WHERE station_id='board-in'").fetchone()[0]
                    db.execute("INSERT INTO station_versions VALUES('board-in',?,'OUT')", (version,))
        return dict(event_id=str(uuid.uuid4()), station_id="board-in",
                    direction_version=version if version is not None else (1 if direction == "IN" else 2),
                    stream_id=str(uuid.uuid4()),
                    track_id=7, observed_at=at, person_id=person,
                    display_name=name, decision=decision)

    def ingest(self, **kwargs):
        event = self.event(**kwargs)
        self.store.ingest(event)
        return event

    def test_in_out_forms_one_visit(self):
        entry = self.ingest()
        leave = self.ingest(at="2026-10-26T02:00:00.000Z", direction="OUT")
        visits = self.store.list_visits({})
        self.assertEqual(visits["total"], 1)
        self.assertEqual(visits["items"][0]["entry_event_id"], entry["event_id"])
        self.assertEqual(visits["items"][0]["exit_event_id"], leave["event_id"])
        self.assertEqual(visits["items"][0]["status"], "matched")
        self.assertEqual(self.store.list_anomalies({})["total"], 0)

    def test_fresh_database_exposes_only_the_board_station(self):
        self.assertEqual([station["id"] for station in self.store.list_stations()], ["board-in"])

    def test_legacy_pc_history_survives_but_new_pc_events_are_rejected(self):
        with self.store.connection(write=True) as db:
            db.execute("INSERT OR IGNORE INTO stations VALUES('pc-out','电脑摄像头','OUT',1,1)")
            db.execute("INSERT OR IGNORE INTO station_versions VALUES('pc-out',1,'OUT')")
        historic = self.event(at="2026-10-26T01:00:00.000Z", direction="OUT")
        historic["station_id"] = "pc-out"
        historic["direction_version"] = 1
        self.store.ingest(historic)

        AuthManager(self.store)
        current = self.event(at=(datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat(), direction="OUT")
        current["station_id"] = "pc-out"
        current["direction_version"] = 1
        with self.assertRaises(StoreValidationError):
            self.store.ingest(current)
        self.assertTrue(self.store.ingest(historic)["duplicate"])
        self.assertEqual(self.store.list_events({"station_id": "pc-out"})["total"], 1)
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT enabled FROM stations WHERE id='pc-out'").fetchone()[0], 0)

    def test_duplicate_entry_does_not_replace_open_visit(self):
        first = self.ingest()
        duplicate = self.ingest(at="2026-10-26T01:10:00.000Z")
        self.ingest(at="2026-10-26T02:00:00.000Z", direction="OUT")
        self.assertEqual(self.store.list_visits({})["items"][0]["entry_event_id"], first["event_id"])
        anomaly = self.store.list_anomalies({})["items"][0]
        self.assertEqual((anomaly["kind"], anomaly["event_id"]),
                         ("duplicate_entry", duplicate["event_id"]))

    def test_exit_without_entry_is_anomalous(self):
        event = self.ingest(direction="OUT")
        self.assertEqual(self.store.list_visits({})["total"], 0)
        anomaly = self.store.list_anomalies({})["items"][0]
        self.assertEqual((anomaly["kind"], anomaly["event_id"]),
                         ("exit_without_entry", event["event_id"]))

    def test_retried_uuid_is_idempotent(self):
        event = self.event()
        self.assertEqual(self.store.ingest(event),
                         dict(event_id=event["event_id"], persisted=True, duplicate=False))
        self.assertTrue(self.store.ingest(event)["duplicate"])
        self.assertEqual(self.store.list_events({})["total"], 1)
        with self.assertRaises(StoreConflict):
            self.store.ingest(dict(event, display_name="另一姓名"))
        self.assertEqual(self.store.list_events({})["items"][0]["display_name"], "张三")

    def test_pending_is_not_missing_exit_until_deadline(self):
        self.ingest()
        self.assertEqual(self.store.expire_pending(datetime(2026, 10, 27, 0, 59, 59, 999000, tzinfo=timezone.utc)), 0)
        self.assertEqual(self.store.list_visits({})["items"][0]["status"], "pending")
        deadline = datetime(2026, 10, 27, 1, tzinfo=timezone.utc)
        self.assertEqual(self.store.expire_pending(deadline), 1)
        self.assertEqual(self.store.expire_pending(deadline), 0)
        self.assertEqual(self.store.list_visits({})["items"][0]["status"], "overdue")
        self.assertEqual(self.store.list_anomalies({})["items"][0]["kind"], "missing_exit")

    def test_late_exit_closes_visit_and_preserves_anomaly(self):
        self.ingest()
        self.store.expire_pending(datetime(2026, 10, 27, 1, tzinfo=timezone.utc))
        anomaly = self.store.list_anomalies({})["items"][0]
        self.store.resolve_anomaly(anomaly["id"], "已联系本人", "admin")
        self.ingest(at="2026-10-27T02:00:00.000Z", direction="OUT")
        self.assertEqual(self.store.list_visits({})["items"][0]["status"], "matched")
        self.assertEqual(self.store.list_anomalies({})["total"], 0)
        historical = self.store.list_anomalies({"include_history": True})["items"][0]
        self.assertFalse(historical["active"])
        self.assertEqual((historical["note"], historical["resolved_by"]), ("已联系本人", "admin"))

    def test_out_arrives_before_earlier_in(self):
        self.ingest(at="2026-10-26T02:00:00.000Z", direction="OUT")
        anomaly = self.store.list_anomalies({})["items"][0]
        self.store.resolve_anomaly(anomaly["id"], "记录延迟，待核对", "admin")
        self.ingest()
        self.assertEqual(self.store.list_visits({})["items"][0]["status"], "matched")
        self.assertEqual(self.store.list_anomalies({})["total"], 0)
        self.assertEqual(self.store.list_anomalies({"include_history": True})["items"][0]["note"],
                         "记录延迟，待核对")

    def test_same_timestamp_and_cross_midnight_pairing(self):
        self.ingest(at="2026-10-27T00:00:01.000Z", direction="OUT")
        self.ingest(at="2026-10-26T23:59:59.000Z")
        self.assertEqual(self.store.list_visits({})["items"][0]["status"], "matched")
        self.ingest(at="2026-10-27T01:00:00.000Z", direction="OUT", person="p2")
        self.ingest(at="2026-10-27T09:00:00.000+08:00", person="p2")
        self.assertEqual(self.store.list_visits({"person_id": "p2"})["items"][0]["status"], "matched")

    def test_denied_and_unknown_never_modify_visit(self):
        self.ingest()
        self.ingest(at="2026-10-26T01:10:00.000Z", person=None, decision="denied", name="")
        self.ingest(at="2026-10-26T01:20:00.000Z", direction="OUT", person=None, decision="denied", name="")
        self.ingest(at="2026-10-26T01:30:00.000Z", direction="OUT", person="p1", decision="denied")
        self.assertEqual(self.store.list_visits({})["items"][0]["status"], "pending")
        self.assertEqual(self.store.list_anomalies({})["total"], 0)
        self.assertEqual(self.store.list_events({"decision": "denied"})["total"], 3)

    def test_snapshots_survive_registry_changes(self):
        self.ingest(name="旧姓名")
        self.ingest(at="2026-10-26T02:00:00.000Z", direction="OUT", name="新姓名")
        self.assertEqual({e["display_name"] for e in self.store.list_events({})["items"]},
                         {"旧姓名", "新姓名"})
        reopened = AccessStore(self.path)
        self.assertEqual(reopened.list_events({})["total"], 2)
        self.assertEqual(reopened.list_visits({})["items"][0]["status"], "matched")

    def test_parallel_retries_commit_once(self):
        event = self.event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: self.store.ingest(event), range(16)))
        self.assertEqual(sum(not result["duplicate"] for result in results), 1)
        self.assertEqual(self.store.list_events({})["total"], 1)
        self.assertEqual(self.store.list_visits({})["total"], 1)

    def test_direction_history_is_used_for_delayed_events(self):
        station = self.store.update_station("board-in", {"direction": "OUT"})
        self.assertEqual(station["version"], 2)
        self.ingest(version=1)
        self.ingest(at="2026-10-26T02:00:00.000Z", version=2)
        self.assertEqual(self.store.list_visits({})["items"][0]["status"], "matched")
        self.assertEqual([e["direction"] for e in self.store.list_events({})["items"]], ["OUT", "IN"])

    def test_timeout_can_be_disabled_without_disabling_sequence_checks(self):
        self.store.update_settings({"missing_exit_seconds": 0})
        self.ingest()
        self.store.expire_pending(datetime(2027, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(self.store.list_anomalies({})["total"], 0)
        self.ingest(at="2026-10-26T02:00:00.000Z")
        self.assertEqual(self.store.list_anomalies({})["items"][0]["kind"], "duplicate_entry")

    def test_invalid_event_does_not_partially_write(self):
        cases = [dict(person_id=None), dict(direction_version=999), dict(track_id=-1),
                 dict(observed_at="2026-10-26T01:00:00"), dict(station_id="../../data"),
                 dict(event_id="invalid"), dict(decision="admin")]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(StoreValidationError):
                self.store.ingest(dict(self.event(), **change))
        self.assertEqual(self.store.list_events({})["total"], 0)

    def test_filter_pagination_and_sql_values_are_bound(self):
        for index in range(55):
            self.ingest(at=f"2026-10-26T01:{index:02d}:00.000Z", person=f"p{index}")
        self.assertEqual(len(self.store.list_events({})["items"]), 50)
        second = self.store.list_events({}, page=2, limit=50)
        self.assertEqual((second["total"], len(second["items"])), (55, 5))
        self.assertEqual(self.store.list_events({"person_id": "' OR 1=1 --"})["total"], 0)
        with self.assertRaises(StoreValidationError):
            self.store.list_events({}, page=0)
        with self.assertRaises(StoreValidationError):
            self.store.list_events({}, limit=201)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_export_limit_rejects_instead_of_silently_truncating(self):
        self.ingest(person="p1")
        self.ingest(person="p2")
        self.ingest(person="p3")
        with self.assertRaises(StoreValidationError):
            self.store.export_events({}, maximum=2)
        exported = self.store.export_events({"person_id": "p2"}, maximum=2)
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0]["person_id"], "p2")


if __name__ == "__main__":
    unittest.main()
