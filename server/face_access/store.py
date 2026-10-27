"""SQLite owns events, station history and derived visits; Qt owns face data."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid

from .pairing import derive_visits, utc_timestamp


class StoreValidationError(ValueError):
    pass


class StoreConflict(ValueError):
    pass


class StoreNotFound(LookupError):
    pass


def _text(value, name, maximum=128, empty=False):
    if (not isinstance(value, str) or len(value) > maximum
            or (not empty and not value.strip()) or "\x00" in value):
        raise StoreValidationError(f"{name}格式错误")
    return value


class AccessStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.connection(write=True) as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise StoreConflict("数据库版本不兼容，不能自动降级")
            if version == 0:
                self._migrate(db)
        self.db_path.chmod(0o600)

    @contextmanager
    def connection(self, write=False):
        db = sqlite3.connect(self.db_path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA busy_timeout=5000")
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA journal_mode=WAL")
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except BaseException:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _migrate(db):
        statements = [
            "CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            """CREATE TABLE stations(id TEXT PRIMARY KEY, name TEXT NOT NULL,
                direction TEXT NOT NULL CHECK(direction IN ('IN','OUT')),
                version INTEGER NOT NULL, enabled INTEGER NOT NULL DEFAULT 1)""",
            """CREATE TABLE station_versions(station_id TEXT NOT NULL REFERENCES stations(id),
                version INTEGER NOT NULL, direction TEXT NOT NULL,
                PRIMARY KEY(station_id,version))""",
            """CREATE TABLE events(event_id TEXT PRIMARY KEY, station_id TEXT NOT NULL,
                direction_version INTEGER NOT NULL, direction TEXT NOT NULL,
                stream_id TEXT NOT NULL, track_id INTEGER NOT NULL,
                observed_at TEXT NOT NULL, received_at TEXT NOT NULL,
                person_id TEXT, display_name TEXT NOT NULL, decision TEXT NOT NULL,
                payload TEXT NOT NULL,
                FOREIGN KEY(station_id,direction_version) REFERENCES station_versions(station_id,version))""",
            "CREATE INDEX events_person_time ON events(person_id,observed_at)",
            "CREATE INDEX events_time ON events(observed_at)",
            """CREATE TABLE visits(id TEXT PRIMARY KEY, person_id TEXT NOT NULL,
                entry_event_id TEXT NOT NULL REFERENCES events(event_id),
                exit_event_id TEXT REFERENCES events(event_id), entered_at TEXT NOT NULL,
                exited_at TEXT, display_name TEXT NOT NULL, status TEXT NOT NULL)""",
            """CREATE TABLE anomalies(id TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES events(event_id),
                person_id TEXT NOT NULL, kind TEXT NOT NULL, active INTEGER NOT NULL,
                resolved_at TEXT, resolved_by TEXT, note TEXT NOT NULL DEFAULT '')""",
            """CREATE TABLE anomaly_resolutions(id INTEGER PRIMARY KEY, anomaly_id TEXT NOT NULL REFERENCES anomalies(id),
                resolved_at TEXT NOT NULL, actor TEXT NOT NULL, note TEXT NOT NULL)""",
        ]
        for statement in statements:
            db.execute(statement)
        db.execute("INSERT INTO settings VALUES('missing_exit_seconds','86400')")
        db.execute("INSERT INTO stations(id,name,direction,version) VALUES('board-in','板卡摄像头','IN',1)")
        db.execute("INSERT INTO station_versions VALUES('board-in',1,'IN')")
        db.execute("PRAGMA user_version=1")

    def settings(self):
        with self.connection() as db:
            return {"missing_exit_seconds": int(db.execute(
                "SELECT value FROM settings WHERE key='missing_exit_seconds'").fetchone()[0])}

    def update_settings(self, changes):
        if not isinstance(changes, dict) or set(changes) != {"missing_exit_seconds"}:
            raise StoreValidationError("设置字段错误")
        value = changes["missing_exit_seconds"]
        if type(value) is not int or (value != 0 and not 60 <= value <= 31536000):
            raise StoreValidationError("超时须为0或60—31536000秒")
        with self.connection(write=True) as db:
            db.execute("UPDATE settings SET value=? WHERE key='missing_exit_seconds'", (str(value),))
            for row in db.execute("SELECT DISTINCT person_id FROM events WHERE person_id IS NOT NULL").fetchall():
                self._rebuild(db, row[0])
        return self.settings()

    def list_stations(self):
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT * FROM stations WHERE id='board-in'")]

    def get_station(self, station_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM stations WHERE id=?", (station_id,)).fetchone()
            if row is None:
                raise StoreNotFound("终端不存在")
            return dict(row)

    def update_station(self, station_id, changes):
        if station_id != "board-in":
            raise StoreNotFound("终端不存在")
        if not isinstance(changes, dict) or not changes or set(changes) - {"direction", "name", "enabled"}:
            raise StoreValidationError("终端设置字段错误")
        if "direction" in changes and changes["direction"] not in ("IN", "OUT"):
            raise StoreValidationError("方向只能是IN或OUT")
        if "name" in changes:
            _text(changes["name"], "终端名称", 64)
        if "enabled" in changes and type(changes["enabled"]) is not bool:
            raise StoreValidationError("启用状态错误")
        with self.connection(write=True) as db:
            found = db.execute("SELECT * FROM stations WHERE id=?", (station_id,)).fetchone()
            if found is None:
                raise StoreNotFound("终端不存在")
            station = dict(found)
            station.update(changes)
            if changes.get("direction", found["direction"]) != found["direction"] or changes.get("enabled", bool(found["enabled"])) != bool(found["enabled"]):
                station["version"] += 1
                db.execute("INSERT INTO station_versions VALUES(?,?,?)",
                           (station_id, station["version"], station["direction"]))
            db.execute("UPDATE stations SET name=?,direction=?,version=?,enabled=? WHERE id=?",
                       (station["name"], station["direction"], station["version"], station["enabled"], station_id))
        return self.get_station(station_id)

    @staticmethod
    def _validate_event(event):
        required = {"event_id", "station_id", "direction_version", "stream_id", "track_id",
                    "observed_at", "person_id", "display_name", "decision"}
        if not isinstance(event, dict) or set(event) != required:
            raise StoreValidationError("识别事件字段错误")
        clean = dict(event)
        try:
            for field in ("event_id", "stream_id"):
                text = _text(event[field], field, 36)
                clean[field] = str(uuid.UUID(text))
            clean["observed_at"] = utc_timestamp(event["observed_at"])
        except (ValueError, TypeError, OverflowError) as error:
            raise StoreValidationError("事件ID或时间格式错误") from error
        _text(event["station_id"], "终端ID", 64)
        _text(event["display_name"], "姓名", 128, empty=True)
        if event["person_id"] is not None:
            _text(event["person_id"], "人员ID", 128)
        if (event["decision"] not in ("allowed", "denied")
                or (event["decision"] == "allowed" and event["person_id"] is None)):
            raise StoreValidationError("识别授权结果错误")
        if type(event["track_id"]) is not int or not 0 <= event["track_id"] <= 2147483647:
            raise StoreValidationError("轨迹ID错误")
        if type(event["direction_version"]) is not int or event["direction_version"] < 1:
            raise StoreValidationError("终端方向版本错误")
        return clean

    def ingest(self, event):
        clean = self._validate_event(event)
        payload = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.connection(write=True) as db:
            existing = db.execute("SELECT payload FROM events WHERE event_id=?", (clean["event_id"],)).fetchone()
            if existing is not None:
                if existing[0] != payload:
                    raise StoreConflict("相同事件ID不能覆盖原始内容")
                return dict(event_id=clean["event_id"], persisted=True, duplicate=True)
            if clean["station_id"] == "pc-out":
                cutoff = db.execute("SELECT value FROM settings WHERE key='pc_out_decommissioned_at'").fetchone()
                if cutoff is not None and clean["observed_at"] > cutoff[0]:
                    raise StoreValidationError("电脑识别终端已停用")
            station = db.execute("SELECT direction FROM station_versions WHERE station_id=? AND version=?",
                                 (clean["station_id"], clean["direction_version"])).fetchone()
            if station is None:
                raise StoreValidationError("终端或方向版本不存在")
            values = dict(clean, direction=station[0], received_at=utc_timestamp(datetime.now(timezone.utc).isoformat()),
                          payload=payload)
            db.execute("""INSERT INTO events VALUES(:event_id,:station_id,:direction_version,:direction,
                :stream_id,:track_id,:observed_at,:received_at,:person_id,:display_name,:decision,:payload)""", values)
            if clean["decision"] == "allowed":
                self._rebuild(db, clean["person_id"])
        return dict(event_id=clean["event_id"], persisted=True, duplicate=False)

    @staticmethod
    def _rebuild(db, person_id):
        events = [dict(row) for row in db.execute(
            "SELECT * FROM events WHERE person_id=? AND decision='allowed'", (person_id,))]
        timeout = int(db.execute("SELECT value FROM settings WHERE key='missing_exit_seconds'").fetchone()[0])
        evaluated = db.execute("SELECT value FROM settings WHERE key='last_evaluated_at'").fetchone()
        visits, anomalies = derive_visits(events, timeout, evaluated[0] if evaluated else None)
        db.execute("DELETE FROM visits WHERE person_id=?", (person_id,))
        db.execute("UPDATE anomalies SET active=0 WHERE person_id=?", (person_id,))
        db.executemany("""INSERT INTO visits VALUES(:id,:person_id,:entry_event_id,:exit_event_id,
            :entered_at,:exited_at,:display_name,:status)""", visits)
        db.executemany("""INSERT INTO anomalies(id,event_id,person_id,kind,active)
            VALUES(:id,:event_id,:person_id,:kind,:active)
            ON CONFLICT(id) DO UPDATE SET active=excluded.active""", anomalies)

    def expire_pending(self, now_utc: datetime):
        try:
            now = utc_timestamp(now_utc.isoformat())
        except (ValueError, AttributeError) as error:
            raise StoreValidationError("扫描时间错误") from error
        with self.connection(write=True) as db:
            before = db.execute("SELECT count(*) FROM anomalies WHERE kind='missing_exit' AND active=1").fetchone()[0]
            db.execute("INSERT INTO settings VALUES('last_evaluated_at',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (now,))
            for row in db.execute("SELECT DISTINCT person_id FROM events WHERE person_id IS NOT NULL AND decision='allowed'").fetchall():
                self._rebuild(db, row[0])
            after = db.execute("SELECT count(*) FROM anomalies WHERE kind='missing_exit' AND active=1").fetchone()[0]
        return max(0, after - before)

    def resolve_anomaly(self, anomaly_id, note, actor):
        _text(note, "处理备注", 2000)
        _text(actor, "处理人", 128)
        with self.connection(write=True) as db:
            now = utc_timestamp(datetime.now(timezone.utc).isoformat())
            updated = db.execute("UPDATE anomalies SET resolved_at=?,resolved_by=?,note=? WHERE id=?",
                                 (now, actor, note, anomaly_id))
            if not updated.rowcount:
                raise StoreNotFound("异常不存在")
            db.execute("INSERT INTO anomaly_resolutions(anomaly_id,resolved_at,actor,note) VALUES(?,?,?,?)",
                       (anomaly_id, now, actor, note))
            return dict(db.execute("SELECT * FROM anomalies WHERE id=?", (anomaly_id,)).fetchone())

    def list_events(self, filters, page=1, limit=50):
        return self._list("events", filters, page, limit)

    def list_visits(self, filters, page=1, limit=50):
        return self._list("visits", filters, page, limit)

    def list_anomalies(self, filters, page=1, limit=50):
        return self._list("anomalies", filters, page, limit)

    def _list(self, table, filters, page, limit):
        if type(page) is not int or page < 1 or type(limit) is not int or not 1 <= limit <= 200:
            raise StoreValidationError("分页参数错误")
        return self._select(table, filters, page, limit)

    def export_events(self, filters, maximum=10000):
        if type(maximum) is not int or not 1 <= maximum <= 10000:
            raise StoreValidationError("导出上限错误")
        result = self._select("events", filters, 1, maximum + 1)
        if result["total"] > maximum:
            raise StoreValidationError(f"单次最多导出{maximum}条，请缩小筛选范围")
        return result["items"]

    def _select(self, table, filters, page, limit):
        if not isinstance(filters, dict):
            raise StoreValidationError("筛选参数错误")
        conditions, parameters = [], []
        available = {
            "events": {"person_id", "station_id", "direction", "decision"},
            "visits": {"person_id", "status"},
            "anomalies": {"person_id", "kind"},
        }[table]
        time_field = "observed_at" if table == "events" else "entered_at" if table == "visits" else None
        for key, value in filters.items():
            if key in available:
                conditions.append(f"{key}=?")
                parameters.append(value)
            elif key in ("from", "to") and time_field:
                try:
                    normalized = utc_timestamp(value)
                except ValueError as error:
                    raise StoreValidationError("筛选时间错误") from error
                conditions.append(f"{time_field}{'>=' if key == 'from' else '<='}?")
                parameters.append(normalized)
            elif table == "anomalies" and key in ("include_history", "resolved"):
                if type(value) is not bool:
                    raise StoreValidationError("异常筛选状态错误")
                if key == "resolved":
                    conditions.append("resolved_at IS NOT NULL" if value else "resolved_at IS NULL")
            else:
                raise StoreValidationError("不支持的筛选字段")
        if table == "anomalies" and not filters.get("include_history", False):
            conditions.append("active=1")
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        order = {"events": "observed_at DESC,event_id DESC", "visits": "entered_at DESC,id DESC", "anomalies": "id DESC"}[table]
        columns = "*" if table != "events" else "event_id,station_id,direction_version,direction,stream_id,track_id,observed_at,received_at,person_id,display_name,decision"
        with self.connection() as db:
            db.execute("BEGIN")
            total = db.execute(f"SELECT count(*) FROM {table}{where}", parameters).fetchone()[0]
            rows = db.execute(f"SELECT {columns} FROM {table}{where} ORDER BY {order} LIMIT ? OFFSET ?",
                              [*parameters, limit, (page - 1) * limit]).fetchall()
            return dict(items=[dict(row) for row in rows], total=total, page=page, limit=limit)
