"""Shared board/web administrator authentication; no plaintext credentials."""

import hashlib
import hmac
import json
import secrets
import threading
import time
from datetime import datetime, timezone

from .pairing import utc_timestamp
from .store import AccessStore, StoreConflict, StoreValidationError, _text


class AuthError(Exception):
    def __init__(self, message="请先登录", status=401):
        super().__init__(message)
        self.status = status


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _password_record(password):
    _text(password, "密码", 128)
    if len(password) < 8:
        raise StoreValidationError("密码至少8个字符")
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 600000)
    return json.dumps(dict(algorithm="pbkdf2-sha256", iterations=600000,
                           salt=salt.hex(), derived=derived.hex()), sort_keys=True)


def _password_matches(password, record):
    if not isinstance(password, str) or not 8 <= len(password) <= 128:
        return False
    data = json.loads(record)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                  bytes.fromhex(data["salt"]), data["iterations"])
    return hmac.compare_digest(derived.hex(), data["derived"])


class AuthManager:
    def __init__(self, store: AccessStore, clock=time.time):
        self.store, self.clock = store, clock
        self._attempts = {}
        self._lock = threading.Lock()
        with store.connection(write=True) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS auth_sessions(token_hash TEXT PRIMARY KEY,
                csrf_hash TEXT NOT NULL, created_at REAL NOT NULL, active_at REAL NOT NULL)""")
            db.execute("CREATE TABLE IF NOT EXISTS station_tokens(station_id TEXT PRIMARY KEY REFERENCES stations(id), token_hash TEXT NOT NULL)")
            db.execute("DELETE FROM station_tokens WHERE station_id='pc-out'")
            db.execute("UPDATE stations SET enabled=0 WHERE id='pc-out'")
            cutoff = utc_timestamp(datetime.now(timezone.utc).isoformat())
            db.execute("INSERT INTO settings(key,value) VALUES('pc_out_decommissioned_at',?) ON CONFLICT(key) DO NOTHING",
                       (cutoff,))

    def initialized(self):
        with self.store.connection() as db:
            return db.execute("SELECT 1 FROM settings WHERE key='admin_password'").fetchone() is not None

    def setup(self, password):
        record = _password_record(password)
        with self.store.connection(write=True) as db:
            if db.execute("SELECT 1 FROM settings WHERE key='admin_password'").fetchone():
                raise StoreConflict("管理密码已经设置")
            db.execute("INSERT INTO settings VALUES('admin_password',?)", (record,))

    def _reserve_attempt(self, source):
        now = self.clock()
        with self._lock:
            count, started = self._attempts.get(source, (0, now))
            if now - started >= 60:
                count, started = 0, now
            if count >= 5:
                raise AuthError("登录尝试过多，请稍后重试", 429)
            if source not in self._attempts and len(self._attempts) >= 1024:
                oldest = min(self._attempts, key=lambda k: self._attempts[k][1])
                del self._attempts[oldest]
            self._attempts[source] = (count + 1, started)

    def login(self, password, source):
        self._reserve_attempt(source)
        with self.store.connection() as db:
            row = db.execute("SELECT value FROM settings WHERE key='admin_password'").fetchone()
        if not row or not _password_matches(password, row[0]):
            raise AuthError("密码错误或尚未设置")
        with self._lock:
            self._attempts.pop(source, None)
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        now = self.clock()
        with self.store.connection(write=True) as db:
            # A concurrent password change must not create a fresh old-password session.
            current = db.execute("SELECT value FROM settings WHERE key='admin_password'").fetchone()[0]
            if current != row[0]:
                raise AuthError("密码已经变更，请重新登录")
            db.execute("DELETE FROM auth_sessions WHERE active_at<=? OR created_at<=?", (now - 300, now - 28800))
            if db.execute("SELECT count(*) FROM auth_sessions").fetchone()[0] >= 64:
                db.execute("DELETE FROM auth_sessions WHERE token_hash=(SELECT token_hash FROM auth_sessions ORDER BY active_at LIMIT 1)")
            db.execute("INSERT INTO auth_sessions VALUES(?,?,?,?)", (_digest(token), _digest(csrf), now, now))
        return dict(token=token, csrf_token=csrf, expires_at=now + 300)

    def authorize(self, token, csrf=None, refresh=False):
        if not isinstance(token, str) or not 1 <= len(token) <= 128:
            raise AuthError()
        now = self.clock()
        with self.store.connection(write=refresh) as db:
            row = db.execute("SELECT * FROM auth_sessions WHERE token_hash=?", (_digest(token),)).fetchone()
            if row is None or now - row["active_at"] >= 300 or now - row["created_at"] >= 28800:
                raise AuthError("登录已失效，请重新登录")
            if csrf is not None and (not isinstance(csrf, str) or len(csrf) > 128
                                     or not hmac.compare_digest(_digest(csrf), row["csrf_hash"])):
                raise AuthError("请求校验失败", 403)
            if refresh:
                db.execute("UPDATE auth_sessions SET active_at=? WHERE token_hash=?", (now, _digest(token)))
            return dict(expires_at=min((now if refresh else row["active_at"]) + 300, row["created_at"] + 28800))

    def logout(self, token):
        if isinstance(token, str) and len(token) <= 128:
            with self.store.connection(write=True) as db:
                db.execute("DELETE FROM auth_sessions WHERE token_hash=?", (_digest(token),))

    def change_password(self, token, current_password, new_password):
        self.authorize(token)
        record = _password_record(new_password)
        with self.store.connection(write=True) as db:
            old = db.execute("SELECT value FROM settings WHERE key='admin_password'").fetchone()[0]
            if not _password_matches(current_password, old):
                raise AuthError("原管理密码错误")
            db.execute("UPDATE settings SET value=? WHERE key='admin_password'", (record,))
            db.execute("DELETE FROM auth_sessions")
