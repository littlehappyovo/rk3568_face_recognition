from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from face_access.store import AccessStore, StoreConflict
from face_access.auth import AuthManager, AuthError


class AuthTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = AccessStore(Path(temp.name) / "access.sqlite3")
        self.now = [100000.0]
        self.auth = AuthManager(self.store, clock=lambda: self.now[0])
        self.password = "Test-Password-2026"

    def test_password_is_salted_and_never_stored_plaintext(self):
        self.auth.setup(self.password)
        with self.store.connection() as db:
            saved = db.execute("SELECT value FROM settings WHERE key='admin_password'").fetchone()[0]
        self.assertNotIn(self.password, saved)
        self.assertIn("600000", saved)
        self.assertTrue(self.auth.initialized())
        with self.assertRaises(StoreConflict):
            self.auth.setup("Other-Password-2026")
        session = self.auth.login(self.password, "127.0.0.1")
        self.assertTrue(self.auth.authorize(session["token"]))

    def test_change_password_invalidates_all_sessions(self):
        self.auth.setup(self.password)
        first = self.auth.login(self.password, "one")
        second = self.auth.login(self.password, "two")
        self.auth.change_password(first["token"], self.password, "New-Password-2026")
        for token in (first["token"], second["token"]):
            with self.assertRaises(AuthError):
                self.auth.authorize(token)
        self.assertTrue(self.auth.login("New-Password-2026", "three"))

    def test_rate_limit_and_session_expiry(self):
        self.auth.setup(self.password)
        for _ in range(5):
            with self.assertRaises(AuthError) as failure:
                self.auth.login("wrong-password", "client")
            self.assertEqual(failure.exception.status, 401)
        with self.assertRaises(AuthError) as failure:
            self.auth.login(self.password, "client")
        self.assertEqual(failure.exception.status, 429)
        self.now[0] += 61
        session = self.auth.login(self.password, "client")
        self.now[0] += 299
        self.auth.authorize(session["token"])
        self.now[0] += 2
        with self.assertRaises(AuthError):
            self.auth.authorize(session["token"])

    def test_csrf_is_required_and_explicit_activity_refreshes_idle_only(self):
        self.auth.setup(self.password)
        session = self.auth.login(self.password, "client")
        with self.assertRaises(AuthError) as failure:
            self.auth.authorize(session["token"], csrf="wrong")
        self.assertEqual(failure.exception.status, 403)
        self.now[0] += 290
        self.auth.authorize(session["token"], csrf=session["csrf_token"], refresh=True)
        self.now[0] += 290
        self.assertTrue(self.auth.authorize(session["token"]))
        self.now[0] += 28800
        with self.assertRaises(AuthError):
            self.auth.authorize(session["token"], refresh=True)

    def test_startup_revokes_legacy_pc_station_token_and_disables_station(self):
        with self.store.connection(write=True) as db:
            db.execute("INSERT INTO stations VALUES('pc-out','电脑摄像头','OUT',1,1)")
            db.execute("INSERT INTO station_versions VALUES('pc-out',1,'OUT')")
            db.execute("INSERT INTO station_tokens VALUES('pc-out','legacy-token-hash')")
        from face_access.auth import AuthManager
        self.auth = AuthManager(self.store, clock=lambda: self.now[0])
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM station_tokens WHERE station_id='pc-out'").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT enabled FROM stations WHERE id='pc-out'").fetchone()[0], 0)

    def test_admin_logout_still_invalidates_session(self):
        self.auth.setup(self.password)
        session = self.auth.login(self.password, "client")
        self.auth.logout(session["token"])
        with self.assertRaises(AuthError):
            self.auth.authorize(session["token"])


if __name__ == "__main__":
    unittest.main()
