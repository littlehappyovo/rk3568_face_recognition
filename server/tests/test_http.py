from http.cookiejar import CookieJar
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import uuid
from urllib.error import HTTPError
from urllib.request import build_opener, HTTPCookieProcessor, Request

from waitress import create_server

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from face_access.config import ServerConfig
from face_access.http_app import create_app


class UnavailableQt:
    result = None

    def call(self, operation, payload, timeout_s=10):
        from face_access.qt_bridge import BridgeError
        if self.result is not None:
            return self.result
        raise BridgeError("测试后端未连接")


class HttpTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.runtime_path = Path(temp.name)
        config = ServerConfig(Path(temp.name), Path(temp.name) / "qt.sock", "test-local-secret")
        self.bridge = UnavailableQt()
        self.app = create_app(config, self.bridge)

        def controlled_source(environ, respond):
            # Simulate a LAN source at the WSGI transport boundary, over a real TCP server.
            if environ.get("HTTP_X_TEST_REMOTE_ADDR"):
                environ["REMOTE_ADDR"] = environ["HTTP_X_TEST_REMOTE_ADDR"]
            return self.app(environ, respond)

        server = create_server(controlled_source, host="127.0.0.1", port=0, threads=4)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        self.url = f"http://127.0.0.1:{server.effective_port}"
        self.cookies = CookieJar()
        self.client = build_opener(HTTPCookieProcessor(self.cookies))
        self.csrf = ""

        def close():
            self.app.extensions["access_jobs"].close()
            server.task_dispatcher.shutdown(timeout=3)
            server.close()
            thread.join(3)
        self.addCleanup(close)

    def request(self, path, method="GET", data=None, headers=None, raw=False):
        supplied = dict(headers or {})
        supplied.setdefault("Origin", self.url)
        if data is not None and not isinstance(data, bytes):
            data = json.dumps(data).encode()
            supplied.setdefault("Content-Type", "application/json")
        if self.csrf:
            supplied.setdefault("X-CSRF-Token", self.csrf)
        request = Request(self.url + path, data=data, method=method, headers=supplied)
        try:
            response = self.client.open(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            content = response.read()
            return response.status, content if raw else json.loads(content), response.headers

    def setup_and_login(self):
        status, _, _ = self.request("/api/local/setup", "POST", {"password": "Test-Password-2026"},
                                    {"X-Local-Secret": "test-local-secret"})
        self.assertEqual(status, 201)
        status, body, _ = self.request("/api/auth/login", "POST", {"password": "Test-Password-2026"})
        self.assertEqual(status, 200)
        self.csrf = body["csrf_token"]

    def test_setup_rejects_lan_and_is_single_use(self):
        for headers in ({}, {"X-Local-Secret": "test-local-secret", "X-Test-Remote-Addr": "192.168.1.10", "X-Forwarded-For": "127.0.0.1"}):
            self.assertEqual(self.request("/api/local/setup", "POST", {"password": "Test-Password-2026"}, headers)[0], 403)
        self.setup_and_login()
        self.assertEqual(self.request("/api/local/setup", "POST", {"password": "Other-Password-2026"},
                                      {"X-Local-Secret": "test-local-secret"})[0], 409)

    def test_login_cookie_cannot_mutate_without_csrf(self):
        self.assertEqual(self.request("/api/events")[0], 401)
        self.setup_and_login()
        cookie = next(iter(self.cookies))
        self.assertEqual(cookie.name, "face_admin")
        self.assertTrue(cookie.has_nonstandard_attr("HttpOnly"))
        self.assertEqual(cookie.get_nonstandard_attr("SameSite"), "Strict")
        self.csrf = ""
        self.assertEqual(self.request("/api/settings", "PATCH", {"missing_exit_seconds": 0})[0], 403)
        self.assertEqual(self.request("/api/events")[0], 200)

    def test_browser_can_refresh_session_using_same_origin_post(self):
        self.setup_and_login()
        self.assertEqual(self.request('/api/auth/session','POST',{})[0],200)

    def test_session_check_does_not_extend_idle_timeout(self):
        now = [100000.0]
        self.app.extensions["access_auth"].clock = lambda: now[0]
        self.setup_and_login()
        now[0] += 200
        self.assertEqual(self.request("/api/auth/check")[0], 200)
        now[0] += 101
        self.assertEqual(self.request("/api/auth/check")[0], 401)

    def test_lists_are_paginated(self):
        self.setup_and_login()
        self.assertEqual(self.request("/api/events?limit=201")[0], 400)
        self.assertEqual(self.request("/api/events?page=0")[0], 400)
        status, body, _ = self.request("/api/events?limit=10&page=1")
        self.assertEqual((status, body["limit"], body["total"]), (200, 10, 0))

    def test_csv_formula_prefix_is_escaped(self):
        self.setup_and_login()
        import uuid
        event = dict(event_id=str(uuid.uuid4()), station_id="board-in", direction_version=1,
                     stream_id=str(uuid.uuid4()), track_id=1, observed_at="2026-10-26T00:00:00Z",
                     person_id="p1", display_name="=1+1", decision="allowed")
        self.assertEqual(self.request("/api/local/events", "POST", event,
                                     {"X-Local-Secret": "test-local-secret"})[0], 200)
        status, content, _ = self.request("/api/events.csv", raw=True)
        self.assertEqual(status, 200)
        self.assertIn("'=1+1", content.decode("utf-8-sig"))

    def test_pc_camera_routes_are_gone_and_board_direction_management_remains(self):
        self.setup_and_login()
        self.bridge.result = {"configured": True}
        self.assertEqual(self.request("/api/stations/pc-out/token", "POST", {})[0], 404)
        self.assertEqual(self.request("/api/stations/pc-out/frames", "POST", b"jpeg")[0], 404)
        self.assertEqual(self.request("/api/stations/pc-out/tracks", "POST", {})[0], 404)
        status, body, _ = self.request("/api/stations")
        self.assertEqual((status, [station["id"] for station in body["items"]]), (200, ["board-in"]))
        status, station, _ = self.request("/api/stations/board-in", "PATCH", {"direction": "OUT"})
        self.assertEqual((status, station["direction"]), (200, "OUT"))
        status, page, _ = self.request("/", raw=True)
        self.assertEqual(status, 200)
        self.assertNotIn("电脑相机配对".encode(), page)
        self.assertNotIn(b"issue-token", page)
        self.assertIn("历史电脑终端".encode(), page)

    def test_unknown_routes_do_not_serve_runtime_files(self):
        status, content, _ = self.request("/access.sqlite3", raw=True)
        self.assertEqual(status, 404)
        self.assertNotIn(b"SQLite format", content)
        self.assertEqual(self.request("/api/local/events", "POST", {})[0], 403)

    def test_combined_filters_pagination_and_anomaly_resolution(self):
        self.setup_and_login(); store=self.app.extensions['access_store']
        out_station=store.update_station('board-in', {'direction':'OUT'})
        for i in range(52):
            store.ingest(dict(event_id=str(uuid.uuid4()),station_id='board-in',direction_version=out_station['version'],stream_id=str(uuid.uuid4()),track_id=1,
                observed_at=f'2026-10-26T01:00:{i:02}.000Z',person_id='P1',display_name='<img src=x onerror=alert(1)>',decision='allowed'))
        query='?person_id=P1&station_id=board-in&direction=OUT&decision=allowed&from=2026-10-26T01:00:00Z&to=2026-10-26T01:00:51Z'
        first=self.request('/api/events'+query)[1]; second=self.request('/api/events'+query+'&page=2')[1]
        self.assertEqual((first['total'],len(first['items']),len(second['items'])),(52,50,2))
        self.assertFalse({r['event_id'] for r in first['items']} & {r['event_id'] for r in second['items']})
        anomaly=self.request('/api/anomalies?person_id=P1&kind=exit_without_entry&resolved=false')[1]['items'][0]
        route='/api/anomalies/'+anomaly['id']+'/resolve'
        self.assertEqual(self.request(route,'POST',{'note':''})[0],400)
        self.assertEqual(self.request(route,'POST',{'note':'人工核查：已离开'})[0],200)
        resolved=self.request('/api/anomalies?resolved=true')[1]['items'][0]
        self.assertEqual(resolved['note'],'人工核查：已离开')
        self.cookies.clear();self.assertEqual(self.request(route,'POST',{'note':'again'})[0],401)

    def test_station_direction_is_configurable_without_pc_station(self):
        self.setup_and_login();self.bridge.result={'configured':True}
        stations=self.request('/api/stations')[1]['items']
        self.assertEqual([station['id'] for station in stations], ['board-in'])
        status,station,_=self.request('/api/stations/board-in','PATCH',{'direction':'OUT'})
        self.assertEqual((status,station['direction'],station['version']),(200,'OUT',2))

    def test_failed_gallery_job_never_reports_completed_and_samples_require_login(self):
        import time
        self.assertEqual(self.request('/api/people/P1/samples/S1',raw=True)[0],401)
        self.setup_and_login()
        status,body,_=self.request('/api/gallery/rebuild','POST',{})
        self.assertEqual(status,202)
        for _ in range(50):
            result=self.request('/api/jobs/'+body['job_id'])[1]
            if result['status']=='failed': break
            time.sleep(.01)
        self.assertEqual(result['status'],'failed')
        self.assertIsNone(result['result'])
        self.assertIn('message',result['error'])
        self.assertEqual(self.request('/api/people','POST',{'person_id':'../outside','display_name':'bad'})[0],400)

    def test_historical_pc_events_remain_readable_but_new_pc_events_are_rejected(self):
        self.setup_and_login()
        store = self.app.extensions["access_store"]
        past = dict(event_id=str(uuid.uuid4()), station_id="pc-out", direction_version=1,
                    direction="OUT", stream_id=str(uuid.uuid4()), track_id=1,
                    observed_at="2026-10-26T00:00:00.000Z", received_at="2026-10-26T00:00:00.000Z",
                    person_id="P1", display_name="历史人员", decision="allowed")
        import json as json_module
        with store.connection(write=True) as db:
            db.execute("INSERT INTO stations VALUES('pc-out','电脑摄像头（历史）','OUT',1,0) ON CONFLICT(id) DO UPDATE SET enabled=0")
            db.execute("INSERT OR IGNORE INTO station_versions VALUES('pc-out',1,'OUT')")
            db.execute("INSERT INTO events VALUES(:event_id,:station_id,:direction_version,:direction,:stream_id,:track_id,:observed_at,:received_at,:person_id,:display_name,:decision,:payload)",
                       dict(past, payload=json_module.dumps(past, ensure_ascii=False)))
        status, body, _ = self.request("/api/events?station_id=pc-out")
        self.assertEqual((status, body["total"], body["items"][0]["display_name"]), (200, 1, "历史人员"))
        current = dict(event_id=str(uuid.uuid4()), station_id="pc-out", direction_version=1,
                       stream_id=str(uuid.uuid4()), track_id=1,
                       observed_at=(datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat(), person_id="P1",
                       display_name="历史人员", decision="allowed")
        self.assertEqual(self.request("/api/local/events", "POST", current,
                                      {"X-Local-Secret": "test-local-secret"})[0], 400)


if __name__ == "__main__":
    unittest.main()
