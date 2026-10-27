import json
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import unittest
from urllib.request import urlopen


class StartupTests(unittest.TestCase):
    def test_standalone_server_starts_with_private_stdin_and_stops_cleanly(self):
        script = Path(__file__).resolve().parents[1] / "start_server.py"
        self.assertTrue(script.is_file(), "后台启动器尚未实现")
        with tempfile.TemporaryDirectory() as temp:
            private = "test-private-startup-secret"
            child = subprocess.Popen([sys.executable, "-I", str(script), "--runtime-dir", temp,
                                      "--host", "127.0.0.1", "--port", "0"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                child.stdin.write(json.dumps(dict(local_secret=private, qt_socket=temp + "/qt.sock")).encode() + b"\n")
                child.stdin.flush()
                available, _, _ = select.select([child.stdout], [], [], 5)
                self.assertTrue(available, "后台未在5秒内报告就绪")
                ready = json.loads(child.stdout.readline())
                self.assertNotIn(private, str(ready))
                self.assertEqual(ready["status"], "ready")
                with urlopen(f"http://127.0.0.1:{ready['port']}/api/auth/status", timeout=2) as response:
                    self.assertEqual(json.load(response), {"initialized": False})
                child.terminate()
                _, errors = child.communicate(timeout=5)
                self.assertEqual(child.returncode, 0, errors.decode())
                self.assertNotIn(private.encode(), errors)
            finally:
                if child.poll() is None:
                    child.kill()
                child.communicate()


if __name__ == "__main__":
    unittest.main()
