import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from face_access.qt_bridge import QtBridge, BridgeError


class BridgeTests(unittest.TestCase):
    def test_remote_frame_operations_are_not_supported(self):
        bridge = QtBridge(Path("/no/bridge.sock"), "test-startup-secret")
        with self.assertRaisesRegex(BridgeError, "不支持的板端操作"):
            bridge.call("frame.submit", {})
        with self.assertRaisesRegex(BridgeError, "不支持的板端操作"):
            bridge.call("frame.tracks", {})

    def test_bridge_correlates_replies_and_bounds_messages(self):
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "bridge.sock")
            listener = socket.socket(socket.AF_UNIX)
            self.addCleanup(listener.close)
            listener.bind(path)
            listener.listen(3)

            def serve():
                for behavior in ("ok", "wrong-id", "too-large"):
                    connection, _ = listener.accept()
                    with connection:
                        size = struct.unpack("!I", connection.recv(4))[0]
                        data = b""
                        while len(data) < size:
                            data += connection.recv(size - len(data))
                        request = json.loads(data)
                        if behavior == "too-large":
                            connection.sendall(struct.pack("!I", 13 * 1024 * 1024))
                        else:
                            reply = dict(request_id=request["request_id"] if behavior == "ok" else "wrong",
                                         ok=True, result={"items": [], "operation": request["operation"]}, error=None)
                            encoded = json.dumps(reply).encode()
                            connection.sendall(struct.pack("!I", len(encoded)) + encoded)

            worker = threading.Thread(target=serve, daemon=True)
            worker.start()
            bridge = QtBridge(Path(path), "test-startup-secret")
            self.assertEqual(bridge.call("people.list", {}, timeout_s=1),
                             {"items": [], "operation": "people.list"})
            for _ in range(2):
                with self.assertRaises(BridgeError):
                    bridge.call("people.list", {}, timeout_s=1)
            worker.join(2)
            self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
