"""Bounded request/reply protocol to the local Qt process."""

import json
from pathlib import Path
import socket
import struct
import time
import uuid

MAX_MESSAGE_BYTES = 12 * 1024 * 1024
OPERATIONS = frozenset(("people.list", "people.create", "people.update", "people.delete",
                        "samples.list", "samples.read", "samples.add", "samples.delete",
                        "gallery.rebuild", "station.configure"))


class BridgeError(Exception):
    pass


class QtBridge:
    def __init__(self, socket_path: Path, secret: str):
        self.socket_path, self._secret = str(socket_path), secret

    @staticmethod
    def _read(connection, count, deadline):
        chunks = bytearray()
        while len(chunks) < count:
            left = deadline - time.monotonic()
            if left <= 0:
                raise BridgeError("板端操作超时")
            connection.settimeout(left)
            data = connection.recv(count - len(chunks))
            if not data:
                raise BridgeError("板端连接已关闭")
            chunks.extend(data)
        return bytes(chunks)

    def call(self, operation: str, payload: dict, timeout_s: float = 10):
        if operation not in OPERATIONS or not isinstance(payload, dict):
            raise BridgeError("不支持的板端操作")
        request_id = str(uuid.uuid4())
        encoded = json.dumps(dict(request_id=request_id, operation=operation,
                                  payload=payload, secret=self._secret), ensure_ascii=False).encode("utf-8")
        if len(encoded) > MAX_MESSAGE_BYTES:
            raise BridgeError("板端请求过大")
        deadline = time.monotonic() + timeout_s
        try:
            with socket.socket(socket.AF_UNIX) as connection:
                connection.settimeout(timeout_s)
                connection.connect(self.socket_path)
                connection.sendall(struct.pack("!I", len(encoded)) + encoded)
                size = struct.unpack("!I", self._read(connection, 4, deadline))[0]
                if not 0 < size <= MAX_MESSAGE_BYTES:
                    raise BridgeError("板端响应长度错误")
                reply = json.loads(self._read(connection, size, deadline))
        except (OSError, ValueError, UnicodeError) as error:
            raise BridgeError("板端服务未就绪或响应无效") from error
        if not isinstance(reply, dict) or reply.get("request_id") != request_id:
            raise BridgeError("板端响应请求ID不匹配")
        if reply.get("ok") is not True:
            error = reply.get("error")
            message = error.get("message") if isinstance(error, dict) else None
            raise BridgeError(message if isinstance(message, str) else "板端操作失败")
        if not isinstance(reply.get("result"), dict):
            raise BridgeError("板端响应格式错误")
        return reply["result"]
