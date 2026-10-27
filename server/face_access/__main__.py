"""Production WSGI entrypoint supervised by the Qt application's QProcess."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import threading

from waitress import create_server

from .config import ServerConfig
from .http_app import create_app
from .qt_bridge import QtBridge


def main():
    parser = argparse.ArgumentParser(description="RK3568局域网管理后台")
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8080, type=int)
    args = parser.parse_args()
    try:
        line = sys.stdin.buffer.readline(4097)
        if len(line) > 4096:
            raise ValueError("启动参数过大")
        private = json.loads(line)
        config = ServerConfig(args.runtime_dir, Path(private["qt_socket"]), private["local_secret"], args.host, args.port)
    except (ValueError, KeyError, TypeError):
        print("后台缺少有效的本机启动参数", file=sys.stderr)
        return 2
    app = create_app(config, QtBridge(config.qt_socket, config.local_secret))
    try:
        server = create_server(app, host=config.host, port=config.port, threads=4,
                               connection_limit=32, channel_timeout=15, cleanup_interval=5,
                               max_request_body_size=8 * 1024 * 1024 + 65536,
                               ident="FaceAccess", log_socket_errors=False)
    except OSError:
        app.extensions["access_jobs"].close()
        print("管理服务监听失败，请检查端口是否被占用", file=sys.stderr)
        return 3
    stopped = threading.Event()
    parent_pid = os.getppid()

    def stop(_signum=None, _frame=None):
        stopped.set()
        # Leave selected descriptors alive until Waitress exits its event loop.
        # Its run() handles SystemExit by draining the request dispatcher.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise SystemExit(0)

    def supervise():
        ticks = 0
        while not stopped.wait(1):
            if os.getppid() != parent_pid:
                os.kill(os.getpid(), signal.SIGTERM)
                break
            ticks += 1
            if ticks % 60 == 0:
                try:
                    app.extensions["access_store"].expire_pending(datetime.now(timezone.utc))
                except Exception:
                    print("出入异常扫描失败，请检查数据库或磁盘状态", file=sys.stderr)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    supervisor = threading.Thread(target=supervise, daemon=True)
    supervisor.start()
    try:
        app.extensions["access_store"].expire_pending(datetime.now(timezone.utc))
        print(json.dumps(dict(status="ready", port=int(server.effective_port))), flush=True)
        server.run()
    finally:
        stopped.set()
        server.close()
        supervisor.join(2)
        app.extensions["access_jobs"].close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
