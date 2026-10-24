#!/usr/bin/env bash
# 停止 eglfs 人脸识别程序，并恢复 Ubuntu 图形登录界面。
set -u

ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
PIDFILE="$ROOT/mipi.pid"
WATCHPIDFILE="$ROOT/mipi_gdm_watch.pid"

if [ -f "$WATCHPIDFILE" ]; then
    watch_pid="$(cat "$WATCHPIDFILE" 2>/dev/null || true)"
    [ -z "$watch_pid" ] || kill "$watch_pid" 2>/dev/null || true
    rm -f "$WATCHPIDFILE"
fi

if [ -f "$PIDFILE" ]; then
    pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "停止人脸识别程序 PID=$pid"
        kill "$pid" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi
    rm -f "$PIDFILE"
else
    pid="$(pgrep -f "$ROOT/build-rk3568/face_recognition_system" | head -n 1 || true)"
    if [ -n "$pid" ]; then
        echo "没有 mipi.pid，按程序路径找到人脸识别进程 PID=$pid"
        kill "$pid" 2>/dev/null || true
        sleep 2
    else
        echo '没有找到人脸识别进程，继续恢复 Ubuntu 图形界面'
    fi
fi

if sudo -n systemctl start gdm3.service; then
    echo '已启动 gdm3，Ubuntu 图形界面已恢复'
    echo '触摸映射：gdm3 使用 udev 默认单位矩阵；人脸识别的进程级旋转参数已退出'
else
    echo '启动 gdm3 失败：请检查 sudo 权限' >&2
    exit 1
fi
