#!/usr/bin/env bash
# 把界面推到 MIPI 屏并在后台常驻。
#
#   bash scripts/run_on_mipi.sh [源码根目录]
#   MIPI_ROTATION=270 bash scripts/run_on_mipi.sh     # 换个方向
#
# 板子的 MIPI DSI 屏走 DRM/KMS（card0-DSI-1），**原生尺寸是 1080x1920 竖屏**；
# 系统里没有 X 服务器，所以不能用默认的 xcb 插件：
#   eglfs   —— 直接 DRM + GBM + EGL，GPU 渲染，首选
#   linuxfb —— 直接写 /dev/fb0，软件渲染，兜底
# 两者都不需要 X。
#
# 旋转：界面按 1440x850 的宽屏布局设计，竖着塞进 1080 宽会被横向裁掉
# （光摄像头下拉框最小宽 230 + 右侧控制栏最小宽 330 就超了），所以默认
# QT_QPA_EGLFS_ROTATION=90，把可用画面转成 1920x1080 横向。
# 换方向用 MIPI_ROTATION=0/90/180/270；画面上下颠倒或镜像就试 270。
#
# 脚本会依次试平台插件，第一个能活下来的就留在屏幕上，并把 PID 写到
# mipi.pid、日志写到 mipi_run.log。
#
# 停止：bash scripts/stop_on_mipi.sh <源码根目录>

set -u

ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
BUILD="${FACEAPP_BUILD_DIR:-$ROOT/build-rk3568}"
CONFIG="${FACEAPP_CONFIG:-$ROOT/config/rk3568.json}"
APP="$BUILD/face_recognition_system"
LOG="$ROOT/mipi_run.log"
PIDFILE="$ROOT/mipi.pid"
WATCHPIDFILE="$ROOT/mipi_gdm_watch.pid"
KMSCONF="$ROOT/mipi_kms.json"
ROTATION="${MIPI_ROTATION:-90}"
AUTO_START_ARGS=()
if [ "${FACEAPP_AUTO_START_CAMERA:-1}" = 1 ]; then
    AUTO_START_ARGS=(--auto-start)
fi

case "$ROTATION" in
    0|90|180|270) ;;
    *) echo "MIPI_ROTATION 只支持 0/90/180/270，收到：$ROTATION" >&2; exit 1 ;;
esac

find_touch_device() {
    local dev
    for dev in /dev/input/by-path/*-event* /dev/input/event*; do
        [ -e "$dev" ] || continue
        if udevadm info --query=property --name="$dev" 2>/dev/null \
            | grep -q '^ID_INPUT_TOUCHSCREEN=1$'; then
            printf '%s\n' "$dev"
            return 0
        fi
    done
    return 1
}

TOUCH_DEVICE="$(find_touch_device || true)"
TOUCH_DEVICE="${TOUCH_DEVICE:-/dev/input/event2}"
case "$ROTATION" in
    # 当前触摸坐标相对屏幕按钮整体差 180°，在原映射基础上补偿 180°。
    0)   TOUCH_ROTATION=180 ;;
    90)  TOUCH_ROTATION=90 ;;
    180) TOUCH_ROTATION=0 ;;
    270) TOUCH_ROTATION=270 ;;
esac
TOUCH_PARAMS="$TOUCH_DEVICE"
[ "$TOUCH_ROTATION" = 0 ] || TOUCH_PARAMS="$TOUCH_PARAMS:rotate=$TOUCH_ROTATION"

cd "$ROOT" || exit 1

if [ ! -x "$APP" ]; then
    echo "找不到 $APP，先运行 bash scripts/build_rk3568.sh" >&2
    exit 1
fi

# 人脸识别使用 eglfs 直接占用 DRM；只有在它启动期间停止 Ubuntu 图形登录。
# 通过 sudoers 允许的两条 systemctl 命令完成切换，不把密码写进脚本。
if ! sudo -n systemctl stop gdm3.service; then
    echo '无法停止 gdm3：请先安装 alientek 的 gdm3 启停 sudo 权限' >&2
    exit 1
fi
unset DISPLAY WAYLAND_DISPLAY
echo '已停止 gdm3，准备由 eglfs 接管 MIPI 屏幕'

# 取消上一个实例的监视器，避免旧实例退出时误启动 gdm3。
if [ -f "$WATCHPIDFILE" ]; then
    old_watch="$(cat "$WATCHPIDFILE" 2>/dev/null || true)"
    [ -z "$old_watch" ] || kill "$old_watch" 2>/dev/null || true
    rm -f "$WATCHPIDFILE"
fi

# 先停掉所有残留实例。只靠 pidfile 不可靠 —— 同步源码树时 mipi.pid 会被删掉，
# 旧进程就会继续占着 DRM 和摄像头，新实例报 "Device or resource busy"。
for stale in $(pgrep -f 'face_recognition_system' 2>/dev/null || true); do
    echo "停掉残留实例 $stale"
    kill "$stale" 2>/dev/null || true
done
if [ -f "$PIDFILE" ]; then
    old="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [ -n "${old:-}" ] && kill -0 "$old" 2>/dev/null; then
        kill "$old" 2>/dev/null || true
    fi
fi
rm -f "$PIDFILE"
sleep 3

echo "== 环境 =="
echo "app      $APP"
echo "config   $CONFIG"
echo "旋转     QT_QPA_EGLFS_ROTATION=$ROTATION"
echo "DISPLAY=${DISPLAY:-<unset>}  WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-<unset>}"
echo "DRM 连接器:"
for c in /sys/class/drm/card*-*/status; do
    [ -e "$c" ] || continue
    printf '  %-24s %s\n' "$(basename "$(dirname "$c")")" "$(cat "$c")"
done
echo "图形库  : $(ls /usr/lib/aarch64-linux-gnu/ 2>/dev/null | grep -icE 'mali|libEGL|libGLES|libgbm') 个匹配"
ls /usr/lib/aarch64-linux-gnu/ 2>/dev/null | grep -iE 'mali|libEGL|libGLES|libgbm' | sed 's/^/          /' || true
echo "fb 设备 : $(ls /dev/fb* 2>/dev/null || echo '无 /dev/fb*')"
echo "占用 DRM: $(fuser /dev/dri/card0 2>&1 | tr -d '\n' || echo 无)"
echo "中文字体: $(fc-list :lang=zh 2>/dev/null | wc -l) 个"

# Ubuntu/gdm 使用系统默认单位矩阵；人脸识别进程单独使用 Qt evdev 的旋转参数。
# 这样退出 eglfs 后，gdm/Wayland 不会继承人脸识别的触摸旋转。
HAVE_MATRIX="$(udevadm info --query=property --name="$TOUCH_DEVICE" 2>/dev/null \
    | sed -n 's/^LIBINPUT_CALIBRATION_MATRIX=//p')"
echo "触摸设备: $TOUCH_DEVICE"
echo "触摸后端: Qt evdev（画面=$ROTATION，进程参数=$TOUCH_PARAMS，udev 矩阵=${HAVE_MATRIX:-<默认>}）"

# 清掉可能从上一次调试启动残留的参数；try_platform 会为当前进程重新设置。
unset QT_QPA_EGLFS_NO_LIBINPUT QT_QPA_EVDEV_TOUCHSCREEN_PARAMETERS QT_QPA_EVDEV_DEBUG

# 明确指定用 DSI-1 这块屏，避免 eglfs 挑到 HDMI。
cat > "$KMSCONF" <<EOF
{
  "device": "/dev/dri/card0",
  "outputs": [
    { "name": "DSI-1", "primary": true, "touchDevice": "$TOUCH_DEVICE" }
  ]
}
EOF
echo "KMS 配置 : $KMSCONF"

try_platform() {
    local platform="$1"
    echo
    echo "--- 尝试 QT_QPA_PLATFORM=$platform ---"
    : > "$LOG"
    # 字号：eglfs 会按面板物理尺寸反推 DPI，这块 MIPI 屏算出来 400+ DPI，
    # Qt 把默认字号放大四五倍，界面直接溢出屏幕（控件重叠、右栏要滚动），
    # 手指就按不到"看到的按钮"。改用像素字号，不随 DPI 变化。
    QT_QPA_PLATFORM="$platform" \
    QT_QPA_EGLFS_INTEGRATION=eglfs_kms \
    QT_QPA_EGLFS_KMS_CONFIG="$KMSCONF" \
    QT_QPA_EGLFS_ALWAYS_SET_MODE=1 \
    QT_QPA_EGLFS_ROTATION="$ROTATION" \
    QT_QPA_FB_DRM=1 \
    QT_QPA_EGLFS_NO_LIBINPUT=1 \
    QT_QPA_EVDEV_TOUCHSCREEN_PARAMETERS="$TOUCH_PARAMS" \
    FACEAPP_UI_FONT_PX="${FACEAPP_UI_FONT_PX:-20}" \
    FACEAPP_UI_FONT_FAMILY="${FACEAPP_UI_FONT_FAMILY:-Noto Sans CJK SC}" \
    FACEAPP_CONFIG="$CONFIG" \
    nohup "$APP" "${AUTO_START_ARGS[@]}" --fullscreen --config "$CONFIG" >> "$LOG" 2>&1 &
    local pid=$!
    sleep 7
    if kill -0 "$pid" 2>/dev/null; then
        echo "$pid" > "$PIDFILE"
        echo "PID $pid 仍在运行"
        echo "关键日志:"
        grep -aiE '\[display\]|\[ui-font\]|platform|libinput|evdevtouch|opened|auto-start|camera|connector' "$LOG" | head -20 | sed 's/^/  /'
        echo
        echo ">>> 已用 $platform 启动（旋转 $ROTATION），应该已经显示在 MIPI 屏上了"
        echo ">>> 画面方向不对就换：MIPI_ROTATION=270 bash $0 $ROOT"
        echo ">>> 触摸由 libinput + udev 校准矩阵接管：${HAVE_MATRIX:-<未设置>}"
        echo ">>> 日志: $LOG"
        echo ">>> 停止并恢复 Ubuntu 图形界面: bash $ROOT/scripts/stop_on_mipi.sh $ROOT"

        # 即使用户直接 kill 应用，监视器也会恢复 gdm3；正常停止时由同一逻辑完成。
        (
            watched_pid="$pid"
            while kill -0 "$watched_pid" 2>/dev/null; do
                sleep 2
            done
            if [ -f "$PIDFILE" ] && [ "$(cat "$PIDFILE" 2>/dev/null || true)" = "$watched_pid" ]; then
                rm -f "$PIDFILE"
                sudo -n systemctl start gdm3.service >/dev/null 2>&1 || true
            fi
        ) > "$ROOT/mipi_gdm_watch.log" 2>&1 &
        echo "$!" > "$WATCHPIDFILE"
        return 0
    fi
    echo "启动后 $platform 进程已退出，日志尾部："
    tail -n 15 "$LOG" | sed 's/^/  /'
    return 1
}

# 有 X 就用 xcb，否则按 eglfs -> linuxfb 顺序试。
if [ -n "${DISPLAY:-}" ]; then
    try_platform xcb && exit 0
fi
try_platform eglfs && exit 0
try_platform linuxfb && exit 0

sudo -n systemctl start gdm3.service >/dev/null 2>&1 || true
echo
echo "eglfs 与 linuxfb 都起不来，日志见 $LOG"
exit 1
