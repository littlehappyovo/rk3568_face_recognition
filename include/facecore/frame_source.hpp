#pragma once

#include "facecore/config.hpp"
#include "facecore/export.hpp"
#include "facecore/types.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace facecore {

// 摄像头和图片序列都实现这一套接口。上层只接收 Frame，不需要知道底层
// 使用 ioctl、cv::VideoCapture 还是 imdecode。
class FACECORE_API IFrameSource {
public:
    virtual ~IFrameSource() = default;
    virtual bool open(std::string* error = nullptr) = 0;
    virtual bool read(Frame& output, std::string* error = nullptr) = 0;
    virtual void close() noexcept = 0;
    virtual bool isOpen() const noexcept = 0;
    virtual std::string name() const = 0;
    // 实际生效的帧尺寸（ISP/GStreamer 可能自己取整）；未知时返回空尺寸。
    virtual cv::Size negotiatedSize() const noexcept { return {}; }
};

struct CameraSourceOptions {
    int deviceIndex{0};
    // 非空时按设备路径打开（Linux V4L2，例如 /dev/video-camera0）。
    std::string device;
    // Linux 原生多平面 V4L2 采集：驱动 MMAP 缓冲区并导出 DMA-BUF fd，
    // 同时在用户态直接把 NV12 转成 BGR，绕过 GStreamer/appsink 的额外排队。
    bool nativeV4l2{false};
    // 非空时按这条 GStreamer 管线打开（cv::CAP_GSTREAMER），优先级最高。
    // RK3568 这类 ISP 设备只提供 V4L2_CAP_VIDEO_CAPTURE_MPLANE。当前优先走
    // 原生 V4L2；GStreamer 是兼容性回退，不是唯一可用路径。
    std::string pipeline;
    int width{1280};
    int height{720};
    double fps{30.0};
    // cv::CAP_ANY / CAP_V4L2 / CAP_GSTREAMER；用 GStreamer 管线时会被忽略。
    int apiPreference{0};
    // 仅用于日志与界面提示：这条候选是怎么来的。
    std::string label;
};

FACECORE_API std::unique_ptr<IFrameSource> makeCameraSource(CameraSourceOptions options);
FACECORE_API std::unique_ptr<IFrameSource> makeImageSource(
    std::vector<std::filesystem::path> imageFiles,
    bool loop = false);

// 按配置生成"摄像头打开候选列表"，按优先级排列，调用方依次尝试直到成功。
//
// 板端实测（ATK-DLRK3568 + imx415 + rkisp，见 24_camera_open_probe.log）：
//   - OpenCV 自带的 V4L2 后端打不开多平面 ISP 设备，但原生 V4L2 可以正常采集；
//   - GStreamer 管线可用：v4l2src + video/x-raw,format=NV12,WxH + videoconvert + appsink
//     能拿到 800x600 BGR、约 20 fps（缩放由 ISP 完成，最省 CPU）；
//   - 不限定输入格式、靠 videoscale 兜底的管线同样可用，但缩放吃 CPU，作为备选。
// 因此 Linux 上把原生 V4L2 DMA-BUF 导出排在自动 GStreamer 前面，GStreamer 作为回退。
FACECORE_API std::vector<CameraSourceOptions> buildCameraCandidates(const CameraConfig& config);

}  // namespace facecore
