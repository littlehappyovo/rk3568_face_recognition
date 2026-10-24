#include "facecore/frame_source.hpp"

#include "io_internal.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <array>
#include <chrono>
#include <limits>
#include <sstream>
#include <utility>

#if !defined(_WIN32)
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace facecore {
namespace {

// 这一文件把不同的输入方式统一成 IFrameSource：普通 OpenCV/GStreamer 路径、
// RK3568 原生多平面 V4L2 路径，以及离线图片序列。上层不应该把 V4L2 ioctl
// 和业务逻辑混在一起，否则摄像头更换后会牵连检测和识别代码。
std::int64_t nowMicroseconds() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void setError(std::string* destination, std::string message) {
    if (destination) *destination = std::move(message);
}

class CameraFrameSource final : public IFrameSource {
public:
    explicit CameraFrameSource(CameraSourceOptions options) : options_(options) {}

    bool open(std::string* error) override {
        // 这个实现把“管线、设备路径、设备索引”三种 cv::VideoCapture 打开方式
        // 统一处理。GStreamer 管线已经包含尺寸/格式协商，因此不能再盲目 set()
        // 一遍属性；非管线方式才尝试设置宽、高和 FPS。
        close();
        try {
            const bool byPipeline = !options_.pipeline.empty();
            const bool byDevice = !options_.device.empty();
            bool opened = false;
            if (byPipeline) {
                opened = capture_.open(options_.pipeline, cv::CAP_GSTREAMER);
            } else if (byDevice) {
                opened = options_.apiPreference == cv::CAP_ANY
                             ? capture_.open(options_.device)
                             : capture_.open(options_.device, options_.apiPreference);
            } else {
                opened = options_.apiPreference == cv::CAP_ANY
                             ? capture_.open(options_.deviceIndex)
                             : capture_.open(options_.deviceIndex, options_.apiPreference);
            }
            if (!opened) {
                if (byPipeline) {
                    setError(error, "Unable to start the camera pipeline: " + options_.pipeline);
                } else if (byDevice) {
                    setError(error, "Unable to open camera device " + options_.device);
                } else {
                    setError(error, "Unable to open camera index " +
                                        std::to_string(options_.deviceIndex));
                }
                return false;
            }
            // GStreamer 管线里已经写死了尺寸，再 set 一次没有意义（有些后端会报不支持）。
            if (!byPipeline) {
                if (options_.width > 0) capture_.set(cv::CAP_PROP_FRAME_WIDTH, options_.width);
                if (options_.height > 0) capture_.set(cv::CAP_PROP_FRAME_HEIGHT, options_.height);
                if (options_.fps > 0.0) capture_.set(cv::CAP_PROP_FPS, options_.fps);
            }
            negotiatedWidth_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
            negotiatedHeight_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
            // GStreamer 的 appsink 常把 fps 报成 0 或 120 之类的无意义值，这里不采信；
            // 真正读到第一帧时再按实际尺寸修正（见 read()）。
            sequence_ = 0;
            return true;
        } catch (const cv::Exception& exception) {
            setError(error, std::string("OpenCV camera open error: ") + exception.what());
            close();
            return false;
        }
    }

    bool read(Frame& output, std::string* error) override {
        // VideoCapture::read() 会阻塞到下一帧。读取成功后 clone 一份，保证输出
        // Frame 不依赖 VideoCapture 内部下一次 read() 可能复用的缓冲区。
        output = {};
        if (!capture_.isOpened()) {
            setError(error, "Camera is not open");
            return false;
        }
        try {
            cv::Mat image;
            if (!capture_.read(image) || image.empty()) {
                setError(error, "Camera returned an empty frame or was disconnected");
                return false;
            }
            output.image = image.clone();
            output.format = image.channels() == 3 ? PixelFormat::Bgr8 : PixelFormat::Unknown;
            // GStreamer 后端报不出可信尺寸，用真实帧尺寸覆盖一次。
            if (negotiatedWidth_ != output.image.cols || negotiatedHeight_ != output.image.rows) {
                negotiatedWidth_ = output.image.cols;
                negotiatedHeight_ = output.image.rows;
            }
            output.timestampUs = nowMicroseconds();
            output.sequence = sequence_++;
            output.source = name();
            return true;
        } catch (const cv::Exception& exception) {
            setError(error, std::string("OpenCV camera read error: ") + exception.what());
            return false;
        }
    }

    void close() noexcept override {
        try {
            capture_.release();
        } catch (...) {
        }
    }

    bool isOpen() const noexcept override { return capture_.isOpened(); }

    std::string name() const override {
        if (!options_.pipeline.empty()) return "gstreamer:" + options_.device;
        if (!options_.device.empty()) return "camera:" + options_.device;
        return "camera:" + std::to_string(options_.deviceIndex);
    }

    // 驱动协商后的实际尺寸；未打开时为 0。
    int negotiatedWidth() const noexcept { return negotiatedWidth_; }
    int negotiatedHeight() const noexcept { return negotiatedHeight_; }

    cv::Size negotiatedSize() const noexcept override {
        return cv::Size(negotiatedWidth_, negotiatedHeight_);
    }

private:
    CameraSourceOptions options_;
    cv::VideoCapture capture_;
    std::uint64_t sequence_{0};
    int negotiatedWidth_{0};
    int negotiatedHeight_{0};
};

#if !defined(_WIN32)

int v4l2Ioctl(int fd, unsigned long request, void* argument) {
    int result;
    do {
        result = ::ioctl(fd, request, argument);
    } while (result < 0 && errno == EINTR);
    return result;
}

struct NativeV4L2Buffer {
    void* address{MAP_FAILED};
    std::size_t length{0};
    int dmaBufFd{-1};
};

class NativeV4L2FrameSource final : public IFrameSource {
public:
    explicit NativeV4L2FrameSource(CameraSourceOptions options) : options_(std::move(options)) {}

    ~NativeV4L2FrameSource() override { close(); }

    bool open(std::string* error) override {
        // 原生路径对应的生命周期是：打开设备 -> 查询能力 -> 设置 NV12 格式
        // -> 申请 MMAP 缓冲区 -> mmap/EXPBUF -> QBUF -> STREAMON。
        // 任何一步失败都调用 close() 回收已经申请的资源，再让候选列表尝试回退。
        close();
        fd_ = ::open(options_.device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (fd_ < 0) {
            setError(error, "Unable to open native V4L2 device " + options_.device + ": "
                               + std::strerror(errno));
            return false;
        }

        v4l2_capability capability{};
        if (v4l2Ioctl(fd_, VIDIOC_QUERYCAP, &capability) < 0) {
            setError(error, "VIDIOC_QUERYCAP failed: " + std::string(std::strerror(errno)));
            close();
            return false;
        }
        const std::uint32_t deviceCapabilities = capability.device_caps != 0
            ? capability.device_caps
            : capability.capabilities;
        if (!(deviceCapabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE) ||
            !(deviceCapabilities & V4L2_CAP_STREAMING)) {
            setError(error, "V4L2 device is not a streaming multi-planar capture device");
            close();
            return false;
        }

        v4l2_format format{};
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        format.fmt.pix_mp.width = options_.width > 0 ? options_.width : 800;
        format.fmt.pix_mp.height = options_.height > 0 ? options_.height : 600;
        format.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
        format.fmt.pix_mp.field = V4L2_FIELD_NONE;
        if (v4l2Ioctl(fd_, VIDIOC_S_FMT, &format) < 0) {
            setError(error, "VIDIOC_S_FMT NV12 failed: " + std::string(std::strerror(errno)));
            close();
            return false;
        }
        width_ = static_cast<int>(format.fmt.pix_mp.width);
        height_ = static_cast<int>(format.fmt.pix_mp.height);
        pixelFormat_ = format.fmt.pix_mp.pixelformat;
        planeCount_ = format.fmt.pix_mp.num_planes;
        if (pixelFormat_ != V4L2_PIX_FMT_NV12 || planeCount_ != 1 || width_ <= 0 || height_ <= 0) {
            setError(error, "Native V4L2 returned an unsupported NV12 layout");
            close();
            return false;
        }
        bytesPerLine_ = format.fmt.pix_mp.plane_fmt[0].bytesperline;
        sizeImage_ = format.fmt.pix_mp.plane_fmt[0].sizeimage;
        if (bytesPerLine_ < static_cast<unsigned>(width_) || sizeImage_ == 0) {
            setError(error, "Native V4L2 returned invalid stride or image size");
            close();
            return false;
        }

        if (options_.fps > 0.0) {
            v4l2_streamparm streamParameter{};
            streamParameter.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            streamParameter.parm.capture.timeperframe.numerator = 1;
            streamParameter.parm.capture.timeperframe.denominator =
                static_cast<unsigned>(options_.fps + 0.5);
            // Some ISP nodes do not expose S_PARM; the format and poll path remain valid.
            v4l2Ioctl(fd_, VIDIOC_S_PARM, &streamParameter);
        }

        // 这里让驱动拥有采集缓冲区（MMAP），同时用 EXPBUF 导出 DMA-BUF fd。
        // 当前实现的主要目标是绕过 GStreamer/appsink 的排队；后续如果接入
        // 硬件转换器，可以继续利用这些 fd 做真正的零拷贝链路。
        v4l2_requestbuffers request{};
        request.count = 4;
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        request.memory = V4L2_MEMORY_MMAP;
        if (v4l2Ioctl(fd_, VIDIOC_REQBUFS, &request) < 0 || request.count < 2) {
            setError(error, "VIDIOC_REQBUFS MMAP failed: " + std::string(std::strerror(errno)));
            close();
            return false;
        }
        buffers_.resize(request.count);

        for (unsigned index = 0; index < request.count; ++index) {
            v4l2_buffer buffer{};
            v4l2_plane plane{};
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            buffer.memory = V4L2_MEMORY_MMAP;
            buffer.index = index;
            buffer.length = 1;
            buffer.m.planes = &plane;
            if (v4l2Ioctl(fd_, VIDIOC_QUERYBUF, &buffer) < 0) {
                setError(error, "VIDIOC_QUERYBUF failed: " + std::string(std::strerror(errno)));
                close();
                return false;
            }

            buffers_[index].length = plane.length;
            buffers_[index].address = ::mmap(nullptr, plane.length, PROT_READ | PROT_WRITE,
                                              MAP_SHARED, fd_, plane.m.mem_offset);
            if (buffers_[index].address == MAP_FAILED) {
                setError(error, "mmap camera buffer failed: " + std::string(std::strerror(errno)));
                close();
                return false;
            }

            v4l2_exportbuffer exportBuffer{};
            exportBuffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            exportBuffer.index = index;
            exportBuffer.plane = 0;
            exportBuffer.flags = O_CLOEXEC;
            if (v4l2Ioctl(fd_, VIDIOC_EXPBUF, &exportBuffer) < 0) {
                setError(error, "VIDIOC_EXPBUF failed: " + std::string(std::strerror(errno)));
                close();
                return false;
            }
            buffers_[index].dmaBufFd = exportBuffer.fd;

            if (!queue(index)) {
                setError(error, "VIDIOC_QBUF failed: " + std::string(std::strerror(errno)));
                close();
                return false;
            }
        }

        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        if (v4l2Ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
            setError(error, "VIDIOC_STREAMON failed: " + std::string(std::strerror(errno)));
            close();
            return false;
        }
        streaming_ = true;
        sequence_ = 0;
        return true;
    }

    bool read(Frame& output, std::string* error) override {
        // 只取一个已经完成的驱动缓冲区。DQBUF 后必须在使用完像素数据后 QBUF，
        // 否则 4 个缓冲区很快耗尽，摄像头会停止产生新帧。
        output = {};
        if (fd_ < 0 || !streaming_) {
            setError(error, "Native V4L2 camera is not open");
            return false;
        }

        pollfd descriptor{};
        descriptor.fd = fd_;
        descriptor.events = POLLIN;
        int pollResult;
        do {
            pollResult = ::poll(&descriptor, 1, 1000);
        } while (pollResult < 0 && errno == EINTR);
        if (pollResult <= 0) {
            setError(error, pollResult == 0 ? "Native V4L2 frame wait timed out"
                                            : "Native V4L2 poll failed: "
                                                  + std::string(std::strerror(errno)));
            return false;
        }

        v4l2_buffer buffer{};
        v4l2_plane plane{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.length = 1;
        buffer.m.planes = &plane;
        if (v4l2Ioctl(fd_, VIDIOC_DQBUF, &buffer) < 0) {
            if (errno == EAGAIN) return read(output, error);
            setError(error, "VIDIOC_DQBUF failed: " + std::string(std::strerror(errno)));
            return false;
        }
        if (buffer.index >= buffers_.size() || buffers_[buffer.index].address == MAP_FAILED) {
            setError(error, "Native V4L2 returned an invalid buffer index");
            return false;
        }

        bool converted = false;
        try {
            const NativeV4L2Buffer& captureBuffer = buffers_[buffer.index];
            // NV12 是 Y 平面 + 交错 UV 平面；OpenCV 需要把它看成 height*1.5 行。
            // stride 可能大于 width，所以不能简单假设每行没有填充。
            cv::Mat nv12(height_ + height_ / 2, width_, CV_8UC1, captureBuffer.address,
                         bytesPerLine_);
            cv::Mat bgr;
            cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
            if (bgr.empty()) {
                setError(error, "Native V4L2 returned an empty converted frame");
            } else {
                output.image = std::move(bgr);
                output.format = PixelFormat::Bgr8;
                output.timestampUs = nowMicroseconds();
                output.sequence = sequence_++;
                output.source = name();
                converted = true;
            }
        } catch (const cv::Exception& exception) {
            setError(error, std::string("Native V4L2 NV12 conversion failed: ") + exception.what());
        }

        if (!queue(buffer.index)) {
            setError(error, "VIDIOC_QBUF after conversion failed: "
                               + std::string(std::strerror(errno)));
            return false;
        }
        return converted;
    }

    void close() noexcept override {
        // 释放顺序与 open() 相反：先 STREAMOFF，再解除映射、关闭导出的 fd，
        // 最后释放驱动缓冲区并关闭设备 fd。close() 必须可重复调用。
        if (fd_ >= 0 && streaming_) {
            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            v4l2Ioctl(fd_, VIDIOC_STREAMOFF, &type);
            streaming_ = false;
        }
        for (NativeV4L2Buffer& buffer : buffers_) {
            if (buffer.address != MAP_FAILED) {
                ::munmap(buffer.address, buffer.length);
                buffer.address = MAP_FAILED;
            }
            if (buffer.dmaBufFd >= 0) {
                ::close(buffer.dmaBufFd);
                buffer.dmaBufFd = -1;
            }
        }
        buffers_.clear();
        if (fd_ >= 0) {
            v4l2_requestbuffers request{};
            request.count = 0;
            request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            request.memory = V4L2_MEMORY_MMAP;
            v4l2Ioctl(fd_, VIDIOC_REQBUFS, &request);
            ::close(fd_);
            fd_ = -1;
        }
        width_ = 0;
        height_ = 0;
        bytesPerLine_ = 0;
        sizeImage_ = 0;
        planeCount_ = 0;
        sequence_ = 0;
    }

    bool isOpen() const noexcept override { return fd_ >= 0 && streaming_; }

    std::string name() const override {
        return "v4l2-dmabuf-export:" + options_.device;
    }

    cv::Size negotiatedSize() const noexcept override {
        return cv::Size(width_, height_);
    }

private:
    bool queue(unsigned index) {
        // QBUF 把一个空闲缓冲区重新交还给驱动。MMAP 模式下这里仍需填写 plane
        // 的长度和 buffer index，不能只传一个空 v4l2_buffer。
        v4l2_buffer buffer{};
        v4l2_plane plane{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        buffer.length = 1;
        buffer.m.planes = &plane;
        plane.length = static_cast<unsigned>(buffers_[index].length);
        return v4l2Ioctl(fd_, VIDIOC_QBUF, &buffer) == 0;
    }

    CameraSourceOptions options_;
    int fd_{-1};
    bool streaming_{false};
    int width_{0};
    int height_{0};
    unsigned bytesPerLine_{0};
    unsigned sizeImage_{0};
    unsigned planeCount_{0};
    std::uint32_t pixelFormat_{0};
    std::uint64_t sequence_{0};
    std::vector<NativeV4L2Buffer> buffers_;
};

#endif

class ImageFrameSource final : public IFrameSource {
public:
    ImageFrameSource(std::vector<std::filesystem::path> files, bool loop)
        : files_(std::move(files)), loop_(loop) {}

    bool open(std::string* error) override {
        index_ = 0;
        sequence_ = 0;
        open_ = !files_.empty();
        if (!open_) setError(error, "Image source contains no files");
        return open_;
    }

    bool read(Frame& output, std::string* error) override {
        output = {};
        if (!open_) {
            setError(error, "Image source is not open");
            return false;
        }
        if (index_ >= files_.size()) {
            if (!loop_) {
                setError(error, "End of image sequence");
                return false;
            }
            index_ = 0;
        }
        const std::filesystem::path file = files_[index_++];
        try {
            const auto bytes = detail::readBinaryFile(file);
            if (bytes.empty()) {
                setError(error, "Image file is empty: " + file.string());
                return false;
            }
            if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                setError(error, "Image file is too large for OpenCV decoding");
                return false;
            }
            cv::Mat encoded(1, static_cast<int>(bytes.size()), CV_8UC1,
                            const_cast<unsigned char*>(bytes.data()));
            cv::Mat image = cv::imdecode(encoded, cv::IMREAD_COLOR);
            if (image.empty()) {
                setError(error, "Unsupported or corrupt image: " + file.string());
                return false;
            }
            output.image = std::move(image);
            output.format = PixelFormat::Bgr8;
            output.timestampUs = nowMicroseconds();
            output.sequence = sequence_++;
            output.source = detail::pathToUtf8(file);
            return true;
        } catch (const std::exception& exception) {
            setError(error, exception.what());
            return false;
        }
    }

    void close() noexcept override { open_ = false; }
    bool isOpen() const noexcept override { return open_; }
    std::string name() const override { return "image-sequence"; }

private:
    std::vector<std::filesystem::path> files_;
    bool loop_{false};
    bool open_{false};
    std::size_t index_{0};
    std::uint64_t sequence_{0};
};

}  // namespace

std::unique_ptr<IFrameSource> makeCameraSource(CameraSourceOptions options) {
#if !defined(_WIN32)
    if (options.nativeV4l2) {
        return std::make_unique<NativeV4L2FrameSource>(std::move(options));
    }
#endif
    return std::make_unique<CameraFrameSource>(std::move(options));
}

namespace {

std::string gstreamerPipelineWithCaps(const std::string& device, int width, int height,
                                      double fps, bool forceInputCaps) {
    std::string pipeline = "v4l2src device=" + device;
    const int fpsRounded = fps > 0.0 ? static_cast<int>(fps + 0.5) : 0;
    const std::string frameRateCaps = fpsRounded > 0
        ? ",framerate=" + std::to_string(fpsRounded) + "/1"
        : std::string();
    if (forceInputCaps) {
        // 让 ISP 直接输出目标尺寸的 NV12，缩放不占 CPU。
        pipeline += " ! video/x-raw,format=NV12";
        if (width > 0 && height > 0) {
            pipeline += ",width=" + std::to_string(width) + ",height=" + std::to_string(height);
        }
        pipeline += frameRateCaps;
    }
    pipeline += " ! videoconvert";
    if (!forceInputCaps) pipeline += " ! videoscale";
    pipeline += " ! video/x-raw,format=BGR";
    if (width > 0 && height > 0) {
        pipeline += ",width=" + std::to_string(width) + ",height=" + std::to_string(height);
    }
    if (!forceInputCaps) {
        pipeline += frameRateCaps;
    }
    // 转换线程与 OpenCV 拉帧解耦，只保留一个最新帧，避免读取端追赶旧帧。
    pipeline += " ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream";
    pipeline += " ! appsink drop=true max-buffers=1 sync=false";
    return pipeline;
}

}  // namespace

std::vector<CameraSourceOptions> buildCameraCandidates(const CameraConfig& config) {
    // 候选顺序很重要：先尊重用户显式管线，再优先原生 V4L2，最后才用兼容性
    // 回退。每个候选只描述“怎么打开”，实际尝试由 OpenCvFaceBackend::openCamera()
    // 完成，这样失败原因可以逐项打印到日志。
    std::vector<CameraSourceOptions> candidates;
    const auto make = [&config]() {
        CameraSourceOptions options;
        options.deviceIndex = config.index;
        options.device = config.device;
        options.width = config.width;
        options.height = config.height;
        options.fps = config.fps;
        return options;
    };

    // 1) 配置里显式给的 GStreamer 管线，最优先，尊重用户的明确选择。
    if (!config.gstreamerPipeline.empty()) {
        CameraSourceOptions options = make();
        options.pipeline = config.gstreamerPipeline;
        options.label = "配置指定的 GStreamer 管线";
        candidates.push_back(std::move(options));
    }

#if !defined(_WIN32)
    if (!config.device.empty()) {
        // 2) 原生 V4L2：MMAP 采集缓冲区并导出 DMA-BUF fd，绕过 GStreamer/appsink。
        {
            CameraSourceOptions options = make();
            options.nativeV4l2 = true;
            options.label = "原生 V4L2 DMA-BUF（NV12）";
            candidates.push_back(std::move(options));
        }
        // 3) 自动 GStreamer：ISP 输出目标尺寸 NV12，作为兼容性回退。
        {
            CameraSourceOptions options = make();
            options.pipeline = gstreamerPipelineWithCaps(config.device, config.width,
                                                         config.height, config.fps, true);
            options.label = "自动 GStreamer（ISP 缩放回退）";
            candidates.push_back(std::move(options));
        }
        // 4) 自动 GStreamer：不限定输入格式，靠 videoscale 兜底。
        {
            CameraSourceOptions options = make();
            options.pipeline = gstreamerPipelineWithCaps(config.device, config.width,
                                                         config.height, config.fps, false);
            options.label = "自动 GStreamer（videoscale 兜底）";
            candidates.push_back(std::move(options));
        }
        // 5) 直接 V4L2 打开设备路径（单平面设备的回退）。
        {
            CameraSourceOptions options = make();
            options.apiPreference = cv::CAP_V4L2;
            options.label = "V4L2 直接打开设备路径";
            candidates.push_back(std::move(options));
        }
    }
#endif

    // 5) 设备索引（最后兜底，也是 Windows 上的主路径）。
    {
        CameraSourceOptions options = make();
        options.device.clear();
#if defined(_WIN32)
        options.apiPreference = cv::CAP_MSMF;
#else
        options.apiPreference = cv::CAP_ANY;
#endif
        options.label = "设备索引 " + std::to_string(config.index);
        candidates.push_back(std::move(options));
    }
    return candidates;
}

std::unique_ptr<IFrameSource> makeImageSource(std::vector<std::filesystem::path> imageFiles,
                                               bool loop) {
    return std::make_unique<ImageFrameSource>(std::move(imageFiles), loop);
}

}  // namespace facecore
