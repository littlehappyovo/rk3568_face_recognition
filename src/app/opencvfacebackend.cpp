#include "opencvfacebackend.h"
#include "voiceannouncer.h"

#include <QColor>
#include <QCoreApplication>
#include <QDateTime>
#include <QDebug>
#include <QDir>
#include <QFileInfo>
#include <QFile>
#include <QSaveFile>
#include <QStandardPaths>
#include <QTimer>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <condition_variable>
#include <mutex>
#include <set>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

namespace {

// 这些常量控制应用层策略，而不是模型本身：识别需要 5 次结果中的 3 次一致，
// 检测允许少量连续失败，采集和原始帧保存有最小时间间隔。
constexpr int kMaximumDetectionMisses = 2;
constexpr std::int64_t kMinimumCaptureIntervalUs = 180000;
constexpr std::int64_t kMinimumRawCaptureIntervalUs = 200000;
constexpr int kRawFrameCaptureTarget = 20;
constexpr qint64 kSystemMetricsIntervalMs = 1000;

struct CpuSnapshot {
    std::uint64_t total = 0;
    std::uint64_t idle = 0;
};

std::optional<CpuSnapshot> readCpuSnapshot()
{
    std::ifstream input("/proc/stat");
    std::string label;
    std::uint64_t user = 0;
    std::uint64_t nice = 0;
    std::uint64_t system = 0;
    std::uint64_t idle = 0;
    std::uint64_t iowait = 0;
    std::uint64_t irq = 0;
    std::uint64_t softirq = 0;
    std::uint64_t steal = 0;
    if (!input || !(input >> label >> user >> nice >> system >> idle >> iowait >> irq
                    >> softirq >> steal) || label != "cpu") {
        return std::nullopt;
    }

    CpuSnapshot snapshot;
    snapshot.idle = idle + iowait;
    snapshot.total = user + nice + system + idle + iowait + irq + softirq + steal;
    return snapshot.total > 0 ? std::optional<CpuSnapshot>(snapshot) : std::nullopt;
}

std::optional<double> readMemoryUsagePercent()
{
    std::ifstream input("/proc/meminfo");
    std::string key;
    std::uint64_t value = 0;
    std::string unit;
    std::uint64_t totalKb = 0;
    std::uint64_t availableKb = 0;
    while (input >> key >> value >> unit) {
        if (key == "MemTotal:") {
            totalKb = value;
        } else if (key == "MemAvailable:") {
            availableKb = value;
        }
    }
    if (totalKb == 0 || availableKb > totalKb) {
        return std::nullopt;
    }
    const double usage = 100.0 * static_cast<double>(totalKb - availableKb)
                         / static_cast<double>(totalKb);
    return std::clamp(usage, 0.0, 100.0);
}

std::optional<double> readNpuUsagePercent()
{
    // RK3568 exposes the NPU devfreq load as e.g. "100@900000000Hz".
    // Some kernels expose a debugfs text file with one or more values ending in '%'.
    const char *paths[] = {
        "/sys/class/devfreq/fde40000.npu/load",
        "/sys/kernel/debug/rknpu/load",
    };
    for (const char *path : paths) {
        std::ifstream input(path);
        std::string text;
        std::getline(input, text);
        if (text.empty()) {
            continue;
        }

        double sum = 0.0;
        int count = 0;
        std::size_t percent = text.find('%');
        while (percent != std::string::npos) {
            std::size_t begin = percent;
            while (begin > 0
                   && (std::isdigit(static_cast<unsigned char>(text[begin - 1]))
                       || text[begin - 1] == '.')) {
                --begin;
            }
            if (begin < percent) {
                try {
                    sum += std::stod(text.substr(begin, percent - begin));
                    ++count;
                } catch (const std::exception &) {
                    // Ignore malformed fields and keep looking for another core.
                }
            }
            percent = text.find('%', percent + 1);
        }

        if (count == 0) {
            const std::size_t at = text.find('@');
            const std::size_t end = at == std::string::npos ? text.size() : at;
            std::size_t begin = 0;
            while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) {
                ++begin;
            }
            if (begin < end) {
                try {
                    sum = std::stod(text.substr(begin, end - begin));
                    count = 1;
                } catch (const std::exception &) {
                    count = 0;
                }
            }
        }
        if (count > 0) {
            return std::clamp(sum / static_cast<double>(count), 0.0, 100.0);
        }
    }
    return std::nullopt;
}

std::optional<double> readTemperatureCelsius()
{
    std::error_code error;
    const std::filesystem::path thermalRoot("/sys/class/thermal");
    if (!std::filesystem::is_directory(thermalRoot, error)) {
        return std::nullopt;
    }

    double bestTemperature = -1.0;
    bool preferredZoneFound = false;
    for (const auto &entry : std::filesystem::directory_iterator(thermalRoot, error)) {
        if (error || !entry.is_directory(error)) {
            continue;
        }
        const std::string zoneName = entry.path().filename().string();
        if (zoneName.rfind("thermal_zone", 0) != 0) {
            continue;
        }

        std::string type;
        std::ifstream typeInput(entry.path() / "type");
        std::getline(typeInput, type);
        std::ifstream temperatureInput(entry.path() / "temp");
        long long milliCelsius = 0;
        if (!temperatureInput || !(temperatureInput >> milliCelsius)) {
            continue;
        }
        const double temperature = static_cast<double>(milliCelsius) / 1000.0;
        if (!std::isfinite(temperature) || temperature < -20.0 || temperature > 150.0) {
            continue;
        }

        const bool preferred = type.find("cpu") != std::string::npos
                               || type.find("soc") != std::string::npos
                               || type.find("package") != std::string::npos;
        if ((preferred && !preferredZoneFound)
            || (preferred == preferredZoneFound && temperature > bestTemperature)) {
            bestTemperature = temperature;
            preferredZoneFound = preferredZoneFound || preferred;
        }
    }
    return bestTemperature >= 0.0 ? std::optional<double>(bestTemperature) : std::nullopt;
}

void rotateCameraFrameForInference(facecore::Frame &frame)
{
    // 摄像头安装方向和 Qt 屏幕旋转是两件事。算法入口统一旋转，才能保证检测、
    // 采集和识别不依赖板子当前是平放、竖放还是屏幕被 Qt 旋转过。
    if (frame.image.empty()) {
        return;
    }

    // 摄像头原始帧相对正对观察者的方向顺时针偏 90°。在检测、跟踪、
    // 质量评估、采集和识别之前统一逆时针旋转，避免识别结果依赖板卡摆放方向。
    cv::Mat rotated;
    cv::rotate(frame.image, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
    frame.image = std::move(rotated);
}

QString exceptionText(const std::exception &exception)
{
    return QString::fromUtf8(exception.what());
}

} // namespace

class OpenCvFaceBackend::AsyncFrameReader
{
public:
    explicit AsyncFrameReader(std::unique_ptr<facecore::IFrameSource> source)
        : source_(std::move(source))
    {
        if (source_) {
            name_ = source_->name();
            negotiatedSize_ = source_->negotiatedSize();
        }
    }

    ~AsyncFrameReader() { close(); }

    AsyncFrameReader(const AsyncFrameReader &) = delete;
    AsyncFrameReader &operator=(const AsyncFrameReader &) = delete;

    const std::string &name() const noexcept { return name_; }
    cv::Size negotiatedSize() const noexcept { return negotiatedSize_; }

    bool isOpen() const noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return open_;
    }

    void start()
    {
        // 采集线程只负责从 IFrameSource 取帧，不做检测和识别；重复 start() 时
        // 如果线程已经存在就直接返回，避免一个摄像头被多个线程同时读取。
        std::lock_guard<std::mutex> lock(mutex_);
        if (!source_ || !open_ || failed_ || thread_.joinable()) {
            return;
        }
        stopping_ = false;
        workerRunning_ = true;
        thread_ = std::thread([this] { run(); });
    }

    // 暂停采集但保留已经打开的设备，供导入图片/训练后快速恢复。
    void pause() noexcept
    {
        stopWorker();
        std::lock_guard<std::mutex> lock(mutex_);
        latestFrame_.reset();
    }

    void close() noexcept
    {
        stopWorker();
        std::lock_guard<std::mutex> lock(mutex_);
        latestFrame_.reset();
        open_ = false;
        if (source_) {
            source_->close();
        }
    }

    bool takeLatest(facecore::Frame &output)
    {
        // 处理线程只取最新帧。算法如果暂时变慢，旧帧会被覆盖，延迟不会随着
        // 队列积累；这比逐帧处理更适合实时门禁画面。
        std::lock_guard<std::mutex> lock(mutex_);
        if (!latestFrame_.has_value()) {
            return false;
        }
        output = std::move(*latestFrame_);
        latestFrame_.reset();
        return true;
    }

    std::optional<std::string> takeError()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_.empty()) {
            return std::nullopt;
        }
        std::string error = std::move(error_);
        error_.clear();
        return error;
    }

private:
    void stopWorker() noexcept
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        // CameraFrameSource::read() normally returns on the next camera frame;
        // join here so close/restart never races with VideoCapture.
        if (thread_.joinable()) {
            thread_.join();
        }
        std::lock_guard<std::mutex> lock(mutex_);
        workerRunning_ = false;
    }

    void run() noexcept
    {
        // run() 是采集线程入口。它把 IFrameSource::read() 得到的 Frame 放入一个
        // 有界槽位，并通过 mutex 保证采集线程和 Qt 后端线程安全交换数据。
        for (;;) {
            facecore::Frame frame;
            std::string error;
            bool success = false;
            try {
                success = source_ && source_->read(frame, &error);
            } catch (const std::exception &exception) {
                error = exception.what();
            } catch (...) {
                error = "摄像头采集线程发生未知异常";
            }

            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                workerRunning_ = false;
                return;
            }
            if (!success || frame.image.empty()) {
                failed_ = true;
                error_ = error.empty() ? "摄像头读取失败" : std::move(error);
                workerRunning_ = false;
                return;
            }
            // 有界最新帧槽位：处理线程慢时丢弃旧帧，不积压延迟。
            latestFrame_ = std::move(frame);
        }
    }

    std::unique_ptr<facecore::IFrameSource> source_;
    std::string name_;
    cv::Size negotiatedSize_;
    std::thread thread_;
    mutable std::mutex mutex_;
    std::optional<facecore::Frame> latestFrame_;
    std::string error_;
    bool open_ = true;
    bool stopping_ = false;
    bool workerRunning_ = false;
    bool failed_ = false;
};

class OpenCvFaceBackend::AsyncDetector
{
public:
    struct Result {
        std::uint64_t sequence = 0;
        std::vector<facecore::FaceDetection> detections;
        std::string error;
    };

    explicit AsyncDetector(std::unique_ptr<facecore::IFaceDetector> detector)
        : detector_(std::move(detector))
        , detectorName_(detector_ ? detector_->name() : std::string("unavailable"))
    {
        // All mutexes, condition variables and state must exist before run().
        thread_ = std::thread([this] { run(); });
    }

    ~AsyncDetector() { stop(); }

    AsyncDetector(const AsyncDetector &) = delete;
    AsyncDetector &operator=(const AsyncDetector &) = delete;

    const std::string &name() const noexcept { return detectorName_; }

    bool submit(std::uint64_t sequence, const cv::Mat &image)
    {
        // 只允许一个待处理任务和一个未消费结果。返回 false 通常表示上一次
        // 检测尚未完成，并不是摄像头或模型报错。
        if (image.empty()) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        // 只保留一个待处理帧和一个未消费结果，避免检测线程追赶旧帧，
        // 也避免结果还没有被主线程取走时被新任务覆盖。
        if (stopping_ || !detector_ || pending_ || running_ || latestResult_.has_value()) {
            return false;
        }
        pendingSequence_ = sequence;
        pendingGeneration_ = generation_;
        pendingImage_ = image.clone();
        pending_ = true;
        condition_.notify_one();
        return true;
    }

    std::optional<Result> takeResult()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!latestResult_.has_value()) {
            return std::nullopt;
        }
        std::optional<Result> result = std::move(latestResult_);
        latestResult_.reset();
        return result;
    }

    std::vector<facecore::FaceDetection> detectSynchronously(const cv::Mat &image)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_ || !detector_) {
                throw std::runtime_error("异步检测线程已停止");
            }
        }
        std::lock_guard<std::mutex> inferenceLock(inferenceMutex_);
        return detector_->detect(image);
    }

    void discardPendingResults() noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++generation_;
        pending_ = false;
        pendingImage_.release();
        latestResult_.reset();
    }

    void stop() noexcept
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return;
            }
            stopping_ = true;
            pending_ = false;
            pendingImage_.release();
        }
        condition_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    void run() noexcept
    {
        // 检测线程取出图像副本后独立执行 detector_->detect()；完成后把 sequence
        // 和结果放回槽位，由 updateTracks() 在下一次处理帧时消费。
        for (;;) {
            cv::Mat image;
            std::uint64_t sequence = 0;
            std::uint64_t generation = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] { return stopping_ || pending_; });
                if (stopping_ && !pending_) {
                    return;
                }
                image = std::move(pendingImage_);
                sequence = pendingSequence_;
                generation = pendingGeneration_;
                pending_ = false;
                running_ = true;
            }

            Result result;
            result.sequence = sequence;
            try {
                if (!detector_) {
                    result.error = "人脸检测器不可用";
                } else {
                    std::lock_guard<std::mutex> inferenceLock(inferenceMutex_);
                    result.detections = detector_->detect(image);
                }
            } catch (const std::exception &exception) {
                result.error = exception.what();
            } catch (...) {
                result.error = "异步人脸检测线程发生未知异常";
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                running_ = false;
                if (!stopping_ && generation == generation_) {
                    latestResult_ = std::move(result);
                }
            }
        }
    }

    std::unique_ptr<facecore::IFaceDetector> detector_;
    std::string detectorName_;
    std::thread thread_;
    std::mutex mutex_;
    std::mutex inferenceMutex_;
    std::condition_variable condition_;
    cv::Mat pendingImage_;
    std::uint64_t pendingSequence_ = 0;
    std::uint64_t pendingGeneration_ = 0;
    std::uint64_t generation_ = 0;
    bool pending_ = false;
    bool running_ = false;
    bool stopping_ = false;
    std::optional<Result> latestResult_;
};

OpenCvFaceBackend::OpenCvFaceBackend(QString applicationDirectory, QObject *parent)
    : FaceBackendService(parent)
    , m_applicationDirectory(std::move(applicationDirectory))
    , m_frameTimer(new QTimer(this))
{
    // 该 QObject 会在 main.cpp 中移动到 FaceBackendThread；因此这里创建的 QTimer
    // 也属于后台线程，timeout 信号将在后台线程调用 processNextFrame()。
    // 摄像头和后端目标均为 30fps；轮询略快于 30fps，才能抵消 Qt 主线程
    // 刷新和 eglfs 调度抖动。采集线程仍只保留最新帧，不会积压旧帧。
    m_frameTimer->setInterval(10);
    m_frameTimer->setTimerType(Qt::PreciseTimer);
    connect(m_frameTimer, &QTimer::timeout,
            this, &OpenCvFaceBackend::processNextFrame);
    m_systemMetricsClock.start();
}

OpenCvFaceBackend::~OpenCvFaceBackend() = default;

void OpenCvFaceBackend::requestShutdown() noexcept
{
    m_shutdownRequested.store(true, std::memory_order_relaxed);
}

void OpenCvFaceBackend::updateSystemMetrics()
{
    if (!m_systemMetricsClock.isValid()) {
        m_systemMetricsClock.start();
    }
    const qint64 elapsedMs = m_systemMetricsClock.elapsed();
    if (m_lastSystemMetricsMs >= 0
        && elapsedMs - m_lastSystemMetricsMs < kSystemMetricsIntervalMs) {
        return;
    }
    m_lastSystemMetricsMs = elapsedMs;

    if (const auto snapshot = readCpuSnapshot()) {
        if (m_previousCpuTotal > 0 && snapshot->total > m_previousCpuTotal
            && snapshot->idle >= m_previousCpuIdle) {
            const std::uint64_t totalDelta = snapshot->total - m_previousCpuTotal;
            const std::uint64_t idleDelta = snapshot->idle - m_previousCpuIdle;
            if (totalDelta > 0 && idleDelta <= totalDelta) {
                const double usage = 100.0
                    * static_cast<double>(totalDelta - idleDelta)
                    / static_cast<double>(totalDelta);
                m_systemMetrics.cpuUsagePercent = std::clamp(usage, 0.0, 100.0);
            }
        }
        m_previousCpuTotal = snapshot->total;
        m_previousCpuIdle = snapshot->idle;
    }

    if (const auto memory = readMemoryUsagePercent()) {
        m_systemMetrics.memoryUsagePercent = *memory;
    }
    if (const auto npu = readNpuUsagePercent()) {
        m_systemMetrics.npuUsagePercent = *npu;
    }
    if (const auto temperature = readTemperatureCelsius()) {
        m_systemMetrics.temperatureCelsius = *temperature;
    }
}

void OpenCvFaceBackend::configureBackend(const QString &backendId)
{
    // UI 传来的只是字符串 ID，真正的模型对象在 initializeCore() 中由工厂创建。
    // 切换后清空识别记忆，防止旧后端的分数和新后端混在同一条轨迹里。
    if (m_shutdownRequested.load(std::memory_order_relaxed)) {
        return;
    }
    if (backendId != QStringLiteral("opencv_lbph")
        && backendId != QStringLiteral("onnx_embedding")
        && backendId != QStringLiteral("mobilefacenet_rknn")) {
        emit errorOccurred(tr("后端配置失败"), tr("不支持的识别后端：%1").arg(backendId));
        return;
    }

    try {
        initializeCore();
        m_backendId = backendId;
        if (m_voice) m_voice->reset();
        m_recognitionMemory.clear();
        clearReferenceCards();
        if (m_recognitionEnabled && !backendReady()) {
            m_recognitionEnabled = false;
            emit recognitionEnabledChanged(false);
        }
        const QString summary = backendId == QStringLiteral("opencv_lbph")
                                    ? tr("LBPH：灰度均衡训练与距离拒识")
                                    : backendId == QStringLiteral("mobilefacenet_rknn")
                                          ? m_mobileFaceNetSummary
                                          : m_backendSummary;
        const QString state = backendReady()
                                  ? tr("已加载")
                                  : ((backendId == QStringLiteral("onnx_embedding") && !m_sface)
                                         || (backendId == QStringLiteral("mobilefacenet_rknn")
                                             && !m_mobileFaceNet)
                                         ? tr("模型待投放")
                                         : tr("尚未训练"));
        emit modelChanged(state, summary);
        setState(m_streaming ? ServiceState::Streaming : ServiceState::Idle,
                 tr("识别后端已切换"));
    } catch (const std::exception &exception) {
        reportException(tr("初始化后端失败"), exception);
    }
}

void OpenCvFaceBackend::refreshCameras()
{
    if (m_shutdownRequested.load(std::memory_order_relaxed)) {
        return;
    }
    if (!m_initialized) {
        try {
            initializeCore();
        } catch (const std::exception &exception) {
            reportException(tr("读取摄像头配置失败"), exception);
            return;
        }
    }
    // OpenCV 没有可移植的、不会真正打开设备的枚举接口，所以只列出一小组索引，
    // 只有用户按下“启动”时才真正申请摄像头权限。配置里给了设备路径时优先列出它，
    // 因为 RK3568 的 ISP 主通路是固定的 /dev/video-camera0。
    CameraList cameras;
    if (!m_config.camera.device.empty()) {
        const QString device = QString::fromStdString(m_config.camera.device);
        cameras.push_back({device,
                           tr("配置设备 %1（%2×%3 @ %4 fps）")
                               .arg(device)
                               .arg(m_config.camera.width)
                               .arg(m_config.camera.height)
                               .arg(m_config.camera.fps, 0, 'g', 3)});
    }
    for (int index = 0; index < 4; ++index) {
        cameras.push_back({QString::number(index),
                           index == 0 ? tr("摄像头索引 0（默认）")
                                      : tr("摄像头索引 %1").arg(index)});
    }
    emit camerasChanged(cameras);
    setState(m_streaming ? ServiceState::Streaming : ServiceState::Idle,
             tr("已列出摄像头；启动时验证设备是否可用"));
}

void OpenCvFaceBackend::startAutomatically()
{
    if (m_shutdownRequested.load(std::memory_order_relaxed)) return;
    try {
        initializeCore();
        const QString camera = m_config.camera.device.empty()
            ? QString::number(m_config.camera.index) : utf8(m_config.camera.device);
        qInfo().noquote() << "[startup] opening camera" << camera;
        openCamera(camera);
        if (!m_streaming) return;
        if (backendReady()) {
            setRecognitionEnabled(true);
            qInfo() << "[startup] camera and recognition enabled";
        } else {
            emit recognitionEnabledChanged(false);
            setState(ServiceState::Streaming, tr("摄像头已自动开启；请采集样本并重建模型以启用识别"));
            qInfo() << "[startup] detection only; selected gallery unavailable or requires rebuild";
        }
    } catch (const std::exception &exception) {
        reportException(tr("自动启动失败"), exception);
    }
}

void OpenCvFaceBackend::refreshFaceDatabase()
{
    if (m_shutdownRequested.load(std::memory_order_relaxed)) return;
    try {
        initializeCore();
        const auto records = m_dataset->loadManifest();
        std::unordered_map<std::string, int> counts;
        for (const auto &sample : records) ++counts[sample.personId];
        FaceDatabasePeople people;
        for (const auto &person : m_dataset->loadRegistry()) {
            people.push_back({{utf8(person.personId), utf8(person.displayName),
                                person.group == facecore::PersonGroup::OutGroup ? "out" : "in"},
                              person.enabled, counts[person.personId]});
        }
        emit faceDatabaseChanged(people);
    } catch (const std::exception &exception) {
        emit errorOccurred(tr("读取人脸库失败"), exceptionText(exception));
    }
}

void OpenCvFaceBackend::loadPersonSamples(const QString &personId)
{
    if (m_shutdownRequested.load(std::memory_order_relaxed)) return;
    try {
        initializeCore();
        FaceDatabaseSamples samples;
        for (const auto &record : m_dataset->loadManifest()) {
            if (utf8(record.personId) != personId) continue;
            FaceDatabaseSample sample;
            sample.sampleId = utf8(record.sampleId);
            sample.sessionId = utf8(record.sessionId);
            sample.details = tr("样本：%1\n批次：%2\n来源：%3\n质量：%4")
                .arg(sample.sampleId, sample.sessionId, utf8(record.source))
                .arg(record.qualityScore, 0, 'f', 2);
            cv::Mat image;
            if (!record.alignedRelativePath.empty()) {
                try { image = m_dataset->loadAlignedSample(record); }
                catch (const std::exception &) { /* Fall back to the gray crop. */ }
            }
            if (image.empty()) {
                try { image = m_dataset->loadGraySample(record); }
                catch (const std::exception &exception) {
                    sample.details += tr("\n照片不可用：%1").arg(exceptionText(exception));
                }
            }
            if (!image.empty()) sample.image = toQImage(image);
            samples.push_back(std::move(sample));
        }
        emit personSamplesChanged(personId, samples);
    } catch (const std::exception &exception) {
        emit errorOccurred(tr("读取人脸样本失败"), exceptionText(exception));
    }
}

void OpenCvFaceBackend::requireDatabaseWritable()
{
    throwIfShutdownRequested();
    initializeCore();
    if (m_training || m_captureActive || m_rawCaptureActive)
        throw std::runtime_error("请先结束采集或训练，再修改人脸库");
}

void OpenCvFaceBackend::finishDatabaseChange(const QString &message)
{
    m_recognitionMemory.clear();
    loadRegistry();
    const bool stale = std::filesystem::exists(m_config.datasetRoot /
        (".gallery-stale-" + m_backendId.toStdString()));
    if (stale) {
        if (m_voice) m_voice->reset();
        m_recognitionEnabled = false;
        emit recognitionEnabledChanged(false);
        emit modelChanged(tr("需要重建"), tr("人脸库已变更，请执行“训练 / 重建模型”后开启识别。"));
    }
    refreshFaceDatabase();
    emit databaseOperationFinished(message + (stale ? tr("；请重建模型后开启识别。") : QString()));
}

void OpenCvFaceBackend::updateDatabasePerson(const PersonProfile &person, bool enabled)
{
    try {
        requireDatabaseWritable();
        auto record = toPersonRecord(person);
        record.enabled = enabled;
        m_dataset->updatePerson(record);
        finishDatabaseChange(tr("人员资料已保存"));
    } catch (const std::exception &exception) {
        emit errorOccurred(tr("保存人员失败"), exceptionText(exception));
    }
}

void OpenCvFaceBackend::deleteDatabasePerson(const QString &personId)
{
    try {
        requireDatabaseWritable();
        m_dataset->removePerson(personId.toStdString());
        finishDatabaseChange(tr("人员及样本索引已归档，照片已保留"));
    } catch (const std::exception &exception) {
        emit errorOccurred(tr("删除人员失败"), exceptionText(exception));
    }
}

void OpenCvFaceBackend::deleteDatabaseSample(const QString &personId, const QString &sampleId)
{
    try {
        requireDatabaseWritable();
        m_dataset->removeSample(personId.toStdString(), sampleId.toStdString());
        finishDatabaseChange(tr("样本索引已归档，照片已保留"));
        loadPersonSamples(personId);
    } catch (const std::exception &exception) {
        emit errorOccurred(tr("删除样本失败"), exceptionText(exception));
    }
}

void OpenCvFaceBackend::openCamera(const QString &cameraId)
{
    // 每次打开前先停止旧 reader、清空旧轨迹和采集状态，避免旧线程继续占用
    // 摄像头，或把上一次画面的 trackId/身份带进新画面。
    if (m_shutdownRequested.load(std::memory_order_relaxed)) {
        return;
    }

    try {
        initializeCore();
        const QString configuredDevice = QString::fromStdString(m_config.camera.device);
        const bool useConfiguredDevice =
            !configuredDevice.isEmpty() && cameraId == configuredDevice;
        bool valid = false;
        const int index = cameraId.toInt(&valid);
        if (!useConfiguredDevice && (!valid || index < 0 || index > 32)) {
            emit errorOccurred(tr("无法启动摄像头"), tr("无效的摄像头标识：%1").arg(cameraId));
            return;
        }

        if (m_cameraReader) {
            m_cameraReader->close();
            m_cameraReader.reset();
        }
        m_frameTimer->stop();
        m_captureActive = false;
        m_rawCaptureActive = false;
        m_rawCaptureAccepted = 0;
        m_rawCaptureTarget = 0;
        m_rawCaptureSessionId.clear();
        m_selectedTrackId.reset();
        m_duplicateFilter->clear();
        resetTrackingState();
        m_recognitionMemory.clear();
        clearReferenceCards();
        m_activeTrackIds.clear();

        // 按优先级依次尝试：配置指定的 GStreamer 管线 → 自动 GStreamer（ISP 缩放）
        // → 自动 GStreamer（videoscale 兜底）→ 直接 V4L2 → 设备索引。
        //
        // 板端实测（24_camera_open_probe.log）：rkisp 设备只提供
        // V4L2_CAP_VIDEO_CAPTURE_MPLANE，OpenCV 自带的 V4L2 后端六种打开方式全部失败，
        // 必须走 GStreamer 的 v4l2src + videoconvert，所以候选顺序不能随意调换。
        std::vector<facecore::CameraSourceOptions> candidates =
            facecore::buildCameraCandidates(m_config.camera);
        if (!useConfiguredDevice) {
            // 用户在界面上选的是索引，去掉所有带设备路径/管线的候选。
            candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                            [](const facecore::CameraSourceOptions &item) {
                                                return !item.pipeline.empty()
                                                       || !item.device.empty();
                                            }),
                             candidates.end());
            for (facecore::CameraSourceOptions &item : candidates) {
                item.deviceIndex = index;
            }
        }

        std::string lastError;
        std::string usedLabel;
        std::unique_ptr<facecore::IFrameSource> openedSource;
        for (const facecore::CameraSourceOptions &candidate : candidates) {
            std::string attemptError;
            std::unique_ptr<facecore::IFrameSource> source =
                facecore::makeCameraSource(candidate);
            if (source->open(&attemptError)) {
                openedSource = std::move(source);
                usedLabel = candidate.label;
                break;
            }
            lastError = candidate.label + "：" + attemptError;
            qInfo().noquote() << "[camera] 打开失败（" << utf8(candidate.label) << "）："
                              << utf8(attemptError);
        }
        if (!openedSource) {
            throw std::runtime_error("所有摄像头打开方式都失败，最后一次是 " + lastError);
        }
        m_cameraReader = std::make_unique<AsyncFrameReader>(std::move(openedSource));
        const cv::Size negotiated = m_cameraReader->negotiatedSize();
        qInfo().noquote() << "[camera] opened:" << utf8(usedLabel) << '|'
                          << QString::fromStdString(m_cameraReader->name()) << '|'
                          << QStringLiteral("%1x%2").arg(negotiated.width).arg(negotiated.height);
        m_cameraReader->start();
        m_streaming = true;
        m_sequence = 0;
        m_smoothedFps = 0.0;
        m_lastSystemMetricsMs = -1;
        m_previousCpuTotal = 0;
        m_previousCpuIdle = 0;
        m_systemMetrics = {};
        m_systemMetricsClock.restart();
        m_frameCadenceClock.invalidate();
        m_frameTimer->start();
        setState(ServiceState::Streaming,
                 tr("摄像头已启动：%1（%2×%3；配置 %4×%5 @ %6 fps）")
                     .arg(utf8(usedLabel))
                     .arg(negotiated.width)
                     .arg(negotiated.height)
                     .arg(m_config.camera.width)
                     .arg(m_config.camera.height)
                     .arg(m_config.camera.fps, 0, 'g', 3));
    } catch (const std::exception &exception) {
        m_streaming = false;
        reportException(tr("摄像头启动失败"), exception);
    }
}

void OpenCvFaceBackend::closeInput()
{
    // 关闭顺序是：停止 Qt 定时器 -> 关闭采集 reader -> 清理轨迹和识别记忆 ->
    // 通知 UI 回到 Idle。这样 timeout 不会在 reader 释放后继续取帧。
    m_frameTimer->stop();
    if (m_cameraReader) {
        m_cameraReader->close();
        m_cameraReader.reset();
    }
    m_streaming = false;
    m_captureActive = false;
    m_rawCaptureActive = false;
    m_rawCaptureAccepted = 0;
    m_rawCaptureTarget = 0;
    m_rawCaptureSessionId.clear();
    m_selectedTrackId.reset();
    if (m_duplicateFilter) {
        m_duplicateFilter->clear();
    }
    resetTrackingState();
    m_recognitionMemory.clear();
    clearReferenceCards();
    m_activeTrackIds.clear();
    emit captureProgress(0, 0);
    emit rawCaptureProgress(0, 0);
    emit frameReady(FrameResult{});
    setState(ServiceState::Idle, tr("视频输入已停止"));
}

void OpenCvFaceBackend::importImages(const QStringList &paths,
                                     const PersonProfile &person,
                                     bool saveToDataset)
{
    if (m_shutdownRequested.load(std::memory_order_relaxed)) {
        return;
    }
    if (paths.isEmpty()) {
        return;
    }
    if (saveToDataset && (person.personId.trimmed().isEmpty()
                          || person.displayName.trimmed().isEmpty())) {
        emit errorOccurred(tr("无法保存图片人脸"), tr("请先填写人员 ID 和姓名。"));
        return;
    }

    const bool resumeStreaming = m_streaming && m_cameraReader
                                 && m_cameraReader->isOpen();
    bool operationFailed = false;
    m_frameTimer->stop();
    if (resumeStreaming) {
        m_cameraReader->pause();
    }
    m_recognitionMemory.clear();
    clearReferenceCards();
    m_activeTrackIds.clear();
    try {
        initializeCore();
        setState(ServiceState::Importing, tr("正在处理 %1 张图片").arg(paths.size()));
        std::vector<std::filesystem::path> files;
        files.reserve(static_cast<std::size_t>(paths.size()));
        for (const QString &path : paths) {
            files.push_back(nativePath(path));
        }

        if (saveToDataset) {
            registerPersonOrThrow(person);
        }
        const std::string sessionId = makeSessionId("IMG");
        facecore::DuplicateFilter importDuplicate(m_config.quality.duplicateMeanAbsoluteDifference,
                                                  8);
        auto imageSource = facecore::makeImageSource(std::move(files));
        std::string error;
        if (!imageSource->open(&error)) {
            throw std::runtime_error(error);
        }

        int accepted = 0;
        int noFace = 0;
        int multipleFaces = 0;
        int lowQuality = 0;
        int duplicates = 0;
        for (int index = 0; index < paths.size(); ++index) {
            throwIfShutdownRequested();
            facecore::Frame frame;
            if (!imageSource->read(frame, &error)) {
                throw std::runtime_error(error);
            }
            resetTrackingState();
            m_recognitionMemory.clear();
            clearReferenceCards();
            m_activeTrackIds.clear();
            const auto detections = m_asyncDetector->detectSynchronously(frame.image);
            const FrameResult displayed = processFrame(frame, false, &detections);
            if (index == paths.size() - 1) {
                emit frameReady(displayed);
            }

            if (!saveToDataset) {
                continue;
            }
            if (detections.empty()) {
                ++noFace;
                continue;
            }
            if (detections.size() != 1) {
                ++multipleFaces;
                continue;
            }
            facecore::TrackedFace track;
            track.trackId = 1;
            track.detection = detections.front();
            track.colorBgr = facecore::FaceTracker::colorForTrack(1);
            const facecore::QualityReport quality = m_quality->evaluate(frame.image,
                                                                         track.detection);
            if (!quality.accepted) {
                ++lowQuality;
                continue;
            }
            facecore::FaceSample sample;
            try {
                sample = m_preprocessor->process(frame.image, track.detection);
            } catch (const std::invalid_argument &) {
                ++lowQuality;
                continue;
            }
            if (!importDuplicate.isNovelAndRemember(sample.grayFace)) {
                ++duplicates;
                continue;
            }
            sample.quality = quality;
            saveSample(frame, track, sample, quality, person, sessionId,
                       "image:" + std::to_string(index));
            ++accepted;
            if (accepted == 1 || accepted % 10 == 0 || index == paths.size() - 1) {
                setState(ServiceState::Importing,
                         tr("正在处理图片：%1/%2，已保存 %3")
                             .arg(index + 1).arg(paths.size()).arg(accepted));
            }
        }
        imageSource->close();
        resetTrackingState();
        m_recognitionMemory.clear();
        clearReferenceCards();
        m_activeTrackIds.clear();
        const QString summary = saveToDataset
            ? tr("图片导入完成：保存 %1，未检测到人脸 %2，多人图片跳过 %3，质量不足 %4，重复 %5")
                  .arg(accepted).arg(noFace).arg(multipleFaces).arg(lowQuality).arg(duplicates)
            : tr("已检测并显示 %1 张图片；未写入数据集").arg(paths.size());
        setState(resumeStreaming ? ServiceState::Streaming : ServiceState::Idle, summary);
    } catch (const std::exception &exception) {
        operationFailed = true;
        reportException(tr("图片导入失败"), exception);
    }
    if (resumeStreaming && !m_shutdownRequested.load(std::memory_order_relaxed)) {
        resetTrackingState();
        m_activeTrackIds.clear();
        m_cameraReader->start();
        m_frameTimer->start();
        if (operationFailed) {
            setState(ServiceState::Streaming, tr("图片导入失败；摄像头视频已恢复"));
        }
    }
}

void OpenCvFaceBackend::beginCapture(const PersonProfile &person, int targetSamples)
{
    if (m_shutdownRequested.load(std::memory_order_relaxed)) {
        return;
    }
    if (!m_streaming || !m_cameraReader || !m_cameraReader->isOpen()) {
        emit errorOccurred(tr("无法开始采集"), tr("请先启动摄像头。"));
        return;
    }
    if (person.personId.trimmed().isEmpty() || person.displayName.trimmed().isEmpty()
        || targetSamples <= 0) {
        emit errorOccurred(tr("人员信息不完整"), tr("请填写人员 ID、姓名和有效样本数。"));
        return;
    }
    try {
        initializeCore();
        registerPersonOrThrow(person);
        m_capturePerson = person;
        m_captureSessionId = makeSessionId("CAM");
        m_captureAccepted = 0;
        m_captureTarget = targetSamples;
        m_captureActive = true;
        m_selectedTrackId.reset();
        m_lastCaptureTimestampUs = 0;
        m_duplicateFilter->clear();
        emit captureProgress(0, targetSamples);
        setState(ServiceState::Capturing,
                 tr("正在为 %1 采集；请点击右侧人脸框确认目标，确认前不会保存")
                     .arg(person.displayName));
    } catch (const std::exception &exception) {
        reportException(tr("无法开始采集"), exception);
    }
}

void OpenCvFaceBackend::selectCaptureTrack(qint64 trackId)
{
    if (!m_captureActive) {
        return;
    }
    if (trackId <= 0) {
        emit errorOccurred(tr("目标选择失败"), tr("无效的跟踪编号。"));
        return;
    }
    const std::uint64_t selected = static_cast<std::uint64_t>(trackId);
    if (m_activeTrackIds.count(selected) == 0) {
        emit errorOccurred(tr("目标选择失败"),
                           tr("该目标已离开当前画面，请点击最新画面中的人脸框。"));
        return;
    }
    m_selectedTrackId = selected;
    m_duplicateFilter->clear();
    setState(ServiceState::Capturing,
             tr("正在为 %1 采集，已锁定目标 #%2")
                 .arg(m_capturePerson.displayName).arg(trackId));
}

void OpenCvFaceBackend::endCapture()
{
    if (!m_captureActive) {
        return;
    }
    m_captureActive = false;
    m_selectedTrackId.reset();
    setState(m_streaming ? ServiceState::Streaming : ServiceState::Idle,
             tr("采集已停止：%1/%2").arg(m_captureAccepted).arg(m_captureTarget));
}

void OpenCvFaceBackend::beginRawFrameCapture()
{
    if (m_shutdownRequested.load(std::memory_order_relaxed)) {
        return;
    }
    if (!m_streaming || !m_cameraReader || !m_cameraReader->isOpen()) {
        emit errorOccurred(tr("无法采集原始图像"), tr("请先启动摄像头。"));
        return;
    }
    if (m_captureActive || m_training) {
        emit errorOccurred(tr("无法采集原始图像"),
                           tr("请先结束当前人脸采集或训练任务。"));
        return;
    }
    if (m_rawCaptureActive) {
        return;
    }

    try {
        initializeCore();
        m_rawCaptureSessionId = makeSessionId("RAW");
        const std::filesystem::path directory =
            m_config.datasetRoot / "raw_frames" / m_rawCaptureSessionId;
        std::filesystem::create_directories(directory);
        m_rawCaptureAccepted = 0;
        m_rawCaptureTarget = kRawFrameCaptureTarget;
        m_lastRawCaptureTimestampUs = 0;
        m_rawCaptureActive = true;
        emit rawCaptureProgress(0, m_rawCaptureTarget);
        setState(ServiceState::Streaming,
                 tr("正在采集原始摄像头图像：0/%1").arg(m_rawCaptureTarget));
    } catch (const std::exception &exception) {
        reportException(tr("无法开始采集原始图像"), exception);
    }
}

void OpenCvFaceBackend::trainModel()
{
    // 这里的“训练”不是重新训练 MobileFaceNet/SFace 权重，而是使用已有样本
    // 重建 LBPH 模型和 embedding 人员特征库。训练期间暂停摄像头，避免同时读写。
    if (m_shutdownRequested.load(std::memory_order_relaxed)) {
        return;
    }
    if (m_training) {
        emit errorOccurred(tr("训练正在进行"), tr("请等待当前任务完成。"));
        return;
    }
    if (m_captureActive) {
        emit errorOccurred(tr("暂不能训练"), tr("请先结束人脸采集。"));
        return;
    }

    const bool resumeStreaming = m_streaming && m_cameraReader
                                 && m_cameraReader->isOpen();
    bool operationFailed = false;
    m_frameTimer->stop();
    if (resumeStreaming) {
        m_cameraReader->pause();
    }
    m_training = true;
    try {
        initializeCore();
        emit trainingProgress(5, tr("检查数据集"));
        loadRegistry();
        const auto records = m_dataset->loadManifest();
        if (records.empty()) {
            throw std::runtime_error("数据集为空，请先采集或导入人脸样本");
        }

        std::vector<facecore::LabeledFaceSample> samples;
        samples.reserve(records.size());
        std::vector<facecore::LabeledFaceSample> alignedSamples;
        alignedSamples.reserve(records.size());
        int skippedDisabledOrOrphan = 0;
        int skippedVersion = 0;
        int skippedReservedSplit = 0;
        int skippedInvalidGray = 0;
        int skippedInvalidAligned = 0;
        for (const facecore::SampleRecord &record : records) {
            throwIfShutdownRequested();
            if (m_people.count(record.personId) == 0) {
                ++skippedDisabledOrOrphan;
                continue;
            }
            if (record.preprocessingVersion != m_config.preprocess.version) {
                ++skippedVersion;
                continue;
            }
            if (record.split == facecore::DataSplit::Validation
                || record.split == facecore::DataSplit::Test) {
                ++skippedReservedSplit;
                continue;
            }
            facecore::LabeledFaceSample sample;
            sample.personId = record.personId;
            sample.sessionId = record.sessionId;
            try {
                sample.face.grayFace = m_dataset->loadGraySample(record);
            } catch (const std::exception &exception) {
                // 灰度图是 LBPH 的必需数据；坏记录只影响这一条样本，
                // 不能让同一批次里其余有效样本也无法训练。
                ++skippedInvalidGray;
                qWarning() << "训练时跳过无效灰度图"
                           << utf8(record.relativePath)
                           << exceptionText(exception);
                continue;
            }
            if (!record.alignedRelativePath.empty()) {
                try {
                    sample.face.alignedBgrFace = m_dataset->loadAlignedSample(record);
                } catch (const std::exception &exception) {
                    // 灰度图足够训练 LBPH；历史版本可能留下了空的彩色对齐图，
                    // 不能因为这类可选文件损坏而让整次训练失败。
                    ++skippedInvalidAligned;
                    qWarning() << "训练时跳过无效彩色对齐图"
                               << utf8(record.alignedRelativePath)
                               << exceptionText(exception);
                }
            }
            samples.push_back(std::move(sample));
            if (!samples.back().face.alignedBgrFace.empty()) {
                alignedSamples.push_back(samples.back());
            }
        }
        if (samples.empty()) {
            throw std::runtime_error(
                "没有符合当前预处理版本且属于启用人员的训练样本");
        }

        emit trainingProgress(25, tr("训练 LBPH"));
        auto replacementLbph = std::make_unique<facecore::LbphFaceRecognizer>(m_config.lbph);
        replacementLbph->train(samples);
        throwIfShutdownRequested();

        const std::filesystem::path modelDirectory = m_runtimeDirectory / "models";
        std::filesystem::create_directories(modelDirectory);
        const std::string version = QDateTime::currentDateTimeUtc()
                                        .toString(QStringLiteral("yyyyMMdd_HHmmss_zzz"))
                                        .toStdString();
        const auto lbphFile = modelDirectory / ("lbph_" + version + ".yml");
        replacementLbph->save(lbphFile);

        emit trainingProgress(60, tr("建立 SFace 特征库"));
        std::shared_ptr<facecore::IEmbeddingFaceRecognizer> replacementSface;
        std::shared_ptr<facecore::IEmbeddingFaceRecognizer> replacementMobileFaceNet;
        const bool hasAlignedSamples = !alignedSamples.empty();
        bool sfaceUpdated = false;
        bool mobileFaceNetUpdated = false;
        if (hasAlignedSamples) {
            std::string recognizerWarning;
            replacementSface =
                facecore::createEmbeddingRecognizer(m_config.sface, &recognizerWarning);
            if (replacementSface) {
                replacementSface->buildGallery(alignedSamples);
                throwIfShutdownRequested();
                replacementSface->save(modelDirectory / ("sface_" + version + ".yml"));
                sfaceUpdated = true;
            } else {
                qWarning() << "SFace 后端不可用，跳过 SFace，只保留 LBPH："
                           << utf8(recognizerWarning);
            }

            std::string mobileFaceNetWarning;
            replacementMobileFaceNet = facecore::createEmbeddingRecognizer(
                m_config.mobilefacenet, &mobileFaceNetWarning, false);
            if (replacementMobileFaceNet) {
                replacementMobileFaceNet->buildGallery(alignedSamples);
                throwIfShutdownRequested();
                replacementMobileFaceNet->save(
                    modelDirectory / ("mobilefacenet_" + version + ".yml"));
                mobileFaceNetUpdated = true;
            } else {
                qWarning() << "MobileFaceNet 后端不可用，跳过 MobileFaceNet 特征库："
                           << utf8(mobileFaceNetWarning);
            }
        }

        emit trainingProgress(90, tr("切换模型版本"));
        m_lbph = std::move(replacementLbph);
        if (replacementSface) {
            m_sface = std::move(replacementSface);
        }
        if (replacementMobileFaceNet) {
            m_mobileFaceNet = std::move(replacementMobileFaceNet);
        }
        m_recognitionMemory.clear();
        clearReferenceCards();
        loadRegistry();
        emit trainingProgress(100, tr("训练完成"));
        completeGalleryRebuild("opencv_lbph", lbphFile);
        if (sfaceUpdated) completeGalleryRebuild("onnx_embedding", modelDirectory / ("sface_" + version + ".yml"));
        if (mobileFaceNetUpdated) completeGalleryRebuild("mobilefacenet_rknn", modelDirectory / ("mobilefacenet_" + version + ".yml"));
        const QString embeddingNames =
            sfaceUpdated && mobileFaceNetUpdated
                ? tr("SFace 与 MobileFaceNet")
                : sfaceUpdated ? tr("SFace")
                               : mobileFaceNetUpdated ? tr("MobileFaceNet") : QString();
        emit modelChanged(QString::fromStdString(version),
                          embeddingNames.isEmpty()
                              ? tr("LBPH 已生成；embedding 特征库未更新；有效彩色对齐图 %1 张，无效灰度图 %2，无效对齐图 %3；跳过孤儿/禁用 %4，旧版本 %5，验证/测试集 %6")
                                    .arg(alignedSamples.size())
                                    .arg(skippedInvalidGray)
                                    .arg(skippedInvalidAligned)
                                    .arg(skippedDisabledOrOrphan)
                                    .arg(skippedVersion)
                                    .arg(skippedReservedSplit)
                              : tr("LBPH 与 %1 特征库已生成；使用有效彩色对齐图 %2 张；跳过无效灰度图 %3、孤儿/禁用 %4，旧版本 %5，验证/测试集 %6，无效对齐图 %7；正式验收前应以验证集标定")
                                    .arg(embeddingNames)
                                    .arg(alignedSamples.size())
                                    .arg(skippedInvalidGray)
                                    .arg(skippedDisabledOrOrphan)
                                    .arg(skippedVersion)
                                    .arg(skippedReservedSplit)
                                    .arg(skippedInvalidAligned));
        setState(resumeStreaming ? ServiceState::Streaming : ServiceState::Idle,
                 tr("模型训练完成，共使用 %1 张样本").arg(samples.size()));
    } catch (const std::exception &exception) {
        operationFailed = true;
        reportException(tr("模型训练失败"), exception);
    }
    m_training = false;
    if (resumeStreaming && !m_shutdownRequested.load(std::memory_order_relaxed)) {
        m_cameraReader->start();
        m_frameTimer->start();
        if (operationFailed) {
            setState(ServiceState::Streaming, tr("模型训练失败；摄像头视频已恢复"));
        }
    }
}

void OpenCvFaceBackend::setRecognitionEnabled(bool enabled)
{
    if (m_shutdownRequested.load(std::memory_order_relaxed)) {
        return;
    }
    try {
        initializeCore();
        if (enabled && !backendReady()) {
            m_recognitionEnabled = false;
            if (m_voice) m_voice->reset();
            emit recognitionEnabledChanged(false);
            emit errorOccurred(tr("尚无可用模型"),
                               tr("请先采集人员数据并执行“训练 / 重建模型”。"));
            return;
        }
        m_recognitionEnabled = enabled;
        if (!enabled && m_voice) m_voice->reset();
        emit recognitionEnabledChanged(enabled);
        m_recognitionMemory.clear();
        clearReferenceCards();
        setState(m_captureActive ? ServiceState::Capturing
                                 : (m_streaming ? ServiceState::Streaming : ServiceState::Idle),
                 enabled ? tr("实时识别已开启") : tr("当前仅进行人脸检测"));
    } catch (const std::exception &exception) {
        m_recognitionEnabled = false;
        if (m_voice) m_voice->reset();
        emit recognitionEnabledChanged(false);
        reportException(tr("切换识别状态失败"), exception);
    }
}

void OpenCvFaceBackend::setVoiceVolume(int volume, bool muted)
{
    if (m_shutdownRequested.load(std::memory_order_relaxed)) return;
    try {
        initializeCore();
        m_voice->setVolume(volume,muted);
    } catch (const std::exception &exception) {
        qWarning().noquote() << "[voice]" << exceptionText(exception);
    }
}

void OpenCvFaceBackend::shutdown()
{
    m_voice.reset();
    m_shutdownRequested.store(true, std::memory_order_relaxed);
    m_frameTimer->stop();
    if (m_cameraReader) {
        m_cameraReader->close();
        m_cameraReader.reset();
    }
    m_streaming = false;
    m_captureActive = false;
    m_rawCaptureActive = false;
    m_rawCaptureAccepted = 0;
    m_rawCaptureTarget = 0;
    m_rawCaptureSessionId.clear();
    emit rawCaptureProgress(0, 0);
    m_training = false;
}

void OpenCvFaceBackend::processNextFrame()
{
    // Qt 定时器只负责触发处理；真正的图像来自 AsyncFrameReader 的最新帧槽位。
    // 本函数把取帧、旋转、系统指标、单帧算法、FPS 统计和 frameReady 串起来。
    if (m_shutdownRequested.load(std::memory_order_relaxed)
        || !m_streaming || !m_cameraReader || !m_cameraReader->isOpen()) {
        return;
    }

    QElapsedTimer workElapsed;
    workElapsed.start();
    QElapsedTimer readElapsed;
    readElapsed.start();
    static const bool phaseTimingEnabled = qEnvironmentVariableIntValue(
                                                "FACEAPP_PHASE_TIMING") > 0;
    try {
        facecore::Frame frame;
        if (const auto error = m_cameraReader->takeError()) {
            throw std::runtime_error(*error);
        }
        if (!m_cameraReader->takeLatest(frame)) {
            return;
        }
        const qint64 readNanoseconds = readElapsed.nsecsElapsed();
        rotateCameraFrameForInference(frame);
        captureRawFrameIfRequested(frame);
        updateSystemMetrics();
        QElapsedTimer processElapsed;
        processElapsed.start();
        FrameResult result = processFrame(frame, true);
        const qint64 processNanoseconds = processElapsed.nsecsElapsed();
        result.systemMetrics = m_systemMetrics;
        const qint64 workNanoseconds = std::max<qint64>(1, workElapsed.nsecsElapsed());
        const bool havePreviousFrame = m_frameCadenceClock.isValid();
        const qint64 frameIntervalNanoseconds = havePreviousFrame
            ? std::max<qint64>(1, m_frameCadenceClock.nsecsElapsed())
            : 33333333;
        m_frameCadenceClock.restart();
        const double instantaneous = havePreviousFrame
            ? 1.0e9 / static_cast<double>(frameIntervalNanoseconds)
            : 30.0;
        m_smoothedFps = m_smoothedFps <= 0.0
                            ? instantaneous
                            : m_smoothedFps * 0.90 + instantaneous * 0.10;
        result.fps = m_smoothedFps;
        // Optional low-overhead timing diagnostics. Disabled by default; when
        // FACEAPP_FRAME_TIMING=1 is set, report slow frames and every 30th
        // frame so periodic detector stalls can be distinguished from UI cost.
        static const bool timingEnabled = qEnvironmentVariableIntValue(
                                               "FACEAPP_FRAME_TIMING") > 0;
        if (timingEnabled && (workNanoseconds >= 25000000 || result.sequence % 30 == 0)) {
            const double milliseconds = static_cast<double>(workNanoseconds) / 1.0e6;
            const bool detectorFrame =
                m_lastDetectionSequence == static_cast<std::uint64_t>(result.sequence);
            qInfo().noquote() << QStringLiteral("[timing] frame=%1 total_ms=%2 detector=%3 fps=%4")
                                     .arg(result.sequence)
                                     .arg(milliseconds, 0, 'f', 2)
                                     .arg(detectorFrame ? QStringLiteral("yes")
                                                        : QStringLiteral("no"))
                                     .arg(result.fps, 0, 'f', 1);
        }
        if (phaseTimingEnabled && (workNanoseconds >= 25000000 || result.sequence % 30 == 0)) {
            qInfo().noquote() << QStringLiteral(
                                      "[phase] frame=%1 capture_ms=%2 process_ms=%3 total_ms=%4")
                                     .arg(result.sequence)
                                     .arg(static_cast<double>(readNanoseconds) / 1.0e6,
                                          0, 'f', 2)
                                     .arg(static_cast<double>(processNanoseconds) / 1.0e6,
                                          0, 'f', 2)
                                     .arg(static_cast<double>(workNanoseconds) / 1.0e6,
                                          0, 'f', 2);
        }
        emit frameReady(result);
    } catch (const std::exception &exception) {
        m_frameTimer->stop();
        m_streaming = false;
        if (m_cameraReader) {
            m_cameraReader->close();
        }
        reportException(tr("视频处理失败"), exception);
        if (m_voice) m_voice->reset();
    }
}

void OpenCvFaceBackend::initializeCore()
{
    // 初始化只执行一次。顺序从外到内：读取配置并解析路径，建立检测器/跟踪器/
    // 预处理，再建立数据集和识别器，最后读取人员登记表与最近生成的模型。
    if (m_initialized) {
        return;
    }
    const QString configured = QString::fromLocal8Bit(qgetenv("FACEAPP_CONFIG"));
    const QString configPath = configured.isEmpty()
        ? QDir(m_applicationDirectory).filePath(QStringLiteral("config/default.json"))
        : configured;
    if (!QFileInfo::exists(configPath)) {
        throw std::runtime_error(
            tr("找不到配置文件：%1").arg(configPath).toUtf8().constData());
    }
    m_config = facecore::loadConfig(nativePath(configPath));
    const std::filesystem::path root = nativePath(m_applicationDirectory);
    const QString configuredDataDirectory =
        QString::fromLocal8Bit(qgetenv("FACEAPP_DATA_DIR"));
    QString runtimeDirectory = configuredDataDirectory;
    if (runtimeDirectory.isEmpty()) {
        runtimeDirectory = QStandardPaths::writableLocation(
            QStandardPaths::AppLocalDataLocation);
    }
    if (runtimeDirectory.isEmpty()) {
        runtimeDirectory = QDir(m_applicationDirectory).filePath(QStringLiteral("runtime"));
    }
    m_runtimeDirectory = nativePath(runtimeDirectory);
    std::filesystem::create_directories(m_runtimeDirectory / "data");
    std::filesystem::create_directories(m_runtimeDirectory / "models");
    std::filesystem::create_directories(m_runtimeDirectory / "logs");
    // 相对模型路径先按可执行文件目录解析；那里没有时再退回配置文件所在目录。
    // 这样用户把 .rknn 放进源码树的 models/ 后用 --config 指到源码配置就能生效，
    // 不必重新构建、也不必手工往构建目录里拷贝模型。
    const std::filesystem::path configDirectory = nativePath(configPath).parent_path();
    const auto resolveModelPath = [&root, &configDirectory](std::filesystem::path &path) {
        if (path.empty() || !path.is_relative()) {
            return;
        }
        const std::filesystem::path nearExecutable = root / path;
        const std::filesystem::path nearConfig = configDirectory / path;
        std::error_code error;
        if (!std::filesystem::is_regular_file(nearExecutable, error)
            && std::filesystem::is_regular_file(nearConfig, error)) {
            path = nearConfig;
        } else {
            path = nearExecutable;
        }
    };
    resolveModelPath(m_config.detector.modelPath);
    resolveModelPath(m_config.sface.modelPath);
    resolveModelPath(m_config.detector.rknn.modelPath);
    resolveModelPath(m_config.sface.rknn.modelPath);
    resolveModelPath(m_config.mobilefacenet.modelPath);
    resolveModelPath(m_config.mobilefacenet.rknn.modelPath);
    if (m_config.datasetRoot.is_relative()) {
        m_config.datasetRoot = m_runtimeDirectory / m_config.datasetRoot;
    }
    facecore::validateConfig(m_config);

    qInfo().noquote() << "[backend] capabilities:" << utf8(facecore::describeBackendCapabilities());
    std::string detectorWarning;
    auto detector = facecore::createFaceDetector(m_config.detector, &detectorWarning);
    if (!detectorWarning.empty()) {
        qInfo().noquote() << "[detector]" << utf8(detectorWarning);
    }
    if (detector) {
        qInfo().noquote() << "[detector] active:" << QString::fromStdString(detector->name());
    }
    m_asyncDetector = std::make_unique<AsyncDetector>(std::move(detector));
    qInfo().noquote() << "[detector] asynchronous worker enabled";
    m_tracker = std::make_unique<facecore::FaceTracker>(m_config.tracker);
    m_preprocessor = std::make_unique<facecore::FacePreprocessor>(m_config.preprocess);
    m_quality = std::make_unique<facecore::QualityEvaluator>(m_config.quality);
    m_duplicateFilter = std::make_unique<facecore::DuplicateFilter>(
        m_config.quality.duplicateMeanAbsoluteDifference, 8);
    m_dataset = std::make_unique<facecore::DatasetStore>(
        m_config.datasetRoot,
        cv::Size(m_config.preprocess.faceWidth, m_config.preprocess.faceHeight),
        cv::Size(112, 112));
    m_dataset->initialize();
    m_lbph = std::make_unique<facecore::LbphFaceRecognizer>(m_config.lbph);
    std::string recognizerWarning;
    m_sface = facecore::createEmbeddingRecognizer(m_config.sface, &recognizerWarning);
    if (!recognizerWarning.empty()) {
        qInfo().noquote() << "[recognizer]" << utf8(recognizerWarning);
    }
    if (m_sface) {
        m_backendSummary = QStringLiteral("%1（%2）")
                               .arg(QString::fromStdString(m_sface->name()),
                                    utf8(m_sface->extractorName()));
    } else {
        m_backendSummary = utf8(recognizerWarning);
        if (m_backendSummary.isEmpty()) {
            m_backendSummary = tr("模型待投放：没有可用的 embedding 识别后端");
        }
        qWarning().noquote() << "[recognizer]" << m_backendSummary;
    }
    std::string mobileFaceNetWarning;
    m_mobileFaceNet = facecore::createEmbeddingRecognizer(
        m_config.mobilefacenet, &mobileFaceNetWarning, false);
    if (!mobileFaceNetWarning.empty()) {
        qInfo().noquote() << "[recognizer/mobilefacenet]" << utf8(mobileFaceNetWarning);
    }
    if (m_mobileFaceNet) {
        m_mobileFaceNetSummary = QStringLiteral("%1（%2）")
                                    .arg(QString::fromStdString(m_mobileFaceNet->name()),
                                         utf8(m_mobileFaceNet->extractorName()));
    } else {
        m_mobileFaceNetSummary = utf8(mobileFaceNetWarning);
        if (m_mobileFaceNetSummary.isEmpty()) {
            m_mobileFaceNetSummary = tr("模型待投放：MobileFaceNet RKNN 不可用");
        }
        qWarning().noquote() << "[recognizer/mobilefacenet]" << m_mobileFaceNetSummary;
    }
    loadRegistry();
    loadNewestModels();
    auto voiceSettings=VoiceSettings::load(m_applicationDirectory);
    voiceSettings.preferencesFile=QString::fromStdString((m_runtimeDirectory/"voice-preferences.ini").string());
    m_voice = std::make_unique<VoiceAnnouncer>(std::move(voiceSettings));
    connect(m_voice.get(),&VoiceAnnouncer::volumeChanged,this,&FaceBackendService::voiceVolumeChanged);
    emit voiceVolumeChanged(m_voice->volume(),m_voice->muted());
    m_initialized = true;
}

void OpenCvFaceBackend::loadRegistry()
{
    // registry 决定哪些 personId 是启用人员，manifest 决定每个人有哪些样本。
    // 同时建立“人员 -> 第一张参考照片”的索引，供右侧静态照片卡片使用。
    m_people.clear();
    m_referenceSamples.clear();
    m_referenceImageCache.clear();
    m_referenceImageMisses.clear();
    clearReferenceCards();
    if (!m_dataset) {
        return;
    }
    for (const facecore::PersonRecord &person : m_dataset->loadRegistry()) {
        if (person.enabled) {
            m_people[person.personId] = person;
        }
    }
    for (const facecore::SampleRecord &record : m_dataset->loadManifest()) {
        if (m_people.count(record.personId) == 0 || record.relativePath.empty()) {
            continue;
        }
        // The first valid sample is enough for the static reference card. Prefer
        // the color-aligned photo, with the gray sample as a compatibility fallback.
        m_referenceSamples.emplace(record.personId, record);
    }
}

void OpenCvFaceBackend::clearReferenceCards() noexcept
{
    m_referenceCardPersonIds.clear();
    m_referenceCards.clear();
    ++m_referenceRevision;
}

QImage OpenCvFaceBackend::referenceImageForPerson(const std::string &personId)
{
    const auto cached = m_referenceImageCache.find(personId);
    if (cached != m_referenceImageCache.end()) {
        return cached->second;
    }
    if (m_referenceImageMisses.count(personId) != 0) {
        return {};
    }

    const auto sample = m_referenceSamples.find(personId);
    if (sample == m_referenceSamples.end()) {
        m_referenceImageMisses.insert(personId);
        return {};
    }

    cv::Mat image;
    try {
        if (!sample->second.alignedRelativePath.empty()) {
            image = m_dataset->loadAlignedSample(sample->second);
        }
        if (image.empty()) {
            image = m_dataset->loadGraySample(sample->second);
        }
    } catch (const std::exception &exception) {
        qWarning().noquote() << "[reference-photo]" << utf8(personId)
                             << exceptionText(exception);
        m_referenceImageMisses.insert(personId);
        return {};
    }

    if (image.empty()) {
        m_referenceImageMisses.insert(personId);
        return {};
    }
    const QImage converted = toQImage(image);
    if (converted.isNull()) {
        m_referenceImageMisses.insert(personId);
        return {};
    }
    m_referenceImageCache.emplace(personId, converted);
    return converted;
}

void OpenCvFaceBackend::completeGalleryRebuild(const std::string &backendId,
                                               const std::filesystem::path &modelFile)
{
    const auto stale = m_config.datasetRoot / (".gallery-stale-" + backendId);
    const auto compatible = m_config.datasetRoot / (".gallery-compatible-" + backendId);
    const bool invalidated = std::filesystem::exists(stale);
    if (!invalidated && !std::filesystem::exists(compatible)) return;
    QByteArray content;
    if (!invalidated) {
        QFile previous(utf8(compatible.string()));
        if (!previous.open(QIODevice::ReadOnly)) throw std::runtime_error("Cannot read gallery compatibility index");
        content = previous.readAll();
    }
    content += QByteArray::fromStdString(modelFile.filename().string()) + '\n';
    QSaveFile output(utf8(compatible.string()));
    if (!output.open(QIODevice::WriteOnly) || output.write(content) != content.size() || !output.commit())
        throw std::runtime_error("Cannot save gallery compatibility index");
    // Persist the generation boundary before allowing recognition again. Historical
    // galleries remain available as backups but cannot undo a dataset deletion.
    std::filesystem::remove(stale);
}

void OpenCvFaceBackend::loadNewestModels()
{
    const std::filesystem::path directory = m_runtimeDirectory / "models";
    const auto loadFirstValid = [this, &directory](const std::string &prefix,
                                                  const QString &label,
                                                  const std::string &backendId, auto load) {
        QStringList failures;
        const auto compatible = m_config.datasetRoot / (".gallery-compatible-" + backendId);
        const bool guarded = std::filesystem::exists(compatible);
        std::set<std::string> allowed;
        if (guarded) {
            std::ifstream input(compatible);
            if (!input) throw std::runtime_error("Cannot read gallery compatibility index");
            std::string filename;
            while (std::getline(input, filename)) allowed.insert(filename);
        }
        for (const auto &file : modelFilesNewestFirst(directory, prefix, ".yml")) {
            if (guarded && allowed.count(file.filename().string()) == 0) continue;
            try {
                load(file);
                qInfo().noquote() << "[models] loaded" << label << utf8(file.string());
                return;
            } catch (const std::exception &exception) {
                const QString failure = utf8(file.filename().string()) + ": "
                                        + exceptionText(exception);
                failures.push_back(failure);
                qWarning().noquote() << "[models] skipped" << label << failure;
            }
        }
        if (!failures.isEmpty()) {
            emit errorOccurred(tr("没有可加载的 %1 模型").arg(label), failures.join('\n'));
        }
    };
    loadFirstValid("lbph_", QStringLiteral("LBPH"), "opencv_lbph",
                   [this](const auto &file) { m_lbph->load(file); });
    if (m_sface) {
        loadFirstValid("sface_", QStringLiteral("SFace"), "onnx_embedding",
                       [this](const auto &file) { m_sface->load(file); });
    }
    if (m_mobileFaceNet) {
        loadFirstValid("mobilefacenet_", QStringLiteral("MobileFaceNet"), "mobilefacenet_rknn",
                       [this](const auto &file) { m_mobileFaceNet->load(file); });
    }
}

void OpenCvFaceBackend::registerPersonOrThrow(const PersonProfile &person)
{
    const facecore::PersonRecord candidate = toPersonRecord(person);
    for (const facecore::PersonRecord &existing : m_dataset->loadRegistry()) {
        if (existing.personId != candidate.personId) {
            continue;
        }
        if (existing.displayName != candidate.displayName
            || existing.group != candidate.group) {
            throw std::invalid_argument(
                tr("人员 ID %1 已绑定到“%2”（%3），不能写入另一人员。请使用新 ID，或改回原资料。")
                    .arg(person.personId,
                         utf8(existing.displayName),
                         groupName(existing.group))
                    .toUtf8().constData());
        }
        if (!existing.enabled) {
            throw std::invalid_argument("该人员已禁用，请先在人脸数据库管理中启用并保存");
        }
        break;
    }
    m_dataset->upsertPerson(candidate);
    loadRegistry();
}

void OpenCvFaceBackend::resetTrackingState() noexcept
{
    if (m_voice) m_voice->reset();
    if (m_asyncDetector) {
        m_asyncDetector->discardPendingResults();
    }
    if (m_tracker) {
        m_tracker->reset();
    }
    m_lastTracks.clear();
    m_previousTrackingGray.release();
    m_lastDetectionSequence = 0;
    m_lastDetectionRequestSequence = 0;
    m_consecutiveDetectionMisses = 0;
}

std::vector<facecore::FaceDetection> OpenCvFaceBackend::predictDetections(
    const cv::Mat &previousGray, const cv::Mat &currentGray) const
{
    std::vector<facecore::FaceDetection> predictions;
    predictions.reserve(m_lastTracks.size());
    if (m_lastTracks.empty()) {
        return predictions;
    }

    const bool canTrack = !previousGray.empty() && !currentGray.empty()
                          && previousGray.size() == currentGray.size()
                          && previousGray.type() == CV_8UC1
                          && currentGray.type() == CV_8UC1;
    const auto median = [](std::vector<float> values) {
        if (values.empty()) return 0.0F;
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };

    for (const facecore::TrackedFace &track : m_lastTracks) {
        facecore::FaceDetection prediction = track.detection;
        if (!canTrack) {
            predictions.push_back(std::move(prediction));
            continue;
        }

        const cv::Rect2f box = track.detection.box;
        const float left = std::clamp(box.x, 0.0F,
                                      std::max(0.0F, static_cast<float>(currentGray.cols - 1)));
        const float top = std::clamp(box.y, 0.0F,
                                     std::max(0.0F, static_cast<float>(currentGray.rows - 1)));
        const float right = std::clamp(box.x + box.width, left + 1.0F,
                                       static_cast<float>(currentGray.cols));
        const float bottom = std::clamp(box.y + box.height, top + 1.0F,
                                        static_cast<float>(currentGray.rows));
        const float width = right - left;
        const float height = bottom - top;

        std::vector<cv::Point2f> previousPoints;
        previousPoints.reserve(9);
        for (const float y : {0.20F, 0.50F, 0.80F}) {
            for (const float x : {0.20F, 0.50F, 0.80F}) {
                previousPoints.emplace_back(left + width * x, top + height * y);
            }
        }

        std::vector<cv::Point2f> currentPoints;
        std::vector<unsigned char> status;
        std::vector<float> errors;
        try {
            cv::calcOpticalFlowPyrLK(previousGray, currentGray, previousPoints,
                                     currentPoints, status, errors, cv::Size(15, 15), 2,
                                     cv::TermCriteria(cv::TermCriteria::COUNT
                                                          | cv::TermCriteria::EPS,
                                                      10, 0.03));
        } catch (const cv::Exception &) {
            currentPoints.clear();
            status.clear();
            errors.clear();
        }

        std::vector<float> shiftsX;
        std::vector<float> shiftsY;
        for (std::size_t index = 0; index < currentPoints.size()
                                      && index < previousPoints.size(); ++index) {
            if (index >= status.size() || status[index] == 0
                || !std::isfinite(currentPoints[index].x)
                || !std::isfinite(currentPoints[index].y)) {
                continue;
            }
            if (index < errors.size() && (!std::isfinite(errors[index]) || errors[index] > 64.0F)) {
                continue;
            }
            shiftsX.push_back(currentPoints[index].x - previousPoints[index].x);
            shiftsY.push_back(currentPoints[index].y - previousPoints[index].y);
        }

        if (shiftsX.size() >= 2) {
            const float shiftX = median(std::move(shiftsX));
            const float shiftY = median(std::move(shiftsY));
            const float maximumShift = std::max(24.0F,
                                                0.50F * static_cast<float>(
                                                    std::max(currentGray.cols, currentGray.rows)));
            if (std::abs(shiftX) <= maximumShift && std::abs(shiftY) <= maximumShift) {
                prediction.box.x += shiftX;
                prediction.box.y += shiftY;
                prediction.box.x = std::clamp(
                    prediction.box.x, 0.0F,
                    std::max(0.0F, static_cast<float>(currentGray.cols) - prediction.box.width));
                prediction.box.y = std::clamp(
                    prediction.box.y, 0.0F,
                    std::max(0.0F, static_cast<float>(currentGray.rows) - prediction.box.height));
                if (prediction.hasLandmarks) {
                    for (cv::Point2f &landmark : prediction.landmarks) {
                        landmark.x += shiftX;
                        landmark.y += shiftY;
                    }
                }
            }
        }
        predictions.push_back(std::move(prediction));
    }
    return predictions;
}

std::vector<facecore::TrackedFace> OpenCvFaceBackend::updateTracks(
    const cv::Mat &image, std::uint64_t sequence)
{
    // 先消费后台检测结果，再用光流预测检测间隔内的人脸位置，最后交给
    // FaceTracker 分配/更新 trackId。检测器较慢时，画面仍能继续跟随人脸框。
    const std::uint64_t detectionInterval = static_cast<std::uint64_t>(
        std::max(1, m_config.tracker.detectionIntervalFrames));
    std::vector<facecore::FaceDetection> detections;
    cv::Mat currentGray;
    if (!m_lastTracks.empty()) {
        if (image.channels() == 1) {
            currentGray = image;
        } else {
            cv::cvtColor(image, currentGray, cv::COLOR_BGR2GRAY);
        }
    }
    // 先消费后台检测结果，再提交新任务。检测线程只处理最新的一帧，
    // 当前画面线程始终使用光流预测继续绘制，避免检测器阻塞画面刷新。
    if (m_asyncDetector) {
        const auto completed = m_asyncDetector->takeResult();
        if (completed.has_value()) {
            if (!completed->error.empty()) {
                throw std::runtime_error(completed->error);
            }
            m_lastDetectionSequence = completed->sequence;
            if (completed->detections.empty()) {
                ++m_consecutiveDetectionMisses;
                if (m_consecutiveDetectionMisses <= kMaximumDetectionMisses) {
                    detections = predictDetections(m_previousTrackingGray, currentGray);
                } else {
                    resetTrackingState();
                }
            } else {
                m_consecutiveDetectionMisses = 0;
                detections = completed->detections;
            }
        }
    }

    // 没有上一帧跟踪目标且后台返回了新检测结果时，当前帧仍需要保存
    // 灰度图供下一帧光流使用；无脸画面则完全跳过这次转换。
    if (currentGray.empty() && !detections.empty()) {
        if (image.channels() == 1) {
            currentGray = image;
        } else {
            cv::cvtColor(image, currentGray, cv::COLOR_BGR2GRAY);
        }
    }

    if (detections.empty()) {
        detections = predictDetections(m_previousTrackingGray, currentGray);
    }

    const bool detectionDue = m_lastDetectionRequestSequence == 0
                              || sequence - m_lastDetectionRequestSequence >= detectionInterval;
    if (detectionDue && m_asyncDetector
        && m_asyncDetector->submit(sequence, image)) {
        m_lastDetectionRequestSequence = sequence;
    }

    std::vector<facecore::TrackedFace> tracks;
    if (m_tracker) {
        tracks = m_tracker->update(detections);
    }
    m_lastTracks = tracks;
    m_previousTrackingGray = std::move(currentGray);
    return tracks;
}

FrameResult OpenCvFaceBackend::processFrame(
    const facecore::Frame &frame, bool allowCapture,
    const std::vector<facecore::FaceDetection> *imageDetections)
{
    // 单帧业务总入口：跟踪 -> 采集 -> 质量评估 -> 必要时识别 -> 组织 UI 结果。
    // 这里不直接绘制文字，FrameView 会在主线程的 paintEvent() 中绘制。
    const std::uint64_t sequence = ++m_sequence;
    static const bool phaseTimingEnabled = qEnvironmentVariableIntValue(
                                                "FACEAPP_PHASE_TIMING") > 0;
    QElapsedTimer phaseElapsed;
    if (phaseTimingEnabled) {
        phaseElapsed.start();
    }
    const bool stillImage = imageDetections != nullptr;
    const std::vector<facecore::TrackedFace> tracks = stillImage
        ? m_tracker->update(*imageDetections) : updateTracks(frame.image, sequence);
    const qint64 trackingNanoseconds = phaseTimingEnabled ? phaseElapsed.nsecsElapsed() : 0;
    if (phaseTimingEnabled) {
        phaseElapsed.restart();
    }

    m_activeTrackIds.clear();
    for (const facecore::TrackedFace &track : tracks) {
        m_activeTrackIds.insert(track.trackId);
    }

    if (allowCapture && m_captureActive) {
        captureIfEligible(frame, tracks);
    }

    std::set<std::uint64_t> activeTracks;
    FrameResult output;
    output.sequence = static_cast<qint64>(sequence);
    output.originalImage = toQImage(frame.image);
    output.processedImage = output.originalImage;
    const qint64 qimageNanoseconds = phaseTimingEnabled ? phaseElapsed.nsecsElapsed() : 0;
    if (phaseTimingEnabled) {
        phaseElapsed.restart();
    }
    output.faces.reserve(static_cast<int>(tracks.size()));
    for (const facecore::TrackedFace &track : tracks) {
        activeTracks.insert(track.trackId);
        facecore::QualityReport quality = m_quality->evaluate(frame.image, track.detection);
        std::optional<facecore::RecognitionResult> recognition;
        if (m_recognitionEnabled && quality.accepted && backendReady()) {
            const auto memory = m_recognitionMemory.find(track.trackId);
            if (stillImage || memory == m_recognitionMemory.end()
                || recognitionDue(memory->second, frame.timestampUs)) {
                try {
                    facecore::FaceSample sample = m_preprocessor->process(frame.image,
                                                                           track.detection);
                    sample.quality = quality;
                    recognition = recognizeTrack(track, sample, sequence, frame.timestampUs,
                                                  !stillImage);
                } catch (const std::invalid_argument &) {
                    quality.accepted = false;
                    quality.reasons.push_back("invalid_alignment");
                    m_recognitionMemory.erase(track.trackId);
                }
            } else {
                recognition = cachedRecognition(track.trackId);
            }
        }
        FaceObservation observation = makeObservation(track, quality, recognition);
        observation.normalizedBox = QRectF(
            track.detection.box.x / static_cast<qreal>(frame.image.cols),
            track.detection.box.y / static_cast<qreal>(frame.image.rows),
            track.detection.box.width / static_cast<qreal>(frame.image.cols),
            track.detection.box.height / static_cast<qreal>(frame.image.rows));
        output.faces.push_back(std::move(observation));
    }

    // Only live, enabled recognition speaks. Imported still images never do.
    // The optical-flow/tracker lifecycle defines loss, with no wall-clock cooldown.
    if (m_voice && !stillImage && allowCapture && m_recognitionEnabled) {
        QSet<int> alive;
        for (const auto id : m_tracker->activeTrackIds()) alive.insert(static_cast<int>(id));
        m_voice->onFrame(output.faces, QDateTime::currentDateTime().time().hour(), alive);
    }

    for (auto iterator = m_recognitionMemory.begin(); iterator != m_recognitionMemory.end();) {
        if (activeTracks.count(iterator->first) == 0) {
            iterator = m_recognitionMemory.erase(iterator);
        } else {
            ++iterator;
        }
    }

    // Build the right-side static reference cards only when the set of confirmed
    // people changes. The live camera image stays exclusively on the left, so a
    // stable identity does not cause another right-side image refresh every frame.
    std::vector<std::string> currentReferencePeople;
    for (const facecore::TrackedFace &track : tracks) {
        const auto memory = m_recognitionMemory.find(track.trackId);
        if (memory == m_recognitionMemory.end()
            || !memory->second.consensusReached
            || memory->second.stablePersonId.empty()) {
            continue;
        }
        if (std::find(currentReferencePeople.begin(), currentReferencePeople.end(),
                      memory->second.stablePersonId) == currentReferencePeople.end()) {
            currentReferencePeople.push_back(memory->second.stablePersonId);
        }
    }
    if (currentReferencePeople != m_referenceCardPersonIds) {
        m_referenceCardPersonIds = currentReferencePeople;
        m_referenceCards.clear();
        for (const std::string &personId : m_referenceCardPersonIds) {
            ReferenceFaceCard card;
            card.personId = utf8(personId);
            card.image = referenceImageForPerson(personId);
            const auto person = m_people.find(personId);
            if (person != m_people.end()) {
                card.displayName = utf8(person->second.displayName);
                card.group = groupName(person->second.group);
            }
            m_referenceCards.push_back(std::move(card));
        }
        ++m_referenceRevision;
        // Only the frame that announces a card-set change needs to carry the
        // QImage payload across the worker/UI boundary. Later frames carry the
        // revision number only, avoiding repeated QVector/QImage ref-counting.
        output.referenceCards = m_referenceCards;
    }
    output.referenceRevision = static_cast<qint64>(m_referenceRevision);
    if (phaseTimingEnabled && (sequence % 30 == 0 || trackingNanoseconds >= 25000000
                               || qimageNanoseconds >= 10000000)) {
        const qint64 facesNanoseconds = phaseElapsed.nsecsElapsed();
        qInfo().noquote() << QStringLiteral(
                                  "[phase-inner] frame=%1 track_ms=%2 qimage_ms=%3 faces_ms=%4")
                                 .arg(static_cast<qulonglong>(sequence))
                                 .arg(static_cast<double>(trackingNanoseconds) / 1.0e6,
                                      0, 'f', 2)
                                 .arg(static_cast<double>(qimageNanoseconds) / 1.0e6,
                                      0, 'f', 2)
                                 .arg(static_cast<double>(facesNanoseconds) / 1.0e6,
                                      0, 'f', 2);
    }
    return output;
}

FaceObservation OpenCvFaceBackend::makeObservation(
    const facecore::TrackedFace &track,
    const facecore::QualityReport &quality,
    const std::optional<facecore::RecognitionResult> &recognition) const
{
    FaceObservation output;
    output.trackId = static_cast<int>(track.trackId);
    const QColor color(static_cast<int>(track.colorBgr[2]),
                       static_cast<int>(track.colorBgr[1]),
                       static_cast<int>(track.colorBgr[0]));
    output.color = color;
    output.selectedForCapture = m_captureActive
                                && m_selectedTrackId.has_value()
                                && *m_selectedTrackId == track.trackId;

    if (!quality.accepted) {
        output.identityState = IdentityState::DetectionOnly;
        output.scoreText = qualityMessage(quality);
        return output;
    }
    if (!recognition.has_value()) {
        output.identityState = IdentityState::DetectionOnly;
        return output;
    }

    const facecore::RecognitionResult &result = *recognition;
    if (result.state == facecore::RecognitionState::Known && m_people.count(result.personId) == 0) {
        output.identityState = IdentityState::Unknown;
    } else if (result.state == facecore::RecognitionState::Known) {
        output.identityState = IdentityState::Known;
        output.personId = utf8(result.personId);
        const auto found = m_people.find(result.personId);
        if (found != m_people.end()) {
            output.displayName = utf8(found->second.displayName);
            output.group = groupName(found->second.group);
        }
    } else if (result.state == facecore::RecognitionState::Unknown) {
        output.identityState = IdentityState::Unknown;
    } else {
        output.identityState = IdentityState::DetectionOnly;
    }
    if (std::isfinite(result.bestScore)) {
        output.scoreText = result.backend == "opencv-lbph"
            ? tr("距离 %1").arg(result.bestScore, 0, 'f', 1)
            : tr("相似度 %1").arg(result.bestScore, 0, 'f', 2);
    }
    return output;
}

bool OpenCvFaceBackend::recognitionDue(const RecognitionMemory &memory,
                                      std::int64_t timestampUs) const noexcept
{
    return memory.due(timestampUs, m_config.tracker.recognitionIntervalMs);
}

std::optional<facecore::RecognitionResult> OpenCvFaceBackend::cachedRecognition(
    std::uint64_t trackId) const
{
    const auto found = m_recognitionMemory.find(trackId);
    if (found == m_recognitionMemory.end()) {
        return std::nullopt;
    }
    return found->second.cached();
}

std::optional<facecore::RecognitionResult> OpenCvFaceBackend::recognizeTrack(
    const facecore::TrackedFace &track,
    const facecore::FaceSample &sample,
    std::uint64_t sequence,
    std::int64_t timestampUs,
    bool temporalVoting)
{
    // 根据 backendId 选择 LBPH、SFace 或 MobileFaceNet，将结果加入该 track 的
    // 5 次投票窗口。多数结果达到 3 次后写入 stablePersonId。
    RecognitionMemory &memory = m_recognitionMemory[track.trackId];
    static const bool timingEnabled = qEnvironmentVariableIntValue(
                                           "FACEAPP_FRAME_TIMING") > 0;
    QElapsedTimer recognitionElapsed;
    if (timingEnabled) {
        recognitionElapsed.start();
    }
    if (m_backendId == QStringLiteral("opencv_lbph")) {
        memory.latest = m_lbph->recognize(sample);
    } else if (m_backendId == QStringLiteral("mobilefacenet_rknn")
               ? static_cast<bool>(m_mobileFaceNet)
               : static_cast<bool>(m_sface)) {
        memory.latest = m_backendId == QStringLiteral("mobilefacenet_rknn")
                            ? m_mobileFaceNet->recognize(sample)
                            : m_sface->recognize(sample);
    } else {
        memory.latest = {};
        memory.latest.backend = "unavailable";
        memory.latest.state = facecore::RecognitionState::Error;
        memory.latest.message = "模型待投放：没有可用的 embedding 识别后端";
    }
    if (timingEnabled) {
        qInfo().noquote() << QStringLiteral("[timing] recognition_ms=%1 backend=%2 track=%3")
                                 .arg(static_cast<double>(recognitionElapsed.nsecsElapsed()) / 1.0e6,
                                      0, 'f', 2)
                                 .arg(m_backendId)
                                 .arg(static_cast<qulonglong>(track.trackId));
    }
    if (qEnvironmentVariableIntValue("FACEAPP_RECOGNITION_LOG") > 0) {
        const auto &result = memory.latest;
        qInfo().noquote() << "[recognition] track=" << static_cast<qulonglong>(track.trackId)
                         << "person=" << utf8(result.personId)
                         << "best=" << result.bestScore << "second=" << result.secondBestScore
                         << "gap=" << result.bestScore - result.secondBestScore
                         << "threshold=" << result.threshold << "reason=" << utf8(result.message);
    }
    memory.update(memory.latest, sequence, timestampUs);
    if (!temporalVoting && (memory.latest.state == facecore::RecognitionState::Known
                            || memory.latest.state == facecore::RecognitionState::Unknown)) {
        memory.consensusReached = true;
        memory.stablePersonId = memory.latest.known() ? memory.latest.personId : std::string();
    }

    return cachedRecognition(track.trackId);
}

void OpenCvFaceBackend::captureRawFrameIfRequested(const facecore::Frame &frame)
{
    if (!m_rawCaptureActive || frame.image.empty()) {
        return;
    }
    if (m_rawCaptureAccepted >= m_rawCaptureTarget) {
        m_rawCaptureActive = false;
        emit rawCaptureProgress(m_rawCaptureTarget, m_rawCaptureTarget);
        return;
    }
    if (m_lastRawCaptureTimestampUs > 0
        && frame.timestampUs - m_lastRawCaptureTimestampUs < kMinimumRawCaptureIntervalUs) {
        return;
    }

    const std::filesystem::path directory =
        m_config.datasetRoot / "raw_frames" / m_rawCaptureSessionId;
    const std::filesystem::path filename =
        directory / ("raw_" + std::to_string(m_rawCaptureAccepted + 1) + ".jpg");
    try {
        const bool saved = cv::imwrite(filename.u8string(), frame.image,
                                       {cv::IMWRITE_JPEG_QUALITY, 95});
        if (!saved) {
            throw std::runtime_error("OpenCV 无法写入原始摄像头图像");
        }
    } catch (const std::exception &exception) {
        m_rawCaptureActive = false;
        emit rawCaptureProgress(0, 0);
        emit errorOccurred(tr("原始图像保存失败"), exceptionText(exception));
        return;
    }

    m_lastRawCaptureTimestampUs = frame.timestampUs;
    ++m_rawCaptureAccepted;
    emit rawCaptureProgress(m_rawCaptureAccepted, m_rawCaptureTarget);
    if (m_rawCaptureAccepted >= m_rawCaptureTarget) {
        m_rawCaptureActive = false;
        setState(ServiceState::Streaming,
                 tr("原始摄像头图像采集完成：共 %1 张，已保存到 data/raw_frames/%2")
                     .arg(m_rawCaptureAccepted)
                     .arg(utf8(m_rawCaptureSessionId)));
    } else if (m_rawCaptureAccepted == 1 || m_rawCaptureAccepted % 5 == 0) {
        setState(ServiceState::Streaming,
                 tr("正在采集原始摄像头图像：%1/%2")
                     .arg(m_rawCaptureAccepted)
                     .arg(m_rawCaptureTarget));
    }
}

void OpenCvFaceBackend::captureIfEligible(
    const facecore::Frame &frame,
    const std::vector<facecore::TrackedFace> &tracks)
{
    if (!m_selectedTrackId.has_value()) {
        return;
    }
    const auto found = std::find_if(tracks.begin(), tracks.end(), [this](const auto &track) {
        return track.trackId == *m_selectedTrackId;
    });
    if (found == tracks.end()) {
        m_selectedTrackId.reset();
        m_duplicateFilter->clear();
        setState(ServiceState::Capturing,
                 tr("采集目标已离开画面；请重新点击 %1 的人脸框")
                     .arg(m_capturePerson.displayName));
        return;
    }
    if (frame.timestampUs - m_lastCaptureTimestampUs < kMinimumCaptureIntervalUs) {
        return;
    }
    const facecore::QualityReport quality = m_quality->evaluate(frame.image, found->detection);
    if (!quality.accepted) {
        return;
    }
    facecore::FaceSample sample;
    try {
        sample = m_preprocessor->process(frame.image, found->detection);
    } catch (const std::invalid_argument &) {
        return;
    }
    if (!m_duplicateFilter->isNovelAndRemember(sample.grayFace)) {
        return;
    }
    sample.quality = quality;
    saveSample(frame, *found, sample, quality, m_capturePerson, m_captureSessionId,
               frame.source);
    m_lastCaptureTimestampUs = frame.timestampUs;
    ++m_captureAccepted;
    emit captureProgress(m_captureAccepted, m_captureTarget);
    if (m_captureAccepted >= m_captureTarget) {
        m_captureActive = false;
        m_selectedTrackId.reset();
        setState(ServiceState::Streaming,
                 tr("采集完成：%1，共 %2 张")
                     .arg(m_capturePerson.displayName)
                     .arg(m_captureAccepted));
    }
}

void OpenCvFaceBackend::saveSample(const facecore::Frame &frame,
                                   const facecore::TrackedFace &track,
                                   const facecore::FaceSample &sample,
                                   const facecore::QualityReport &quality,
                                   const PersonProfile &person,
                                   const std::string &sessionId,
                                   const std::string &source)
{
    facecore::SampleRecord metadata;
    metadata.personId = person.personId.trimmed().toStdString();
    metadata.sessionId = sessionId;
    metadata.source = source;
    metadata.timestampUs = frame.timestampUs;
    metadata.sourceBox = track.detection.box;
    metadata.qualityScore = quality.sharpness;
    metadata.preprocessingVersion = m_config.preprocess.version;
    metadata.split = facecore::DataSplit::Unassigned;
    m_dataset->saveSample(sample.grayFace, sample.alignedBgrFace, std::move(metadata));
}

bool OpenCvFaceBackend::backendReady() const
{
    if (std::filesystem::exists(m_config.datasetRoot /
        (".gallery-stale-" + m_backendId.toStdString()))) return false;
    if (m_backendId == QStringLiteral("opencv_lbph")) {
        return m_lbph && m_lbph->trained();
    }
    if (m_backendId == QStringLiteral("mobilefacenet_rknn")) {
        return m_mobileFaceNet && m_mobileFaceNet->identityCount() > 0;
    }
    return m_sface && m_sface->identityCount() > 0;
}

void OpenCvFaceBackend::setState(ServiceState state, const QString &message)
{
    emit stateChanged(state, message);
}

void OpenCvFaceBackend::reportException(const QString &title,
                                        const std::exception &exception)
{
    const QString details = exceptionText(exception);
    emit errorOccurred(title, details);
    setState(ServiceState::Error, title + QStringLiteral("：") + details);
}

void OpenCvFaceBackend::throwIfShutdownRequested() const
{
    if (m_shutdownRequested.load(std::memory_order_relaxed)) {
        throw std::runtime_error("Operation cancelled because the application is closing");
    }
}

std::filesystem::path OpenCvFaceBackend::nativePath(const QString &path)
{
#if defined(_WIN32)
    return std::filesystem::path(path.toStdWString());
#else
    return std::filesystem::u8path(path.toUtf8().constData());
#endif
}

QString OpenCvFaceBackend::utf8(const std::string &text)
{
    return QString::fromUtf8(text.data(), static_cast<int>(text.size()));
}

QImage OpenCvFaceBackend::toQImage(const cv::Mat &image)
{
    if (image.empty()) {
        return {};
    }
    QImage::Format format = QImage::Format_Invalid;
    if (image.type() == CV_8UC3) {
        format = QImage::Format_RGB888;
    } else if (image.type() == CV_8UC1) {
        format = QImage::Format_Grayscale8;
    } else if (image.type() == CV_8UC4) {
        format = QImage::Format_RGBA8888;
    } else {
        throw std::invalid_argument("Unsupported cv::Mat format for QImage conversion");
    }

    // 直接让 QImage 持有最终缓冲区，避免“OpenCV 转换缓冲区 -> QImage.copy()”
    // 的第二次整帧复制。QImage 会按值管理自己的内存，返回后不依赖 cv::Mat。
    QImage output(image.cols, image.rows, format);
    cv::Mat destination(image.rows, image.cols, image.type(), output.bits(),
                        static_cast<std::size_t>(output.bytesPerLine()));
    if (image.type() == CV_8UC3) {
        cv::cvtColor(image, destination, cv::COLOR_BGR2RGB);
    } else if (image.type() == CV_8UC4) {
        cv::cvtColor(image, destination, cv::COLOR_BGRA2RGBA);
    } else {
        image.copyTo(destination);
    }
    return output;
}

facecore::PersonRecord OpenCvFaceBackend::toPersonRecord(const PersonProfile &person)
{
    facecore::PersonRecord output;
    output.personId = person.personId.trimmed().toStdString();
    output.displayName = person.displayName.trimmed().toUtf8().toStdString();
    if (person.group == QStringLiteral("in")) {
        output.group = facecore::PersonGroup::InGroup;
    } else if (person.group == QStringLiteral("out")) {
        output.group = facecore::PersonGroup::OutGroup;
    } else {
        throw std::invalid_argument("Person group must be 'in' or 'out'");
    }
    output.enabled = true;
    return output;
}

QString OpenCvFaceBackend::groupName(facecore::PersonGroup group)
{
    return group == facecore::PersonGroup::OutGroup ? tr("组外") : tr("组内");
}

std::string OpenCvFaceBackend::makeSessionId(const char *prefix)
{
    return std::string(prefix) + "_" +
           QDateTime::currentDateTimeUtc()
               .toString(QStringLiteral("yyyyMMdd_HHmmss_zzz"))
               .toStdString();
}

QString OpenCvFaceBackend::qualityMessage(const facecore::QualityReport &quality)
{
    if (quality.reasons.empty()) {
        return {};
    }
    const std::string &reason = quality.reasons.front();
    if (reason == "face_too_small") return tr("人脸过小");
    if (reason == "face_clipped") return tr("人脸超出画面");
    if (reason == "too_dark") return tr("光线过暗");
    if (reason == "too_bright") return tr("光线过亮");
    if (reason == "too_blurry") return tr("图像模糊");
    if (reason == "invalid_landmarks" || reason == "invalid_alignment") return tr("人脸对齐无效");
    return tr("质量不足");
}

std::vector<std::filesystem::path> OpenCvFaceBackend::modelFilesNewestFirst(
    const std::filesystem::path &directory,
    const std::string &prefix,
    const std::string &extension)
{
    if (!std::filesystem::is_directory(directory)) {
        return {};
    }
    std::vector<std::filesystem::path> candidates;
    for (const auto &entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string filename = entry.path().filename().string();
        if (filename.rfind(prefix, 0) != 0 || entry.path().extension() != extension) {
            continue;
        }
        candidates.push_back(entry.path());
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto &first, const auto &second) {
        const auto a = std::filesystem::last_write_time(first);
        const auto b = std::filesystem::last_write_time(second);
        return a != b ? a > b : first.filename().native() > second.filename().native();
    });
    return candidates;
}
