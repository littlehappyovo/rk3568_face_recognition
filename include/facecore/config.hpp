#pragma once

#include "facecore/export.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace facecore {

// 配置层负责把 JSON 中的字符串和数字转换为强类型结构体。算法对象
// 创建时还会再次校验范围，这样配置错误能在启动阶段暴露出来。
// 计算后端。Auto 表示“能用 NPU 就用 NPU”，否则回退 CPU/OpenCV。
enum class ComputeBackend { Auto, OpenCv, Rknn };

FACECORE_API const char* toString(ComputeBackend value) noexcept;
FACECORE_API ComputeBackend parseComputeBackend(const std::string& text);

// RKNN（.rknn）模型契约。
//
// 这些字段必须与用户用 RKNN-Toolkit2 转换模型时使用的参数一致：应用不会猜
// 色序、layout、mean/std 或量化方式。任何一项不一致通常不会报错，只会让数值
// 结果变错，所以加载时会把解析结果完整打印出来，便于逐项核对。
struct RknnModelConfig {
    std::filesystem::path modelPath;
    // bgr | rgb | gray
    std::string inputColorOrder{"bgr"};
    // nchw | nhwc | auto（按 RKNN 实际张量布局）
    std::string inputLayout{"nchw"};
    // model_tensor: existing native tensor path; raw_uint8: interleaved pixels,
    // with normalization/quantization performed by RKNN using the compiled model.
    std::string inputMode{"model_tensor"};
    // 像素先乘 inputScale，再减 inputMean，最后除 inputStd。
    double inputScale{1.0};
    std::vector<double> inputMean{0.0, 0.0, 0.0};
    std::vector<double> inputStd{1.0, 1.0, 1.0};
    // -1 自动：量化输入直接喂 0..255 像素，浮点输入按 mean/std 归一化。
    // 0 强制喂浮点，1 强制喂量化像素。
    int inputQuantized{-1};
    // 0 表示 NPU_CORE_AUTO。
    int coreMask{0};
    // 可选：显式指定输出张量名；留空时按张量形状自动识别。
    std::vector<std::string> outputNames;
    // YuNet 专用：0 自动（检测到两个 1 通道 head 时取 sqrt(cls*obj)），
    // 1 强制乘积，2 只用单 head。
    int objectnessMode{0};
};

struct DetectorConfig {
    std::filesystem::path modelPath;
    // 检测网络的输入尺寸，CPU 与 NPU 两条通路共用（都会向上对齐到 32 的倍数）。
    // 用 RKNN 时该值必须与 .rknn 的输入张量一致，否则加载时会直接报错。
    int inputWidth{320};
    int inputHeight{320};
    float scoreThreshold{0.85F};
    float nmsThreshold{0.30F};
    int topK{5000};
    int backendId{0};
    int targetId{0};
    ComputeBackend backend{ComputeBackend::Auto};
    RknnModelConfig rknn;
};

struct TrackerConfig {
    // 这些参数决定检测结果怎样匹配为同一个 trackId，以及轨迹允许丢失多久。
    float minimumIou{0.20F};
    float maximumNormalizedCenterDistance{1.25F};
    int maximumMissedFrames{8};
    float smoothingAlpha{0.65F};
    int detectionIntervalFrames{10};
    int recognitionIntervalMs{1000};
};

struct PreprocessConfig {
    // grayFace 给 LBPH 使用，alignedBgrFace 给 embedding 后端使用。
    int faceWidth{128};
    int faceHeight{128};
    float boxExpansion{0.18F};
    bool alignByEyes{true};
    bool equalizeHistogram{true};
    std::string version{"gray128-eq-v1"};
};

struct QualityConfig {
    int minimumFaceWidth{80};
    double minimumBrightness{35.0};
    double maximumBrightness{225.0};
    double minimumLaplacianVariance{45.0};
    float maximumClippedFraction{0.10F};
    double duplicateMeanAbsoluteDifference{3.0};
};

struct LbphConfig {
    int radius{1};
    int neighbors{8};
    int gridX{8};
    int gridY{8};
    double distanceThreshold{75.0};
    double minimumDistanceMargin{3.0};
    std::string preprocessingVersion;
};

struct SFaceConfig {
    // SFace 和 MobileFaceNet 共用这套配置形状，但模型、维度和输入契约不同。
    std::filesystem::path modelPath;
    double cosineThreshold{0.363};
    double minimumCosineMargin{0.04};
    int backendId{0};
    int targetId{0};
    std::string preprocessingVersion;
    std::string modelIdentity;
    ComputeBackend backend{ComputeBackend::Auto};
    RknnModelConfig rknn;
    int inputWidth{112};
    int inputHeight{112};
    int embeddingDim{128};
};

// MobileFaceNet 与 SFace 复用同一套 embedding 图库和阈值契约；单独的类型名
// 让配置和 UI 能清楚区分两种模型，避免把 512 维 MobileFaceNet 特征库误当成
// 128 维 SFace 特征库。
using MobileFaceNetConfig = SFaceConfig;

// 摄像头参数。device 非空时优先按设备路径打开（例如 /dev/video-camera0），
// 否则退回 OpenCV 设备索引。
//
// Linux/ISP 说明：RK3568 的 rkisp 设备只提供 V4L2_CAP_VIDEO_CAPTURE_MPLANE。
// 当前版本优先尝试原生 V4L2，失败后才使用自动 GStreamer 和 OpenCV 回退路径。
// gstreamerPipeline 留空时会由 buildCameraCandidates() 自动生成。
struct CameraConfig {
    std::string device;
    std::string gstreamerPipeline;
    int index{0};
    int width{1280};
    int height{720};
    double fps{30.0};
};

struct SystemConfig {
    DetectorConfig detector;
    TrackerConfig tracker;
    PreprocessConfig preprocess;
    QualityConfig quality;
    LbphConfig lbph;
    SFaceConfig sface;
    MobileFaceNetConfig mobilefacenet;
    CameraConfig camera;
    std::filesystem::path datasetRoot{"data"};
};

FACECORE_API SystemConfig loadConfig(const std::filesystem::path& file);
FACECORE_API void saveConfig(const SystemConfig& config, const std::filesystem::path& file);
FACECORE_API void validateConfig(const SystemConfig& config);

}  // namespace facecore
