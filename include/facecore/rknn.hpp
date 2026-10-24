#pragma once

#include "facecore/config.hpp"
#include "facecore/detector.hpp"
#include "facecore/embedding.hpp"
#include "facecore/types.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace facecore {

// 一个 RKNN 张量的可读描述。
struct RknnTensorReport {
    std::string name;
    std::string shape;         // 例如 [1,3,320,320]
    std::string dataType;      // FP32 / UINT8 / INT8 ...
    std::string format;        // NCHW / NHWC / NC1HWC2 / UNDEFINED
    std::string quantization;  // none / affine(zp=.., scale=..) / dfp(fl=..)
    std::uint32_t elements{0};
    std::uint32_t bytes{0};
};

// RKNN 运行库与驱动环境。不加载模型也能得到大部分信息。
struct RknnRuntimeInfo {
    bool compiledIn{false};
    bool libraryAvailable{false};
    std::string libraryPath;
    std::uint64_t librarySize{0};
    std::string libraryVersion;  // 从 librknnrt.so 里提取的版本字符串
    std::string driverVersion;   // /sys/kernel/debug/rknpu/version
    std::string npuLoad;         // /sys/kernel/debug/rknpu/load
    std::string userGroups;      // render 组等，便于排查 /dev/dri 权限
    std::string notes;
};

FACECORE_API bool rknnBackendCompiledIn() noexcept;
FACECORE_API RknnRuntimeInfo probeRknnRuntime();
FACECORE_API std::string describeRknnRuntime(const RknnRuntimeInfo& info);

// 加载 .rknn 并报告输入输出张量契约；加载后立即释放。
// 用于在正式推理前逐项核对量化参数与形状。
struct RknnModelReport {
    std::string modelPath;
    std::uint64_t modelBytes{0};
    std::string apiVersion;
    std::string driverVersion;
    std::vector<RknnTensorReport> inputs;
    std::vector<RknnTensorReport> outputs;
};

FACECORE_API RknnModelReport inspectRknnModel(const std::filesystem::path& modelPath);
FACECORE_API std::string describeRknnModel(const RknnModelReport& report);

// RK3568 主通路：YuNet 检测（NPU）。
class FACECORE_API RknnYuNetFaceDetector final : public IFaceDetector {
public:
    explicit RknnYuNetFaceDetector(DetectorConfig config);
    ~RknnYuNetFaceDetector() override;

    RknnYuNetFaceDetector(RknnYuNetFaceDetector&&) noexcept;
    RknnYuNetFaceDetector& operator=(RknnYuNetFaceDetector&&) noexcept;
    RknnYuNetFaceDetector(const RknnYuNetFaceDetector&) = delete;
    RknnYuNetFaceDetector& operator=(const RknnYuNetFaceDetector&) = delete;

    std::vector<FaceDetection> detect(const cv::Mat& bgrFrame) override;
    std::string name() const override;
    std::string describe() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// RK3568 主通路：SFace embedding（NPU）。返回的抽取器可直接交给
// EmbeddingFaceRecognizer 使用，图库格式与 CPU 版完全一致。
FACECORE_API std::shared_ptr<const IEmbeddingExtractor> makeRknnSFaceExtractor(
    const SFaceConfig& config);

}  // namespace facecore
