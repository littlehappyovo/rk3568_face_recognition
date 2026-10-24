#pragma once

#include "facecore/config.hpp"
#include "facecore/export.hpp"
#include "facecore/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace facecore {

class FACECORE_API IFaceDetector {
public:
    virtual ~IFaceDetector() = default;
    virtual std::vector<FaceDetection> detect(const cv::Mat& bgrFrame) = 0;
    virtual std::string name() const = 0;
};

class FACECORE_API YuNetFaceDetector final : public IFaceDetector {
public:
    explicit YuNetFaceDetector(DetectorConfig config);
    ~YuNetFaceDetector() override;

    YuNetFaceDetector(YuNetFaceDetector&&) noexcept;
    YuNetFaceDetector& operator=(YuNetFaceDetector&&) noexcept;
    YuNetFaceDetector(const YuNetFaceDetector&) = delete;
    YuNetFaceDetector& operator=(const YuNetFaceDetector&) = delete;

    std::vector<FaceDetection> detect(const cv::Mat& bgrFrame) override;
    std::string name() const override;
    // 可读说明：网络输入尺寸与各 stride 的 head 数量，用于启动日志与验收工具。
    std::string describe() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace facecore

