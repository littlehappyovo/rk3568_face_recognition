#pragma once

#include "facecore/config.hpp"
#include "facecore/detector.hpp"
#include "facecore/embedding.hpp"

#include <memory>
#include <string>

namespace facecore {

// 工厂把“配置想用什么”和“当前编译/运行环境能用什么”连接起来。
// 应用层只保存接口指针，因此 processFrame() 不需要知道具体是 CPU 还是 NPU。
// 检测后端工厂。
//
// backend = rknn  ：必须用 NPU；加载失败时记录警告并尽量回退（回退不可用才抛错）。
// backend = opencv：只用 OpenCV DNN。
// backend = auto  ：.rknn 存在且本构建包含 RKNN 时优先 NPU，否则 OpenCV。
FACECORE_API std::unique_ptr<IFaceDetector> createFaceDetector(const DetectorConfig& config,
                                                               std::string* warning = nullptr);

// embedding 识别器工厂。返回 nullptr 表示“没有可用的 embedding 后端”（例如用户
// 还没投放 .rknn，且本构建也没有 OpenCV DNN），此时 LBPH 通路仍可正常工作，
// 界面应明确提示“模型待投放”，而不是崩溃或假装在用 NPU。
FACECORE_API std::shared_ptr<IEmbeddingFaceRecognizer> createEmbeddingRecognizer(
    const SFaceConfig& config,
    std::string* warning = nullptr,
    bool allowOpenCvFallback = true);

// 当前构建到底带哪些后端，用于启动日志与自检。
FACECORE_API bool rknnBackendAvailable() noexcept;
FACECORE_API bool openCvDnnBackendAvailable() noexcept;
FACECORE_API bool lbphBackendAvailable() noexcept;
FACECORE_API std::string describeBackendCapabilities();

}  // namespace facecore
