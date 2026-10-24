#include "facecore/backend_factory.hpp"

#include "facecore/rknn.hpp"

#include <exception>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace facecore {
namespace {

bool modelFileReady(const std::filesystem::path& path) {
    if (path.empty()) return false;
    std::error_code error;
    return std::filesystem::is_regular_file(path, error);
}

bool wantsRknn(ComputeBackend backend, const RknnModelConfig& rknn) {
    if (backend == ComputeBackend::Rknn) return true;
    if (backend == ComputeBackend::OpenCv) return false;
    return rknnBackendAvailable() && modelFileReady(rknn.modelPath);
}

void appendWarning(std::string* warning, const std::string& text) {
    if (warning == nullptr || text.empty()) return;
    if (!warning->empty()) *warning += " | ";
    *warning += text;
}

bool isMissingModelError(const std::string& message) {
    return message.find("does not exist") != std::string::npos
           || message.find("is empty") != std::string::npos
           || message.find("model is empty") != std::string::npos
           || message.find("modelPath") != std::string::npos;
}

std::string placeholderHint(const RknnModelConfig& rknn) {
    return "模型待投放：请把转换好的 .rknn 放到 " +
           (rknn.modelPath.empty() ? std::string("<未配置路径>")
                                   : rknn.modelPath.string()) +
           "，并确认它由 RKNN-Toolkit2 2.3.2、target_platform='rk3568' 转换得到";
}

}  // namespace

bool rknnBackendAvailable() noexcept {
    return rknnBackendCompiledIn() && probeRknnRuntime().libraryAvailable;
}

bool openCvDnnBackendAvailable() noexcept {
#if FACEAPP_HAVE_OPENCV_DNN
    return true;
#else
    return false;
#endif
}

bool lbphBackendAvailable() noexcept {
#if FACEAPP_HAVE_LBPH
    return true;
#else
    return false;
#endif
}

std::string describeBackendCapabilities() {
    std::ostringstream text;
    text << "RKNN(compile)=" << (rknnBackendCompiledIn() ? "yes" : "no")
         << " RKNN(runtime)=" << (rknnBackendAvailable() ? "yes" : "no")
         << " OpenCVDNN=" << (openCvDnnBackendAvailable() ? "yes" : "no")
         << " LBPH=" << (lbphBackendAvailable() ? "yes" : "no");
    return text.str();
}

std::unique_ptr<IFaceDetector> createFaceDetector(const DetectorConfig& config,
                                                  std::string* warning) {
    if (wantsRknn(config.backend, config.rknn)) {
        try {
            auto detector = std::make_unique<RknnYuNetFaceDetector>(config);
            appendWarning(warning, "检测后端：RKNN/NPU");
            return detector;
        } catch (const std::exception& exception) {
            const std::string reason = exception.what();
            if (config.backend == ComputeBackend::Rknn && isMissingModelError(reason)) {
                appendWarning(warning, placeholderHint(config.rknn));
            }
            appendWarning(warning, std::string("RKNN YuNet 不可用，已回退 OpenCV：") + reason);
        }
    } else if (config.backend == ComputeBackend::Rknn && !rknnBackendCompiledIn()) {
        appendWarning(warning, "配置要求 RKNN 检测后端，但本构建没有编译 RKNN 支持");
    }

#if FACEAPP_HAVE_OPENCV_DNN
    try {
        auto detector = std::make_unique<YuNetFaceDetector>(config);
        appendWarning(warning, "检测后端：OpenCV DNN YuNet（CPU）");
        return detector;
    } catch (const std::exception& exception) {
        // 回退也失败时把两边的原因都带上，避免只看到“OpenCV 起不来”而丢掉 RKNN 的真实原因。
        const std::string detail = std::string("OpenCV DNN YuNet 也不可用：") + exception.what();
        if (warning == nullptr || warning->empty()) throw std::runtime_error(detail);
        throw std::runtime_error(detail + " | " + *warning);
    }
#else
    throw std::runtime_error(
        "No face detector backend is available. Enable FACEAPP_ENABLE_RKNN with a valid "
        "detector.rknn.model, or enable FACEAPP_ENABLE_OPENCV_DNN. " +
        describeBackendCapabilities());
#endif
}

std::shared_ptr<IEmbeddingFaceRecognizer> createEmbeddingRecognizer(const SFaceConfig& config,
                                                                   std::string* warning,
                                                                   bool allowOpenCvFallback) {
    if (wantsRknn(config.backend, config.rknn)) {
        try {
            auto extractor = makeRknnSFaceExtractor(config);
            auto recognizer = std::make_shared<EmbeddingFaceRecognizer>(extractor, config);
            appendWarning(warning, "识别后端：RKNN/NPU");
            return recognizer;
        } catch (const std::exception& exception) {
            const std::string reason = exception.what();
            if (config.backend == ComputeBackend::Rknn && isMissingModelError(reason)) {
                appendWarning(warning, placeholderHint(config.rknn));
            }
            appendWarning(warning, std::string("RKNN embedding 不可用：") + reason);
            if (!allowOpenCvFallback) {
                appendWarning(warning, "已禁止回退到不匹配的 OpenCV SFace 预处理");
                return nullptr;
            }
        }
    }

#if FACEAPP_HAVE_OPENCV_DNN
    try {
        auto extractor = makeOpenCvSFaceExtractor(config);
        auto recognizer = std::make_shared<EmbeddingFaceRecognizer>(extractor, config);
        appendWarning(warning, "识别后端：OpenCV DNN SFace（CPU）");
        return recognizer;
    } catch (const std::exception& exception) {
        appendWarning(warning, std::string("OpenCV DNN SFace 不可用：") + exception.what());
    }
#else
    appendWarning(warning, "本构建没有编译 OpenCV DNN SFace 后端");
#endif

    appendWarning(warning,
                  "没有可用的 embedding 识别后端，实时识别将不可用（LBPH 通路不受影响）。" +
                      describeBackendCapabilities());
    return nullptr;
}

}  // namespace facecore
