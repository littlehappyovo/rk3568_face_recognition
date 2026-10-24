#include "facecore/embedding.hpp"

#include "io_internal.hpp"

#if FACEAPP_HAVE_OPENCV_DNN
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace facecore {
namespace {

#if FACEAPP_HAVE_OPENCV_DNN

// CPU 特征抽取后端：直接用 OpenCV 的通用 dnn 模块跑 SFace ONNX。
//
// 预处理与 OpenCV 4.5.4 的 cv::FaceRecognizerSF::feature 完全一致（已核对 4.5.4 源码
// modules/objdetect/src/face_recognize.cpp 第 41 行）：
//   blobFromImage(aligned_img, 1.0, Size(112, 112), Scalar(0, 0, 0), swapRB=true, crop=false)
// 即：缩放到 112x112、BGR 转 RGB、原始 0..255（scale=1、mean=0），不裁剪。
// 这里不复用 cv::FaceRecognizerSF：一是 4.5.4 与 4.7+ 的重载不同，二是把预处理写成
// 显式代码后，NPU 侧的 mean/std/色序契约就有唯一的权威参照。
class OpenCvDnnSFaceExtractor final : public IEmbeddingExtractor {
public:
    explicit OpenCvDnnSFaceExtractor(const SFaceConfig& config)
        : inputWidth(config.inputWidth), inputHeight(config.inputHeight) {
        if (config.modelPath.empty()) throw std::invalid_argument("SFace model path is empty");
        if (inputWidth <= 0 || inputHeight <= 0) {
            throw std::invalid_argument("SFace input size must be positive");
        }
        if (!std::filesystem::is_regular_file(config.modelPath)) {
            throw std::runtime_error("SFace model file does not exist: " +
                                     config.modelPath.string());
        }
        try {
            net = cv::dnn::readNetFromONNX(detail::pathToUtf8(config.modelPath));
        } catch (const cv::Exception& exception) {
            throw std::runtime_error(std::string("Failed to load the SFace ONNX model: ") +
                                     exception.what());
        }
        if (net.empty()) throw std::runtime_error("OpenCV returned an empty SFace network");
        net.setPreferableBackend(config.backendId);
        net.setPreferableTarget(config.targetId);

        // 启动阶段空跑一次，确认模型输出维度与配置一致，避免第一帧才发现不匹配。
        const cv::Mat probe =
            featureOf(cv::Mat::zeros(cv::Size(inputWidth, inputHeight), CV_8UC3));
        if (probe.total() == 0) {
            throw std::runtime_error("SFace ONNX produced an empty feature");
        }
        if (probe.total() != static_cast<std::size_t>(config.embeddingDim)) {
            throw std::runtime_error(
                "SFace ONNX outputs " + std::to_string(probe.total()) +
                " values but sface.embedding_dim is " + std::to_string(config.embeddingDim));
        }
    }

    cv::Mat extract(const cv::Mat& alignedFace) const override {
        if (alignedFace.empty()) throw std::invalid_argument("Aligned SFace input is empty");
        cv::Mat bgr;
        if (alignedFace.type() == CV_8UC3) {
            bgr = alignedFace;
        } else if (alignedFace.type() == CV_8UC1) {
            cv::cvtColor(alignedFace, bgr, cv::COLOR_GRAY2BGR);
        } else if (alignedFace.type() == CV_8UC4) {
            cv::cvtColor(alignedFace, bgr, cv::COLOR_BGRA2BGR);
        } else {
            throw std::invalid_argument("SFace input must be an 8-bit image");
        }
        const cv::Size expected(inputWidth, inputHeight);
        if (bgr.size() != expected) cv::resize(bgr, bgr, expected, 0.0, 0.0, cv::INTER_LINEAR);
        return normalizeEmbedding(featureOf(bgr));
    }

    std::string name() const override { return "opencv-dnn-sface-112"; }

    std::string describe() const override {
        return "OpenCV DNN SFace ONNX (" + std::to_string(inputWidth) + "x" +
               std::to_string(inputHeight) + ", RGB 0..255, no mean/std)";
    }

private:
    cv::Mat featureOf(const cv::Mat& bgr) const {
        cv::Mat blob = cv::dnn::blobFromImage(bgr, 1.0, cv::Size(inputWidth, inputHeight),
                                              cv::Scalar(0, 0, 0), /*swapRB=*/true,
                                              /*crop=*/false);
        net.setInput(blob);
        cv::Mat output;
        try {
            net.forward(output);
        } catch (const cv::Exception& exception) {
            throw std::runtime_error(std::string("SFace forward failed: ") + exception.what());
        }
        if (output.empty()) throw std::runtime_error("SFace ONNX produced no output");
        cv::Mat flat;
        if (output.type() == CV_32F) {
            flat = output;
        } else {
            output.convertTo(flat, CV_32F);
        }
        if (!flat.isContinuous()) flat = flat.clone();
        return flat.reshape(1, 1);
    }

    // extract() 是接口要求的 const 成员，而 cv::dnn::Net::setInput/forward 都不是 const，
    // 所以这里必须 mutable。抽取器本身在多个线程间共享时由调用方负责加锁
    // （EmbeddingFaceRecognizer 内部已有 mutex）。
    mutable cv::dnn::Net net;
    int inputWidth{112};
    int inputHeight{112};
};

#endif  // FACEAPP_HAVE_OPENCV_DNN

}  // namespace

std::shared_ptr<const IEmbeddingExtractor> makeOpenCvSFaceExtractor(const SFaceConfig& config) {
#if FACEAPP_HAVE_OPENCV_DNN
    return std::make_shared<OpenCvDnnSFaceExtractor>(config);
#else
    (void)config;
    throw std::runtime_error(
        "The OpenCV DNN SFace backend is not available in this build; rebuild with "
        "FACEAPP_ENABLE_OPENCV_DNN=ON against an OpenCV that ships the dnn module, "
        "or use sface.backend=rknn");
#endif
}

}  // namespace facecore
