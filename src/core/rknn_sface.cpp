#include "facecore/rknn.hpp"

#include "io_internal.hpp"
#include "rknn_session.hpp"

#include <opencv2/imgproc.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace facecore {
namespace {

#if FACEAPP_HAVE_RKNN

// 这个抽取器只负责“对齐脸 -> RKNN 输入 -> 特征向量”。SFaceConfig 和
// EmbeddingGallery 是通用接口，因此实际模型可以是 SFace RKNN，也可以是
// MobileFaceNet RKNN；图库和阈值判定统一在 embedding.cpp 中完成。
// RKNN 版 embedding 抽取器。图库、阈值与持久化都交给
// EmbeddingFaceRecognizer，这里只负责对齐人脸 -> 归一化 embedding。
class RknnEmbeddingExtractor final : public IEmbeddingExtractor {
public:
    explicit RknnEmbeddingExtractor(SFaceConfig value) : config(std::move(value)) {
        if (config.rknn.modelPath.empty()) {
            throw std::invalid_argument("embedding.rknn.model is empty");
        }
        if (config.embeddingDim <= 0) {
            throw std::invalid_argument("sface.embedding_dim must be positive");
        }
        session.load(config.rknn.modelPath, config.rknn.coreMask);
        if (session.inputs().size() != 1) {
            throw std::runtime_error("RKNN embedding expects exactly one input tensor but the model "
                                     "reports " + std::to_string(session.inputs().size()));
        }
        inputDesc = session.inputs().front();
        layout = resolveRknnInputLayout(inputDesc, config.rknn);
        if (layout.channels != 3) {
            throw std::runtime_error("RKNN embedding expects a 3-channel input tensor but the model "
                                     "reports " + std::to_string(layout.channels) + " channel(s)");
        }
        rawPixels = config.rknn.inputMode == "raw_uint8";
        if (rawPixels && config.rknn.inputColorOrder != "rgb"
                      && config.rknn.inputColorOrder != "bgr") {
            throw std::invalid_argument("raw_uint8 embedding expects rgb or bgr input_color");
        }

        const std::vector<RknnTensorDesc>& outputs = session.outputs();
        if (outputs.empty()) throw std::runtime_error("RKNN embedding has no output tensors");
        outputIndex = 0;
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            if (outputs[index].elements == static_cast<std::uint32_t>(config.embeddingDim)) {
                outputIndex = index;
                break;
            }
        }
        if (outputs[outputIndex].elements != static_cast<std::uint32_t>(config.embeddingDim)) {
            throw std::runtime_error("No RKNN output matches embedding_dim=" +
                                     std::to_string(config.embeddingDim));
        }
    }

    cv::Mat extract(const cv::Mat& alignedFace) const override {
        // 先统一成 BGR 8-bit，再按照模型实际 layout、色序和量化类型填充
        // inputBuffer。输出维度不符合 embeddingDim 时拒绝继续，避免错误向量
        // 被写入特征库。
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
        const cv::Size expected(layout.width, layout.height);
        if (bgr.size() != expected) cv::resize(bgr, bgr, expected, 0.0, 0.0, cv::INTER_LINEAR);

        const bool quantized = config.rknn.inputQuantized == 1
                               || (config.rknn.inputQuantized < 0 && inputDesc.quantized);
        if (rawPixels) {
            cv::Mat pixels;
            if (config.rknn.inputColorOrder == "rgb") {
                cv::cvtColor(bgr, pixels, cv::COLOR_BGR2RGB);
            } else {
                pixels = bgr.clone();
            }
            // No scale/mean/std, FP16 conversion or native quantization here.
            // Standard RKNN input conversion performs these exactly once.
            inputBuffer.assign(pixels.data, pixels.data + pixels.total() * pixels.elemSize());
        } else if (quantized) {
            fillQuantizedInput(bgr, config.rknn, inputDesc, inputBuffer);
        } else {
            fillFloatInput(bgr, config.rknn, inputDesc, inputBuffer);
        }

        std::vector<std::vector<float>> outputs;
        session.infer({inputBuffer}, outputs, rawPixels ? RknnInputMode::RawUint8Nhwc
                                                      : RknnInputMode::ModelTensor);
        if (outputIndex >= outputs.size()) {
            throw std::runtime_error("RKNN embedding returned fewer outputs than expected");
        }
        const std::vector<float>& raw = outputs[outputIndex];
        if (raw.size() != static_cast<std::size_t>(config.embeddingDim)) {
            throw std::runtime_error(
                "RKNN embedding output has " + std::to_string(raw.size()) +
                " elements but embedding_dim is " + std::to_string(config.embeddingDim) +
                "; fix the configuration or re-convert the model");
        }
        cv::Mat feature(1, static_cast<int>(raw.size()), CV_32F,
                        const_cast<float*>(raw.data()));
        return normalizeEmbedding(feature);
    }

    std::string name() const override {
        return "rknn-embedding-" + std::to_string(layout.width);
    }

    std::string inputContract() const override {
        return rawPixels ? "rknn-u8-nhwc-" + config.rknn.inputColorOrder + "-v1" : "";
    }

    std::string describe() const override {
        std::ostringstream text;
        text << "RKNN embedding on NPU (" << config.rknn.modelPath.filename().string() << ")\n";
        text << "  rknn api      : "
             << (session.apiVersion().empty() ? "unknown" : session.apiVersion()) << '\n';
        text << "  rknn driver   : "
             << (session.driverVersion().empty() ? "unknown" : session.driverVersion()) << '\n';
        text << "  input tensor  : " << inputDesc.name << ' ' << inputDesc.formatName << ' '
             << layout.width << 'x' << layout.height << " type=" << inputDesc.dataTypeName
             << " color=" << config.rknn.inputColorOrder << " scale=" << config.rknn.inputScale
             << '\n';
        text << "  supplied input: " << (rawPixels ? "UINT8 NHWC; normalization in RKNN"
                                                   : "native tensor; application preprocessing") << '\n';
        text << "  output tensor : " << session.outputs()[outputIndex].name << " elements="
             << session.outputs()[outputIndex].elements << '\n';
        if (!note_.empty()) text << "  note          : " << note_ << '\n';
        if (!session.loadNotes().empty()) text << "  note          : " << session.loadNotes() << '\n';
        return text.str();
    }

private:
    SFaceConfig config;
    RknnSession session;
    RknnTensorDesc inputDesc;
    RknnInputLayout layout{};
    std::size_t outputIndex{0};
    bool rawPixels{false};
    std::string note_;
    mutable std::vector<std::uint8_t> inputBuffer;
};

#endif  // FACEAPP_HAVE_RKNN

}  // namespace

std::shared_ptr<const IEmbeddingExtractor> makeRknnSFaceExtractor(const SFaceConfig& config) {
#if FACEAPP_HAVE_RKNN
    return std::make_shared<RknnEmbeddingExtractor>(config);
#else
    (void)config;
    throw std::runtime_error(
        "The RKNN backend is not compiled into this build; rebuild with FACEAPP_ENABLE_RKNN=ON "
        "and point FACEAPP_RKNN_INCLUDE_DIR / FACEAPP_RKNN_LIBRARY at librknnrt");
#endif
}

}  // namespace facecore
