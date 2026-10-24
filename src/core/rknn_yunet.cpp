#include "facecore/rknn.hpp"

#include "facecore/yunet_decode.hpp"
#include "io_internal.hpp"
#include "rknn_session.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
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

// RKNN YuNet 与 CPU YuNet 的外层流程相同，差别只在“模型输入/输出如何交给
// RKNN Runtime”。构造函数先读取真实张量契约，再建立输出 head 映射；如果
// 配置写成 320 而模型实际是 640，这里会在启动时明确报错。
void validateDetectorConfig(const DetectorConfig& config) {
    if (config.rknn.modelPath.empty()) {
        throw std::invalid_argument("detector.rknn.model is empty");
    }
    if (config.inputWidth <= 0 || config.inputHeight <= 0) {
        throw std::invalid_argument("detector.input_width/height must be positive");
    }
    if (config.topK <= 0) throw std::invalid_argument("detector.top_k must be positive");
    if (!std::isfinite(config.scoreThreshold) || config.scoreThreshold < 0.0F
        || config.scoreThreshold > 1.0F) {
        throw std::invalid_argument("detector.score_threshold must be within [0, 1]");
    }
    if (!std::isfinite(config.nmsThreshold) || config.nmsThreshold < 0.0F
        || config.nmsThreshold > 1.0F) {
        throw std::invalid_argument("detector.nms_threshold must be within [0, 1]");
    }
}

}  // namespace

class RknnYuNetFaceDetector::Impl {
public:
    explicit Impl(DetectorConfig value) : config(std::move(value)) {
        // session.load() 会加载 .rknn；随后必须用 Runtime 返回的输入尺寸校验
        // 配置，不能根据文件名猜测模型尺寸。
        validateDetectorConfig(config);
        session.load(config.rknn.modelPath, config.rknn.coreMask);
        if (session.inputs().size() != 1) {
            throw std::runtime_error("RKNN YuNet expects exactly one input tensor but the model "
                                     "reports " + std::to_string(session.inputs().size()));
        }
        inputDesc = session.inputs().front();
        inputLayout = resolveRknnInputLayout(inputDesc, config.rknn);
        networkSize = cv::Size(inputLayout.width, inputLayout.height);
        if (networkSize.width != config.inputWidth || networkSize.height != config.inputHeight) {
            throw std::runtime_error(
                "detector.input_width/height is " + std::to_string(config.inputWidth) + "x" +
                std::to_string(config.inputHeight) + " but the .rknn input tensor is " +
                std::to_string(networkSize.width) + "x" + std::to_string(networkSize.height) +
                "; update config/default.json to match the converted model");
        }
        if (inputLayout.channels != 3) {
            throw std::runtime_error("RKNN YuNet expects a 3-channel input tensor but the model "
                                     "reports " + std::to_string(inputLayout.channels) +
                                     " channel(s)");
        }

        heads.reserve(session.outputs().size());
        for (std::size_t index = 0; index < session.outputs().size(); ++index) {
            const RknnTensorDesc& desc = session.outputs()[index];
            YunetRawHead head;
            head.dims.assign(desc.dims.begin(), desc.dims.end());
            head.name = desc.name.empty() ? ("output" + std::to_string(index)) : desc.name;
            heads.push_back(std::move(head));
        }
        if (!config.rknn.outputNames.empty()) {
            std::vector<YunetRawHead> filtered;
            for (const YunetRawHead& head : heads) {
                if (std::find(config.rknn.outputNames.begin(), config.rknn.outputNames.end(),
                              head.name) != config.rknn.outputNames.end()) {
                    filtered.push_back(head);
                }
            }
            if (filtered.empty()) {
                std::string available;
                for (const YunetRawHead& head : heads) {
                    if (!available.empty()) available += ", ";
                    available += head.name;
                }
                throw std::runtime_error("detector.rknn.output_names matches no output tensor; "
                                         "available outputs: " + available);
            }
            heads = std::move(filtered);
        }
        mapping = classifyYunetHeads(heads, networkSize.width, networkSize.height);
    }

    std::vector<FaceDetection> detect(const cv::Mat& frame) {
        // CPU 与 NPU 共用 letterbox 和 decode。这里唯一新增的是把图像按 RKNN
        // 张量类型填入 inputBuffer，以及把输出张量转成 float 供解码器使用。
        // 与 CPU 通路共用同一套 letterbox，保证两条路径的预处理完全一致。
        const YunetLetterbox letterbox = makeYunetNetworkInput(frame, networkSize);
        const float inverseScale = static_cast<float>(1.0 / letterbox.scale);
        const bool quantized = config.rknn.inputQuantized == 1
                               || (config.rknn.inputQuantized < 0 && !inputDesc.float32);
        if (quantized) {
            fillQuantizedInput(letterbox.canvas, config.rknn, inputDesc, inputBuffer);
        } else {
            fillFloatInput(letterbox.canvas, config.rknn, inputDesc, inputBuffer);
        }

        std::vector<std::vector<float>> outputs;
        session.infer({inputBuffer}, outputs);
        if (outputs.size() != heads.size()) {
            throw std::runtime_error("RKNN returned " + std::to_string(outputs.size()) +
                                     " output(s) but " + std::to_string(heads.size()) +
                                     " were declared");
        }
        for (std::size_t index = 0; index < heads.size(); ++index) {
            const std::size_t expected = static_cast<std::size_t>(heads[index].count)
                                         * (heads[index].channels > 0 ? heads[index].channels : 1);
            if (outputs[index].size() != expected) {
                throw std::runtime_error(
                    "RKNN output " + heads[index].name + " has " +
                    std::to_string(outputs[index].size()) + " elements but its shape implies " +
                    std::to_string(expected) + "; disable want_float or check the model outputs");
            }
            heads[index].data = std::move(outputs[index]);
        }

        YunetDecodeOptions options;
        options.scoreThreshold = config.scoreThreshold;
        options.nmsThreshold = config.nmsThreshold;
        options.topK = config.topK;
        options.inverseScale = inverseScale;
        options.objectnessMode = config.rknn.objectnessMode;
        std::vector<FaceDetection> detections = decodeYunetHeads(mapping, options);
        clampDetectionsToFrame(detections, frame.size());
        return detections;
    }

    std::string describe() const {
        std::ostringstream text;
        text << "RKNN YuNet on NPU (" << config.rknn.modelPath.filename().string() << ")\n";
        text << "  rknn api      : " << (session.apiVersion().empty() ? "unknown" : session.apiVersion()) << '\n';
        text << "  rknn driver   : " << (session.driverVersion().empty() ? "unknown" : session.driverVersion()) << '\n';
        text << "  input tensor  : " << inputDesc.name << ' ' << inputDesc.formatName << ' '
             << std::to_string(inputLayout.width) << 'x' << std::to_string(inputLayout.height)
             << " type=" << inputDesc.dataTypeName
             << " color=" << config.rknn.inputColorOrder
             << " scale=" << config.rknn.inputScale << '\n';
        text << "  head mapping  :";
        for (const YunetStrideHeads& stride : mapping.strides) {
            text << " stride" << stride.stride << "(n=" << stride.count << ",scores="
                 << stride.scores.size() << ")";
        }
        text << '\n';
        for (const std::string& note : mapping.notes) text << "  note          : " << note << '\n';
        if (!session.loadNotes().empty()) text << "  note          : " << session.loadNotes() << '\n';
        return text.str();
    }

    DetectorConfig config;
    RknnSession session;
    RknnTensorDesc inputDesc;
    RknnInputLayout inputLayout;
    cv::Size networkSize{0, 0};
    std::vector<YunetRawHead> heads;
    YunetHeadMapping mapping;
    std::vector<std::uint8_t> inputBuffer;
};

RknnYuNetFaceDetector::RknnYuNetFaceDetector(DetectorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

RknnYuNetFaceDetector::~RknnYuNetFaceDetector() = default;
RknnYuNetFaceDetector::RknnYuNetFaceDetector(RknnYuNetFaceDetector&&) noexcept = default;
RknnYuNetFaceDetector& RknnYuNetFaceDetector::operator=(RknnYuNetFaceDetector&&) noexcept = default;

std::vector<FaceDetection> RknnYuNetFaceDetector::detect(const cv::Mat& bgrFrame) {
    return impl_->detect(bgrFrame);
}

std::string RknnYuNetFaceDetector::name() const { return "rknn-yunet"; }

std::string RknnYuNetFaceDetector::describe() const { return impl_->describe(); }

}  // namespace facecore
