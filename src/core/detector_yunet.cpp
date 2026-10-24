#include "facecore/detector.hpp"

#include "facecore/yunet_decode.hpp"
#include "io_internal.hpp"

#if FACEAPP_HAVE_OPENCV_DNN
#include <opencv2/dnn.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace facecore {
namespace {

// CPU YuNet 后端的核心特点：只负责 ONNX forward，输出解释交给公共的
// yunet_decode.cpp。这样 CPU 和 RKNN 两条路径使用同一套候选框、关键点和 NMS
// 逻辑，出现差异时可以先排除“解码实现不同”这个变量。
void validateDetectorConfig(const DetectorConfig& config) {
    if (config.modelPath.empty()) {
        throw std::invalid_argument("YuNet model path is empty");
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

// CPU 检测后端：直接用 OpenCV 的通用 dnn 模块跑 YuNet 2023mar ONNX，
// 再用与 NPU 通路完全相同的 head 归类 + 解码 + NMS。
//
// 为什么不用 cv::FaceDetectorYN：OpenCV 4.5.4（Ubuntu 22.04 自带）的实现里
// 把输出层名写死成 { "loc", "conf", "iou" } 并配 anchor priors 解码，那是老版 YuNet；
// 对 2023mar 的 12 个 anchor-free 输出，forward 会报
//   "Layer with requested id=-1 not found"。
// 用通用 dnn + 自己的解码器后，4.5.4 与 4.7+ 行为一致，也让 CPU/NPU 两条路径的
// letterbox、解码、NMS 完全相同，可以直接逐项对比。
class YuNetFaceDetector::Impl {
public:
    explicit Impl(DetectorConfig value) : config(std::move(value)) {
        // 构造时完成模型加载和输出头映射，detect() 只做逐帧推理，不重复解析
        // 网络结构。启动阶段失败能比运行中突然失败更容易定位。
        validateDetectorConfig(config);
        if (!std::filesystem::is_regular_file(config.modelPath)) {
            throw std::runtime_error("YuNet model file does not exist: " +
                                     config.modelPath.string());
        }
#if !FACEAPP_HAVE_OPENCV_DNN
        throw std::runtime_error(
            "The OpenCV DNN backend is not available in this build; rebuild with "
            "FACEAPP_ENABLE_OPENCV_DNN=ON against an OpenCV that ships the dnn module");
#else
        networkSize = alignYunetNetworkInput(cv::Size(config.inputWidth, config.inputHeight));
        try {
            net = cv::dnn::readNetFromONNX(detail::pathToUtf8(config.modelPath));
        } catch (const cv::Exception& exception) {
            throw std::runtime_error(std::string("Failed to load the YuNet ONNX model: ") +
                                     exception.what());
        }
        if (net.empty()) throw std::runtime_error("OpenCV returned an empty YuNet network");
        net.setPreferableBackend(config.backendId);
        net.setPreferableTarget(config.targetId);
        outputNames = net.getUnconnectedOutLayersNames();
        if (outputNames.empty()) {
            throw std::runtime_error("The YuNet ONNX model exposes no output layers");
        }

        // 启动阶段先空跑一次：一方面让 dnn 完成形状推断（首帧更快），
        // 另一方面把模型与解码器不匹配的问题提前暴露出来，而不是等第一帧。
        std::vector<YunetRawHead> probeHeads = forwardHeads(cv::Mat::zeros(networkSize, CV_8UC3));
        const YunetHeadMapping probeMapping =
            classifyYunetHeads(probeHeads, networkSize.width, networkSize.height);
        for (const std::string& note : probeMapping.notes) notes += note + "; ";
        std::ostringstream summary;
        summary << "OpenCV DNN YuNet (" << outputNames.size() << " heads, input "
                << networkSize.width << 'x' << networkSize.height << ')';
        for (const YunetStrideHeads& stride : probeMapping.strides) {
            summary << " stride" << stride.stride << "(n=" << stride.count << ')';
        }
        description = summary.str();
#endif
    }

    std::vector<FaceDetection> detect(const cv::Mat& frame) {
        // makeYunetNetworkInput() 会按网络尺寸做 letterbox；decode 时再用
        // inverseScale 把网络坐标还原回原始摄像头帧坐标。
#if !FACEAPP_HAVE_OPENCV_DNN
        (void)frame;
        throw std::runtime_error("The OpenCV DNN backend is not available in this build");
#else
        if (frame.empty()) throw std::invalid_argument("YuNet input frame is empty");
        const YunetLetterbox letterbox = makeYunetNetworkInput(frame, networkSize);
        std::vector<YunetRawHead> heads = forwardHeads(letterbox.canvas);
        const YunetHeadMapping mapping =
            classifyYunetHeads(heads, networkSize.width, networkSize.height);

        YunetDecodeOptions options;
        options.scoreThreshold = config.scoreThreshold;
        options.nmsThreshold = config.nmsThreshold;
        options.topK = config.topK;
        options.inverseScale = static_cast<float>(1.0 / letterbox.scale);
        // 2023mar 每个 stride 都有 cls 与 obj 两个 1 通道 head，自动按 sqrt(cls*obj) 融合。
        options.objectnessMode = 0;

        std::vector<FaceDetection> detections = decodeYunetHeads(mapping, options);
        clampDetectionsToFrame(detections, frame.size());
        return detections;
#endif
    }

    std::string describe() const { return description; }

private:
#if FACEAPP_HAVE_OPENCV_DNN
    // 跑一次前向并把 12 个输出转成解码器认识的 head。
    // 注意：OpenCV 对 3 维以上 blob 返回的 Mat，rows/cols 是 -1，
    // 维度只能通过 size[d] 读（实测 dims=3 size=[1,1600,4] rows=-1 cols=-1）。
    std::vector<YunetRawHead> forwardHeads(const cv::Mat& canvas) {
        // scale=1、无 mean、不交换通道：YuNet 2023mar ONNX 期望 BGR、原始 0..255。
        cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0, cv::Size(), cv::Scalar(), false,
                                              false);
        net.setInput(blob);
        std::vector<cv::Mat> blobs;
        try {
            net.forward(blobs, outputNames);
        } catch (const cv::Exception& exception) {
            throw std::runtime_error(std::string("YuNet forward failed: ") + exception.what());
        }

        std::vector<YunetRawHead> heads;
        heads.reserve(blobs.size());
        for (std::size_t index = 0; index < blobs.size(); ++index) {
            const cv::Mat& out = blobs[index];
            if (out.empty()) {
                throw std::runtime_error("YuNet returned an empty output for layer '" +
                                         std::string(outputNames[index]) + "'");
            }
            cv::Mat floatBlob;
            if (out.type() == CV_32F) {
                floatBlob = out;
            } else {
                out.convertTo(floatBlob, CV_32F);
            }
            if (!floatBlob.isContinuous()) floatBlob = floatBlob.clone();

            YunetRawHead head;
            head.name = outputNames[index];
            if (floatBlob.dims >= 2) {
                head.dims = {floatBlob.size[floatBlob.dims - 2],
                             floatBlob.size[floatBlob.dims - 1]};
            } else {
                head.dims = {1, static_cast<int>(floatBlob.total())};
            }
            if (static_cast<std::size_t>(head.dims[0]) * head.dims[1] != floatBlob.total()) {
                throw std::runtime_error(
                    "YuNet output '" + head.name + "' has " +
                    std::to_string(floatBlob.total()) + " elements but shape " +
                    std::to_string(head.dims[0]) + "x" + std::to_string(head.dims[1]) +
                    "; the model is not a supported YuNet head layout");
            }
            const cv::Mat flat = floatBlob.reshape(1, 1);
            const float* data = flat.ptr<float>(0);
            head.data.assign(data, data + flat.total());
            heads.push_back(std::move(head));
        }
        return heads;
    }
#endif

    DetectorConfig config;
#if FACEAPP_HAVE_OPENCV_DNN
    cv::dnn::Net net;
    std::vector<cv::String> outputNames;
#endif
    cv::Size networkSize{0, 0};
    std::string description;
    std::string notes;
};

YuNetFaceDetector::YuNetFaceDetector(DetectorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

YuNetFaceDetector::~YuNetFaceDetector() = default;
YuNetFaceDetector::YuNetFaceDetector(YuNetFaceDetector&&) noexcept = default;
YuNetFaceDetector& YuNetFaceDetector::operator=(YuNetFaceDetector&&) noexcept = default;

std::vector<FaceDetection> YuNetFaceDetector::detect(const cv::Mat& bgrFrame) {
    return impl_->detect(bgrFrame);
}

std::string YuNetFaceDetector::name() const { return "opencv-dnn-yunet"; }

std::string YuNetFaceDetector::describe() const { return impl_->describe(); }

}  // namespace facecore
