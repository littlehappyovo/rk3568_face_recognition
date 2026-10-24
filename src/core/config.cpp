#include "facecore/config.hpp"

#include "io_internal.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace facecore {
namespace {

// 配置读取采用“默认值 + 有则覆盖”。loadConfig() 负责解析文件，最后由
// validateConfig() 统一检查尺寸、阈值、路径、后端和模型输入契约。
template <typename T>
void readIfPresent(const cv::FileNode& node, const char* key, T& value) {
    const cv::FileNode item = node[key];
    if (!item.empty()) {
        item >> value;
    }
}

void readBoolIfPresent(const cv::FileNode& node, const char* key, bool& value) {
    const cv::FileNode item = node[key];
    if (!item.empty()) {
        int numeric = value ? 1 : 0;
        item >> numeric;
        value = numeric != 0;
    }
}

void readDoubleListIfPresent(const cv::FileNode& node, const char* key,
                             std::vector<double>& values) {
    const cv::FileNode item = node[key];
    if (item.empty() || !item.isSeq()) return;
    std::vector<double> parsed;
    for (const cv::FileNode& entry : item) {
        double value = 0.0;
        entry >> value;
        parsed.push_back(value);
    }
    if (!parsed.empty()) values = parsed;
}

void readStringListIfPresent(const cv::FileNode& node, const char* key,
                             std::vector<std::string>& values) {
    const cv::FileNode item = node[key];
    if (item.empty() || !item.isSeq()) return;
    std::vector<std::string> parsed;
    for (const cv::FileNode& entry : item) {
        std::string value;
        entry >> value;
        if (!value.empty()) parsed.push_back(value);
    }
    if (!parsed.empty()) values = parsed;
}

void readBackendIfPresent(const cv::FileNode& node, const char* key, ComputeBackend& value) {
    const cv::FileNode item = node[key];
    if (item.empty()) return;
    std::string text;
    item >> text;
    if (text.empty()) return;
    value = parseComputeBackend(text);
}

void readRknnIfPresent(const cv::FileNode& node, RknnModelConfig& config) {
    std::string path;
    readIfPresent(node, "model", path);
    if (!path.empty()) config.modelPath = std::filesystem::u8path(path);
    readIfPresent(node, "input_color", config.inputColorOrder);
    readIfPresent(node, "input_layout", config.inputLayout);
    readIfPresent(node, "input_mode", config.inputMode);
    readIfPresent(node, "input_scale", config.inputScale);
    readDoubleListIfPresent(node, "input_mean", config.inputMean);
    readDoubleListIfPresent(node, "input_std", config.inputStd);
    readIfPresent(node, "input_quantized", config.inputQuantized);
    readIfPresent(node, "core_mask", config.coreMask);
    readStringListIfPresent(node, "output_names", config.outputNames);
    readIfPresent(node, "objectness_mode", config.objectnessMode);
}

void readEmbeddingIfPresent(const cv::FileNode& node, SFaceConfig& config) {
    std::string path;
    readIfPresent(node, "model", path);
    if (!path.empty()) config.modelPath = std::filesystem::u8path(path);
    readIfPresent(node, "cosine_threshold", config.cosineThreshold);
    readIfPresent(node, "minimum_cosine_margin", config.minimumCosineMargin);
    readIfPresent(node, "model_identity", config.modelIdentity);
    readIfPresent(node, "backend_id", config.backendId);
    readIfPresent(node, "target_id", config.targetId);
    readIfPresent(node, "input_width", config.inputWidth);
    readIfPresent(node, "input_height", config.inputHeight);
    readIfPresent(node, "embedding_dim", config.embeddingDim);
    readBackendIfPresent(node, "backend", config.backend);
    readRknnIfPresent(node["rknn"], config.rknn);
}

void writeDoubleList(cv::FileStorage& storage, const char* key, const std::vector<double>& values) {
    storage << key << "[";
    for (double value : values) storage << value;
    storage << "]";
}

void writeStringList(cv::FileStorage& storage, const char* key,
                     const std::vector<std::string>& values) {
    storage << key << "[";
    for (const std::string& value : values) storage << value;
    storage << "]";
}

void writeRknn(cv::FileStorage& storage, const RknnModelConfig& config) {
    storage << "rknn" << "{"
            << "model" << detail::pathToUtf8(config.modelPath)
            << "input_color" << config.inputColorOrder
            << "input_layout" << config.inputLayout
            << "input_mode" << config.inputMode
            << "input_scale" << config.inputScale;
    writeDoubleList(storage, "input_mean", config.inputMean);
    writeDoubleList(storage, "input_std", config.inputStd);
    storage << "input_quantized" << config.inputQuantized
            << "core_mask" << config.coreMask;
    writeStringList(storage, "output_names", config.outputNames);
    storage << "objectness_mode" << config.objectnessMode << "}";
}

void writeEmbedding(cv::FileStorage& storage, const char* section,
                    const SFaceConfig& config) {
    storage << section << "{"
            << "model" << detail::pathToUtf8(config.modelPath)
            << "cosine_threshold" << config.cosineThreshold
            << "minimum_cosine_margin" << config.minimumCosineMargin
            << "model_identity" << config.modelIdentity
            << "backend_id" << config.backendId
            << "target_id" << config.targetId
            << "input_width" << config.inputWidth
            << "input_height" << config.inputHeight
            << "embedding_dim" << config.embeddingDim
            << "backend" << toString(config.backend);
    writeRknn(storage, config.rknn);
    storage << "}";
}

void requirePositive(int value, const char* field) {
    if (value <= 0) {
        throw std::invalid_argument(std::string(field) + " must be positive");
    }
}

void requireRange(double value, double minimum, double maximum, const char* field) {
    if (!std::isfinite(value) || value < minimum || value > maximum) {
        throw std::invalid_argument(std::string(field) + " is outside the allowed range");
    }
}

void requireNonNegative(double value, const char* field) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(std::string(field) + " must be finite and non-negative");
    }
}

void validateRknn(const RknnModelConfig& config, const char* field) {
    if (config.inputMode != "model_tensor" && config.inputMode != "raw_uint8") {
        throw std::invalid_argument(std::string(field) + ".input_mode must be model_tensor or raw_uint8");
    }
    if (config.inputMode == "raw_uint8"
        && (config.inputScale != 1.0 || config.inputQuantized != -1)) {
        throw std::invalid_argument(std::string(field) +
            ": raw_uint8 requires input_scale=1 and input_quantized=-1; mean/std belong to the compiled model");
    }
    if (config.inputColorOrder != "bgr" && config.inputColorOrder != "rgb"
        && config.inputColorOrder != "gray") {
        throw std::invalid_argument(std::string(field) +
                                    ".input_color must be bgr, rgb or gray");
    }
    if (config.inputLayout != "nchw" && config.inputLayout != "nhwc"
        && config.inputLayout != "auto") {
        throw std::invalid_argument(std::string(field) +
                                    ".input_layout must be nchw, nhwc or auto");
    }
    if (!std::isfinite(config.inputScale) || config.inputScale <= 0.0) {
        throw std::invalid_argument(std::string(field) +
                                    ".input_scale must be finite and positive");
    }
    for (double value : config.inputMean) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(field) + ".input_mean must be finite");
        }
    }
    for (double value : config.inputStd) {
        if (!std::isfinite(value) || value == 0.0) {
            throw std::invalid_argument(std::string(field) +
                                        ".input_std must be finite and non-zero");
        }
    }
    if (config.inputQuantized < -1 || config.inputQuantized > 1) {
        throw std::invalid_argument(std::string(field) +
                                    ".input_quantized must be -1, 0 or 1");
    }
    if (config.coreMask < 0) {
        throw std::invalid_argument(std::string(field) + ".core_mask cannot be negative");
    }
    if (config.objectnessMode < 0 || config.objectnessMode > 2) {
        throw std::invalid_argument(std::string(field) +
                                    ".objectness_mode must be 0, 1 or 2");
    }
}

void validateEmbedding(const SFaceConfig& config, const char* field) {
    const std::string prefix = std::string(field) + ".";
    requireRange(config.cosineThreshold, -1.0, 1.0,
                 (prefix + "cosine_threshold").c_str());
    requireRange(config.minimumCosineMargin, 0.0, 2.0,
                 (prefix + "minimum_cosine_margin").c_str());
    requirePositive(config.inputWidth, (prefix + "input_width").c_str());
    requirePositive(config.inputHeight, (prefix + "input_height").c_str());
    requirePositive(config.embeddingDim, (prefix + "embedding_dim").c_str());
    validateRknn(config.rknn, (prefix + "rknn").c_str());
    if (config.backend == ComputeBackend::Rknn && config.rknn.modelPath.empty()) {
        throw std::invalid_argument(std::string(field) +
                                    ".backend is \"rknn\" but " + field +
                                    ".rknn.model is empty");
    }
}

}  // namespace

const char* toString(ComputeBackend value) noexcept {
    switch (value) {
        case ComputeBackend::OpenCv: return "opencv";
        case ComputeBackend::Rknn: return "rknn";
        default: return "auto";
    }
}

ComputeBackend parseComputeBackend(const std::string& text) {
    std::string normalized = text;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                   });
    if (normalized == "auto") return ComputeBackend::Auto;
    if (normalized == "opencv" || normalized == "cpu" || normalized == "dnn") {
        return ComputeBackend::OpenCv;
    }
    if (normalized == "rknn" || normalized == "npu") return ComputeBackend::Rknn;
    throw std::invalid_argument("Unknown compute backend: '" + text +
                                "' (expected auto, opencv or rknn)");
}

void validateConfig(const SystemConfig& config) {
    // 这里集中验证所有跨模块前置条件。尤其是 detector.input_width/height、
    // embeddingDim 和 RKNN input layout；这些错误如果不在启动时发现，往往会
    // 表现为“能推理但框/身份完全不对”，排查成本更高。
    requirePositive(config.detector.inputWidth, "detector.input_width");
    requirePositive(config.detector.inputHeight, "detector.input_height");
    requireRange(config.detector.scoreThreshold, 0.0, 1.0, "detector.score_threshold");
    requireRange(config.detector.nmsThreshold, 0.0, 1.0, "detector.nms_threshold");
    requirePositive(config.detector.topK, "detector.top_k");
    validateRknn(config.detector.rknn, "detector.rknn");
    if (config.detector.backend == ComputeBackend::Rknn
        && config.detector.rknn.modelPath.empty()) {
        throw std::invalid_argument(
            "detector.backend is \"rknn\" but detector.rknn.model is empty");
    }
    requireRange(config.tracker.minimumIou, 0.0, 1.0, "tracker.minimum_iou");
    requireRange(config.tracker.smoothingAlpha, 0.0, 1.0, "tracker.smoothing_alpha");
    requireNonNegative(config.tracker.maximumNormalizedCenterDistance,
                       "tracker.maximum_normalized_center_distance");
    if (config.tracker.maximumMissedFrames < 0)
        throw std::invalid_argument("tracker.maximum_missed_frames cannot be negative");
    requirePositive(config.tracker.detectionIntervalFrames,
                    "tracker.detection_interval_frames");
    requirePositive(config.tracker.recognitionIntervalMs,
                    "tracker.recognition_interval_ms");
    requirePositive(config.preprocess.faceWidth, "preprocess.face_width");
    requirePositive(config.preprocess.faceHeight, "preprocess.face_height");
    requireRange(config.preprocess.boxExpansion, 0.0, 1.0, "preprocess.box_expansion");
    requirePositive(config.quality.minimumFaceWidth, "quality.minimum_face_width");
    requireRange(config.quality.minimumBrightness, 0.0, 255.0, "quality.minimum_brightness");
    requireRange(config.quality.maximumBrightness, 0.0, 255.0, "quality.maximum_brightness");
    if (config.quality.minimumBrightness >= config.quality.maximumBrightness) {
        throw std::invalid_argument("quality brightness minimum must be below maximum");
    }
    requireNonNegative(config.quality.minimumLaplacianVariance,
                       "quality.minimum_laplacian_variance");
    requireNonNegative(config.quality.duplicateMeanAbsoluteDifference,
                       "quality.duplicate_mean_absolute_difference");
    requireRange(config.quality.maximumClippedFraction, 0.0, 1.0,
                 "quality.maximum_clipped_fraction");
    requirePositive(config.lbph.radius, "lbph.radius");
    requirePositive(config.lbph.neighbors, "lbph.neighbors");
    requirePositive(config.lbph.gridX, "lbph.grid_x");
    requirePositive(config.lbph.gridY, "lbph.grid_y");
    if (!std::isfinite(config.lbph.distanceThreshold)
        || config.lbph.distanceThreshold <= 0.0) {
        throw std::invalid_argument("LBPH thresholds are invalid");
    }
    requireNonNegative(config.lbph.minimumDistanceMargin, "lbph.minimum_distance_margin");
    validateEmbedding(config.sface, "sface");
    validateEmbedding(config.mobilefacenet, "mobilefacenet");
    if (config.preprocess.version.empty()) {
        throw std::invalid_argument("preprocess.version cannot be empty");
    }
    if (config.camera.index < 0) {
        throw std::invalid_argument("camera.index cannot be negative");
    }
    if (config.camera.width < 0 || config.camera.height < 0) {
        throw std::invalid_argument("camera.width/height cannot be negative");
    }
    requireNonNegative(config.camera.fps, "camera.fps");
    if (config.datasetRoot.empty()) {
        throw std::invalid_argument("dataset_root cannot be empty");
    }
}

SystemConfig loadConfig(const std::filesystem::path& file) {
    // OpenCV FileStorage 支持 JSON/YAML。先在内存中读取，避免中文路径在某些
    // OpenCV 版本的文件接口上出现编码问题；解析完成后再调用 validateConfig。
    const std::string content = detail::readTextFile(file);
    cv::FileStorage storage(content, cv::FileStorage::READ | cv::FileStorage::MEMORY);
    if (!storage.isOpened()) {
        throw std::runtime_error("Invalid OpenCV JSON/YAML configuration: " + file.string());
    }

    SystemConfig config;
    std::string path;
    const cv::FileNode detector = storage["detector"];
    readIfPresent(detector, "model", path);
    if (!path.empty()) config.detector.modelPath = std::filesystem::u8path(path);
    readIfPresent(detector, "input_width", config.detector.inputWidth);
    readIfPresent(detector, "input_height", config.detector.inputHeight);
    readIfPresent(detector, "score_threshold", config.detector.scoreThreshold);
    readIfPresent(detector, "nms_threshold", config.detector.nmsThreshold);
    readIfPresent(detector, "top_k", config.detector.topK);
    readIfPresent(detector, "backend_id", config.detector.backendId);
    readIfPresent(detector, "target_id", config.detector.targetId);
    readBackendIfPresent(detector, "backend", config.detector.backend);
    readRknnIfPresent(detector["rknn"], config.detector.rknn);

    const cv::FileNode tracker = storage["tracker"];
    readIfPresent(tracker, "minimum_iou", config.tracker.minimumIou);
    readIfPresent(tracker, "maximum_normalized_center_distance",
                  config.tracker.maximumNormalizedCenterDistance);
    readIfPresent(tracker, "maximum_missed_frames", config.tracker.maximumMissedFrames);
    readIfPresent(tracker, "smoothing_alpha", config.tracker.smoothingAlpha);
    readIfPresent(tracker, "detection_interval_frames",
                  config.tracker.detectionIntervalFrames);
    readIfPresent(tracker, "recognition_interval_ms",
                  config.tracker.recognitionIntervalMs);

    const cv::FileNode preprocess = storage["preprocess"];
    readIfPresent(preprocess, "face_width", config.preprocess.faceWidth);
    readIfPresent(preprocess, "face_height", config.preprocess.faceHeight);
    readIfPresent(preprocess, "box_expansion", config.preprocess.boxExpansion);
    readBoolIfPresent(preprocess, "align_by_eyes", config.preprocess.alignByEyes);
    readBoolIfPresent(preprocess, "equalize_histogram", config.preprocess.equalizeHistogram);
    readIfPresent(preprocess, "version", config.preprocess.version);

    const cv::FileNode quality = storage["quality"];
    readIfPresent(quality, "minimum_face_width", config.quality.minimumFaceWidth);
    readIfPresent(quality, "minimum_brightness", config.quality.minimumBrightness);
    readIfPresent(quality, "maximum_brightness", config.quality.maximumBrightness);
    readIfPresent(quality, "minimum_laplacian_variance",
                  config.quality.minimumLaplacianVariance);
    readIfPresent(quality, "maximum_clipped_fraction", config.quality.maximumClippedFraction);
    readIfPresent(quality, "duplicate_mean_absolute_difference",
                  config.quality.duplicateMeanAbsoluteDifference);

    const cv::FileNode lbph = storage["lbph"];
    readIfPresent(lbph, "radius", config.lbph.radius);
    readIfPresent(lbph, "neighbors", config.lbph.neighbors);
    readIfPresent(lbph, "grid_x", config.lbph.gridX);
    readIfPresent(lbph, "grid_y", config.lbph.gridY);
    readIfPresent(lbph, "distance_threshold", config.lbph.distanceThreshold);
    readIfPresent(lbph, "minimum_distance_margin", config.lbph.minimumDistanceMargin);

    readEmbeddingIfPresent(storage["sface"], config.sface);
    readEmbeddingIfPresent(storage["mobilefacenet"], config.mobilefacenet);

    const cv::FileNode camera = storage["camera"];
    readIfPresent(camera, "device", config.camera.device);
    readIfPresent(camera, "gstreamer_pipeline", config.camera.gstreamerPipeline);
    readIfPresent(camera, "index", config.camera.index);
    readIfPresent(camera, "width", config.camera.width);
    readIfPresent(camera, "height", config.camera.height);
    readIfPresent(camera, "fps", config.camera.fps);

    path.clear();
    readIfPresent(storage.root(), "dataset_root", path);
    if (!path.empty()) config.datasetRoot = std::filesystem::u8path(path);
    config.lbph.preprocessingVersion = config.preprocess.version;
    config.sface.preprocessingVersion = config.preprocess.version;
    config.mobilefacenet.preprocessingVersion = config.preprocess.version;
    validateConfig(config);
    return config;
}

void saveConfig(const SystemConfig& config, const std::filesystem::path& file) {
    validateConfig(config);
    cv::FileStorage storage("", cv::FileStorage::WRITE | cv::FileStorage::MEMORY |
                                    cv::FileStorage::FORMAT_JSON);
    storage << "detector" << "{"
            << "model" << detail::pathToUtf8(config.detector.modelPath)
            << "input_width" << config.detector.inputWidth
            << "input_height" << config.detector.inputHeight
            << "score_threshold" << config.detector.scoreThreshold
            << "nms_threshold" << config.detector.nmsThreshold
            << "top_k" << config.detector.topK
            << "backend_id" << config.detector.backendId
            << "target_id" << config.detector.targetId
            << "backend" << toString(config.detector.backend);
    writeRknn(storage, config.detector.rknn);
    storage << "}";
    storage << "tracker" << "{"
            << "minimum_iou" << config.tracker.minimumIou
            << "maximum_normalized_center_distance" << config.tracker.maximumNormalizedCenterDistance
            << "maximum_missed_frames" << config.tracker.maximumMissedFrames
            << "smoothing_alpha" << config.tracker.smoothingAlpha
            << "detection_interval_frames" << config.tracker.detectionIntervalFrames
            << "recognition_interval_ms" << config.tracker.recognitionIntervalMs << "}";
    storage << "preprocess" << "{"
            << "face_width" << config.preprocess.faceWidth
            << "face_height" << config.preprocess.faceHeight
            << "box_expansion" << config.preprocess.boxExpansion
            << "align_by_eyes" << config.preprocess.alignByEyes
            << "equalize_histogram" << config.preprocess.equalizeHistogram
            << "version" << config.preprocess.version << "}";
    storage << "quality" << "{"
            << "minimum_face_width" << config.quality.minimumFaceWidth
            << "minimum_brightness" << config.quality.minimumBrightness
            << "maximum_brightness" << config.quality.maximumBrightness
            << "minimum_laplacian_variance" << config.quality.minimumLaplacianVariance
            << "maximum_clipped_fraction" << config.quality.maximumClippedFraction
            << "duplicate_mean_absolute_difference" << config.quality.duplicateMeanAbsoluteDifference
            << "}";
    storage << "lbph" << "{"
            << "radius" << config.lbph.radius
            << "neighbors" << config.lbph.neighbors
            << "grid_x" << config.lbph.gridX
            << "grid_y" << config.lbph.gridY
            << "distance_threshold" << config.lbph.distanceThreshold
            << "minimum_distance_margin" << config.lbph.minimumDistanceMargin << "}";
    writeEmbedding(storage, "sface", config.sface);
    writeEmbedding(storage, "mobilefacenet", config.mobilefacenet);
    storage << "camera" << "{"
            << "device" << config.camera.device
            << "gstreamer_pipeline" << config.camera.gstreamerPipeline
            << "index" << config.camera.index
            << "width" << config.camera.width
            << "height" << config.camera.height
            << "fps" << config.camera.fps << "}";
    storage << "dataset_root" << detail::pathToUtf8(config.datasetRoot);
    detail::writeTextFile(file, storage.releaseAndGetString());
}

}  // namespace facecore
