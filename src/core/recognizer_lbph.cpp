#include "facecore/recognizer.hpp"

#include "io_internal.hpp"

#include <opencv2/face.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>

namespace facecore {

// LBPH 是传统 CPU 基线：它接收 128x128 均衡化灰度脸，OpenCV 返回的是距离，
// 距离越小越像。它和 embedding 的“相似度越大越像”不能共用同一个阈值。
class LbphFaceRecognizer::Impl {
public:
    static cv::Ptr<cv::face::LBPHFaceRecognizer> createModel(const LbphConfig& config) {
        return cv::face::LBPHFaceRecognizer::create(
            config.radius, config.neighbors, config.gridX, config.gridY,
            std::numeric_limits<double>::max());
    }

    explicit Impl(LbphConfig value)
        : config(std::move(value)), recognizer(createModel(config)) {}

    LbphConfig config;
    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer;
    std::map<int, std::string> labels;
    cv::Size inputSize;
    bool hasModel{false};
    mutable std::mutex mutex;
};

LbphFaceRecognizer::LbphFaceRecognizer(LbphConfig config)
    : impl_(std::make_unique<Impl>(config)) {
    if (config.radius <= 0 || config.neighbors <= 0 || config.gridX <= 0 ||
        config.gridY <= 0 || !std::isfinite(config.distanceThreshold)
        || config.distanceThreshold <= 0.0
        || !std::isfinite(config.minimumDistanceMargin)
        || config.minimumDistanceMargin < 0.0) {
        throw std::invalid_argument("Invalid LBPH configuration");
    }
}

LbphFaceRecognizer::~LbphFaceRecognizer() = default;
LbphFaceRecognizer::LbphFaceRecognizer(LbphFaceRecognizer&&) noexcept = default;
LbphFaceRecognizer& LbphFaceRecognizer::operator=(LbphFaceRecognizer&&) noexcept = default;

void LbphFaceRecognizer::train(const std::vector<LabeledFaceSample>& samples) {
    if (samples.empty()) throw std::invalid_argument("LBPH training set is empty");

    std::map<std::string, int> numericLabels;
    for (const LabeledFaceSample& sample : samples) {
        if (sample.personId.empty()) throw std::invalid_argument("Training personId is empty");
        numericLabels.emplace(sample.personId, 0);
    }
    int nextLabel = 0;
    for (auto& item : numericLabels) item.second = nextLabel++;

    std::vector<cv::Mat> images;
    std::vector<int> labels;
    cv::Size expectedSize;
    images.reserve(samples.size());
    labels.reserve(samples.size());
    for (const LabeledFaceSample& sample : samples) {
        const cv::Mat& image = sample.face.grayFace;
        if (image.empty() || image.type() != CV_8UC1) {
            throw std::invalid_argument("Every LBPH sample must be non-empty CV_8UC1");
        }
        if (expectedSize.empty()) expectedSize = image.size();
        if (image.size() != expectedSize) {
            throw std::invalid_argument("All LBPH samples must have exactly the same size");
        }
        images.push_back(image.clone());
        labels.push_back(numericLabels.at(sample.personId));
    }

    cv::Ptr<cv::face::LBPHFaceRecognizer> replacement = Impl::createModel(impl_->config);
    try {
        replacement->train(images, labels);
    } catch (const cv::Exception& exception) {
        throw std::runtime_error(std::string("LBPH training failed: ") + exception.what());
    }
    std::map<int, std::string> replacementLabels;
    for (const auto& item : numericLabels) replacementLabels.emplace(item.second, item.first);

    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->recognizer = std::move(replacement);
    impl_->labels = std::move(replacementLabels);
    impl_->inputSize = expectedSize;
    impl_->hasModel = true;
}

RecognitionResult LbphFaceRecognizer::recognize(const FaceSample& sample) const {
    RecognitionResult output;
    output.backend = name();
    output.threshold = impl_->config.distanceThreshold;
    if (sample.grayFace.empty() || sample.grayFace.type() != CV_8UC1) {
        output.state = RecognitionState::Error;
        output.message = "LBPH requires a non-empty CV_8UC1 face";
        return output;
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->hasModel) {
        output.state = RecognitionState::Error;
        output.message = "LBPH model is not trained or loaded";
        return output;
    }
    if (sample.grayFace.size() != impl_->inputSize) {
        output.state = RecognitionState::Error;
        output.message = "LBPH input dimensions do not match the trained model";
        return output;
    }
    try {
        const auto collector = cv::face::StandardCollector::create(
            std::numeric_limits<double>::max());
        impl_->recognizer->predict(sample.grayFace, collector);
        std::map<int, double> bestDistanceByLabel;
        for (const auto& result : collector->getResults(false)) {
            const auto found = bestDistanceByLabel.find(result.first);
            if (found == bestDistanceByLabel.end() || result.second < found->second) {
                bestDistanceByLabel[result.first] = result.second;
            }
        }
        std::vector<std::pair<int, double>> ranking(bestDistanceByLabel.begin(),
                                                    bestDistanceByLabel.end());
        std::sort(ranking.begin(), ranking.end(), [](const auto& first, const auto& second) {
            return first.second < second.second;
        });
        if (ranking.empty()) {
            output.state = RecognitionState::Error;
            output.message = "LBPH returned no candidates";
            return output;
        }
        output.bestScore = ranking[0].second;
        output.secondBestScore = ranking.size() > 1
                                     ? ranking[1].second
                                     : std::numeric_limits<double>::infinity();
        const auto label = impl_->labels.find(ranking[0].first);
        if (label == impl_->labels.end()) {
            output.state = RecognitionState::Error;
            output.message = "LBPH returned an unmapped numeric label";
            return output;
        }
        const bool thresholdAccepted = output.bestScore <= impl_->config.distanceThreshold;
        const bool marginAccepted =
            !std::isfinite(output.secondBestScore) ||
            output.secondBestScore - output.bestScore >= impl_->config.minimumDistanceMargin;
        if (thresholdAccepted && marginAccepted) {
            output.state = RecognitionState::Known;
            output.personId = label->second;
            output.message = "accepted";
        } else {
            output.state = RecognitionState::Unknown;
            output.message = thresholdAccepted ? "ambiguous_best_match" : "distance_above_threshold";
        }
        return output;
    } catch (const cv::Exception& exception) {
        output.state = RecognitionState::Error;
        output.message = std::string("LBPH inference failed: ") + exception.what();
        return output;
    }
}

void LbphFaceRecognizer::save(const std::filesystem::path& modelFile) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->hasModel) throw std::runtime_error("Cannot save an untrained LBPH model");
    cv::FileStorage storage("", cv::FileStorage::WRITE | cv::FileStorage::MEMORY |
                                    cv::FileStorage::FORMAT_YAML);
    storage << "facecore_format" << 2;
    storage << "distance_threshold" << impl_->config.distanceThreshold;
    storage << "minimum_distance_margin" << impl_->config.minimumDistanceMargin;
    storage << "input_width" << impl_->inputSize.width;
    storage << "input_height" << impl_->inputSize.height;
    storage << "preprocessing_version"
            << (impl_->config.preprocessingVersion.empty()
                    ? std::string("unspecified")
                    : impl_->config.preprocessingVersion);
    storage << "lbph_model" << "{";
    impl_->recognizer->write(storage);
    storage << "}";
    storage << "labels" << "[";
    for (const auto& label : impl_->labels) {
        storage << "{" << "numeric" << label.first << "person_id" << label.second << "}";
    }
    storage << "]";
    detail::writeTextFile(modelFile, storage.releaseAndGetString());
}

void LbphFaceRecognizer::load(const std::filesystem::path& modelFile) {
    const std::string content = detail::readTextFile(modelFile);
    cv::FileStorage storage(content, cv::FileStorage::READ | cv::FileStorage::MEMORY);
    if (!storage.isOpened() || storage["lbph_model"].empty() || storage["labels"].empty()) {
        throw std::runtime_error("Invalid facecore LBPH model file: " + modelFile.string());
    }
    int format = 0;
    int inputWidth = 0;
    int inputHeight = 0;
    std::string preprocessingVersion;
    storage["facecore_format"] >> format;
    storage["input_width"] >> inputWidth;
    storage["input_height"] >> inputHeight;
    storage["preprocessing_version"] >> preprocessingVersion;
    if (format != 2 || inputWidth <= 0 || inputHeight <= 0
        || preprocessingVersion.empty()) {
        throw std::runtime_error("Unsupported or incomplete LBPH model contract");
    }
    if (!impl_->config.preprocessingVersion.empty()
        && preprocessingVersion != impl_->config.preprocessingVersion) {
        throw std::runtime_error("LBPH preprocessing version does not match current configuration");
    }
    double persistedThreshold = impl_->config.distanceThreshold;
    double persistedMargin = impl_->config.minimumDistanceMargin;
    if (!storage["distance_threshold"].empty()) {
        storage["distance_threshold"] >> persistedThreshold;
    }
    if (!storage["minimum_distance_margin"].empty()) {
        storage["minimum_distance_margin"] >> persistedMargin;
    }
    if (!std::isfinite(persistedThreshold) || persistedThreshold <= 0.0
        || !std::isfinite(persistedMargin) || persistedMargin < 0.0) {
        throw std::runtime_error("LBPH model contains invalid thresholds");
    }

    cv::Ptr<cv::face::LBPHFaceRecognizer> replacement = Impl::createModel(impl_->config);
    try {
        replacement->read(storage["lbph_model"]);
    } catch (const cv::Exception& exception) {
        throw std::runtime_error(std::string("Cannot read LBPH model: ") + exception.what());
    }
    std::map<int, std::string> replacementLabels;
    std::set<std::string> personIds;
    for (const cv::FileNode& node : storage["labels"]) {
        int numeric = -1;
        std::string personId;
        node["numeric"] >> numeric;
        node["person_id"] >> personId;
        if (numeric < 0 || personId.empty()) {
            throw std::runtime_error("Invalid LBPH label entry");
        }
        if (!replacementLabels.emplace(numeric, personId).second
            || !personIds.insert(personId).second) {
            throw std::runtime_error("Duplicate LBPH label entry");
        }
    }
    if (replacementLabels.empty()) throw std::runtime_error("LBPH model has no label mappings");

    const cv::Mat storedLabels = replacement->getLabels();
    std::set<int> modelLabels;
    for (std::size_t index = 0; index < storedLabels.total(); ++index) {
        modelLabels.insert(storedLabels.at<int>(static_cast<int>(index)));
    }
    std::set<int> mappedLabels;
    for (const auto& item : replacementLabels) mappedLabels.insert(item.first);
    if (modelLabels != mappedLabels) {
        throw std::runtime_error("LBPH model labels do not match the label mapping");
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->recognizer = std::move(replacement);
    impl_->labels = std::move(replacementLabels);
    impl_->inputSize = cv::Size(inputWidth, inputHeight);
    impl_->hasModel = true;
}

std::string LbphFaceRecognizer::name() const { return "opencv-lbph"; }
bool LbphFaceRecognizer::trained() const {
    if (!impl_) return false;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->hasModel;
}

}  // namespace facecore
