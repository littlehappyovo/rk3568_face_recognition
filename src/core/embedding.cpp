#include "facecore/embedding.hpp"

#include "feature_cache.hpp"
#include "io_internal.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace facecore {
namespace {

// embedding 的公共层不关心特征是由 OpenCV DNN 还是 RKNN 得到的，只要求抽取器
// 返回一个向量。这里负责 L2 归一化、人员原型图库、余弦相似度、阈值和 margin。
const char* kFormatKey = "facecore_format";
constexpr int kFormatVersion = 2;

void validateThresholds(double cosineThreshold, double minimumCosineMargin) {
    if (!std::isfinite(cosineThreshold) || cosineThreshold < -1.0 || cosineThreshold > 1.0
        || !std::isfinite(minimumCosineMargin) || minimumCosineMargin < 0.0
        || minimumCosineMargin > 2.0) {
        throw std::invalid_argument("Invalid embedding cosine thresholds");
    }
}

}  // namespace

cv::Mat normalizeEmbedding(const cv::Mat& input) {
    if (input.empty()) throw std::runtime_error("Embedding backend produced an empty feature");
    cv::Mat feature;
    input.reshape(1, 1).convertTo(feature, CV_32F);
    const double norm = cv::norm(feature, cv::NORM_L2);
    if (!std::isfinite(norm) || norm <= 1e-12) {
        throw std::runtime_error("Embedding backend produced a zero or invalid feature");
    }
    return feature / norm;
}

double embeddingCosineSimilarity(const cv::Mat& first, const cv::Mat& second) {
    if (first.size() != second.size() || first.type() != CV_32F || second.type() != CV_32F) {
        throw std::runtime_error("Embedding gallery feature shape mismatch");
    }
    return first.dot(second);
}

void EmbeddingGallery::clear() { identities_.clear(); }

void EmbeddingGallery::build(const std::vector<LabeledFaceSample>& samples,
                             const IEmbeddingExtractor& extractor) {
    // 同一人员可能有多张照片。先分别提取特征，再求平均并重新归一化，得到一个
    // 人员原型；识别时比较查询特征与每个人员原型，而不是保存每张照片的结果。
    if (samples.empty()) throw std::invalid_argument("Embedding gallery input is empty");
    std::vector<std::pair<std::string, cv::Mat>> features;
    for (const LabeledFaceSample& sample : samples) {
        if (sample.personId.empty()) throw std::invalid_argument("Gallery personId is empty");
        features.emplace_back(sample.personId, extractor.extract(sample.face.alignedBgrFace));
    }
    buildFromFeatures(features);
}

void EmbeddingGallery::buildFromFeatures(const std::vector<std::pair<std::string, cv::Mat>>& entries) {
    if (entries.empty()) throw std::invalid_argument("Embedding gallery input is empty");
    std::map<std::string, std::vector<cv::Mat>> features;
    for (const auto& entry : entries) {
        if (entry.first.empty() || entry.second.empty() || entry.second.type() != CV_32F
            || entry.second.rows != 1 || !cv::checkRange(entry.second)) {
            throw std::invalid_argument("Invalid embedding gallery feature");
        }
        features[entry.first].push_back(entry.second);
    }

    std::vector<Identity> replacement;
    replacement.reserve(features.size());
    for (const auto& item : features) {
        cv::Mat mean = cv::Mat::zeros(item.second.front().size(), CV_32F);
        for (const cv::Mat& feature : item.second) {
            if (feature.size() != mean.size()) {
                throw std::runtime_error("Embedding backend emitted inconsistent feature dimensions");
            }
            mean += feature;
        }
        replacement.push_back({item.first, normalizeEmbedding(mean)});
    }
    identities_ = std::move(replacement);
}

RecognitionResult EmbeddingGallery::recognize(const FaceSample& sample,
                                              const IEmbeddingExtractor& extractor,
                                              double cosineThreshold,
                                              double minimumCosineMargin,
                                              const std::string& backendName) const {
    // ranking 按余弦相似度从高到低排列。除了绝对阈值，还检查第一名和第二名
    // 的差距，防止两个相似人员分数都很高时强行接受第一名。
    RecognitionResult output;
    output.backend = backendName;
    output.threshold = cosineThreshold;
    if (identities_.empty()) {
        output.state = RecognitionState::Error;
        output.message = "Embedding gallery is empty";
        return output;
    }
    try {
        const cv::Mat query = extractor.extract(sample.alignedBgrFace);
        std::vector<std::pair<std::string, double>> ranking;
        ranking.reserve(identities_.size());
        for (const Identity& identity : identities_) {
            ranking.emplace_back(identity.personId, embeddingCosineSimilarity(query, identity.feature));
        }
        std::sort(ranking.begin(), ranking.end(), [](const auto& first, const auto& second) {
            return first.second > second.second;
        });
        output.bestScore = ranking.front().second;
        output.secondBestScore = ranking.size() > 1
                                     ? ranking[1].second
                                     : -std::numeric_limits<double>::infinity();
        const bool thresholdAccepted = output.bestScore >= cosineThreshold;
        const bool marginAccepted =
            !std::isfinite(output.secondBestScore) ||
            output.bestScore - output.secondBestScore >= minimumCosineMargin;
        if (thresholdAccepted && marginAccepted) {
            output.state = RecognitionState::Known;
            output.personId = ranking.front().first;
            output.message = "accepted";
        } else {
            output.state = RecognitionState::Unknown;
            output.message = thresholdAccepted ? "ambiguous_best_match"
                                               : "similarity_below_threshold";
        }
        return output;
    } catch (const std::exception& exception) {
        output.state = RecognitionState::Error;
        output.message = std::string("Embedding inference failed: ") + exception.what();
        return output;
    }
}

void EmbeddingGallery::save(const std::filesystem::path& file,
                            double cosineThreshold,
                            double minimumCosineMargin,
                            const std::string& modelIdentity,
                            const std::string& preprocessingVersion,
                            const std::string& modelFingerprint,
                            const std::string& featureContract) const {
    if (identities_.empty()) throw std::runtime_error("Cannot save an empty embedding gallery");
    cv::FileStorage storage("", cv::FileStorage::WRITE | cv::FileStorage::MEMORY |
                                    cv::FileStorage::FORMAT_YAML);
    storage << kFormatKey << kFormatVersion;
    storage << "cosine_threshold" << cosineThreshold;
    storage << "minimum_cosine_margin" << minimumCosineMargin;
    storage << "model_identity"
            << (modelIdentity.empty() ? std::string("unspecified") : modelIdentity);
    storage << "preprocessing_version"
            << (preprocessingVersion.empty() ? std::string("unspecified")
                                             : preprocessingVersion);
    storage << "feature_columns" << identities_.front().feature.cols;
    if (!modelFingerprint.empty()) storage << "model_fingerprint" << modelFingerprint;
    if (!featureContract.empty()) storage << "feature_contract" << featureContract;
    storage << "gallery" << "[";
    for (const Identity& identity : identities_) {
        storage << "{" << "person_id" << identity.personId << "feature" << identity.feature << "}";
    }
    storage << "]";
    detail::writeTextFile(file, storage.releaseAndGetString());
}

void EmbeddingGallery::load(const std::filesystem::path& file,
                            const std::string& modelIdentity,
                            const std::string& preprocessingVersion,
                            double* cosineThreshold,
                            double* minimumCosineMargin,
                            int expectedFeatureColumns,
                            const std::string& expectedModelFingerprint,
                            const std::string& expectedFeatureContract) {
    const std::string content = detail::readTextFile(file);
    cv::FileStorage storage(content, cv::FileStorage::READ | cv::FileStorage::MEMORY);
    if (!storage.isOpened() || storage["gallery"].empty()) {
        throw std::runtime_error("Invalid embedding gallery file: " + file.string());
    }
    int format = 0;
    int featureColumns = 0;
    std::string storedModelIdentity;
    std::string storedPreprocessingVersion;
    storage[kFormatKey] >> format;
    storage["feature_columns"] >> featureColumns;
    storage["model_identity"] >> storedModelIdentity;
    storage["preprocessing_version"] >> storedPreprocessingVersion;
    if (format != kFormatVersion || featureColumns <= 0 || storedModelIdentity.empty()
        || storedPreprocessingVersion.empty()) {
        throw std::runtime_error("Unsupported or incomplete embedding gallery contract");
    }
    if (!modelIdentity.empty() && storedModelIdentity != modelIdentity) {
        throw std::runtime_error("Embedding model identity does not match the gallery");
    }
    if (!expectedModelFingerprint.empty()) {
        std::string storedFingerprint;
        if (!storage["model_fingerprint"].empty()) storage["model_fingerprint"] >> storedFingerprint;
        if (storedFingerprint != expectedModelFingerprint) throw std::runtime_error(
            "Embedding gallery model fingerprint is missing or mismatched; rebuild the gallery");
    }
    if (!expectedFeatureContract.empty()) {
        std::string storedContract;
        if (!storage["feature_contract"].empty()) storage["feature_contract"] >> storedContract;
        if (storedContract != expectedFeatureContract) throw std::runtime_error(
            "Embedding gallery feature contract is missing or mismatched; rebuild the gallery");
    }
    if (!preprocessingVersion.empty() && storedPreprocessingVersion != preprocessingVersion) {
        throw std::runtime_error("Embedding preprocessing version does not match the gallery");
    }
    if (expectedFeatureColumns > 0 && featureColumns != expectedFeatureColumns) {
        throw std::runtime_error("Embedding gallery dimension does not match the model: expected " +
                                 std::to_string(expectedFeatureColumns) + ", got " +
                                 std::to_string(featureColumns));
    }
    double storedThreshold = cosineThreshold ? *cosineThreshold : 0.0;
    double storedMargin = minimumCosineMargin ? *minimumCosineMargin : 0.0;
    if (!storage["cosine_threshold"].empty()) storage["cosine_threshold"] >> storedThreshold;
    if (!storage["minimum_cosine_margin"].empty()) {
        storage["minimum_cosine_margin"] >> storedMargin;
    }
    validateThresholds(storedThreshold, storedMargin);

    std::vector<Identity> replacement;
    std::set<std::string> personIds;
    for (const cv::FileNode& node : storage["gallery"]) {
        std::string personId;
        cv::Mat feature;
        node["person_id"] >> personId;
        node["feature"] >> feature;
        if (personId.empty() || feature.empty() || !personIds.insert(personId).second) {
            throw std::runtime_error("Invalid or duplicate embedding gallery entry");
        }
        cv::Mat normalized = normalizeEmbedding(feature);
        if (normalized.cols != featureColumns) {
            throw std::runtime_error("Embedding gallery feature dimensions are inconsistent");
        }
        replacement.push_back({personId, std::move(normalized)});
    }
    if (replacement.empty()) throw std::runtime_error("Embedding gallery contains no identities");
    identities_ = std::move(replacement);
    if (cosineThreshold) *cosineThreshold = storedThreshold;
    if (minimumCosineMargin) *minimumCosineMargin = storedMargin;
}

class EmbeddingFaceRecognizer::Impl {
public:
    Impl(std::shared_ptr<const IEmbeddingExtractor> value, SFaceConfig configuration)
        : extractor(std::move(value)), config(std::move(configuration)) {
        if (!extractor) throw std::invalid_argument("Embedding extractor is null");
        const std::string contract = extractor->inputContract();
        if (!contract.empty()) config.preprocessingVersion += ":" + contract;
        validateThresholds(config.cosineThreshold, config.minimumCosineMargin);
    }

    std::string featureContract() const {
        std::ostringstream key;
        key << "embedding-cache-v1:" << std::quoted(config.modelIdentity) << ':'
            << std::quoted(config.preprocessingVersion) << ':' << std::quoted(modelFingerprint) << ':'
            << std::quoted(extractor->name()) << ':' << std::quoted(extractor->describe())
            << ':' << CV_VERSION << ':' << config.embeddingDim << ':'
            << config.inputWidth << ':' << config.inputHeight << ':' << config.backendId << ':' << config.targetId
            << ':' << std::quoted(config.rknn.inputColorOrder) << ':' << std::quoted(config.rknn.inputLayout)
            << ':' << std::quoted(config.rknn.inputMode) << ':' << std::hexfloat << config.rknn.inputScale
            << ':' << config.rknn.inputQuantized;
        for (double value : config.rknn.inputMean) key << ':' << std::hexfloat << value;
        key << ":std";
        for (double value : config.rknn.inputStd) key << ':' << std::hexfloat << value;
        for (const auto& name : config.rknn.outputNames) key << ':' << std::quoted(name);
        return key.str();
    }

    std::shared_ptr<const IEmbeddingExtractor> extractor;
    SFaceConfig config;
    std::string modelFingerprint;
    EmbeddingGallery gallery;
    mutable std::mutex mutex;
};

EmbeddingFaceRecognizer::EmbeddingFaceRecognizer(std::shared_ptr<const IEmbeddingExtractor> extractor,
                                                 SFaceConfig config)
    : impl_(std::make_unique<Impl>(std::move(extractor), std::move(config))) {}

EmbeddingFaceRecognizer::~EmbeddingFaceRecognizer() = default;
EmbeddingFaceRecognizer::EmbeddingFaceRecognizer(EmbeddingFaceRecognizer&&) noexcept = default;
EmbeddingFaceRecognizer& EmbeddingFaceRecognizer::operator=(EmbeddingFaceRecognizer&&) noexcept =
    default;

void EmbeddingFaceRecognizer::buildGallery(const std::vector<LabeledFaceSample>& samples) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    EmbeddingGallery replacement;
    replacement.build(samples, *impl_->extractor);
    impl_->gallery = std::move(replacement);
}

IncrementalTrainingStats EmbeddingFaceRecognizer::buildGalleryIncremental(
    const std::vector<LabeledFaceSample>& samples, const std::filesystem::path& cacheFile,
    const std::string& modelFingerprint) {
    if (modelFingerprint.empty()) throw std::invalid_argument("Model fingerprint is empty");
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->modelFingerprint != modelFingerprint) {
        impl_->gallery.clear();
        impl_->modelFingerprint = modelFingerprint;
    }
    const auto& config = impl_->config;
    const auto valid = [&config](const cv::Mat& feature) {
        if (feature.empty() || feature.type() != CV_32F || feature.rows != 1
            || feature.cols != config.embeddingDim || !cv::checkRange(feature)) return false;
        const double norm = cv::norm(feature, cv::NORM_L2);
        return std::isfinite(norm) && std::abs(norm - 1.0) < 0.001;
    };
    auto update = detail::updateFeatureCache(samples, cacheFile, impl_->featureContract(), true,
        [this](const auto& sample) { return impl_->extractor->extract(sample.face.alignedBgrFace); }, valid);
    std::vector<std::pair<std::string, cv::Mat>> entries;
    entries.reserve(update.entries.size());
    for (const auto& entry : update.entries) entries.emplace_back(entry.personId, entry.feature);
    EmbeddingGallery replacement;
    replacement.buildFromFeatures(entries);
    impl_->gallery = std::move(replacement);
    return update.stats;
}

void EmbeddingFaceRecognizer::setModelFingerprint(const std::string& modelFingerprint) {
    if (modelFingerprint.empty()) throw std::invalid_argument("Model fingerprint is empty");
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->modelFingerprint != modelFingerprint) {
        impl_->gallery.clear();
        impl_->modelFingerprint = modelFingerprint;
    }
}

RecognitionResult EmbeddingFaceRecognizer::recognize(const FaceSample& sample) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->gallery.recognize(sample, *impl_->extractor, impl_->config.cosineThreshold,
                                    impl_->config.minimumCosineMargin, name());
}

void EmbeddingFaceRecognizer::save(const std::filesystem::path& galleryFile) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->gallery.save(galleryFile, impl_->config.cosineThreshold, impl_->config.minimumCosineMargin,
                        impl_->config.modelIdentity, impl_->config.preprocessingVersion,
                        impl_->modelFingerprint,
                        impl_->modelFingerprint.empty() ? std::string{} : impl_->featureContract());
}

void EmbeddingFaceRecognizer::load(const std::filesystem::path& galleryFile) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->gallery.load(galleryFile, impl_->config.modelIdentity, impl_->config.preprocessingVersion,
                        &impl_->config.cosineThreshold, &impl_->config.minimumCosineMargin,
                        impl_->config.embeddingDim, impl_->modelFingerprint,
                        impl_->modelFingerprint.empty() ? std::string{} : impl_->featureContract());
}

std::string EmbeddingFaceRecognizer::name() const { return impl_->extractor->name(); }

std::size_t EmbeddingFaceRecognizer::identityCount() const {
    if (!impl_) return 0;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->gallery.size();
}

std::string EmbeddingFaceRecognizer::extractorName() const { return impl_->extractor->name(); }

std::string EmbeddingFaceRecognizer::describe() const { return impl_->extractor->describe(); }

double EmbeddingFaceRecognizer::cosineThreshold() const { return impl_->config.cosineThreshold; }

double EmbeddingFaceRecognizer::minimumCosineMargin() const {
    return impl_->config.minimumCosineMargin;
}

}  // namespace facecore
