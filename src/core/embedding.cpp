#include "facecore/embedding.hpp"

#include "io_internal.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
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
    std::map<std::string, std::vector<cv::Mat>> features;
    for (const LabeledFaceSample& sample : samples) {
        if (sample.personId.empty()) throw std::invalid_argument("Gallery personId is empty");
        features[sample.personId].push_back(extractor.extract(sample.face.alignedBgrFace));
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
                            const std::string& preprocessingVersion) const {
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
                            int expectedFeatureColumns) {
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

    std::shared_ptr<const IEmbeddingExtractor> extractor;
    SFaceConfig config;
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

RecognitionResult EmbeddingFaceRecognizer::recognize(const FaceSample& sample) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->gallery.recognize(sample, *impl_->extractor, impl_->config.cosineThreshold,
                                    impl_->config.minimumCosineMargin, name());
}

void EmbeddingFaceRecognizer::save(const std::filesystem::path& galleryFile) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->gallery.save(galleryFile, impl_->config.cosineThreshold, impl_->config.minimumCosineMargin,
                        impl_->config.modelIdentity, impl_->config.preprocessingVersion);
}

void EmbeddingFaceRecognizer::load(const std::filesystem::path& galleryFile) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->gallery.load(galleryFile, impl_->config.modelIdentity, impl_->config.preprocessingVersion,
                        &impl_->config.cosineThreshold, &impl_->config.minimumCosineMargin,
                        impl_->config.embeddingDim);
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
