#include "facecore/recognizer.hpp"

#include "facecore/embedding.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace facecore {

// OpenCV DNN 版 SFace 只是“OpenCV 抽取器 + 通用图库”的组合，保留这个类是为了
// 兼容既有调用方与既有 facecore_format 2 图库文件。
class SFaceRecognizer::Impl {
public:
    explicit Impl(SFaceConfig config)
        : extractor(makeOpenCvSFaceExtractor(config)),
          recognizer(extractor, std::move(config)) {}

    std::shared_ptr<const IEmbeddingExtractor> extractor;
    EmbeddingFaceRecognizer recognizer;
};

SFaceRecognizer::SFaceRecognizer(SFaceConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

SFaceRecognizer::~SFaceRecognizer() = default;
SFaceRecognizer::SFaceRecognizer(SFaceRecognizer&&) noexcept = default;
SFaceRecognizer& SFaceRecognizer::operator=(SFaceRecognizer&&) noexcept = default;

void SFaceRecognizer::buildGallery(const std::vector<LabeledFaceSample>& samples) {
    impl_->recognizer.buildGallery(samples);
}

RecognitionResult SFaceRecognizer::recognize(const FaceSample& sample) const {
    return impl_->recognizer.recognize(sample);
}

void SFaceRecognizer::save(const std::filesystem::path& galleryFile) const {
    impl_->recognizer.save(galleryFile);
}

void SFaceRecognizer::load(const std::filesystem::path& galleryFile) {
    impl_->recognizer.load(galleryFile);
}

std::string SFaceRecognizer::name() const { return impl_->recognizer.name(); }

std::size_t SFaceRecognizer::identityCount() const { return impl_->recognizer.identityCount(); }

std::string SFaceRecognizer::extractorName() const { return impl_->recognizer.extractorName(); }

}  // namespace facecore
