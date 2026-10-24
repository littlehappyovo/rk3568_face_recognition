#pragma once

#include "facecore/config.hpp"
#include "facecore/export.hpp"
#include "facecore/types.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace facecore {

class FACECORE_API IFaceRecognizer {
public:
    virtual ~IFaceRecognizer() = default;
    virtual RecognitionResult recognize(const FaceSample& sample) const = 0;
    virtual void save(const std::filesystem::path& modelFile) const = 0;
    virtual void load(const std::filesystem::path& modelFile) = 0;
    virtual std::string name() const = 0;
};

// embedding 类识别器（SFace：ONNX/OpenCV 与 RKNN/NPU 两条通路共用）。
// 抽出这个接口是为了让 app 层不关心 embedding 是 CPU 还是 NPU 算出来的。
class FACECORE_API IEmbeddingFaceRecognizer : public IFaceRecognizer {
public:
    // 用彩色对齐人脸样本建立特征库（每种身份取均值后做 L2 归一化）。
    virtual void buildGallery(const std::vector<LabeledFaceSample>& samples) = 0;
    // 图库中的身份数量；0 表示尚未建立图库。
    virtual std::size_t identityCount() const = 0;
    // 实际使用的特征抽取器名称，例如 opencv-sface-112 / rknn-sface-112。
    virtual std::string extractorName() const = 0;
};

class FACECORE_API LbphFaceRecognizer final : public IFaceRecognizer {
public:
    explicit LbphFaceRecognizer(LbphConfig config = {});
    ~LbphFaceRecognizer() override;

    LbphFaceRecognizer(LbphFaceRecognizer&&) noexcept;
    LbphFaceRecognizer& operator=(LbphFaceRecognizer&&) noexcept;
    LbphFaceRecognizer(const LbphFaceRecognizer&) = delete;
    LbphFaceRecognizer& operator=(const LbphFaceRecognizer&) = delete;

    void train(const std::vector<LabeledFaceSample>& samples);
    RecognitionResult recognize(const FaceSample& sample) const override;
    void save(const std::filesystem::path& modelFile) const override;
    void load(const std::filesystem::path& modelFile) override;
    std::string name() const override;
    bool trained() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// OpenCV DNN 版 SFace。内部复用 EmbeddingFaceRecognizer，只是把抽取器固定为
// cv::FaceRecognizerSF，保留该类是为了兼容既有调用方与图库文件。
class FACECORE_API SFaceRecognizer final : public IEmbeddingFaceRecognizer {
public:
    explicit SFaceRecognizer(SFaceConfig config);
    ~SFaceRecognizer() override;

    SFaceRecognizer(SFaceRecognizer&&) noexcept;
    SFaceRecognizer& operator=(SFaceRecognizer&&) noexcept;
    SFaceRecognizer(const SFaceRecognizer&) = delete;
    SFaceRecognizer& operator=(const SFaceRecognizer&) = delete;

    void buildGallery(const std::vector<LabeledFaceSample>& samples) override;
    RecognitionResult recognize(const FaceSample& sample) const override;
    void save(const std::filesystem::path& galleryFile) const override;
    void load(const std::filesystem::path& galleryFile) override;
    std::string name() const override;
    std::size_t identityCount() const override;
    std::string extractorName() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace facecore
