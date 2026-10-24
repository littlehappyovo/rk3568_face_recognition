#pragma once

#include "facecore/config.hpp"
#include "facecore/export.hpp"
#include "facecore/recognizer.hpp"
#include "facecore/types.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace facecore {

// 对齐人脸（通常是 112x112 彩色）-> L2 归一化 embedding。
//
// 检测/识别后端（OpenCV DNN、RKNN NPU）只需要实现这个接口，图库匹配、阈值、
// margin、持久化格式和投票逻辑都由 EmbeddingGallery / EmbeddingFaceRecognizer
// 复用，避免为 NPU 复制一份识别逻辑。
class FACECORE_API IEmbeddingExtractor {
public:
    virtual ~IEmbeddingExtractor() = default;
    virtual cv::Mat extract(const cv::Mat& alignedFace) const = 0;
    virtual std::string name() const = 0;
    // 一段可读的说明，用于启动日志，例如输入尺寸、量化方式和 RKNN 张量名。
    virtual std::string describe() const { return name(); }
    // Changes to numerical input processing must invalidate previously saved
    // embeddings, independently of the version of the stored aligned photos.
    virtual std::string inputContract() const { return {}; }
};

// 展平成 1xN 的 CV_32F 并做 L2 归一化；零向量或非有限值会抛异常。
FACECORE_API cv::Mat normalizeEmbedding(const cv::Mat& input);
// 要求两个特征形状一致，返回点积（已归一化时即为余弦相似度）。
FACECORE_API double embeddingCosineSimilarity(const cv::Mat& first, const cv::Mat& second);

// 通用 embedding 图库。文件格式与既有 facecore_format 2 完全一致，
// CPU 与 NPU 生成的 .yml 特征库可以互换使用（前提是模型身份/预处理版本一致）。
class FACECORE_API EmbeddingGallery {
public:
    struct Identity {
        std::string personId;
        cv::Mat feature;
    };

    void clear();
    bool empty() const noexcept { return identities_.empty(); }
    std::size_t size() const noexcept { return identities_.size(); }
    const std::vector<Identity>& identities() const noexcept { return identities_; }

    // 同一 personId 的多个样本先取均值再归一化。
    void build(const std::vector<LabeledFaceSample>& samples, const IEmbeddingExtractor& extractor);
    // 余弦排序 + 阈值 + margin 判定；图库为空或推理失败返回 Error 状态。
    RecognitionResult recognize(const FaceSample& sample,
                                const IEmbeddingExtractor& extractor,
                                double cosineThreshold,
                                double minimumCosineMargin,
                                const std::string& backendName) const;

    void save(const std::filesystem::path& file,
              double cosineThreshold,
              double minimumCosineMargin,
              const std::string& modelIdentity,
              const std::string& preprocessingVersion) const;
    // 校验 facecore_format、feature_columns、model_identity 与预处理版本；
    // 文件里记录的阈值会通过输出参数带回，缺省时保持入参不变。
    void load(const std::filesystem::path& file,
              const std::string& modelIdentity,
              const std::string& preprocessingVersion,
              double* cosineThreshold,
              double* minimumCosineMargin,
              int expectedFeatureColumns = 0);

private:
    std::vector<Identity> identities_;
};

// 任意抽取器 + 通用图库 = 一个完整的 embedding 识别后端。
class FACECORE_API EmbeddingFaceRecognizer final : public IEmbeddingFaceRecognizer {
public:
    EmbeddingFaceRecognizer(std::shared_ptr<const IEmbeddingExtractor> extractor, SFaceConfig config);
    ~EmbeddingFaceRecognizer() override;

    EmbeddingFaceRecognizer(EmbeddingFaceRecognizer&&) noexcept;
    EmbeddingFaceRecognizer& operator=(EmbeddingFaceRecognizer&&) noexcept;
    EmbeddingFaceRecognizer(const EmbeddingFaceRecognizer&) = delete;
    EmbeddingFaceRecognizer& operator=(const EmbeddingFaceRecognizer&) = delete;

    void buildGallery(const std::vector<LabeledFaceSample>& samples) override;
    RecognitionResult recognize(const FaceSample& sample) const override;
    void save(const std::filesystem::path& galleryFile) const override;
    void load(const std::filesystem::path& galleryFile) override;
    std::string name() const override;
    std::size_t identityCount() const override;
    std::string extractorName() const override;
    std::string describe() const;
    double cosineThreshold() const;
    double minimumCosineMargin() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// OpenCV DNN 版 SFace 抽取器。未编译 OpenCV DNN 后端时抛出 runtime_error。
FACECORE_API std::shared_ptr<const IEmbeddingExtractor> makeOpenCvSFaceExtractor(
    const SFaceConfig& config);

}  // namespace facecore
