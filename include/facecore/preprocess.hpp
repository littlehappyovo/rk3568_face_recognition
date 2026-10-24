#pragma once

#include "facecore/config.hpp"
#include "facecore/export.hpp"
#include "facecore/types.hpp"

#include <deque>

namespace facecore {

class FACECORE_API FacePreprocessor {
public:
    explicit FacePreprocessor(PreprocessConfig config = {});

    FaceSample process(const cv::Mat& bgrFrame, const FaceDetection& detection) const;
    cv::Mat makeGrayFace(const cv::Mat& bgrFrame, const FaceDetection& detection) const;
    cv::Mat makeAlignedColorFace(const cv::Mat& bgrFrame, const FaceDetection& detection,
                                 cv::Size outputSize = {112, 112}) const;
    cv::Rect expandedClampedBox(cv::Size frameSize, const cv::Rect2f& box) const;
    const PreprocessConfig& config() const noexcept { return config_; }

private:
    PreprocessConfig config_;
};

class FACECORE_API QualityEvaluator {
public:
    explicit QualityEvaluator(QualityConfig config = {});
    QualityReport evaluate(const cv::Mat& bgrFrame, const FaceDetection& detection) const;

private:
    QualityConfig config_;
};

class FACECORE_API DuplicateFilter {
public:
    explicit DuplicateFilter(double minimumMeanAbsoluteDifference = 3.0,
                             std::size_t historySize = 5);
    bool isNovelAndRemember(const cv::Mat& normalizedGrayFace);
    void clear() noexcept;

private:
    double threshold_;
    std::size_t historySize_;
    std::deque<cv::Mat> history_;
};

}  // namespace facecore

