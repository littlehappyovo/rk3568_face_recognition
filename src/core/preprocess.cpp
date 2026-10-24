#include "facecore/preprocess.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace facecore {
namespace {

// 预处理是识别稳定性的基础：检测框只说明“脸在哪里”，这里把它变成固定尺寸、
// 固定姿态的人脸输入。LBPH 使用均衡化灰度图，embedding 使用 112x112 彩色图。
cv::Mat toBgr(const cv::Mat& input) {
    if (input.empty()) throw std::invalid_argument("Input image is empty");
    if (input.type() == CV_8UC3) return input;
    cv::Mat output;
    if (input.type() == CV_8UC1) {
        cv::cvtColor(input, output, cv::COLOR_GRAY2BGR);
    } else if (input.type() == CV_8UC4) {
        cv::cvtColor(input, output, cv::COLOR_BGRA2BGR);
    } else {
        throw std::invalid_argument("Input image must be CV_8UC1, CV_8UC3, or CV_8UC4");
    }
    return output;
}

bool validLandmarks(const FaceDetection& detection, cv::Size frameSize) {
    if (!detection.hasLandmarks) return false;
    const auto& box = detection.box;
    if (!std::isfinite(box.x) || !std::isfinite(box.y) || !std::isfinite(box.width)
        || !std::isfinite(box.height) || box.width <= 0 || box.height <= 0) return false;
    const cv::Rect2f allowed(box.x - box.width * .25F, box.y - box.height * .25F,
                             box.width * 1.5F, box.height * 1.5F);
    for (const auto& point : detection.landmarks) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)
            || point.x < 0 || point.y < 0 || point.x >= frameSize.width
            || point.y >= frameSize.height || !allowed.contains(point)) return false;
    }
    // Reject collapsed points without imposing a new head-pose policy.
    const auto eyeVector = detection.landmarks[1] - detection.landmarks[0];
    const auto eyeCenter = (detection.landmarks[0] + detection.landmarks[1]) * .5F;
    const auto mouthCenter = (detection.landmarks[3] + detection.landmarks[4]) * .5F;
    const auto eyeToMouth = mouthCenter - eyeCenter;
    const double eyeDistance = cv::norm(eyeVector);
    return eyeDistance >= 1.0 && cv::norm(detection.landmarks[3] - detection.landmarks[4]) >= 1.0
        && std::abs(eyeVector.x * eyeToMouth.y - eyeVector.y * eyeToMouth.x) / eyeDistance >= 1.0;
}

std::vector<cv::Point2f> referenceLandmarks(cv::Size size) {
    const float scaleX = static_cast<float>(size.width) / 112.0F;
    const float scaleY = static_cast<float>(size.height) / 112.0F;
    return {{38.2946F * scaleX, 51.6963F * scaleY},
            {73.5318F * scaleX, 51.5014F * scaleY},
            {56.0252F * scaleX, 71.7366F * scaleY},
            {41.5493F * scaleX, 92.3655F * scaleY},
            {70.7299F * scaleX, 92.2041F * scaleY}};
}

}  // namespace

FacePreprocessor::FacePreprocessor(PreprocessConfig config) : config_(std::move(config)) {
    if (config_.faceWidth <= 0 || config_.faceHeight <= 0
        || !std::isfinite(config_.boxExpansion) || config_.boxExpansion < 0.0F
        || config_.boxExpansion > 1.0F) {
        throw std::invalid_argument("Invalid face preprocessing configuration");
    }
}

cv::Rect FacePreprocessor::expandedClampedBox(cv::Size frameSize,
                                               const cv::Rect2f& box) const {
    if (frameSize.width <= 0 || frameSize.height <= 0 || !std::isfinite(box.x)
        || !std::isfinite(box.y) || !std::isfinite(box.width)
        || !std::isfinite(box.height) || box.width <= 0.0F || box.height <= 0.0F) {
        return {};
    }
    const float horizontal = box.width * config_.boxExpansion;
    const float vertical = box.height * config_.boxExpansion;
    const int left = static_cast<int>(std::floor(box.x - horizontal));
    const int top = static_cast<int>(std::floor(box.y - vertical));
    const int right = static_cast<int>(std::ceil(box.x + box.width + horizontal));
    const int bottom = static_cast<int>(std::ceil(box.y + box.height + vertical));
    return cv::Rect(left, top, std::max(0, right - left), std::max(0, bottom - top)) &
           cv::Rect(0, 0, frameSize.width, frameSize.height);
}

cv::Mat FacePreprocessor::makeAlignedColorFace(const cv::Mat& input,
                                                const FaceDetection& detection,
                                                cv::Size outputSize) const {
    // Present-but-invalid landmarks are rejected for this face. Missing
    // landmarks retain the box-crop fallback used by non-landmark detectors.
    if (outputSize.width <= 0 || outputSize.height <= 0) {
        throw std::invalid_argument("Aligned face output size must be positive");
    }
    const cv::Mat bgr = toBgr(input);
    if (config_.alignByEyes && detection.hasLandmarks) {
        if (!validLandmarks(detection, bgr.size())) {
            throw std::invalid_argument("invalid_alignment: invalid landmarks");
        }
        std::vector<cv::Point2f> source(detection.landmarks.begin(), detection.landmarks.end());
        const std::vector<cv::Point2f> destination = referenceLandmarks(outputSize);
        cv::Mat inliers;
        const cv::Mat transform = cv::estimateAffinePartial2D(source, destination, inliers,
                                                               cv::LMEDS);
        if (!transform.empty() && cv::checkRange(transform)) {
            std::vector<cv::Point2f> fitted;
            cv::transform(source, fitted, transform);
            double squaredError = 0;
            for (std::size_t i = 0; i < fitted.size(); ++i) {
                const auto error = fitted[i] - destination[i];
                squaredError += error.dot(error);
            }
            if (std::sqrt(squaredError / fitted.size()) > .25 * outputSize.width) {
                throw std::invalid_argument("invalid_alignment: inconsistent landmarks");
            }
            cv::Mat mask(bgr.size(), CV_8UC1, cv::Scalar(255));
            cv::Mat coverage;
            cv::warpAffine(mask, coverage, transform, outputSize, cv::INTER_NEAREST,
                           cv::BORDER_CONSTANT, cv::Scalar(0));
            if (cv::countNonZero(coverage) < .50 * coverage.total()) {
                throw std::invalid_argument("invalid_alignment: insufficient source coverage");
            }
            cv::Mat aligned;
            cv::warpAffine(bgr, aligned, transform, outputSize, cv::INTER_LINEAR,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            if (cv::norm(aligned, cv::NORM_INF) <= 0) {
                throw std::invalid_argument("invalid_alignment: empty aligned content");
            }
            return aligned;
        }
        throw std::invalid_argument("invalid_alignment: cannot fit landmarks");
    }

    const cv::Rect crop = expandedClampedBox(bgr.size(), detection.box);
    if (crop.empty()) {
        throw std::invalid_argument("Face bounding box lies outside the image");
    }
    cv::Mat resized;
    cv::resize(bgr(crop), resized, outputSize, 0.0, 0.0,
               crop.width > outputSize.width ? cv::INTER_AREA : cv::INTER_LINEAR);
    return resized;
}

cv::Mat FacePreprocessor::makeGrayFace(const cv::Mat& bgrFrame,
                                       const FaceDetection& detection) const {
    cv::Mat aligned = makeAlignedColorFace(
        bgrFrame, detection, cv::Size(config_.faceWidth, config_.faceHeight));
    cv::Mat gray;
    cv::cvtColor(aligned, gray, cv::COLOR_BGR2GRAY);
    if (config_.equalizeHistogram) cv::equalizeHist(gray, gray);
    return gray;
}

FaceSample FacePreprocessor::process(const cv::Mat& bgrFrame,
                                     const FaceDetection& detection) const {
    // 一次 process 同时生成两套输入，避免训练时 LBPH 和 embedding 使用不同
    // 的检测框。grayFace 尺寸由配置决定，alignedBgrFace 固定为 112x112。
    FaceSample sample;
    sample.grayFace = makeGrayFace(bgrFrame, detection);
    sample.alignedBgrFace = makeAlignedColorFace(bgrFrame, detection, cv::Size(112, 112));
    return sample;
}

QualityEvaluator::QualityEvaluator(QualityConfig config) : config_(std::move(config)) {
    if (config_.minimumFaceWidth <= 0 || !std::isfinite(config_.minimumBrightness)
        || !std::isfinite(config_.maximumBrightness)
        || !std::isfinite(config_.minimumLaplacianVariance)
        || !std::isfinite(config_.maximumClippedFraction)
        || config_.minimumBrightness < 0.0 ||
        config_.maximumBrightness > 255.0 ||
        config_.minimumBrightness >= config_.maximumBrightness ||
        config_.minimumLaplacianVariance < 0.0 || config_.maximumClippedFraction < 0.0F ||
        config_.maximumClippedFraction > 1.0F) {
        throw std::invalid_argument("Invalid face quality configuration");
    }
}

QualityReport QualityEvaluator::evaluate(const cv::Mat& input,
                                          const FaceDetection& detection) const {
    // 质量判断只决定“这张脸是否适合采集/识别”，不会改变检测框本身；因此
    // 低质量目标仍可以显示检测框，但不会写入数据集或送入识别模型。
    QualityReport report;
    if (input.empty()) {
        report.reasons.push_back("empty_frame");
        return report;
    }
    if (!std::isfinite(detection.box.x) || !std::isfinite(detection.box.y)
        || !std::isfinite(detection.box.width) || !std::isfinite(detection.box.height)
        || detection.box.width <= 0.0F || detection.box.height <= 0.0F) {
        report.reasons.push_back("invalid_box");
        return report;
    }
    report.faceWidthPixels = detection.box.width;
    if (detection.hasLandmarks && !validLandmarks(detection, input.size())) {
        report.reasons.push_back("invalid_landmarks");
    }
    if (detection.box.width < static_cast<float>(config_.minimumFaceWidth)) {
        report.reasons.push_back("face_too_small");
    }

    const cv::Rect2f frameRect(0.0F, 0.0F, static_cast<float>(input.cols),
                               static_cast<float>(input.rows));
    const cv::Rect2f clipped = detection.box & frameRect;
    const double sourceArea = std::max(1.0, static_cast<double>(detection.box.area()));
    const double clippedFraction = 1.0 - static_cast<double>(clipped.area()) / sourceArea;
    report.clipped = clippedFraction > config_.maximumClippedFraction;
    if (report.clipped) report.reasons.push_back("face_clipped");
    const cv::Rect crop = cv::Rect(static_cast<int>(std::floor(clipped.x)),
                                   static_cast<int>(std::floor(clipped.y)),
                                   static_cast<int>(std::ceil(clipped.width)),
                                   static_cast<int>(std::ceil(clipped.height))) &
                          cv::Rect(0, 0, input.cols, input.rows);
    if (crop.empty()) {
        report.reasons.push_back("face_outside_frame");
        return report;
    }

    cv::Mat gray;
    const cv::Mat face = input(crop);
    if (face.channels() == 1) {
        gray = face;
    } else if (face.channels() == 3) {
        cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
    } else if (face.channels() == 4) {
        cv::cvtColor(face, gray, cv::COLOR_BGRA2GRAY);
    } else {
        report.reasons.push_back("unsupported_pixel_format");
        return report;
    }
    report.brightness = cv::mean(gray)[0];
    if (report.brightness < config_.minimumBrightness) report.reasons.push_back("too_dark");
    if (report.brightness > config_.maximumBrightness) report.reasons.push_back("too_bright");

    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean;
    cv::Scalar standardDeviation;
    cv::meanStdDev(laplacian, mean, standardDeviation);
    report.sharpness = standardDeviation[0] * standardDeviation[0];
    if (report.sharpness < config_.minimumLaplacianVariance) {
        report.reasons.push_back("too_blurry");
    }
    report.accepted = report.reasons.empty();
    return report;
}

DuplicateFilter::DuplicateFilter(double minimumMeanAbsoluteDifference,
                                 std::size_t historySize)
    : threshold_(minimumMeanAbsoluteDifference), historySize_(historySize) {
    if (!std::isfinite(threshold_) || threshold_ < 0.0 || historySize_ == 0) {
        throw std::invalid_argument("Invalid duplicate filter configuration");
    }
}

bool DuplicateFilter::isNovelAndRemember(const cv::Mat& face) {
    if (face.empty() || face.type() != CV_8UC1) {
        throw std::invalid_argument("DuplicateFilter requires a non-empty CV_8UC1 face");
    }
    for (const cv::Mat& previous : history_) {
        if (previous.size() != face.size()) continue;
        cv::Mat difference;
        cv::absdiff(previous, face, difference);
        if (cv::mean(difference)[0] < threshold_) return false;
    }
    history_.push_back(face.clone());
    while (history_.size() > historySize_) history_.pop_front();
    return true;
}

void DuplicateFilter::clear() noexcept { history_.clear(); }

}  // namespace facecore
