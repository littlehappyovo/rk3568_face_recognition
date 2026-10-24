#include "facecore/tracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace facecore {
namespace {

// 跟踪器不是重新识别人脸，而是把“这一帧的检测框”与“上一帧的轨迹”配对。
// IoU 适合框重叠，中心距离适合轻微移动；两者结合可以在检测间隔内保持稳定 ID。
float intersectionOverUnion(const cv::Rect2f& first, const cv::Rect2f& second) {
    const float left = std::max(first.x, second.x);
    const float top = std::max(first.y, second.y);
    const float right = std::min(first.x + first.width, second.x + second.width);
    const float bottom = std::min(first.y + first.height, second.y + second.height);
    const float intersection = std::max(0.0F, right - left) * std::max(0.0F, bottom - top);
    const float unionArea = first.area() + second.area() - intersection;
    return unionArea > 0.0F ? intersection / unionArea : 0.0F;
}

float normalizedCenterDistance(const cv::Rect2f& first, const cv::Rect2f& second) {
    const cv::Point2f firstCenter(first.x + first.width * 0.5F,
                                  first.y + first.height * 0.5F);
    const cv::Point2f secondCenter(second.x + second.width * 0.5F,
                                   second.y + second.height * 0.5F);
    const float distance = cv::norm(firstCenter - secondCenter);
    const float scale = std::max(1.0F, 0.5F * (std::sqrt(std::max(0.0F, first.area())) +
                                                std::sqrt(std::max(0.0F, second.area()))));
    return distance / scale;
}

float blend(float previous, float current, float alpha) {
    return previous * (1.0F - alpha) + current * alpha;
}

FaceDetection smooth(const FaceDetection& previous, const FaceDetection& current, float alpha) {
    FaceDetection output = current;
    output.box.x = blend(previous.box.x, current.box.x, alpha);
    output.box.y = blend(previous.box.y, current.box.y, alpha);
    output.box.width = blend(previous.box.width, current.box.width, alpha);
    output.box.height = blend(previous.box.height, current.box.height, alpha);
    output.score = blend(previous.score, current.score, alpha);
    if (previous.hasLandmarks && current.hasLandmarks) {
        for (std::size_t index = 0; index < output.landmarks.size(); ++index) {
            output.landmarks[index].x =
                blend(previous.landmarks[index].x, current.landmarks[index].x, alpha);
            output.landmarks[index].y =
                blend(previous.landmarks[index].y, current.landmarks[index].y, alpha);
        }
    }
    return output;
}

}  // namespace

struct FaceTracker::TrackState {
    std::uint64_t id{0};
    FaceDetection detection;
    int age{0};
    int missedFrames{0};
};

FaceTracker::FaceTracker(TrackerConfig config) : config_(config) {
    if (!std::isfinite(config_.minimumIou) || config_.minimumIou < 0.0F ||
        config_.minimumIou > 1.0F ||
        !std::isfinite(config_.maximumNormalizedCenterDistance) ||
        config_.maximumNormalizedCenterDistance < 0.0F ||
        config_.maximumMissedFrames < 0 || config_.smoothingAlpha < 0.0F ||
        config_.smoothingAlpha > 1.0F || !std::isfinite(config_.smoothingAlpha)) {
        throw std::invalid_argument("Invalid face tracker configuration");
    }
}

FaceTracker::~FaceTracker() = default;
FaceTracker::FaceTracker(FaceTracker&&) noexcept = default;
FaceTracker& FaceTracker::operator=(FaceTracker&&) noexcept = default;

std::vector<TrackedFace> FaceTracker::update(const std::vector<FaceDetection>& detections) {
    // 先给所有旧轨迹增加 age/missedFrames，再计算所有可能的匹配候选；随后按
    // 匹配分数贪心地一对一分配。未匹配检测创建新 track，超过丢失上限的旧 track
    // 在函数末尾删除。
    struct Candidate {
        std::size_t track;
        std::size_t detection;
        float score;
    };
    const auto validDetection = [](const FaceDetection& detection) {
        return std::isfinite(detection.box.x) && std::isfinite(detection.box.y)
               && std::isfinite(detection.box.width) && std::isfinite(detection.box.height)
               && detection.box.width > 0.0F && detection.box.height > 0.0F;
    };

    for (TrackState& track : tracks_) {
        ++track.age;
        ++track.missedFrames;
    }

    std::vector<Candidate> candidates;
    for (std::size_t track = 0; track < tracks_.size(); ++track) {
        for (std::size_t detection = 0; detection < detections.size(); ++detection) {
            if (!validDetection(detections[detection])) {
                continue;
            }
            const float iou = intersectionOverUnion(tracks_[track].detection.box,
                                                    detections[detection].box);
            const float centerDistance = normalizedCenterDistance(
                tracks_[track].detection.box, detections[detection].box);
            if (iou >= config_.minimumIou ||
                centerDistance <= config_.maximumNormalizedCenterDistance) {
                // IoU dominates, while center distance breaks ties during gentle motion.
                candidates.push_back({track, detection, iou - 0.10F * centerDistance});
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& first,
                                                       const Candidate& second) {
        return first.score > second.score;
    });

    std::vector<bool> trackUsed(tracks_.size(), false);
    std::vector<bool> detectionUsed(detections.size(), false);
    std::vector<std::size_t> detectionToTrack(detections.size(),
                                               static_cast<std::size_t>(-1));
    for (const Candidate& candidate : candidates) {
        if (trackUsed[candidate.track] || detectionUsed[candidate.detection]) continue;
        TrackState& track = tracks_[candidate.track];
        track.detection = smooth(track.detection, detections[candidate.detection],
                                 config_.smoothingAlpha);
        track.missedFrames = 0;
        trackUsed[candidate.track] = true;
        detectionUsed[candidate.detection] = true;
        detectionToTrack[candidate.detection] = candidate.track;
    }

    for (std::size_t detection = 0; detection < detections.size(); ++detection) {
        if (detectionUsed[detection] || !validDetection(detections[detection])) continue;
        TrackState track;
        track.id = nextId_++;
        track.detection = detections[detection];
        track.age = 1;
        track.missedFrames = 0;
        tracks_.push_back(track);
        detectionToTrack[detection] = tracks_.size() - 1;
    }

    std::vector<TrackedFace> output;
    output.reserve(detections.size());
    for (std::size_t detection = 0; detection < detections.size(); ++detection) {
        const std::size_t trackIndex = detectionToTrack[detection];
        if (trackIndex == static_cast<std::size_t>(-1)) continue;
        const TrackState& track = tracks_[trackIndex];
        output.push_back({track.id, track.detection, colorForTrack(track.id), track.age,
                          track.missedFrames});
    }

    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [this](const TrackState& track) {
                      return track.missedFrames > config_.maximumMissedFrames;
                  }),
                  tracks_.end());
    return output;
}

void FaceTracker::reset() noexcept {
    tracks_.clear();
    nextId_ = 1;
}

cv::Scalar FaceTracker::colorForTrack(std::uint64_t trackId) noexcept {
    static const std::array<cv::Scalar, 12> palette = {
        cv::Scalar(255, 99, 71),   cv::Scalar(60, 179, 113), cv::Scalar(30, 144, 255),
        cv::Scalar(238, 130, 238), cv::Scalar(0, 215, 255),  cv::Scalar(255, 191, 0),
        cv::Scalar(147, 20, 255),  cv::Scalar(50, 205, 50),  cv::Scalar(255, 105, 180),
        cv::Scalar(0, 140, 255),   cv::Scalar(255, 255, 0),  cv::Scalar(180, 105, 255)};
    const std::size_t index = trackId == 0 ? 0 : static_cast<std::size_t>((trackId - 1) % palette.size());
    return palette[index];
}

}  // namespace facecore
