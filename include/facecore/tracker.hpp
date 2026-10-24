#pragma once

#include "facecore/config.hpp"
#include "facecore/export.hpp"
#include "facecore/types.hpp"

#include <cstdint>
#include <vector>

namespace facecore {

class FACECORE_API FaceTracker {
public:
    explicit FaceTracker(TrackerConfig config = {});
    ~FaceTracker();

    FaceTracker(const FaceTracker&) = delete;
    FaceTracker& operator=(const FaceTracker&) = delete;
    FaceTracker(FaceTracker&&) noexcept;
    FaceTracker& operator=(FaceTracker&&) noexcept;

    std::vector<TrackedFace> update(const std::vector<FaceDetection>& detections);
    // Includes temporarily missed tracks until the configured loss limit.
    std::vector<std::uint64_t> activeTrackIds() const;
    void reset() noexcept;
    static cv::Scalar colorForTrack(std::uint64_t trackId) noexcept;

private:
    struct TrackState;
    TrackerConfig config_;
    std::vector<TrackState> tracks_;
    std::uint64_t nextId_{1};
};

}  // namespace facecore
