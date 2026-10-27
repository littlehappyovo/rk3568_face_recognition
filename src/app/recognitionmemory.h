#pragma once

#include <facecore/types.hpp>
#include <algorithm>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <string>

// Per-track temporal voting with periodic checks. Independent of Qt/camera so
// recovery from identity changes and inference failures can be regression tested.
struct TrackRecognitionMemory {
    static constexpr std::int64_t kUnknownGracePeriodUs = 3'000'000;

    std::uint64_t lastInferenceSequence = 0;
    std::int64_t lastInferenceTimestampUs = 0;
    std::int64_t firstValidInferenceTimestampUs = 0;
    bool hasFirstValidInferenceTimestamp = false;
    facecore::RecognitionResult latest;
    std::deque<std::string> votes;
    std::string stablePersonId;
    bool consensusReached = false;

    bool due(std::int64_t timestampUs, int recognitionIntervalMs) const noexcept {
        if (consensusReached && !stablePersonId.empty()) return false;
        if (lastInferenceTimestampUs <= 0 || timestampUs < lastInferenceTimestampUs) return true;
        const std::int64_t intervalUs = static_cast<std::int64_t>(
            std::max(1, recognitionIntervalMs)) * 1000;
        return timestampUs - lastInferenceTimestampUs >= intervalUs;
    }

    void update(const facecore::RecognitionResult& result, std::uint64_t sequence,
                std::int64_t timestampUs) {
        latest = result;
        lastInferenceSequence = sequence;
        lastInferenceTimestampUs = timestampUs;
        if ((result.state != facecore::RecognitionState::Known
             && result.state != facecore::RecognitionState::Unknown)
            || (result.known() && result.personId.empty())) {
            // A failed inference is not evidence of an unknown person. Also
            // hide a previously cached identity until valid votes return.
            votes.clear(); stablePersonId.clear(); consensusReached = false;
            return;
        }
        if (!hasFirstValidInferenceTimestamp || timestampUs < firstValidInferenceTimestampUs) {
            // Start the grace clock at the first usable inference for this
            // track, and restart it if the source timestamp moves backward.
            firstValidInferenceTimestampUs = timestampUs;
            hasFirstValidInferenceTimestamp = true;
        }
        const std::string vote = result.known() ? result.personId : std::string();
        if (consensusReached && vote != stablePersonId) {
            // Do not let votes from the previous occupant of a track override
            // a new person or retain a known label after an unknown result.
            votes.clear(); stablePersonId.clear(); consensusReached = false;
        }
        votes.push_back(vote);
        while (votes.size() > 5) votes.pop_front();
        std::map<std::string, int> counts;
        for (const auto& item : votes) {
            if (++counts[item] >= 3) {
                const bool unknownGraceElapsed = timestampUs >= firstValidInferenceTimestampUs
                    && timestampUs - firstValidInferenceTimestampUs >= kUnknownGracePeriodUs;
                if (item.empty() && !unknownGraceElapsed) continue;
                consensusReached = true;
                stablePersonId = item;
            }
        }
    }

    std::optional<facecore::RecognitionResult> cached() const {
        if (!consensusReached) return std::nullopt;
        auto result = latest;
        result.personId = stablePersonId;
        result.state = stablePersonId.empty() ? facecore::RecognitionState::Unknown
                                            : facecore::RecognitionState::Known;
        return result;
    }
};
