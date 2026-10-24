#pragma once

#include "facecore/export.hpp"
#include "facecore/types.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace facecore {

struct SplitRatios {
    double train{0.60};
    double validation{0.20};
    double test{0.20};
};

FACECORE_API std::vector<SampleRecord> splitBySession(
    const std::vector<SampleRecord>& records,
    SplitRatios ratios = {},
    std::uint64_t seed = 0xface3568ULL);

struct ScoreObservation {
    bool genuine{false};
    double score{0.0};
};

struct ThresholdCalibration {
    double threshold{0.0};
    double falseAcceptRate{0.0};
    double falseRejectRate{0.0};
    std::size_t genuineCount{0};
    std::size_t impostorCount{0};
};

FACECORE_API ThresholdCalibration calibrateThreshold(
    const std::vector<ScoreObservation>& observations,
    bool higherIsBetter,
    double maximumFalseAcceptRate = 0.01);

struct EvaluationItem {
    std::optional<std::string> expectedPersonId;  // nullopt means an unknown identity
    RecognitionResult predicted;
};

struct EvaluationReport {
    std::size_t total{0};
    std::size_t knownTotal{0};
    std::size_t unknownTotal{0};
    std::size_t correctKnown{0};
    std::size_t falseAccepts{0};
    std::size_t falseRejects{0};
    double top1Accuracy{0.0};
    double falseAcceptRate{0.0};
    double falseRejectRate{0.0};
    std::map<std::string, std::map<std::string, std::size_t>> confusion;
};

FACECORE_API EvaluationReport evaluateRecognition(const std::vector<EvaluationItem>& items);

}  // namespace facecore

