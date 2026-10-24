#include "facecore/training.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace facecore {
namespace {

std::uint64_t stableHash(const std::string& value) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (unsigned char character : value) {
        hash ^= character;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string sessionKey(const std::string& personId, const std::string& sessionId) {
    return personId + '\x1f' + sessionId;
}

double rate(std::size_t numerator, std::size_t denominator) {
    return denominator == 0 ? 0.0 : static_cast<double>(numerator) /
                                      static_cast<double>(denominator);
}

}  // namespace

std::vector<SampleRecord> splitBySession(const std::vector<SampleRecord>& records,
                                         SplitRatios ratios,
                                         std::uint64_t seed) {
    if (!std::isfinite(ratios.train) || !std::isfinite(ratios.validation) ||
        !std::isfinite(ratios.test) || ratios.train < 0.0 || ratios.validation < 0.0 ||
        ratios.test < 0.0 || ratios.train + ratios.validation + ratios.test <= 0.0) {
        throw std::invalid_argument("Dataset split ratios must be finite, non-negative, and non-zero");
    }
    if (records.empty()) return {};

    std::map<std::string, std::vector<std::string>> sessionsByPerson;
    for (const SampleRecord& record : records) {
        if (record.personId.empty() || record.sessionId.empty()) {
            throw std::invalid_argument("Every sample needs a personId and sessionId before splitting");
        }
        auto& sessions = sessionsByPerson[record.personId];
        if (std::find(sessions.begin(), sessions.end(), record.sessionId) == sessions.end()) {
            sessions.push_back(record.sessionId);
        }
    }

    const double ratioSum = ratios.train + ratios.validation + ratios.test;
    if (!std::isfinite(ratioSum)) {
        throw std::invalid_argument("Dataset split ratio sum must be finite");
    }
    const double trainRatio = ratios.train / ratioSum;
    const double validationRatio = ratios.validation / ratioSum;
    std::unordered_map<std::string, DataSplit> assignments;
    for (auto& person : sessionsByPerson) {
        auto& sessions = person.second;
        std::mt19937_64 random(seed ^ stableHash(person.first));
        std::shuffle(sessions.begin(), sessions.end(), random);
        const std::size_t count = sessions.size();
        const std::array<double, 3> normalized = {
            trainRatio, validationRatio, ratios.test / ratioSum};
        std::array<std::size_t, 3> counts{};
        std::array<double, 3> remainders{};
        std::size_t assigned = 0;
        for (std::size_t split = 0; split < counts.size(); ++split) {
            const double exact = static_cast<double>(count) * normalized[split];
            counts[split] = static_cast<std::size_t>(std::floor(exact));
            remainders[split] = exact - static_cast<double>(counts[split]);
            assigned += counts[split];
        }
        while (assigned < count) {
            const auto best = static_cast<std::size_t>(std::distance(
                remainders.begin(), std::max_element(remainders.begin(), remainders.end())));
            ++counts[best];
            remainders[best] = -1.0;
            ++assigned;
        }
        const std::size_t positiveSplits = static_cast<std::size_t>(
            std::count_if(normalized.begin(), normalized.end(), [](double value) {
                return value > 0.0;
            }));
        if (count >= positiveSplits) {
            for (std::size_t split = 0; split < counts.size(); ++split) {
                if (normalized[split] <= 0.0 || counts[split] != 0) continue;
                const auto donor = static_cast<std::size_t>(std::distance(
                    counts.begin(), std::max_element(counts.begin(), counts.end())));
                if (counts[donor] > 1) {
                    --counts[donor];
                    ++counts[split];
                }
            }
        }
        const std::size_t trainCount = counts[0];
        const std::size_t validationCount = counts[1];
        for (std::size_t index = 0; index < count; ++index) {
            DataSplit split = DataSplit::Test;
            if (index < trainCount) split = DataSplit::Train;
            else if (index < trainCount + validationCount) split = DataSplit::Validation;
            assignments.emplace(sessionKey(person.first, sessions[index]), split);
        }
    }

    std::vector<SampleRecord> output = records;
    for (SampleRecord& record : output) {
        record.split = assignments.at(sessionKey(record.personId, record.sessionId));
    }
    return output;
}

ThresholdCalibration calibrateThreshold(const std::vector<ScoreObservation>& observations,
                                         bool higherIsBetter,
                                         double maximumFalseAcceptRate) {
    if (!std::isfinite(maximumFalseAcceptRate) || maximumFalseAcceptRate < 0.0 ||
        maximumFalseAcceptRate > 1.0) {
        throw std::invalid_argument("maximumFalseAcceptRate must be in [0, 1]");
    }
    std::vector<double> candidates;
    std::size_t genuineCount = 0;
    std::size_t impostorCount = 0;
    for (const ScoreObservation& observation : observations) {
        if (!std::isfinite(observation.score)) {
            throw std::invalid_argument("Calibration scores must be finite");
        }
        candidates.push_back(observation.score);
        observation.genuine ? ++genuineCount : ++impostorCount;
    }
    if (genuineCount == 0 || impostorCount == 0) {
        throw std::invalid_argument("Calibration needs both genuine and impostor observations");
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    if (higherIsBetter) {
        candidates.push_back(std::nextafter(candidates.back(),
                                             std::numeric_limits<double>::infinity()));
    } else {
        candidates.push_back(std::nextafter(candidates.front(),
                                             -std::numeric_limits<double>::infinity()));
    }

    ThresholdCalibration best;
    best.genuineCount = genuineCount;
    best.impostorCount = impostorCount;
    best.falseRejectRate = std::numeric_limits<double>::infinity();
    best.falseAcceptRate = std::numeric_limits<double>::infinity();
    bool found = false;
    for (double threshold : candidates) {
        std::size_t falseAccepts = 0;
        std::size_t falseRejects = 0;
        for (const ScoreObservation& observation : observations) {
            const bool accepted = higherIsBetter ? observation.score >= threshold
                                                 : observation.score <= threshold;
            if (observation.genuine && !accepted) ++falseRejects;
            if (!observation.genuine && accepted) ++falseAccepts;
        }
        const double falseAcceptRate = rate(falseAccepts, impostorCount);
        const double falseRejectRate = rate(falseRejects, genuineCount);
        if (falseAcceptRate > maximumFalseAcceptRate + 1e-12) continue;
        if (!found || falseRejectRate < best.falseRejectRate - 1e-12 ||
            (std::abs(falseRejectRate - best.falseRejectRate) <= 1e-12 &&
             falseAcceptRate < best.falseAcceptRate - 1e-12)) {
            found = true;
            best.threshold = threshold;
            best.falseAcceptRate = falseAcceptRate;
            best.falseRejectRate = falseRejectRate;
        }
    }
    if (!found) throw std::runtime_error("No threshold satisfies the requested FAR");
    return best;
}

EvaluationReport evaluateRecognition(const std::vector<EvaluationItem>& items) {
    EvaluationReport report;
    report.total = items.size();
    for (const EvaluationItem& item : items) {
        const bool expectedKnown = item.expectedPersonId.has_value();
        const bool predictedKnown = item.predicted.state == RecognitionState::Known;
        if (expectedKnown) {
            ++report.knownTotal;
            const std::string& expected = *item.expectedPersonId;
            const std::string predicted = predictedKnown ? item.predicted.personId : "<unknown>";
            ++report.confusion[expected][predicted];
            if (predictedKnown && item.predicted.personId == expected) ++report.correctKnown;
            if (!predictedKnown) ++report.falseRejects;
        } else {
            ++report.unknownTotal;
            const std::string predicted = predictedKnown ? item.predicted.personId : "<unknown>";
            ++report.confusion["<unknown>"][predicted];
            if (predictedKnown) ++report.falseAccepts;
        }
    }
    report.top1Accuracy = rate(report.correctKnown, report.knownTotal);
    report.falseAcceptRate = rate(report.falseAccepts, report.unknownTotal);
    report.falseRejectRate = rate(report.falseRejects, report.knownTotal);
    return report;
}

}  // namespace facecore
