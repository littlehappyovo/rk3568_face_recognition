#pragma once

#include "facecore/types.hpp"
#include "io_internal.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <functional>
#include <cstdint>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace facecore::detail {

struct CachedSampleFeature {
    std::string sampleId;
    std::string personId;
    std::string fingerprint;
    cv::Mat feature;
};

struct CachedFeatureUpdate {
    std::vector<CachedSampleFeature> entries;
    IncrementalTrainingStats stats;
};

// Detect accidental cache corruption; source-image identity is checked separately
// with the SHA-256 fingerprint supplied by the dataset owner.
inline std::string featureChecksum(const cv::Mat& feature) {
    std::uint64_t value = 14695981039346656037ULL;
    for (int row = 0; row < feature.rows; ++row) {
        const auto* bytes = feature.ptr<unsigned char>(row);
        for (std::size_t i = 0; i < static_cast<std::size_t>(feature.cols) * feature.elemSize(); ++i) {
            value ^= bytes[i];
            value *= 1099511628211ULL;
        }
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << value;
    return output.str();
}

// A cache is an optimization only: a missing, damaged or incompatible cache
// is discarded and reconstructed from the current eligible samples.
inline std::map<std::string, CachedSampleFeature> readFeatureCache(
    const std::filesystem::path& path, const std::string& contract,
    const std::function<bool(const cv::Mat&)>& validFeature) {
    std::map<std::string, CachedSampleFeature> result;
    if (!std::filesystem::exists(path)) return result;
    try {
        cv::FileStorage storage(readTextFile(path), cv::FileStorage::READ | cv::FileStorage::MEMORY);
        int format = 0;
        std::string storedContract;
        storage["cache_format"] >> format;
        storage["contract"] >> storedContract;
        const auto entries = storage["samples"];
        if (!storage.isOpened() || format != 1 || storedContract != contract || !entries.isSeq()) return {};
        for (const auto& node : entries) {
            CachedSampleFeature entry;
            std::string checksum;
            node["sample_id"] >> entry.sampleId;
            node["person_id"] >> entry.personId;
            node["fingerprint"] >> entry.fingerprint;
            node["feature"] >> entry.feature;
            node["checksum"] >> checksum;
            if (entry.sampleId.empty() || entry.personId.empty() || entry.fingerprint.empty()
                || !validFeature(entry.feature) || checksum != featureChecksum(entry.feature)
                || !result.emplace(entry.sampleId, entry).second) {
                return {};
            }
        }
        return result;
    } catch (const std::exception&) {
        return {};
    }
}

inline CachedFeatureUpdate updateFeatureCache(
    const std::vector<LabeledFaceSample>& samples, const std::filesystem::path& path,
    const std::string& contract, bool aligned,
    const std::function<cv::Mat(const LabeledFaceSample&)>& extract,
    const std::function<bool(const cv::Mat&)>& validFeature) {
    if (samples.empty()) throw std::invalid_argument("Incremental training set is empty");
    if (path.empty() || contract.empty()) throw std::invalid_argument("Incremental cache contract is empty");
    const auto previous = readFeatureCache(path, contract, validFeature);
    CachedFeatureUpdate update;
    std::set<std::string> currentIds;
    update.entries.reserve(samples.size());
    for (const auto& sample : samples) {
        const auto& fingerprint = aligned ? sample.alignedFingerprint : sample.grayFingerprint;
        if (sample.sampleId.empty() || sample.personId.empty() || fingerprint.empty()
            || !currentIds.insert(sample.sampleId).second) {
            throw std::invalid_argument("Incremental sample ID, identity or fingerprint is invalid");
        }
        const auto found = previous.find(sample.sampleId);
        CachedSampleFeature entry{sample.sampleId, sample.personId, fingerprint, {}};
        if (found != previous.end() && found->second.personId == sample.personId
            && found->second.fingerprint == fingerprint) {
            entry.feature = found->second.feature;
            ++update.stats.reused;
        } else {
            entry.feature = extract(sample);
            if (!validFeature(entry.feature)) throw std::runtime_error("Extracted feature violates cache contract");
            entry.feature = entry.feature.clone();
            ++update.stats.recomputed;
        }
        update.entries.push_back(std::move(entry));
    }
    for (const auto& old : previous) if (currentIds.count(old.first) == 0) ++update.stats.removed;

    cv::FileStorage storage("", cv::FileStorage::WRITE | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_YAML);
    storage << "cache_format" << 1 << "contract" << contract << "samples" << "[";
    for (const auto& entry : update.entries) {
        storage << "{" << "sample_id" << entry.sampleId << "person_id" << entry.personId
                << "fingerprint" << entry.fingerprint << "feature" << entry.feature
                << "checksum" << featureChecksum(entry.feature) << "}";
    }
    storage << "]";
    writeTextFile(path, storage.releaseAndGetString());
    return update;
}

}  // namespace facecore::detail
