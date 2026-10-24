#pragma once

#include "facecore/export.hpp"
#include "facecore/types.hpp"

#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace facecore {

class FACECORE_API DatasetStore {
public:
    explicit DatasetStore(std::filesystem::path root,
                          cv::Size expectedGraySize = {128, 128},
                          cv::Size expectedAlignedSize = {112, 112});

    void initialize();
    void upsertPerson(const PersonRecord& person);
    // Explicit management API; capture upsert still refuses identity relabeling.
    void updatePerson(const PersonRecord& person);
    // Remove active indexes, retain original images and a metadata archive.
    void removePerson(const std::string& personId);
    void removeSample(const std::string& personId, const std::string& sampleId);
    std::vector<PersonRecord> loadRegistry() const;

    SampleRecord saveSample(const cv::Mat& normalizedGrayFace,
                            const cv::Mat& alignedBgrFace,
                            SampleRecord metadata);
    std::vector<SampleRecord> loadManifest() const;
    cv::Mat loadGraySample(const SampleRecord& sample) const;
    cv::Mat loadAlignedSample(const SampleRecord& sample) const;

    const std::filesystem::path& root() const noexcept { return root_; }

private:
    void removeEntries(const std::string& personId,
                       const std::optional<std::string>& sampleId);
    std::filesystem::path root_;
    cv::Size expectedGraySize_;
    cv::Size expectedAlignedSize_;
    mutable std::mutex mutex_;
    std::uint64_t sequence_{0};
};

FACECORE_API bool isSafeIdentifier(const std::string& value) noexcept;

}  // namespace facecore
