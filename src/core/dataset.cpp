#include "facecore/dataset.hpp"

#include "io_internal.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace facecore {
namespace {

// 数据集层负责人员表、样本图片和 manifest 的一致性。上层只提供 Mat 与元数据，
// 本文件负责 CSV 转义、相对路径安全校验、PNG 编解码和异常回滚。
constexpr const char* kRegistryHeader = "person_id,display_name,group,enabled\n";
constexpr const char* kManifestHeader =
    "sample_id,person_id,session_id,relative_path,aligned_relative_path,source,"
    "timestamp_us,x,y,width,height,quality_score,preprocessing_version,split\n";

std::string csvEscape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char character : value) {
        if (character == '"') escaped.push_back('"');
        escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
}

void requireSingleLine(const std::string& value, const char* field) {
    if (value.find_first_of("\r\n") != std::string::npos) {
        throw std::invalid_argument(std::string(field) + " cannot contain CR or LF");
    }
}

bool isSafeRelativeDataPath(const std::string& value, const char* expectedDirectory) {
    if (value.empty()) return false;
    const std::filesystem::path path = std::filesystem::u8path(value);
    if (path.is_absolute() || path.has_root_path()) return false;
    auto iterator = path.begin();
    if (iterator == path.end() || iterator->generic_u8string() != expectedDirectory) return false;
    for (const auto& component : path) {
        if (component == "." || component == "..") return false;
    }
    return true;
}

std::vector<std::string> parseCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char character = line[index];
        if (quoted) {
            if (character == '"') {
                if (index + 1 < line.size() && line[index + 1] == '"') {
                    field.push_back('"');
                    ++index;
                } else {
                    quoted = false;
                }
            } else {
                field.push_back(character);
            }
        } else if (character == '"' && field.empty()) {
            quoted = true;
        } else if (character == ',') {
            fields.push_back(field);
            field.clear();
        } else if (character != '\r') {
            field.push_back(character);
        }
    }
    if (quoted) throw std::runtime_error("Malformed CSV: unterminated quoted field");
    fields.push_back(field);
    return fields;
}

void ensureLayout(const std::filesystem::path& root) {
    // 初始化是幂等的：目录已存在就复用，文件不存在才写表头，不会清空用户数据。
    if (root.empty()) throw std::invalid_argument("Dataset root cannot be empty");
    std::filesystem::create_directories(root / "faces_gray");
    std::filesystem::create_directories(root / "faces_aligned");
    std::filesystem::create_directories(root / "manifests");
    const auto registry = root / "registry.csv";
    const auto manifest = root / "manifests" / "all.csv";
    if (!std::filesystem::exists(registry)) detail::writeTextFile(registry, kRegistryHeader);
    if (!std::filesystem::exists(manifest)) detail::writeTextFile(manifest, kManifestHeader);
}

std::vector<PersonRecord> readRegistryFile(const std::filesystem::path& file) {
    std::vector<PersonRecord> people;
    if (!std::filesystem::exists(file)) return people;
    std::ifstream input(file, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read registry: " + file.string());
    std::string line;
    bool header = true;
    std::size_t lineNumber = 0;
    std::set<std::string> identifiers;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (header) {
            header = false;
            if (line.rfind("person_id,", 0) == 0) continue;
        }
        if (line.empty()) continue;
        const auto fields = parseCsvLine(line);
        if (fields.size() != 4) {
            throw std::runtime_error("Registry row has wrong field count at line " +
                                     std::to_string(lineNumber));
        }
        PersonRecord person;
        person.personId = fields[0];
        person.displayName = fields[1];
        if (!isSafeIdentifier(person.personId) || person.displayName.empty()) {
            throw std::runtime_error("Invalid person record at registry line " +
                                     std::to_string(lineNumber));
        }
        if (!identifiers.insert(person.personId).second) {
            throw std::runtime_error("Duplicate personId at registry line " +
                                     std::to_string(lineNumber));
        }
        if (fields[2] == "in") person.group = PersonGroup::InGroup;
        else if (fields[2] == "out") person.group = PersonGroup::OutGroup;
        else throw std::runtime_error("Invalid person group at registry line " +
                                      std::to_string(lineNumber));
        if (fields[3] == "1" || fields[3] == "true") person.enabled = true;
        else if (fields[3] == "0" || fields[3] == "false") person.enabled = false;
        else throw std::runtime_error("Invalid enabled flag at registry line " +
                                      std::to_string(lineNumber));
        people.push_back(std::move(person));
    }
    return people;
}

void writeRegistryFile(const std::filesystem::path& file,
                       const std::vector<PersonRecord>& people) {
    std::ostringstream output;
    output << kRegistryHeader;
    for (const PersonRecord& person : people) {
        output << csvEscape(person.personId) << ',' << csvEscape(person.displayName) << ','
               << toString(person.group) << ',' << (person.enabled ? "1" : "0") << '\n';
    }
    detail::writeTextFile(file, output.str());
}

void invalidateGalleries(const std::filesystem::path& root) {
    for (const auto* backend : {"opencv_lbph", "onnx_embedding", "mobilefacenet_rknn"}) {
        detail::writeTextFile(root / (std::string(".gallery-stale-") + backend),
                              "Dataset changed; rebuild this recognition gallery.\n");
    }
}

void writePng(const std::filesystem::path& file, const cv::Mat& image) {
    std::vector<unsigned char> encoded;
    if (!cv::imencode(".png", image, encoded, {cv::IMWRITE_PNG_COMPRESSION, 3})) {
        throw std::runtime_error("OpenCV failed to encode PNG: " + file.string());
    }
    if (encoded.empty()) {
        throw std::runtime_error("OpenCV encoded an empty PNG: " + file.string());
    }
    detail::writeBinaryFile(file, encoded.data(), encoded.size());
}

cv::Mat readImage(const std::filesystem::path& file, int flags) {
    const auto bytes = detail::readBinaryFile(file);
    if (bytes.empty()) throw std::runtime_error("Image file is empty: " + file.string());
    if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Image file is too large for OpenCV decoding");
    }
    cv::Mat encoded(1, static_cast<int>(bytes.size()), CV_8UC1,
                    const_cast<unsigned char*>(bytes.data()));
    cv::Mat image = cv::imdecode(encoded, flags);
    if (image.empty()) throw std::runtime_error("Cannot decode image: " + file.string());
    return image;
}

DataSplit parseSplit(const std::string& value) {
    if (value == "train") return DataSplit::Train;
    if (value == "validation") return DataSplit::Validation;
    if (value == "test") return DataSplit::Test;
    if (value == "unassigned") return DataSplit::Unassigned;
    throw std::runtime_error("Unknown dataset split: " + value);
}

std::int64_t currentTimestampUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

}  // namespace

bool isSafeIdentifier(const std::string& value) noexcept {
    if (value.empty() || value.size() > 80) return false;
    return std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return std::isalnum(character) != 0 || character == '_' || character == '-';
    });
}

DatasetStore::DatasetStore(std::filesystem::path root,
                           cv::Size expectedGraySize,
                           cv::Size expectedAlignedSize)
    : root_(std::move(root)),
      expectedGraySize_(expectedGraySize),
      expectedAlignedSize_(expectedAlignedSize) {
    if (root_.empty()) throw std::invalid_argument("Dataset root cannot be empty");
    if (expectedGraySize_.width <= 0 || expectedGraySize_.height <= 0
        || expectedAlignedSize_.width <= 0 || expectedAlignedSize_.height <= 0) {
        throw std::invalid_argument("Expected dataset image sizes must be positive");
    }
}

void DatasetStore::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    ensureLayout(root_);
}

void DatasetStore::upsertPerson(const PersonRecord& person) {
    if (!isSafeIdentifier(person.personId)) {
        throw std::invalid_argument("personId must contain only letters, digits, '_' or '-'");
    }
    if (person.displayName.empty()) throw std::invalid_argument("displayName cannot be empty");
    requireSingleLine(person.displayName, "displayName");
    std::lock_guard<std::mutex> lock(mutex_);
    ensureLayout(root_);
    std::vector<PersonRecord> people = readRegistryFile(root_ / "registry.csv");
    const auto found = std::find_if(people.begin(), people.end(), [&](const PersonRecord& item) {
        return item.personId == person.personId;
    });
    if (found == people.end()) {
        people.push_back(person);
    } else {
        if (found->displayName != person.displayName || found->group != person.group) {
            throw std::invalid_argument(
                "Refusing to change displayName/group for an existing personId");
        }
        found->enabled = person.enabled;
    }
    std::sort(people.begin(), people.end(), [](const PersonRecord& first, const PersonRecord& second) {
        return first.personId < second.personId;
    });
    writeRegistryFile(root_ / "registry.csv", people);
}

std::vector<PersonRecord> DatasetStore::loadRegistry() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return readRegistryFile(root_ / "registry.csv");
}

void DatasetStore::updatePerson(const PersonRecord& person) {
    if (!isSafeIdentifier(person.personId) || person.displayName.empty()
        || (person.group != PersonGroup::InGroup && person.group != PersonGroup::OutGroup)) {
        throw std::invalid_argument("Invalid person profile");
    }
    requireSingleLine(person.displayName, "displayName");
    std::lock_guard<std::mutex> lock(mutex_);
    auto people = readRegistryFile(root_ / "registry.csv");
    const auto found = std::find_if(people.begin(), people.end(), [&](const auto& entry) {
        return entry.personId == person.personId;
    });
    if (found == people.end()) throw std::invalid_argument("Person does not exist");
    if (found->enabled != person.enabled) invalidateGalleries(root_);
    *found = person;
    writeRegistryFile(root_ / "registry.csv", people);
}

void DatasetStore::removePerson(const std::string& personId) {
    removeEntries(personId, std::nullopt);
}

void DatasetStore::removeSample(const std::string& personId, const std::string& sampleId) {
    if (!isSafeIdentifier(sampleId)) throw std::invalid_argument("Unsafe sample ID");
    removeEntries(personId, sampleId);
}

void DatasetStore::removeEntries(const std::string& personId,
                                 const std::optional<std::string>& sampleId) {
    if (!isSafeIdentifier(personId)) throw std::invalid_argument("Unsafe person ID");
    std::lock_guard<std::mutex> lock(mutex_);
    auto people = readRegistryFile(root_ / "registry.csv");
    const auto found = std::find_if(people.begin(), people.end(), [&](const auto& entry) {
        return entry.personId == personId;
    });
    if (found == people.end()) throw std::invalid_argument("Person does not exist");
    const auto manifestFile = root_ / "manifests" / "all.csv";
    const auto oldManifest = detail::readTextFile(manifestFile);
    std::istringstream input(oldManifest);
    std::ostringstream output;
    std::string line;
    bool removed = false;
    bool header = true;
    while (std::getline(input, line)) {
        if (header) {
            header = false;
            if (line.rfind("sample_id,", 0) == 0) { output << line << '\n'; continue; }
            throw std::runtime_error("Manifest header is invalid");
        }
        if (line.empty()) continue;
        const auto fields = parseCsvLine(line);
        if (fields.size() != 14) throw std::runtime_error("Manifest row is invalid");
        if (fields[1] == personId && (!sampleId || fields[0] == *sampleId)) {
            removed = true;
        } else {
            output << line << '\n';
        }
    }
    if (sampleId && !removed) throw std::invalid_argument("Sample does not belong to this person");
    if (header) throw std::runtime_error("Manifest is empty");

    // Archive metadata before changing indexes. Images stay at their original paths;
    // a deleted ID can be reused without importing its old samples into the new person.
    const auto archive = detail::temporarySibling(root_ / "archive" / personId);
    std::filesystem::create_directories(archive);
    detail::writeTextFile(archive / "registry.csv", detail::readTextFile(root_ / "registry.csv"));
    detail::writeTextFile(archive / "all.csv", oldManifest);
    invalidateGalleries(root_);
    detail::writeTextFile(manifestFile, output.str());
    if (!sampleId) {
        people.erase(found);
        try {
            writeRegistryFile(root_ / "registry.csv", people);
        } catch (...) {
            detail::writeTextFile(manifestFile, oldManifest);
            throw;
        }
    }
}

SampleRecord DatasetStore::saveSample(const cv::Mat& normalizedGrayFace,
                                      const cv::Mat& alignedBgrFace,
                                      SampleRecord metadata) {
    // 先校验人员和路径，再写图像，最后追加 manifest。任一步失败都会删除本次
    // 刚写入的文件，避免出现“CSV 有记录但图片为空/不存在”的半条样本。
    if (normalizedGrayFace.empty() || normalizedGrayFace.type() != CV_8UC1) {
        throw std::invalid_argument("Normalized sample must be a non-empty CV_8UC1 image");
    }
    if (normalizedGrayFace.size() != expectedGraySize_) {
        throw std::invalid_argument("Normalized gray sample has the wrong fixed dimensions");
    }
    if (!alignedBgrFace.empty() && alignedBgrFace.type() != CV_8UC3) {
        throw std::invalid_argument("Aligned sample must be empty or CV_8UC3");
    }
    if (!alignedBgrFace.empty() && alignedBgrFace.size() != expectedAlignedSize_) {
        throw std::invalid_argument("Aligned color sample has the wrong fixed dimensions");
    }
    if (!isSafeIdentifier(metadata.personId) || !isSafeIdentifier(metadata.sessionId)) {
        throw std::invalid_argument("Unsafe personId or sessionId");
    }
    if (metadata.preprocessingVersion.empty()) {
        throw std::invalid_argument("preprocessingVersion cannot be empty");
    }
    requireSingleLine(metadata.source, "source");
    requireSingleLine(metadata.preprocessingVersion, "preprocessingVersion");
    if (!std::isfinite(metadata.qualityScore)
        || !std::isfinite(metadata.sourceBox.x) || !std::isfinite(metadata.sourceBox.y)
        || !std::isfinite(metadata.sourceBox.width)
        || !std::isfinite(metadata.sourceBox.height)) {
        throw std::invalid_argument("Sample metadata contains a non-finite number");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    ensureLayout(root_);
    const std::vector<PersonRecord> people = readRegistryFile(root_ / "registry.csv");
    const auto registered = std::find_if(people.begin(), people.end(), [&](const auto& item) {
        return item.personId == metadata.personId && item.enabled;
    });
    if (registered == people.end()) {
        throw std::invalid_argument("Sample personId is not registered and enabled");
    }
    metadata.timestampUs = metadata.timestampUs == 0 ? currentTimestampUs() : metadata.timestampUs;
    if (metadata.sampleId.empty()) {
        metadata.sampleId = metadata.personId + "_" + std::to_string(metadata.timestampUs) + "_" +
                            std::to_string(sequence_++);
    }
    if (!isSafeIdentifier(metadata.sampleId)) throw std::invalid_argument("Unsafe sampleId");

    const std::filesystem::path grayRelative = std::filesystem::path("faces_gray") /
                                               metadata.personId / metadata.sessionId /
                                               (metadata.sampleId + ".png");
    const std::filesystem::path alignedRelative = std::filesystem::path("faces_aligned") /
                                                  metadata.personId / metadata.sessionId /
                                                  (metadata.sampleId + ".png");
    const auto grayAbsolute = root_ / grayRelative;
    const auto alignedAbsolute = root_ / alignedRelative;
    if (std::filesystem::exists(grayAbsolute) ||
        (!alignedBgrFace.empty() && std::filesystem::exists(alignedAbsolute))) {
        throw std::runtime_error("Refusing to overwrite an existing dataset sample: " +
                                 metadata.sampleId);
    }

    writePng(grayAbsolute, normalizedGrayFace);
    try {
        if (!alignedBgrFace.empty()) writePng(alignedAbsolute, alignedBgrFace);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(grayAbsolute, ignored);
        throw;
    }
    metadata.relativePath = grayRelative.generic_u8string();
    metadata.alignedRelativePath = alignedBgrFace.empty()
                                       ? std::string()
                                       : alignedRelative.generic_u8string();

    const std::filesystem::path manifestFile = root_ / "manifests" / "all.csv";
    std::ostringstream row;
    row << std::setprecision(17) << csvEscape(metadata.sampleId) << ','
        << csvEscape(metadata.personId) << ',' << csvEscape(metadata.sessionId) << ','
        << csvEscape(metadata.relativePath) << ',' << csvEscape(metadata.alignedRelativePath)
        << ',' << csvEscape(metadata.source) << ',' << metadata.timestampUs << ','
        << metadata.sourceBox.x << ',' << metadata.sourceBox.y << ','
        << metadata.sourceBox.width << ',' << metadata.sourceBox.height << ','
        << metadata.qualityScore << ',' << csvEscape(metadata.preprocessingVersion) << ','
        << toString(metadata.split) << '\n';
    try {
        std::string manifest = detail::readTextFile(manifestFile);
        if (!manifest.empty() && manifest.back() != '\n') {
            throw std::runtime_error("Dataset manifest is truncated (missing final newline)");
        }
        manifest += row.str();
        detail::writeTextFile(manifestFile, manifest);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(grayAbsolute, ignored);
        if (!alignedBgrFace.empty()) std::filesystem::remove(alignedAbsolute, ignored);
        throw;
    }
    return metadata;
}

std::vector<SampleRecord> DatasetStore::loadManifest() const {
    // manifest 是“图片文件”和“人员标签”的索引。读取时严格检查列数、重复
    // sampleId、相对路径和文件存在性，训练阶段才能安全地跳过坏记录而不是读到
    // 空 Mat 后才在模型内部报错。
    std::lock_guard<std::mutex> lock(mutex_);
    const auto file = root_ / "manifests" / "all.csv";
    std::vector<SampleRecord> records;
    if (!std::filesystem::exists(file)) return records;
    std::ifstream input(file, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read manifest: " + file.string());
    std::string line;
    std::size_t lineNumber = 0;
    bool header = true;
    std::set<std::string> sampleIds;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (header) {
            header = false;
            if (line.rfind("sample_id,", 0) == 0) continue;
        }
        if (line.empty()) continue;
        const auto fields = parseCsvLine(line);
        if (fields.size() != 14) {
            throw std::runtime_error("Manifest row has wrong field count at line " +
                                     std::to_string(lineNumber));
        }
        try {
            SampleRecord record;
            record.sampleId = fields[0];
            record.personId = fields[1];
            record.sessionId = fields[2];
            record.relativePath = fields[3];
            record.alignedRelativePath = fields[4];
            record.source = fields[5];
            record.timestampUs = std::stoll(fields[6]);
            record.sourceBox.x = std::stof(fields[7]);
            record.sourceBox.y = std::stof(fields[8]);
            record.sourceBox.width = std::stof(fields[9]);
            record.sourceBox.height = std::stof(fields[10]);
            record.qualityScore = std::stod(fields[11]);
            record.preprocessingVersion = fields[12];
            record.split = parseSplit(fields[13]);
            if (!isSafeIdentifier(record.sampleId) || !isSafeIdentifier(record.personId)
                || !isSafeIdentifier(record.sessionId)
                || !isSafeRelativeDataPath(record.relativePath, "faces_gray")
                || (!record.alignedRelativePath.empty()
                    && !isSafeRelativeDataPath(record.alignedRelativePath, "faces_aligned"))
                || record.preprocessingVersion.empty()
                || !std::isfinite(record.qualityScore)
                || !std::isfinite(record.sourceBox.x) || !std::isfinite(record.sourceBox.y)
                || !std::isfinite(record.sourceBox.width)
                || !std::isfinite(record.sourceBox.height)) {
                throw std::runtime_error("Manifest row violates the dataset contract");
            }
            if (!sampleIds.insert(record.sampleId).second) {
                throw std::runtime_error("Manifest contains a duplicate sampleId");
            }
            records.push_back(std::move(record));
        } catch (const std::exception& exception) {
            throw std::runtime_error("Invalid manifest value at line " +
                                     std::to_string(lineNumber) + ": " + exception.what());
        }
    }
    return records;
}

cv::Mat DatasetStore::loadGraySample(const SampleRecord& sample) const {
    if (!isSafeRelativeDataPath(sample.relativePath, "faces_gray")) {
        throw std::invalid_argument("Sample has an unsafe gray path");
    }
    cv::Mat image = readImage(root_ / std::filesystem::u8path(sample.relativePath),
                              cv::IMREAD_GRAYSCALE);
    if (image.type() != CV_8UC1 || image.size() != expectedGraySize_) {
        throw std::runtime_error("Stored gray sample violates the fixed-size contract");
    }
    return image;
}

cv::Mat DatasetStore::loadAlignedSample(const SampleRecord& sample) const {
    if (!isSafeRelativeDataPath(sample.alignedRelativePath, "faces_aligned")) {
        throw std::invalid_argument("Sample has an unsafe aligned color path");
    }
    cv::Mat image = readImage(root_ / std::filesystem::u8path(sample.alignedRelativePath),
                              cv::IMREAD_COLOR);
    if (image.type() != CV_8UC3 || image.size() != expectedAlignedSize_) {
        throw std::runtime_error("Stored aligned sample violates the fixed-size contract");
    }
    return image;
}

}  // namespace facecore
