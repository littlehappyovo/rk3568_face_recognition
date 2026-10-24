#include "facecore/rknn.hpp"

#include "io_internal.hpp"
#include "rknn_session.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace facecore {
namespace {

std::string readTextBestEffort(const std::filesystem::path& file) {
    std::ifstream input(file, std::ios::binary);
    if (!input) return {};
    std::ostringstream buffer;
    buffer << input.rdbuf();
    std::string text = buffer.str();
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
    return text;
}

std::vector<std::filesystem::path> libraryCandidates() {
    std::vector<std::filesystem::path> candidates;
    if (const char* configured = std::getenv("FACEAPP_RKNN_LIBRARY")) {
        if (*configured != '\0') candidates.emplace_back(configured);
    }
    candidates.emplace_back("/lib/librknnrt.so");
    candidates.emplace_back("/usr/lib/librknnrt.so");
    candidates.emplace_back("/usr/lib/aarch64-linux-gnu/librknnrt.so");
    candidates.emplace_back("/usr/local/lib/librknnrt.so");
    return candidates;
}

// librknnrt 自己会把版本字符串编进 .so，例如
// "librknnrt version: 2.3.2 (429f97ae6b@2025-04-09T09:09:27)"。
std::string extractLibraryVersion(const std::filesystem::path& library) {
    std::ifstream input(library, std::ios::binary);
    if (!input) return {};
    const std::string bytes((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    static const std::string marker = "librknnrt version: ";
    const std::size_t position = bytes.find(marker);
    if (position == std::string::npos) return {};
    const std::size_t start = position + marker.size();
    std::size_t end = start;
    while (end < bytes.size() && bytes[end] != '\0' && bytes[end] != '\n'
           && end - start < 120) {
        ++end;
    }
    return bytes.substr(start, end - start);
}

std::string valueAfterColon(const std::string& line) {
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) return line;
    std::string value = line.substr(colon + 1);
    const std::size_t first = value.find_first_not_of(" \t");
    if (first == std::string::npos) return {};
    return value.substr(first);
}

std::string processGroups() {
    std::ifstream status("/proc/self/status");
    if (!status) return {};
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("Groups:", 0) == 0) return valueAfterColon(line);
    }
    return {};
}

}  // namespace

bool rknnBackendCompiledIn() noexcept {
#if FACEAPP_HAVE_RKNN
    return true;
#else
    return false;
#endif
}

RknnRuntimeInfo probeRknnRuntime() {
    RknnRuntimeInfo info;
    info.compiledIn = rknnBackendCompiledIn();

    for (const std::filesystem::path& candidate : libraryCandidates()) {
        std::error_code error;
        if (!std::filesystem::is_regular_file(candidate, error)) continue;
        info.libraryAvailable = true;
        info.libraryPath = detail::pathToUtf8(candidate);
        info.librarySize = static_cast<std::uint64_t>(std::filesystem::file_size(candidate, error));
        info.libraryVersion = extractLibraryVersion(candidate);
        break;
    }

    if (info.libraryAvailable) {
#if !FACEAPP_HAVE_RKNN
        info.notes += "librknnrt.so was found but this build has no RKNN support compiled in; ";
#endif
    } else if (info.compiledIn) {
        info.notes += "this build links librknnrt but no librknnrt.so was found on the default "
                      "library path; ";
    } else {
        info.notes += "librknnrt.so not found; ";
    }

    const std::string driverLine = readTextBestEffort("/sys/kernel/debug/rknpu/version");
    info.driverVersion = driverLine.empty() ? std::string() : valueAfterColon(driverLine);
    if (info.driverVersion.empty()) {
        info.notes += "cannot read /sys/kernel/debug/rknpu/version (needs root or rknpu debugfs); ";
    }

    info.npuLoad = valueAfterColon(readTextBestEffort("/sys/kernel/debug/rknpu/load"));
    info.userGroups = processGroups();
    if (!info.userGroups.empty() && info.userGroups.find("110") == std::string::npos) {
        info.notes += "the current user may not be in the render group (needed for /dev/dri); ";
    }
    return info;
}

std::string describeRknnRuntime(const RknnRuntimeInfo& info) {
    std::ostringstream text;
    text << "RKNN backend compiled in : " << (info.compiledIn ? "yes" : "no") << '\n';
    text << "librknnrt.so available   : " << (info.libraryAvailable ? "yes" : "no");
    if (info.libraryAvailable) {
        text << " (" << info.libraryPath << ", " << info.librarySize << " bytes)";
    }
    text << '\n';
    text << "librknnrt.so version     : "
         << (info.libraryVersion.empty() ? "(unknown)" : info.libraryVersion) << '\n';
    text << "RKNPU driver version     : "
         << (info.driverVersion.empty() ? "(unavailable)" : info.driverVersion) << '\n';
    text << "NPU load                 : "
         << (info.npuLoad.empty() ? "(unavailable)" : info.npuLoad) << '\n';
    text << "process groups           : "
         << (info.userGroups.empty() ? "(unavailable)" : info.userGroups) << '\n';
    if (!info.notes.empty()) text << "notes                    : " << info.notes << '\n';
    return text.str();
}

RknnModelReport inspectRknnModel(const std::filesystem::path& modelPath) {
    RknnModelReport report;
    report.modelPath = detail::pathToUtf8(modelPath);
    std::error_code error;
    if (std::filesystem::is_regular_file(modelPath, error)) {
        report.modelBytes = static_cast<std::uint64_t>(std::filesystem::file_size(modelPath, error));
    }

    RknnSession session;
    session.load(modelPath);
    report.apiVersion = session.apiVersion();
    report.driverVersion = session.driverVersion();
    report.inputs.reserve(session.inputs().size());
    for (const RknnTensorDesc& desc : session.inputs()) report.inputs.push_back(toTensorReport(desc));
    report.outputs.reserve(session.outputs().size());
    for (const RknnTensorDesc& desc : session.outputs()) report.outputs.push_back(toTensorReport(desc));
    return report;
}

std::string describeRknnModel(const RknnModelReport& report) {
    std::ostringstream text;
    text << "model                    : " << report.modelPath << " (" << report.modelBytes
         << " bytes)\n";
    text << "rknn api version         : "
         << (report.apiVersion.empty() ? "(unknown)" : report.apiVersion) << '\n';
    text << "rknn driver version      : "
         << (report.driverVersion.empty() ? "(unknown)" : report.driverVersion) << '\n';
    text << "inputs (" << report.inputs.size() << "):\n";
    for (const RknnTensorReport& tensor : report.inputs) {
        text << "  - " << tensor.name << " shape=" << tensor.shape << " type=" << tensor.dataType
             << " format=" << tensor.format << " quant=" << tensor.quantization
             << " elements=" << tensor.elements << " bytes=" << tensor.bytes << '\n';
    }
    text << "outputs (" << report.outputs.size() << "):\n";
    for (const RknnTensorReport& tensor : report.outputs) {
        text << "  - " << tensor.name << " shape=" << tensor.shape << " type=" << tensor.dataType
             << " format=" << tensor.format << " quant=" << tensor.quantization
             << " elements=" << tensor.elements << " bytes=" << tensor.bytes << '\n';
    }
    return text.str();
}

}  // namespace facecore
