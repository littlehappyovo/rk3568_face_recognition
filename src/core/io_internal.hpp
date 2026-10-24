#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace facecore::detail {

inline std::vector<unsigned char> readBinaryFile(const std::filesystem::path& file) {
    std::ifstream input(file, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open file for reading: " + file.string());
    }
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("Cannot determine file size: " + file.string());
    }
    input.seekg(0, std::ios::beg);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty() &&
        !input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("Cannot read file completely: " + file.string());
    }
    return bytes;
}

inline std::string readTextFile(const std::filesystem::path& file) {
    const auto bytes = readBinaryFile(file);
    return std::string(bytes.begin(), bytes.end());
}

inline std::filesystem::path temporarySibling(const std::filesystem::path& file) {
    static std::atomic<std::uint64_t> sequence{0};
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path temporary = file;
    temporary += ".tmp." + std::to_string(tick) + "." +
                 std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
    return temporary;
}

inline void replaceFileAtomically(const std::filesystem::path& temporary,
                                  const std::filesystem::path& destination) {
#if defined(_WIN32)
    if (!MoveFileExW(temporary.c_str(), destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD error = GetLastError();
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw std::runtime_error("Cannot replace file atomically (Windows error " +
                                 std::to_string(error) + "): " + destination.string());
    }
#else
    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    if (error) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw std::runtime_error("Cannot replace file atomically: " +
                                 destination.string() + ": " + error.message());
    }
#endif
}

inline void writeBinaryFile(const std::filesystem::path& file,
                            const unsigned char* data,
                            std::size_t size) {
    if (file.has_parent_path()) {
        std::filesystem::create_directories(file.parent_path());
    }
    const std::filesystem::path temporary = temporarySibling(file);
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Cannot open temporary file for writing: " + file.string());
    }
    if (size > 0) {
        output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }
    output.flush();
    if (!output) {
        output.close();
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw std::runtime_error("Cannot write file completely: " + file.string());
    }
    output.close();
    replaceFileAtomically(temporary, file);
}

inline void writeTextFile(const std::filesystem::path& file, const std::string& text) {
    writeBinaryFile(file, reinterpret_cast<const unsigned char*>(text.data()), text.size());
}

inline std::string pathToUtf8(const std::filesystem::path& path) {
#if defined(_WIN32)
    return path.u8string();
#else
    return path.string();
#endif
}

}  // namespace facecore::detail
