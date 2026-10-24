#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace facecore {

// 这些结构体是核心算法和 Qt 应用之间共享的数据协议。核心库只依赖
// OpenCV/标准 C++，应用层拿到结果后再转换成 QImage、QRectF 等 Qt 类型。
enum class PixelFormat { Unknown, Gray8, Bgr8, Bgra8, Nv12 };
enum class PersonGroup { InGroup, OutGroup };
enum class DataSplit { Unassigned, Train, Validation, Test };
enum class RecognitionState { Known, Unknown, LowQuality, Error };

struct Frame {
    cv::Mat image;                 // 当前帧；进入算法时通常是 BGR CV_8UC3。
    PixelFormat format{PixelFormat::Unknown}; // 当前像素格式，便于调试和转换。
    std::int64_t timestampUs{0};   // 墙钟时间，采集间隔和日志使用，单位为微秒。
    std::uint64_t sequence{0};     // 单调递增序号，用于异步结果排序和丢帧判断。
    std::string source;            // 例如 v4l2-dmabuf-export:/dev/video-camera0。

    bool empty() const noexcept { return image.empty(); }
};

struct FaceDetection {
    cv::Rect2f box;                // 当前帧坐标系中的人脸框。
    std::array<cv::Point2f, 5> landmarks{};  // right eye, left eye, nose, right/left mouth corner (YuNet order)
    float score{0.0F};             // YuNet 的“是不是脸”置信度。
    bool hasLandmarks{false};      // 关键点有效时才用于仿射对齐。
};

struct TrackedFace {
    std::uint64_t trackId{0};      // 跟踪器分配的 ID，识别缓存也按它索引。
    FaceDetection detection;       // 匹配、平滑或光流预测后的结果。
    cv::Scalar colorBgr{0, 255, 0}; // UI 用固定颜色区分多人。
    int age{0};                    // 轨迹存活帧数。
    int missedFrames{0};           // 连续没有匹配到新检测的帧数。
};

struct QualityReport {
    bool accepted{false};          // 所有条件满足时才为 true。
    double brightness{0.0};        // 人脸区域平均灰度。
    double sharpness{0.0};         // 拉普拉斯方差，越大通常越清晰。
    double faceWidthPixels{0.0};   // 检测框宽度，用于过滤过远的小脸。
    bool clipped{false};           // 人脸是否有较大部分在画面外。
    std::vector<std::string> reasons; // 拒绝原因，便于界面提示和排错。
};

struct FaceSample {
    cv::Mat grayFace;        // normalized grayscale face, normally 128x128 CV_8UC1
    cv::Mat alignedBgrFace;  // aligned color face for an embedding backend
    QualityReport quality;
};

struct LabeledFaceSample {
    std::string personId;
    std::string sessionId;
    FaceSample face;
};

struct RecognitionResult {
    RecognitionState state{RecognitionState::Error};
    std::string personId;
    double bestScore{std::numeric_limits<double>::quiet_NaN()};
    double secondBestScore{std::numeric_limits<double>::quiet_NaN()};
    double threshold{std::numeric_limits<double>::quiet_NaN()};
    std::string backend;           // LBPH 是距离；embedding 是余弦相似度。
    std::string message;           // accepted 或具体的拒识/错误原因。

    bool known() const noexcept { return state == RecognitionState::Known; }
};

struct PersonRecord {
    std::string personId;
    std::string displayName;
    PersonGroup group{PersonGroup::InGroup};
    bool enabled{true};
};

struct SampleRecord {
    std::string sampleId;
    std::string personId;
    std::string sessionId;
    std::string relativePath;
    std::string alignedRelativePath;
    std::string source;
    std::int64_t timestampUs{0};
    cv::Rect2f sourceBox;
    double qualityScore{0.0};
    std::string preprocessingVersion;
    DataSplit split{DataSplit::Unassigned};
};

inline const char* toString(PersonGroup value) noexcept {
    return value == PersonGroup::InGroup ? "in" : "out";
}

inline const char* toString(DataSplit value) noexcept {
    switch (value) {
        case DataSplit::Train: return "train";
        case DataSplit::Validation: return "validation";
        case DataSplit::Test: return "test";
        default: return "unassigned";
    }
}

}  // namespace facecore
