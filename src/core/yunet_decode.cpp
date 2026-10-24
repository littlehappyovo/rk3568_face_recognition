#include "facecore/yunet_decode.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>

namespace facecore {
namespace {

std::string dimsToString(const std::vector<int>& dims) {
    std::string text = "[";
    for (std::size_t index = 0; index < dims.size(); ++index) {
        if (index != 0) text += ',';
        text += std::to_string(dims[index]);
    }
    return text + "]";
}

std::string lowerCopy(const std::string& value) {
    std::string result = value;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return result;
}

bool containsAny(const std::string& haystack, std::initializer_list<const char*> needles) {
    for (const char* needle : needles) {
        if (haystack.find(needle) != std::string::npos) return true;
    }
    return false;
}

// 把 RKNN 报告的维度规约成 (count, channels, interleaved)。
// 支持 [N,C]、[C,N]、[1,N,C]、[1,C,N]、[1,C1,H,W,C2]（NC1HWC2 内存布局）。
bool reduceHeadShape(const std::vector<int>& dims, int* count, int* channels, bool* interleaved) {
    std::vector<int> trimmed = dims;
    while (trimmed.size() > 2 && trimmed.front() == 1) trimmed.erase(trimmed.begin());
    while (trimmed.size() > 2 && trimmed.back() == 1) trimmed.pop_back();
    if (trimmed.size() == 2) {
        const int first = trimmed[0];
        const int second = trimmed[1];
        if (first <= 0 || second <= 0) return false;
        if (second == 1 || second == 4 || second == 10) {
            *count = first;
            *channels = second;
            *interleaved = true;
            return true;
        }
        if (first == 1 || first == 4 || first == 10) {
            *count = second;
            *channels = first;
            *interleaved = false;
            return true;
        }
        // 两个维度都很大：按 [N,C] 处理，C 由邻近 head 推断，此处失败。
        return false;
    }
    if (trimmed.size() == 4) {
        // NC1HWC2: [C1, H, W, C2] -> count = H*W, channels = C1*C2。
        // 内存顺序 ((c1*H + h)*W + w)*C2 + c2 等价于 (h*W + w)*(C1*C2) + (c1*C2 + c2)。
        const int c1 = trimmed[0];
        const int height = trimmed[1];
        const int width = trimmed[2];
        const int c2 = trimmed[3];
        if (c1 <= 0 || height <= 0 || width <= 0 || c2 <= 0) return false;
        *count = height * width;
        *channels = c1 * c2;
        *interleaved = true;
        return true;
    }
    return false;
}

float headValue(const YunetRawHead& head, int row, int channel) {
    const std::size_t index = head.interleaved
                                  ? static_cast<std::size_t>(row) * head.channels + channel
                                  : static_cast<std::size_t>(channel) * head.count + row;
    return head.data[index];
}

float intersectionOverUnion(const cv::Rect2f& first, const cv::Rect2f& second) {
    const float left = std::max(first.x, second.x);
    const float top = std::max(first.y, second.y);
    const float right = std::min(first.x + first.width, second.x + second.width);
    const float bottom = std::min(first.y + first.height, second.y + second.height);
    const float overlapWidth = right - left;
    const float overlapHeight = bottom - top;
    if (overlapWidth <= 0.0F || overlapHeight <= 0.0F) return 0.0F;
    const float intersection = overlapWidth * overlapHeight;
    const float unionArea = first.width * first.height + second.width * second.height - intersection;
    if (unionArea <= 0.0F) return 0.0F;
    return intersection / unionArea;
}

}  // namespace

cv::Size alignYunetNetworkInput(cv::Size networkSize) {
    if (networkSize.width <= 0 || networkSize.height <= 0) {
        throw std::invalid_argument("alignYunetNetworkInput: network size must be positive");
    }
    const auto align32 = [](int value) { return ((std::max(1, value) + 31) / 32) * 32; };
    return cv::Size(align32(networkSize.width), align32(networkSize.height));
}

YunetLetterbox makeYunetNetworkInput(const cv::Mat& frame, cv::Size networkSize) {
    if (frame.empty()) throw std::invalid_argument("makeYunetNetworkInput: frame is empty");
    if (networkSize.width <= 0 || networkSize.height <= 0) {
        throw std::invalid_argument("makeYunetNetworkInput: network size must be positive");
    }
    cv::Mat bgr;
    if (frame.type() == CV_8UC3) {
        bgr = frame;
    } else if (frame.type() == CV_8UC1) {
        cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
    } else if (frame.type() == CV_8UC4) {
        cv::cvtColor(frame, bgr, cv::COLOR_BGRA2BGR);
    } else {
        throw std::invalid_argument("makeYunetNetworkInput: frame must be 8-bit gray, BGR or BGRA");
    }

    YunetLetterbox letterbox;
    letterbox.scale = std::min(static_cast<double>(networkSize.width) / bgr.cols,
                               static_cast<double>(networkSize.height) / bgr.rows);
    const int width = std::max(1, static_cast<int>(std::lround(bgr.cols * letterbox.scale)));
    const int height = std::max(1, static_cast<int>(std::lround(bgr.rows * letterbox.scale)));
    letterbox.canvas = cv::Mat::zeros(networkSize, CV_8UC3);
    if (width == bgr.cols && height == bgr.rows) {
        bgr.copyTo(letterbox.canvas(cv::Rect(0, 0, width, height)));
    } else {
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);
        resized.copyTo(letterbox.canvas(cv::Rect(0, 0, width, height)));
    }
    return letterbox;
}

void clampDetectionsToFrame(std::vector<FaceDetection>& detections, cv::Size frameSize) {
    std::vector<FaceDetection> kept;
    kept.reserve(detections.size());
    for (FaceDetection& detection : detections) {
        const float left = std::max(0.0F, detection.box.x);
        const float top = std::max(0.0F, detection.box.y);
        const float right = std::min(static_cast<float>(frameSize.width),
                                     detection.box.x + detection.box.width);
        const float bottom = std::min(static_cast<float>(frameSize.height),
                                      detection.box.y + detection.box.height);
        if (right - left < 1.0F || bottom - top < 1.0F) continue;
        detection.box = cv::Rect2f(left, top, right - left, bottom - top);
        kept.push_back(detection);
    }
    detections = std::move(kept);
}

std::vector<int> greedyNms(const std::vector<cv::Rect2f>& boxes,
                           const std::vector<float>& scores,
                           float iouThreshold,
                           int topK) {
    if (boxes.size() != scores.size()) {
        throw std::invalid_argument("greedyNms: boxes and scores must have the same length");
    }
    if (boxes.empty()) return {};
    if (!std::isfinite(iouThreshold) || iouThreshold < 0.0F || iouThreshold > 1.0F) {
        throw std::invalid_argument("greedyNms: iouThreshold must be within [0, 1]");
    }

    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&scores](int first, int second) {
        return scores[first] > scores[second];
    });

    std::vector<int> kept;
    std::vector<bool> suppressed(boxes.size(), false);
    for (std::size_t position = 0; position < order.size(); ++position) {
        const int candidate = order[position];
        if (suppressed[candidate]) continue;
        kept.push_back(candidate);
        if (topK > 0 && static_cast<int>(kept.size()) >= topK) break;
        for (std::size_t other = position + 1; other < order.size(); ++other) {
            const int competitor = order[other];
            if (suppressed[competitor]) continue;
            if (intersectionOverUnion(boxes[candidate], boxes[competitor]) > iouThreshold) {
                suppressed[competitor] = true;
            }
        }
    }
    return kept;
}

YunetHeadMapping classifyYunetHeads(std::vector<YunetRawHead>& heads,
                                    int inputWidth,
                                    int inputHeight) {
    if (inputWidth <= 0 || inputHeight <= 0) {
        throw std::invalid_argument("classifyYunetHeads: network input size must be positive");
    }

    struct Resolved {
        const YunetRawHead* head{nullptr};
        int count{0};
        int channels{0};
        std::string role;  // bbox | kps | score | unknown
    };
    std::vector<Resolved> resolved;
    resolved.reserve(heads.size());
    for (YunetRawHead& head : heads) {
        Resolved item;
        item.head = &head;
        int count = 0;
        int channels = 0;
        bool interleaved = true;
        if (reduceHeadShape(head.dims, &count, &channels, &interleaved)) {
            head.count = count;
            head.channels = channels;
            head.interleaved = interleaved;
        }
        item.count = head.count;
        item.channels = head.channels;

        const std::string name = lowerCopy(head.name);
        if (containsAny(name, {"bbox", "box", "reg"})) {
            item.role = "bbox";
        } else if (containsAny(name, {"kps", "landmark", "keypoint", "pts"})) {
            item.role = "kps";
        } else if (containsAny(name, {"cls", "obj", "score", "conf"})) {
            item.role = "score";
        } else {
            item.role = "unknown";
        }
        resolved.push_back(item);
    }

    // 名称没给足信息时，退回按通道数归类。
    for (Resolved& item : resolved) {
        if (item.role != "unknown") continue;
        if (item.channels == 4) {
            item.role = "bbox";
        } else if (item.channels == 10) {
            item.role = "kps";
        } else if (item.channels == 1) {
            item.role = "score";
        }
    }

    YunetHeadMapping mapping;
    // 按 count 分组（同一 count 属于同一个 stride）。
    std::vector<int> counts;
    for (const Resolved& item : resolved) {
        if (item.role == "unknown") {
            mapping.notes.push_back("忽略无法归类的输出 " + item.head->name + " dims=" +
                                    dimsToString(item.head->dims));
            continue;
        }
        if (std::find(counts.begin(), counts.end(), item.count) == counts.end()) {
            counts.push_back(item.count);
        }
    }
    std::sort(counts.begin(), counts.end(), std::greater<int>());

    for (int count : counts) {
        YunetStrideHeads stride;
        stride.count = count;
        for (const Resolved& item : resolved) {
            if (item.count != count) continue;
            if (item.role == "bbox" && stride.bbox == nullptr) {
                stride.bbox = item.head;
            } else if (item.role == "kps" && stride.kps == nullptr) {
                stride.kps = item.head;
            } else if (item.role == "score") {
                stride.scores.push_back(item.head);
            }
        }
        if (stride.bbox == nullptr || stride.scores.empty()) {
            mapping.notes.push_back("忽略候选数 " + std::to_string(count) +
                                    " 的输出组：缺少 bbox 或 score head");
            continue;
        }

        // count = rows * columns，且 rows/columns 与输入尺寸成整数比例。
        const double rawStride = std::sqrt(static_cast<double>(inputWidth) * inputHeight / count);
        int strideValue = static_cast<int>(std::lround(rawStride));
        if (strideValue <= 0) {
            mapping.notes.push_back("候选数 " + std::to_string(count) + " 无法推导 stride");
            continue;
        }
        if (inputWidth % strideValue != 0 || inputHeight % strideValue != 0
            || (inputWidth / strideValue) * (inputHeight / strideValue) != count) {
            // 输入尺寸不是 stride 的整数倍时，按整除向下取整处理（与 OpenCV 一致）。
            const int columns = inputWidth / strideValue;
            const int rows = inputHeight / strideValue;
            mapping.notes.push_back("候选数 " + std::to_string(count) + " 与推断 stride " +
                                    std::to_string(strideValue) + " 的网格 " +
                                    std::to_string(columns) + "x" + std::to_string(rows) +
                                    " 不一致，按整除网格解码");
        }
        stride.stride = strideValue;
        stride.columns = inputWidth / strideValue;
        if (stride.columns <= 0) {
            mapping.notes.push_back("候选数 " + std::to_string(count) + " 的网格宽度为 0");
            continue;
        }
        mapping.strides.push_back(stride);
    }

    std::sort(mapping.strides.begin(), mapping.strides.end(),
              [](const YunetStrideHeads& first, const YunetStrideHeads& second) {
                  return first.stride < second.stride;
              });
    if (mapping.strides.empty()) {
        std::string detail;
        for (const YunetRawHead& head : heads) {
            if (!detail.empty()) detail += ", ";
            detail += head.name + dimsToString(head.dims);
        }
        throw std::runtime_error(
            "Cannot map the RKNN outputs to YuNet heads; observed outputs: " + detail +
            ". Expected bbox(Nx4), kps(Nx10) and cls/obj(Nx1) tensors.");
    }
    return mapping;
}

std::vector<FaceDetection> decodeYunetHeads(const YunetHeadMapping& mapping,
                                            const YunetDecodeOptions& options) {
    if (!std::isfinite(options.inverseScale) || options.inverseScale <= 0.0F) {
        throw std::invalid_argument("decodeYunetHeads: inverseScale must be positive");
    }

    std::vector<cv::Rect2f> boxes;
    std::vector<std::array<cv::Point2f, 5>> landmarks;
    std::vector<float> scores;

    for (const YunetStrideHeads& stride : mapping.strides) {
        const float strideValue = static_cast<float>(stride.stride);
        for (int index = 0; index < stride.count; ++index) {
            const int row = index / stride.columns;
            const int column = index % stride.columns;

            float score = 1.0F;
            if (options.objectnessMode == 2 || stride.scores.size() < 2) {
                score = headValue(*stride.scores.front(), index, 0);
            } else {
                const float cls = std::max(0.0F, headValue(*stride.scores[0], index, 0));
                const float obj = std::max(0.0F, headValue(*stride.scores[1], index, 0));
                score = std::sqrt(cls * obj);
            }
            if (!std::isfinite(score) || score < options.scoreThreshold) continue;

            const float dx = headValue(*stride.bbox, index, 0);
            const float dy = headValue(*stride.bbox, index, 1);
            const float dw = headValue(*stride.bbox, index, 2);
            const float dh = headValue(*stride.bbox, index, 3);
            if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dw)
                || !std::isfinite(dh)) {
                continue;
            }
            const float centerX = (static_cast<float>(column) + dx) * strideValue;
            const float centerY = (static_cast<float>(row) + dy) * strideValue;
            const float width = std::exp(dw) * strideValue;
            const float height = std::exp(dh) * strideValue;
            if (!std::isfinite(width) || !std::isfinite(height) || width <= 0.0F
                || height <= 0.0F) {
                continue;
            }

            std::array<cv::Point2f, 5> points{};
            if (stride.kps != nullptr) {
                bool valid = true;
                for (int point = 0; point < 5; ++point) {
                    const float offsetX = headValue(*stride.kps, index, point * 2);
                    const float offsetY = headValue(*stride.kps, index, point * 2 + 1);
                    if (!std::isfinite(offsetX) || !std::isfinite(offsetY)) {
                        valid = false;
                        break;
                    }
                    points[static_cast<std::size_t>(point)] =
                        cv::Point2f((static_cast<float>(column) + offsetX) * strideValue,
                                    (static_cast<float>(row) + offsetY) * strideValue);
                }
                if (!valid) continue;
            }

            boxes.emplace_back(centerX - width * 0.5F, centerY - height * 0.5F, width, height);
            landmarks.push_back(points);
            scores.push_back(score);
        }
    }

    std::vector<FaceDetection> output;
    if (boxes.empty()) return output;
    const std::vector<int> kept = greedyNms(boxes, scores, options.nmsThreshold, options.topK);
    output.reserve(kept.size());
    for (int index : kept) {
        FaceDetection detection;
        const cv::Rect2f& box = boxes[static_cast<std::size_t>(index)];
        detection.box = cv::Rect2f(box.x * options.inverseScale, box.y * options.inverseScale,
                                   box.width * options.inverseScale,
                                   box.height * options.inverseScale);
        for (std::size_t point = 0; point < detection.landmarks.size(); ++point) {
            const cv::Point2f& source = landmarks[static_cast<std::size_t>(index)][point];
            detection.landmarks[point] =
                cv::Point2f(source.x * options.inverseScale, source.y * options.inverseScale);
        }
        detection.score = scores[static_cast<std::size_t>(index)];
        detection.hasLandmarks = true;
        output.push_back(detection);
    }
    return output;
}

}  // namespace facecore
