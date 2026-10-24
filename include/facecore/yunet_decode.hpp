#pragma once

#include "facecore/export.hpp"
#include "facecore/types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace facecore {

// 一个 YuNet 原始输出头（已转成 float32 展平）。
//
// YuNet 2023mar 有 12 个输出：bbox_{8,16,32}、cls_{8,16,32}、kps_{8,16,32}、
// obj_{8,16,32}。RKNN 转换后张量名和顺序都可能变化，因此这里只按“形状”归类：
//   channels == 4  -> bbox
//   channels == 10 -> kps（5 个点）
//   channels == 1  -> score（cls 或 obj，两者乘积开方即为置信度）
// 这种归类方式对 cls/obj 的顺序不敏感，因为 sqrt(cls*obj) 是对称的。
struct YunetRawHead {
    std::vector<float> data;
    int count{0};        // 候选框数量，通常等于 特征图 H*W
    int channels{0};     // 每个候选框的通道数
    // true 表示内存按 [i][channel + c] 排列（dims 尾部是通道）；
    // false 表示按 [channel][i] 排列（dims 头部是通道）。
    bool interleaved{true};
    std::vector<int> dims;
    std::string name;
};

struct YunetStrideHeads {
    int stride{0};
    int count{0};
    int columns{0};  // 特征图宽度方向的格子数；index = row*columns + column
    const YunetRawHead* bbox{nullptr};
    const YunetRawHead* kps{nullptr};
    std::vector<const YunetRawHead*> scores;
};

struct YunetHeadMapping {
    std::vector<YunetStrideHeads> strides;
    std::vector<std::string> notes;
};

// 按形状把输出头分组到各 stride，并推导出 stride 与特征图宽度。
// inputWidth/inputHeight 是网络输入尺寸（例如 320x320）。
// heads 会被就地补全 count / channels / interleaved；data 可以在之后填充，
// 因为映射里保存的是指向这些 head 的指针（heads 不得再被扩容）。
FACECORE_API YunetHeadMapping classifyYunetHeads(std::vector<YunetRawHead>& heads,
                                                 int inputWidth,
                                                 int inputHeight);

struct YunetDecodeOptions {
    float scoreThreshold{0.85F};
    float nmsThreshold{0.30F};
    int topK{5000};
    // 网络输入坐标 -> 原图坐标的缩放系数（等比例缩放时等于 1/scale）。
    float inverseScale{1.0F};
    // 0 自动：有 2 个 1 通道 head 时取 sqrt(cls*obj)，只有 1 个时直接用它。
    // 1 强制取乘积。2 只用第一个 head。
    int objectnessMode{0};
};

// 解码 + 置信度融合 + NMS，返回的原图坐标不做边界裁剪（由调用方按帧尺寸裁剪）。
FACECORE_API std::vector<FaceDetection> decodeYunetHeads(const YunetHeadMapping& mapping,
                                                         const YunetDecodeOptions& options);

// 独立的 IoU 贪心 NMS，便于单元测试，也避免依赖 OpenCV dnn 模块。
FACECORE_API std::vector<int> greedyNms(const std::vector<cv::Rect2f>& boxes,
                                        const std::vector<float>& scores,
                                        float iouThreshold,
                                        int topK);

// 计算检测网络的实际输入尺寸：向上对齐到 32 的倍数。
//
// YuNet 是 FPN 结构，网络输入尺寸必须是 32 的整数倍，否则 stride 16 与 stride 32
// 两条分支的特征图会差 1 像素，Eltwise 直接断言失败（OpenCV 4.5.4 实测：
// 2048x1150 帧报 input[0]=[1,64,71,128] vs input[1]=[1,64,70,128]）。
FACECORE_API cv::Size alignYunetNetworkInput(cv::Size networkSize);

// 等比缩放到网络输入尺寸并在右下补零的结果。
//
// CPU（OpenCV DNN）与 NPU（RKNN）两条通路共用这一套预处理。注意 OpenCV 4.5.4 的
// FaceDetectorYN::detect 会断言传入图像尺寸必须等于网络输入尺寸，它**不会**自己做
// letterbox，所以缩放与补边必须由调用方完成。
struct YunetLetterbox {
    cv::Mat canvas;       // CV_8UC3，尺寸等于 networkSize
    double scale{1.0};    // 原图 -> canvas 的等比缩放系数；canvas 坐标 * (1/scale) 即原图坐标
};

// 支持 8UC1 / 8UC3 / 8UC4 输入，内部统一转成 BGR。
FACECORE_API YunetLetterbox makeYunetNetworkInput(const cv::Mat& frame, cv::Size networkSize);

// 把检测结果裁剪到帧范围内，丢掉缩到不足 1 像素的框。CPU 与 NPU 通路共用。
FACECORE_API void clampDetectionsToFrame(std::vector<FaceDetection>& detections,
                                         cv::Size frameSize);

}  // namespace facecore
