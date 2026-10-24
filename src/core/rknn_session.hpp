#pragma once

// facecore 内部的 RKNN 会话封装，不对外暴露 rknn_api.h。
//
// 对外只提供 facecore/rknn.hpp 里的高层接口；这里负责 RAII、张量契约查询、
// 输入布局整理与错误码翻译。

#include "facecore/config.hpp"
#include "facecore/rknn.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace facecore {

struct RknnTensorDesc {
    std::string name;
    std::vector<std::uint32_t> dims;
    std::uint32_t elements{0};
    std::uint32_t bytes{0};
    int dataType{0};
    int format{0};
    int quantizationType{0};
    int zeroPoint{0};
    float scale{0.0F};
    bool quantized{false};
    bool float32{true};
    bool float16{false};
    std::string dataTypeName;
    std::string formatName;
    std::string quantizationText;
};

enum class RknnInputMode { ModelTensor, RawUint8Nhwc };

class RknnSession {
public:
    RknnSession();
    ~RknnSession();
    RknnSession(RknnSession&&) noexcept;
    RknnSession& operator=(RknnSession&&) noexcept;
    RknnSession(const RknnSession&) = delete;
    RknnSession& operator=(const RknnSession&) = delete;

    void load(const std::filesystem::path& model, int coreMask = 0);
    void unload() noexcept;
    bool loaded() const noexcept;

    const std::string& apiVersion() const;
    const std::string& driverVersion() const;
    const std::string& loadNotes() const;
    const std::vector<RknnTensorDesc>& inputs() const;
    const std::vector<RknnTensorDesc>& outputs() const;

    // inputData 与 inputs() 一一对应：
    //   float32 张量 -> elements * sizeof(float) 字节的小端 float
    //   量化张量     -> elements 字节的 uint8/int8
    // RawUint8Nhwc instead accepts elements bytes of interleaved raw pixels,
    // independently of the model's native FP16/quantized tensor type.
    // 输出统一按 float32 取回。
    void infer(const std::vector<std::vector<std::uint8_t>>& inputData,
               std::vector<std::vector<float>>& floatOutputs,
               RknnInputMode inputMode = RknnInputMode::ModelTensor) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// 输入张量的逻辑布局（由 RKNN 报告的 dims/format 推导）。
struct RknnInputLayout {
    int channels{0};
    int height{0};
    int width{0};
    bool nhwc{false};
};

// 解析输入布局，并校验它与 config.inputLayout 是否一致。
RknnInputLayout resolveRknnInputLayout(const RknnTensorDesc& input, const RknnModelConfig& config);

// 把 RKNN 张量描述转换成对外的可读报告。
RknnTensorReport toTensorReport(const RknnTensorDesc& desc);

// 把 float 缓冲当作张量数据写入；layout/colorOrder 见 RknnModelConfig。
void fillFloatInput(const cv::Mat& bgrImage,
                    const RknnModelConfig& config,
                    const RknnTensorDesc& input,
                    std::vector<std::uint8_t>& destination);

// 按模型输入张量的 zeroPoint/scale 写入量化输入；浮点输入另行做 mean/std 归一化。
void fillQuantizedInput(const cv::Mat& bgrImage,
                        const RknnModelConfig& config,
                        const RknnTensorDesc& input,
                        std::vector<std::uint8_t>& destination);

}  // namespace facecore
