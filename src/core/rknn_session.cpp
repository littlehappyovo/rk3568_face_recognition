#include "rknn_session.hpp"

#include "io_internal.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#if FACEAPP_HAVE_RKNN
#include <rknn_api.h>
#endif

namespace facecore {
namespace {

// RknnSession 是 RKNN C API 的 RAII 包装器：构造/加载阶段负责上下文、模型和
// 张量属性，析构/卸载阶段负责释放资源。上层 embedding 和 detector 不直接
// 调 rknn_destroy、rknn_inputs_set 等底层细节。
#if FACEAPP_HAVE_RKNN

const char* rknnErrorText(int code) {
    switch (code) {
        case RKNN_SUCC: return "RKNN_SUCC";
        case RKNN_ERR_FAIL: return "RKNN_ERR_FAIL";
        case RKNN_ERR_TIMEOUT: return "RKNN_ERR_TIMEOUT";
        case RKNN_ERR_DEVICE_UNAVAILABLE: return "RKNN_ERR_DEVICE_UNAVAILABLE";
        case RKNN_ERR_MALLOC_FAIL: return "RKNN_ERR_MALLOC_FAIL";
        case RKNN_ERR_PARAM_INVALID: return "RKNN_ERR_PARAM_INVALID";
        case RKNN_ERR_MODEL_INVALID: return "RKNN_ERR_MODEL_INVALID";
        case RKNN_ERR_CTX_INVALID: return "RKNN_ERR_CTX_INVALID";
        case RKNN_ERR_INPUT_INVALID: return "RKNN_ERR_INPUT_INVALID";
        case RKNN_ERR_OUTPUT_INVALID: return "RKNN_ERR_OUTPUT_INVALID";
        case RKNN_ERR_DEVICE_UNMATCH: return "RKNN_ERR_DEVICE_UNMATCH";
        case RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL: return "RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL";
        case RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION:
            return "RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION";
        case RKNN_ERR_TARGET_PLATFORM_UNMATCH: return "RKNN_ERR_TARGET_PLATFORM_UNMATCH";
        default: return "RKNN_ERR_UNKNOWN";
    }
}

std::string rknnError(int code) {
    return std::string(rknnErrorText(code)) + " (" + std::to_string(code) + ")";
}

RknnTensorDesc describeTensor(const rknn_tensor_attr& attr) {
    RknnTensorDesc desc;
    desc.name = attr.name;
    desc.dims.assign(attr.dims, attr.dims + attr.n_dims);
    desc.elements = attr.n_elems;
    desc.bytes = attr.size;
    desc.dataType = static_cast<int>(attr.type);
    desc.format = static_cast<int>(attr.fmt);
    desc.quantizationType = static_cast<int>(attr.qnt_type);
    desc.zeroPoint = attr.zp;
    desc.scale = attr.scale;
    desc.float32 = attr.type == RKNN_TENSOR_FLOAT32;
    desc.float16 = attr.type == RKNN_TENSOR_FLOAT16;
    desc.quantized = attr.type == RKNN_TENSOR_UINT8 || attr.type == RKNN_TENSOR_INT8
                     || attr.type == RKNN_TENSOR_INT16 || attr.type == RKNN_TENSOR_UINT16
                     || attr.type == RKNN_TENSOR_INT4;
    desc.dataTypeName = get_type_string(attr.type);
    desc.formatName = get_format_string(attr.fmt);
    switch (attr.qnt_type) {
        case RKNN_TENSOR_QNT_NONE:
            desc.quantizationText = "none";
            break;
        case RKNN_TENSOR_QNT_DFP:
            desc.quantizationText = "dfp(fl=" + std::to_string(static_cast<int>(attr.fl)) + ")";
            break;
        case RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC:
            desc.quantizationText = "affine(zp=" + std::to_string(attr.zp) + ", scale=" +
                                    std::to_string(attr.scale) + ")";
            break;
        default:
            desc.quantizationText = "unknown";
            break;
    }
    return desc;
}

std::vector<RknnTensorDesc> queryTensors(rknn_context context,
                                         rknn_query_cmd command,
                                         std::uint32_t count) {
    std::vector<RknnTensorDesc> result;
    result.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        rknn_tensor_attr attr{};
        attr.index = index;
        const int status = rknn_query(context, command, &attr, sizeof(attr));
        if (status != RKNN_SUCC) {
            throw std::runtime_error("rknn_query(tensor attr) failed: " + rknnError(status));
        }
        result.push_back(describeTensor(attr));
    }
    return result;
}

#endif  // FACEAPP_HAVE_RKNN

}  // namespace

RknnInputLayout resolveRknnInputLayout(const RknnTensorDesc& input, const RknnModelConfig& config) {
    RknnInputLayout layout;
    const std::vector<std::uint32_t>& dims = input.dims;
    if (input.formatName == "NCHW" && dims.size() == 4) {
        layout.nhwc = false;
        layout.channels = static_cast<int>(dims[1]);
        layout.height = static_cast<int>(dims[2]);
        layout.width = static_cast<int>(dims[3]);
    } else if (input.formatName == "NHWC" && dims.size() == 4) {
        layout.nhwc = true;
        layout.height = static_cast<int>(dims[1]);
        layout.width = static_cast<int>(dims[2]);
        layout.channels = static_cast<int>(dims[3]);
    } else if (input.formatName == "NCHW" && dims.size() == 3) {
        // 已经去掉 batch 维的 1xCxHxW。
        layout.nhwc = false;
        layout.channels = static_cast<int>(dims[0]);
        layout.height = static_cast<int>(dims[1]);
        layout.width = static_cast<int>(dims[2]);
    } else {
        throw std::runtime_error(
            "Unsupported RKNN input tensor layout: name=" + input.name + " format=" +
            input.formatName + " dims=" + std::to_string(dims.size()) +
            ". Re-convert the model so the input tensor is NCHW (recommended) or NHWC.");
    }
    if (layout.channels <= 0 || layout.height <= 0 || layout.width <= 0) {
        throw std::runtime_error("RKNN input tensor has a non-positive dimension");
    }
    if (config.inputLayout == "nhwc" && !layout.nhwc) {
        throw std::runtime_error(
            "the rknn input_layout is set to \"nhwc\" but the model input tensor is NCHW; "
            "fix the configuration or re-convert the model");
    }
    if (config.inputLayout == "nchw" && layout.nhwc) {
        throw std::runtime_error(
            "the rknn input_layout is set to \"nchw\" but the model input tensor is NHWC; "
            "fix the configuration or re-convert the model");
    }
    return layout;
}

namespace {

int channelValue(const cv::Vec3b& pixel, int channel, const std::string& colorOrder) {
    if (colorOrder == "rgb") {
        return pixel[2 - channel];
    }
    if (colorOrder == "gray") {
        const double luminance = 0.114 * pixel[0] + 0.587 * pixel[1] + 0.299 * pixel[2];
        return static_cast<int>(std::lround(luminance));
    }
    return pixel[channel];
}

double coefficientAt(const std::vector<double>& values, int channel, double fallback) {
    if (values.empty()) return fallback;
    if (values.size() == 1) return values.front();
    return values[static_cast<std::size_t>(channel) % values.size()];
}

void validateColorOrder(const std::string& colorOrder) {
    if (colorOrder != "bgr" && colorOrder != "rgb" && colorOrder != "gray") {
        throw std::invalid_argument(
            "rknn input_color must be one of bgr / rgb / gray, got: " + colorOrder);
    }
}

std::size_t layoutOffset(const RknnInputLayout& layout, int channel, int row, int column) {
    const std::size_t plane = static_cast<std::size_t>(layout.height) * layout.width;
    if (layout.nhwc) {
        return (static_cast<std::size_t>(row) * layout.width + column) * layout.channels + channel;
    }
    return static_cast<std::size_t>(channel) * plane +
           static_cast<std::size_t>(row) * layout.width + column;
}

std::uint16_t floatToHalf(float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000U;
    const std::uint32_t exponent = (bits >> 23) & 0xffU;
    const std::uint32_t mantissa = bits & 0x7fffffU;
    if (exponent == 0xffU) {
        return static_cast<std::uint16_t>(sign | 0x7c00U |
                                          (mantissa != 0 ? 0x0200U : 0U));
    }
    const int unbiased = static_cast<int>(exponent) - 127;
    if (unbiased > 15) return static_cast<std::uint16_t>(sign | 0x7c00U);
    if (unbiased < -14) {
        if (unbiased < -24) return static_cast<std::uint16_t>(sign);
        const std::uint32_t shifted = (mantissa | 0x800000U) >>
                                      static_cast<unsigned>(-unbiased - 14 + 1);
        return static_cast<std::uint16_t>(sign | ((shifted + 0x1000U) >> 13));
    }
    const std::uint32_t halfExponent = static_cast<std::uint32_t>(unbiased + 15) << 10;
    const std::uint32_t roundedMantissa = (mantissa + 0x1000U) >> 13;
    if (roundedMantissa >= 0x400U) {
        const std::uint32_t incremented = halfExponent + 0x400U;
        if (incremented >= 0x7c00U) return static_cast<std::uint16_t>(sign | 0x7c00U);
        return static_cast<std::uint16_t>(sign | incremented);
    }
    return static_cast<std::uint16_t>(sign | halfExponent | roundedMantissa);
}

void requireMatchingImage(const cv::Mat& image, const RknnInputLayout& layout) {
    if (image.empty()) throw std::invalid_argument("RKNN input image is empty");
    if (image.type() != CV_8UC3) {
        throw std::invalid_argument("RKNN input image must be an 8-bit 3-channel BGR image");
    }
    if (image.cols != layout.width || image.rows != layout.height) {
        throw std::invalid_argument("RKNN input image is " + std::to_string(image.cols) + "x" +
                                    std::to_string(image.rows) + " but the model expects " +
                                    std::to_string(layout.width) + "x" +
                                    std::to_string(layout.height));
    }
}

}  // namespace

void fillFloatInput(const cv::Mat& bgrImage,
                    const RknnModelConfig& config,
                    const RknnTensorDesc& input,
                    std::vector<std::uint8_t>& destination) {
    validateColorOrder(config.inputColorOrder);
    const RknnInputLayout layout = resolveRknnInputLayout(input, config);
    requireMatchingImage(bgrImage, layout);
    if (!(std::isfinite(config.inputScale) && config.inputScale > 0.0)) {
        throw std::invalid_argument("rknn input_scale must be a positive finite number");
    }

    std::vector<float> buffer(static_cast<std::size_t>(layout.channels) * layout.height *
                              layout.width);
    for (int row = 0; row < layout.height; ++row) {
        const cv::Vec3b* line = bgrImage.ptr<cv::Vec3b>(row);
        for (int column = 0; column < layout.width; ++column) {
            const cv::Vec3b& pixel = line[column];
            for (int channel = 0; channel < layout.channels; ++channel) {
                const int sourceChannel = layout.channels == 1 ? 0 : channel;
                const double raw = channelValue(pixel, sourceChannel, config.inputColorOrder);
                const double mean = coefficientAt(config.inputMean, channel, 0.0);
                const double deviation = coefficientAt(config.inputStd, channel, 1.0);
                if (!std::isfinite(deviation) || deviation == 0.0) {
                    throw std::invalid_argument("rknn input_std must be finite and non-zero");
                }
                buffer[layoutOffset(layout, channel, row, column)] = static_cast<float>(
                    (raw * config.inputScale - mean) / deviation);
            }
        }
    }
    if (input.float16) {
        std::vector<std::uint16_t> half(buffer.size());
        std::transform(buffer.begin(), buffer.end(), half.begin(), floatToHalf);
        destination.resize(half.size() * sizeof(std::uint16_t));
        std::memcpy(destination.data(), half.data(), destination.size());
    } else if (input.float32) {
        destination.resize(buffer.size() * sizeof(float));
        std::memcpy(destination.data(), buffer.data(), destination.size());
    } else {
        throw std::invalid_argument("RKNN floating input tensor is neither FLOAT16 nor FLOAT32");
    }
}

void fillQuantizedInput(const cv::Mat& bgrImage,
                        const RknnModelConfig& config,
                        const RknnTensorDesc& input,
                        std::vector<std::uint8_t>& destination) {
    validateColorOrder(config.inputColorOrder);
    const RknnInputLayout layout = resolveRknnInputLayout(input, config);
    requireMatchingImage(bgrImage, layout);
    if (!std::isfinite(input.scale) || input.scale <= 0.0F) {
        throw std::invalid_argument(
            "RKNN quantized input has no valid affine scale: " + input.dataTypeName);
    }
    if (input.dataTypeName != "INT8" && input.dataTypeName != "UINT8") {
        throw std::invalid_argument("Unsupported one-byte RKNN quantized input type: "
                                    + input.dataTypeName);
    }

    const int quantizedMinimum = input.dataTypeName == "UINT8" ? 0 : -128;
    const int quantizedMaximum = input.dataTypeName == "UINT8" ? 255 : 127;
    destination.assign(static_cast<std::size_t>(layout.channels) * layout.height * layout.width,
                       0);
    for (int row = 0; row < layout.height; ++row) {
        const cv::Vec3b* line = bgrImage.ptr<cv::Vec3b>(row);
        for (int column = 0; column < layout.width; ++column) {
            const cv::Vec3b& pixel = line[column];
            for (int channel = 0; channel < layout.channels; ++channel) {
                const int sourceChannel = layout.channels == 1 ? 0 : channel;
                const double raw = channelValue(pixel, sourceChannel, config.inputColorOrder);
                const double mean = coefficientAt(config.inputMean, channel, 0.0);
                const double deviation = coefficientAt(config.inputStd, channel, 1.0);
                if (!std::isfinite(deviation) || deviation == 0.0) {
                    throw std::invalid_argument("rknn input_std must be finite and non-zero");
                }

                // RKNN 的量化输入不是简单的 0..255 原始像素。先按模型契约完成
                // scale/mean/std 预处理，再使用输入张量自身的 affine 参数量化：
                // q = round(real / tensor_scale) + tensor_zero_point。
                const double realValue = (raw * config.inputScale - mean) / deviation;
                const double quantized = std::lround(realValue / input.scale)
                                          + input.zeroPoint;
                const int clamped = static_cast<int>(std::clamp(
                    quantized, static_cast<double>(quantizedMinimum),
                    static_cast<double>(quantizedMaximum)));
                // vector<uint8_t> 只负责保存字节；INT8 的负值按二进制补码写入，
                // rknn_input.type 会告诉 RKNN runtime 按 INT8 解释这些字节。
                destination[layoutOffset(layout, channel, row, column)] =
                    static_cast<std::uint8_t>(clamped);
            }
        }
    }
}

RknnTensorReport toTensorReport(const RknnTensorDesc& desc) {
    RknnTensorReport report;
    report.name = desc.name;
    std::string shape = "[";
    for (std::size_t index = 0; index < desc.dims.size(); ++index) {
        if (index != 0) shape += ',';
        shape += std::to_string(desc.dims[index]);
    }
    report.shape = shape + "]";
    report.dataType = desc.dataTypeName;
    report.format = desc.formatName;
    report.quantization = desc.quantizationText;
    report.elements = desc.elements;
    report.bytes = desc.bytes;
    return report;
}

class RknnSession::Impl {
public:
    std::string apiVersion;
    std::string driverVersion;
    std::string loadNotes;
    std::vector<RknnTensorDesc> inputs;
    std::vector<RknnTensorDesc> outputs;
    std::vector<unsigned char> modelBytes;
#if FACEAPP_HAVE_RKNN
    rknn_context context{0};
    bool hasContext{false};
#endif
};

RknnSession::RknnSession() : impl_(std::make_unique<Impl>()) {}
RknnSession::~RknnSession() { unload(); }
RknnSession::RknnSession(RknnSession&&) noexcept = default;
RknnSession& RknnSession::operator=(RknnSession&&) noexcept = default;

void RknnSession::load(const std::filesystem::path& model, int coreMask) {
    // 加载阶段把文件字节交给 rknn_init，再查询 SDK/驱动版本、输入输出数量和
    // 每个张量的 shape/type/量化信息。上层 detector/embedding 依赖这些描述，
    // 而不是根据模型文件名猜输入尺寸。
#if !FACEAPP_HAVE_RKNN
    (void)model;
    (void)coreMask;
    throw std::runtime_error(
        "The RKNN backend is not compiled into this build; rebuild with FACEAPP_ENABLE_RKNN=ON "
        "and point FACEAPP_RKNN_INCLUDE_DIR / FACEAPP_RKNN_LIBRARY at librknnrt");
#else
    unload();
    if (model.empty()) throw std::invalid_argument("RKNN model path is empty");
    if (!std::filesystem::is_regular_file(model)) {
        throw std::runtime_error("RKNN model file does not exist: " + detail::pathToUtf8(model));
    }
    impl_->modelBytes = detail::readBinaryFile(model);
    if (impl_->modelBytes.empty()) {
        throw std::runtime_error("RKNN model file is empty: " + detail::pathToUtf8(model));
    }

    rknn_context context = 0;
    const std::uint32_t flags = RKNN_FLAG_PRIOR_MEDIUM;
    int status = rknn_init(&context, impl_->modelBytes.data(),
                           static_cast<std::uint32_t>(impl_->modelBytes.size()), flags, nullptr);
    if (status != RKNN_SUCC) {
        throw std::runtime_error(
            "rknn_init failed for " + detail::pathToUtf8(model) + ": " + rknnError(status) +
            ". Check that the model was converted with target_platform='rk3568' and that the "
            "RKNN-Toolkit2 version matches librknnrt on the board.");
    }
    impl_->context = context;
    impl_->hasContext = true;

    rknn_sdk_version version{};
    if (rknn_query(impl_->context, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)) == RKNN_SUCC) {
        impl_->apiVersion = version.api_version;
        impl_->driverVersion = version.drv_version;
    }

    if (coreMask > 0) {
        const int maskStatus =
            rknn_set_core_mask(impl_->context, static_cast<rknn_core_mask>(coreMask));
        if (maskStatus != RKNN_SUCC) {
            impl_->loadNotes += "rknn_set_core_mask(" + std::to_string(coreMask) +
                                ") failed: " + rknnError(maskStatus) + "; ";
        }
    }

    rknn_input_output_num counts{};
    status = rknn_query(impl_->context, RKNN_QUERY_IN_OUT_NUM, &counts, sizeof(counts));
    if (status != RKNN_SUCC) {
        unload();
        throw std::runtime_error("rknn_query(RKNN_QUERY_IN_OUT_NUM) failed: " + rknnError(status));
    }
    if (counts.n_input == 0 || counts.n_output == 0) {
        unload();
        throw std::runtime_error("RKNN model reports " + std::to_string(counts.n_input) +
                                 " input(s) and " + std::to_string(counts.n_output) + " output(s)");
    }

    try {
        impl_->inputs = queryTensors(impl_->context, RKNN_QUERY_INPUT_ATTR, counts.n_input);
        impl_->outputs = queryTensors(impl_->context, RKNN_QUERY_OUTPUT_ATTR, counts.n_output);
    } catch (...) {
        unload();
        throw;
    }
#endif
}

void RknnSession::unload() noexcept {
    // unload() 要支持“加载失败后清理”和“析构时清理”两种场景，所以所有操作
    // 都设计成可重复调用。先销毁 context，再清空张量描述和模型字节缓存。
#if FACEAPP_HAVE_RKNN
    if (impl_ && impl_->hasContext) {
        rknn_destroy(impl_->context);
        impl_->context = 0;
        impl_->hasContext = false;
    }
#endif
    if (impl_) {
        impl_->inputs.clear();
        impl_->outputs.clear();
        impl_->modelBytes.clear();
        impl_->apiVersion.clear();
        impl_->driverVersion.clear();
        impl_->loadNotes.clear();
    }
}

bool RknnSession::loaded() const noexcept {
#if FACEAPP_HAVE_RKNN
    return impl_ && impl_->hasContext;
#else
    return false;
#endif
}

const std::string& RknnSession::apiVersion() const { return impl_->apiVersion; }
const std::string& RknnSession::driverVersion() const { return impl_->driverVersion; }
const std::string& RknnSession::loadNotes() const { return impl_->loadNotes; }
const std::vector<RknnTensorDesc>& RknnSession::inputs() const { return impl_->inputs; }
const std::vector<RknnTensorDesc>& RknnSession::outputs() const { return impl_->outputs; }

void RknnSession::infer(const std::vector<std::vector<std::uint8_t>>& inputData,
                        std::vector<std::vector<float>>& floatOutputs,
                        RknnInputMode inputMode) const {
    // 公共接口使用字节输入和 float 输出：输入侧由 detector/embedding 根据量化
    // 契约填好，输出侧统一转换成 float，方便 YuNet 解码和 embedding 归一化。
#if !FACEAPP_HAVE_RKNN
    (void)inputData;
    (void)floatOutputs;
    (void)inputMode;
    throw std::runtime_error("The RKNN backend is not compiled into this build");
#else
    if (!impl_->hasContext) throw std::runtime_error("RKNN session is not loaded");
    if (inputData.size() != impl_->inputs.size()) {
        throw std::invalid_argument("RKNN input count mismatch: expected " +
                                    std::to_string(impl_->inputs.size()) + ", got " +
                                    std::to_string(inputData.size()));
    }

    std::vector<rknn_input> nativeInputs(impl_->inputs.size());
    for (std::size_t index = 0; index < nativeInputs.size(); ++index) {
        const RknnTensorDesc& desc = impl_->inputs[index];
        const bool rawPixels = inputMode == RknnInputMode::RawUint8Nhwc;
        const std::size_t expected = rawPixels
                                         ? static_cast<std::size_t>(desc.elements)
                                         : desc.float32
                                         ? static_cast<std::size_t>(desc.elements) * sizeof(float)
                                         : desc.float16
                                             ? static_cast<std::size_t>(desc.elements) * sizeof(std::uint16_t)
                                             : static_cast<std::size_t>(desc.elements);
        if (inputData[index].size() != expected) {
            throw std::invalid_argument("RKNN input " + desc.name + " expects " +
                                        std::to_string(expected) + " bytes but received " +
                                        std::to_string(inputData[index].size()));
        }
        nativeInputs[index].index = static_cast<std::uint32_t>(index);
        nativeInputs[index].buf = const_cast<std::uint8_t*>(inputData[index].data());
        nativeInputs[index].size = static_cast<std::uint32_t>(inputData[index].size());
        nativeInputs[index].pass_through = 0;
        nativeInputs[index].type = rawPixels ? RKNN_TENSOR_UINT8
                                            : static_cast<rknn_tensor_type>(desc.dataType);
        nativeInputs[index].fmt = rawPixels ? RKNN_TENSOR_NHWC
                                           : static_cast<rknn_tensor_format>(desc.format);
    }

    int status = rknn_inputs_set(impl_->context, static_cast<std::uint32_t>(nativeInputs.size()),
                                 nativeInputs.data());
    if (status != RKNN_SUCC) {
        throw std::runtime_error("rknn_inputs_set failed: " + rknnError(status));
    }
    status = rknn_run(impl_->context, nullptr);
    if (status != RKNN_SUCC) {
        throw std::runtime_error("rknn_run failed: " + rknnError(status));
    }

    std::vector<rknn_output> nativeOutputs(impl_->outputs.size());
    for (std::size_t index = 0; index < nativeOutputs.size(); ++index) {
        nativeOutputs[index].index = static_cast<std::uint32_t>(index);
        nativeOutputs[index].want_float = 1;
        nativeOutputs[index].is_prealloc = 0;
    }
    status = rknn_outputs_get(impl_->context, static_cast<std::uint32_t>(nativeOutputs.size()),
                              nativeOutputs.data(), nullptr);
    if (status != RKNN_SUCC) {
        throw std::runtime_error("rknn_outputs_get failed: " + rknnError(status));
    }

    struct ReleaseGuard {
        rknn_context context;
        std::vector<rknn_output>* outputs;
        ~ReleaseGuard() {
            if (outputs != nullptr && !outputs->empty()) {
                rknn_outputs_release(context, static_cast<std::uint32_t>(outputs->size()),
                                     outputs->data());
            }
        }
    } guard{impl_->context, &nativeOutputs};

    floatOutputs.assign(nativeOutputs.size(), {});
    for (std::size_t index = 0; index < nativeOutputs.size(); ++index) {
        const std::size_t count = impl_->outputs[index].elements;
        const float* data = static_cast<const float*>(nativeOutputs[index].buf);
        if (data == nullptr) {
            throw std::runtime_error("RKNN returned a null buffer for output " +
                                     impl_->outputs[index].name);
        }
        floatOutputs[index].assign(data, data + count);
    }
#endif
}

}  // namespace facecore
