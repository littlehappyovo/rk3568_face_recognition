# 模型准备

模型文件独立于源码仓库管理，按实际后端放入本目录。`MODELS.json`记录当前工程使用版本的大小与SHA-256；配置中的`model_identity`和输入参数与这些版本对应。

| 文件 | 用途 |
| --- | --- |
| `face_detection_yunet_2023mar.onnx` | OpenCV/CPU人脸检测 |
| `face_recognition_sface_2021dec.onnx` | OpenCV/CPU的SFace特征提取 |
| `face_detection_yunet_320.rknn` | RK3568/NPU人脸检测 |
| `mobilefacenet_float_opt3.rknn` | 默认MobileFaceNet/NPU识别，512维特征 |
| `face_recognition_sface_112.rknn` | 可选RKNN/SFace后端，当前默认配置不使用 |

CPU模型来自OpenCV Zoo固定版本`47534e27c9851bb1128ccc0102f1145e27f23f98`，下载地址见`MODELS.json`，原许可文本位于`third_party_licenses/`。从仓库根目录可执行：

```bash
curl -fL https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/face_detection_yunet_2023mar.onnx -o models/face_detection_yunet_2023mar.onnx
curl -fL https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/face_recognition_sface_2021dec.onnx -o models/face_recognition_sface_2021dec.onnx
sha256sum models/*.onnx
```

RKNN模型由部署者从原工程或自己的转换流程单独准备。本项目已验证的MobileFaceNet版本接收原始RGB/UINT8输入，由Runtime按模型合同处理；应用侧不能重复执行相同均值/标准差归一化。换模型时需同时核对形状、布局、色序、输入类型、量化方式、特征维度以及图库版本，不能仅替换文件名。

模型准备完成后再重建应用，以将资源复制到构建目录，或者将对应文件直接放入构建目录的`models/`中。`--config`可选择配置文件；程序先在可执行文件附近查找相对模型路径，再尝试配置文件目录附近的路径。

完整离线回归使用临时数据，其中后台适配器测试在初始化时需要`face_detection_yunet_2023mar.onnx`。因此至少下载第一条命令中的YuNet模型，再构建并运行完整回归。SFace和RKNN文件用于对应后端的实际推理，需单独准备并在目标设备验收。
