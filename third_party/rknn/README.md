# RKNN SDK依赖

RKNN后端需要与板端BSP匹配的`rknn_api.h`和`librknnrt.so`。本仓库保留应用侧适配代码，SDK文件由开发者单独准备。

已验证的工程使用RKNN Runtime2.3.2，头文件来源为Rockchip官方`airockchip/rknn-toolkit2`的`v2.3.2`版本，路径为`rknpu2/runtime/Linux/librknn_api/include/rknn_api.h`。

官方源码入口：https://github.com/airockchip/rknn-toolkit2/tree/v2.3.2

构建时通过`-DFACEAPP_RKNN_INCLUDE_DIR=/path/to/sdk/include`指定头文件目录，通过`-DFACEAPP_RKNN_LIBRARY=/lib/librknnrt.so`指定板端Runtime。也可以把自行准备的头文件放入本目录；该文件已加入Git忽略规则。

当前工程核验过的头文件SHA-256为`c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a`。更换SDK、Runtime或模型后应重新核对张量输入合同和运行结果。
