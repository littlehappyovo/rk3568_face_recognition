cmake_minimum_required(VERSION 3.20)

project(FaceRecognitionSystem VERSION 0.2.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXSourceCompiles)

# ---------------------------------------------------------------------------
# 可选项
# ---------------------------------------------------------------------------
option(FACEAPP_BUILD_GUI "Build the Qt Widgets desktop application" ON)
option(FACEAPP_ENABLE_RKNN "Build the RKNN (Rockchip NPU) backends" ON)
option(FACEAPP_ENABLE_OPENCV_DNN "Build the OpenCV DNN (ONNX) detector/recognizer backends" ON)
option(FACEAPP_ENABLE_LBPH "Build the OpenCV LBPH recognizer backend" ON)

set(FACEAPP_OPENCV_MIN_VERSION "4.5.4" CACHE STRING "Minimum acceptable OpenCV version")
set(FACEAPP_RKNN_INCLUDE_DIR "" CACHE PATH "Directory that contains rknn_api.h")
set(FACEAPP_RKNN_LIBRARY "" CACHE FILEPATH "Path to librknnrt.so")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# OpenCV：只强制 core/imgproc/imgcodecs/videoio，其余能力逐项探测。
# RK3568 只有 4.5.4，开发机可能是 4.11；两代 API 的差异由探测结果处理。
# ---------------------------------------------------------------------------
find_package(OpenCV ${FACEAPP_OPENCV_MIN_VERSION} REQUIRED)
# 有些发行版的 OpenCVConfig.cmake 在没有显式 COMPONENTS 时不会填 OpenCV_LIBS，
# 这里做一次兜底，避免后面所有能力探测都因为链接不到符号而误判为“不可用”。
if(NOT OpenCV_LIBS)
    if(OpenCV_LIB_COMPONENTS)
        set(OpenCV_LIBS ${OpenCV_LIB_COMPONENTS})
    else()
        set(OpenCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio
                        opencv_objdetect opencv_dnn opencv_face)
    endif()
endif()
message(STATUS "OpenCV ${OpenCV_VERSION}: ${OpenCV_INCLUDE_DIRS}")

set(CMAKE_REQUIRED_INCLUDES ${OpenCV_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${OpenCV_LIBS})
set(CMAKE_REQUIRED_FLAGS "")

if(FACEAPP_ENABLE_OPENCV_DNN)
    # CPU 检测/识别后端只用 OpenCV 的通用 dnn 模块（readNetFromONNX + blobFromImage +
    # forward），不用 cv::FaceDetectorYN / cv::FaceRecognizerSF。
    # 原因：OpenCV 4.5.4 的 FaceDetectorYN 把输出层名写死成 {loc, conf, iou} 并配
    # anchor 解码，那是老版 YuNet；对本工程的 YuNet 2023mar（12 个 anchor-free 输出）
    # 调 forward 会直接报 "Layer with requested id=-1 not found"。通用 dnn 则一切正常。
    check_cxx_source_compiles("
        #include <opencv2/core.hpp>
        #include <opencv2/dnn.hpp>
        int main() {
            cv::dnn::Net net = cv::dnn::readNetFromONNX(\"model.onnx\");
            cv::Mat blob = cv::dnn::blobFromImage(cv::Mat::zeros(320, 320, CV_8UC3));
            net.setInput(blob);
            cv::Mat out;
            net.forward(out);
            return 0;
        }" FACEAPP_HAVE_OPENCV_DNN)
endif()

if(FACEAPP_ENABLE_LBPH)
    check_cxx_source_compiles("
        #include <opencv2/face.hpp>
        int main() {
            cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer =
                cv::face::LBPHFaceRecognizer::create(1, 8, 8, 8, 75.0);
            return recognizer.empty() ? 1 : 0;
        }" FACEAPP_HAVE_LBPH)
endif()

# ---------------------------------------------------------------------------
# RKNN：官方 rknn_api.h + 板端 librknnrt.so。
# 找不到就自动关闭 NPU 后端，让 CPU/LBPH 通路仍可编译，避免“环境问题”。
# ---------------------------------------------------------------------------
set(FACEAPP_HAVE_RKNN FALSE)
if(FACEAPP_ENABLE_RKNN)
    if(NOT FACEAPP_RKNN_INCLUDE_DIR)
        find_path(FACEAPP_RKNN_INCLUDE_DIR
            NAMES rknn_api.h
            HINTS
                ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknn
                ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknpu2/runtime/Linux/librknn_api/include
            PATHS /usr/include /usr/local/include
            DOC "Directory that contains rknn_api.h")
    endif()
    if(NOT FACEAPP_RKNN_LIBRARY)
        find_library(FACEAPP_RKNN_LIBRARY
            NAMES rknnrt
            HINTS /lib /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/lib
            DOC "Path to librknnrt.so")
    endif()
    if(NOT FACEAPP_RKNN_LIBRARY)
        foreach(_rknn_candidate /lib/librknnrt.so /usr/lib/librknnrt.so
                                /usr/lib/aarch64-linux-gnu/librknnrt.so
                                /usr/local/lib/librknnrt.so)
            if(EXISTS "${_rknn_candidate}")
                set(FACEAPP_RKNN_LIBRARY "${_rknn_candidate}")
                break()
            endif()
        endforeach()
    endif()

    if(FACEAPP_RKNN_INCLUDE_DIR AND EXISTS "${FACEAPP_RKNN_INCLUDE_DIR}/rknn_api.h"
       AND FACEAPP_RKNN_LIBRARY AND EXISTS "${FACEAPP_RKNN_LIBRARY}")
        set(FACEAPP_HAVE_RKNN TRUE)
        message(STATUS "RKNN enabled: rknn_api.h=${FACEAPP_RKNN_INCLUDE_DIR} "
                       "librknnrt=${FACEAPP_RKNN_LIBRARY}")
    else()
        set(FACEAPP_HAVE_RKNN FALSE)
        message(WARNING
            "FACEAPP_ENABLE_RKNN=ON but rknn_api.h (${FACEAPP_RKNN_INCLUDE_DIR}) or "
            "librknnrt.so (${FACEAPP_RKNN_LIBRARY}) was not found; building without NPU support. "
            "Pass -DFACEAPP_RKNN_INCLUDE_DIR=<dir> -DFACEAPP_RKNN_LIBRARY=<path> to enable it.")
    endif()
endif()

# ---------------------------------------------------------------------------
# facecore
# ---------------------------------------------------------------------------
add_library(facecore STATIC
    src/core/backend_factory.cpp
    src/core/config.cpp
    src/core/dataset.cpp
    src/core/detector_yunet.cpp
    src/core/embedding.cpp
    src/core/embedding_opencv.cpp
    src/core/frame_source.cpp
    src/core/preprocess.cpp
    src/core/recognizer_lbph.cpp
    src/core/recognizer_sface.cpp
    src/core/rknn_probe.cpp
    src/core/rknn_sface.cpp
    src/core/rknn_session.cpp
    src/core/rknn_yunet.cpp
    src/core/tracker.cpp
    src/core/training.cpp
    src/core/yunet_decode.cpp
)
add_library(facecore::facecore ALIAS facecore)

target_include_directories(facecore
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(facecore PUBLIC ${OpenCV_LIBS})
target_compile_definitions(facecore PRIVATE FACECORE_BUILDING_LIBRARY)

if(FACEAPP_HAVE_RKNN)
    target_include_directories(facecore PRIVATE ${FACEAPP_RKNN_INCLUDE_DIR})
    target_link_libraries(facecore PUBLIC ${FACEAPP_RKNN_LIBRARY})
    target_compile_definitions(facecore PUBLIC FACEAPP_HAVE_RKNN=1)
else()
    target_compile_definitions(facecore PUBLIC FACEAPP_HAVE_RKNN=0)
endif()

foreach(_feature FACEAPP_HAVE_OPENCV_DNN FACEAPP_HAVE_LBPH)
    if(${_feature})
        target_compile_definitions(facecore PUBLIC ${_feature}=1)
    else()
        target_compile_definitions(facecore PUBLIC ${_feature}=0)
    endif()
endforeach()

if(WIN32)
    target_compile_definitions(facecore PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
if(FACEAPP_BUILD_GUI)
    find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets Network)
    find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Network)

    add_executable(face_recognition_system WIN32
        src/app/main.cpp
        src/app/opencvfacebackend.h
        src/app/opencvfacebackend.cpp
        src/app/opencvfacebackend_access.cpp
        src/app/accessclient.cpp
        src/app/accessclient.h
        src/ui/adminlogindialog.cpp
        src/ui/adminlogindialog.h
        src/app/accessruntime.cpp
        src/app/accessruntime.h
        src/app/accessbridge.cpp
        src/app/accessbridge.h
        src/app/accesspublisher.cpp
        src/app/accesspublisher.h
        src/app/accesseventgate.cpp
        src/app/accesseventgate.h
        src/app/voiceannouncer.cpp
        src/app/voiceannouncer.h
        src/ui/backendtypes.h
        src/ui/facebackendservice.h
        src/ui/frameview.h
        src/ui/frameview.cpp
        src/ui/mainwindow.h
        src/ui/mainwindow.cpp
        src/ui/facedatabasedialog.h
        src/ui/facedatabasedialog.cpp
        src/ui/mockfacebackend.h
        src/ui/mockfacebackend.cpp
    )
    set_target_properties(face_recognition_system PROPERTIES
        AUTOMOC ON
        AUTOUIC ON
        AUTORCC ON
    )
    target_include_directories(face_recognition_system PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/app
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ui
    )
    target_link_libraries(face_recognition_system PRIVATE
        facecore::facecore
        Qt${QT_VERSION_MAJOR}::Widgets
        Qt${QT_VERSION_MAJOR}::Network
    )
endif()

# ---------------------------------------------------------------------------
# 编译告警
# ---------------------------------------------------------------------------
set(_faceapp_warning_targets facecore)
if(FACEAPP_BUILD_GUI)
    list(APPEND _faceapp_warning_targets face_recognition_system)
endif()

if(MSVC)
    foreach(_target ${_faceapp_warning_targets})
        target_compile_options(${_target} PRIVATE /W4 /permissive- /utf-8)
    endforeach()
    if(FACEAPP_BUILD_GUI)
        target_compile_definitions(face_recognition_system PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()
else()
    foreach(_target ${_faceapp_warning_targets})
        target_compile_options(${_target} PRIVATE -Wall -Wextra -Wpedantic)
    endforeach()
endif()

# ---------------------------------------------------------------------------
# 运行期资源
# ---------------------------------------------------------------------------
if(FACEAPP_BUILD_GUI)
    # Python/frontend changes do not relink the executable. Keep these assets current
    # on every application build, including explicit --target builds.
    add_custom_target(face_access_resources
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/server/face_access
                $<TARGET_FILE_DIR:face_recognition_system>/server/face_access
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/server/start_server.py
                $<TARGET_FILE_DIR:face_recognition_system>/server/start_server.py
        VERBATIM
    )
    add_dependencies(face_recognition_system face_access_resources)
    add_custom_command(TARGET face_recognition_system POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/config
                $<TARGET_FILE_DIR:face_recognition_system>/config
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/third_party_licenses
                $<TARGET_FILE_DIR:face_recognition_system>/third_party_licenses
        VERBATIM
    )
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models)
        add_custom_command(TARGET face_recognition_system POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${CMAKE_CURRENT_SOURCE_DIR}/models
                    $<TARGET_FILE_DIR:face_recognition_system>/models
            VERBATIM
        )
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/audio)
        add_custom_command(TARGET face_recognition_system POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${CMAKE_CURRENT_SOURCE_DIR}/audio
                    $<TARGET_FILE_DIR:face_recognition_system>/audio VERBATIM)
    endif()
endif()

# ---------------------------------------------------------------------------
# 安装
# ---------------------------------------------------------------------------
if(FACEAPP_BUILD_GUI)
    install(TARGETS face_recognition_system
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY config third_party_licenses
            DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY audio DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
    install(DIRECTORY server/face_access DESTINATION ${CMAKE_INSTALL_BINDIR}/server)
    install(FILES server/start_server.py server/requirements.lock DESTINATION ${CMAKE_INSTALL_BINDIR}/server)
endif()

message(STATUS "facecore features: RKNN=${FACEAPP_HAVE_RKNN} "
               "OpenCV_DNN=${FACEAPP_HAVE_OPENCV_DNN} "
               "LBPH=${FACEAPP_HAVE_LBPH}")
