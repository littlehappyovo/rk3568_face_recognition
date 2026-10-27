cmake_minimum_required(VERSION 3.20)
project(FaceLibraryRegression LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
enable_testing()
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
add_test(NAME face.access_server COMMAND ${Python3_EXECUTABLE} -I -m unittest discover -s ${CMAKE_CURRENT_LIST_DIR}/../server/tests -v)
find_package(Qt5 REQUIRED COMPONENTS Widgets Test Network)
find_package(OpenCV 4.5.4 REQUIRED)
set(APP_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
add_executable(face_access_event_tests access_event_tests.cpp
    ${APP_ROOT}/src/app/accesseventgate.cpp ${APP_ROOT}/src/app/accesspublisher.cpp)
target_include_directories(face_access_event_tests PRIVATE ${APP_ROOT}/src/app ${APP_ROOT}/src/ui)
target_link_libraries(face_access_event_tests PRIVATE Qt5::Widgets Qt5::Network Qt5::Test)
add_test(NAME face.access_events COMMAND face_access_event_tests)
add_executable(voice_tests voice_tests.cpp ${APP_ROOT}/src/app/voiceannouncer.cpp ${APP_ROOT}/src/core/tracker.cpp)
target_include_directories(voice_tests PRIVATE ${APP_ROOT}/src/app ${APP_ROOT}/src/ui ${APP_ROOT}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(voice_tests PRIVATE Qt5::Widgets Qt5::Test ${OpenCV_LIBS})
target_compile_definitions(voice_tests PRIVATE FACE_VOICE_PROJECT_ROOT="${APP_ROOT}")
add_test(NAME face.voice COMMAND voice_tests)
set(ADMIN_UI_SOURCES
    ${APP_ROOT}/src/app/accessclient.cpp
    ${APP_ROOT}/src/ui/adminlogindialog.cpp)
add_executable(admin_ui_tests admin_ui_tests.cpp ${ADMIN_UI_SOURCES}
    ${APP_ROOT}/src/app/accessruntime.cpp
    ${APP_ROOT}/src/ui/mainwindow.cpp ${APP_ROOT}/src/ui/frameview.cpp
    ${APP_ROOT}/src/ui/facedatabasedialog.cpp ${APP_ROOT}/src/ui/mockfacebackend.cpp
    ${APP_ROOT}/src/ui/facebackendservice.h)
target_include_directories(admin_ui_tests PRIVATE ${APP_ROOT}/src/app ${APP_ROOT}/src/ui)
target_compile_definitions(admin_ui_tests PRIVATE FACE_TEST_PROJECT_ROOT="${APP_ROOT}" FACE_TEST_ACCESS_PYTHON="${Python3_EXECUTABLE}")
target_link_libraries(admin_ui_tests PRIVATE Qt5::Widgets Qt5::Network Qt5::Test)
add_test(NAME face.admin_ui COMMAND admin_ui_tests)
set_tests_properties(face.admin_ui PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
add_executable(face_ui_tests
    feature_ui_tests.cpp ${ADMIN_UI_SOURCES}
    ${APP_ROOT}/src/ui/mainwindow.cpp
    ${APP_ROOT}/src/ui/frameview.cpp
    ${APP_ROOT}/src/ui/facedatabasedialog.cpp
    ${APP_ROOT}/src/ui/mockfacebackend.cpp
    ${APP_ROOT}/src/ui/facebackendservice.h)
target_include_directories(face_ui_tests PRIVATE ${APP_ROOT}/src/ui ${APP_ROOT}/src/app)
target_link_libraries(face_ui_tests PRIVATE Qt5::Widgets Qt5::Network Qt5::Test)
add_test(NAME face.ui COMMAND face_ui_tests)
set_tests_properties(face.ui PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
add_executable(face_dataset_tests feature_dataset_tests.cpp ${APP_ROOT}/src/core/dataset.cpp)
target_include_directories(face_dataset_tests PRIVATE ${APP_ROOT}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(face_dataset_tests PRIVATE Qt5::Test ${OpenCV_LIBS})
add_test(NAME face.dataset COMMAND face_dataset_tests)
add_executable(recognition_memory_tests recognition_memory_test.cpp)
target_include_directories(recognition_memory_tests PRIVATE ${APP_ROOT}/src/app ${APP_ROOT}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(recognition_memory_tests PRIVATE ${OpenCV_LIBS})
add_test(NAME face.recognition_memory COMMAND recognition_memory_tests)

if(FACEAPP_CORE_BUILD)
    add_executable(face_access_integration_tests access_integration_tests.cpp
        ${APP_ROOT}/src/app/accessruntime.cpp ${APP_ROOT}/src/app/accessbridge.cpp ${APP_ROOT}/src/app/accesspublisher.cpp
        ${APP_ROOT}/src/app/opencvfacebackend.cpp ${APP_ROOT}/src/app/opencvfacebackend_access.cpp
        ${APP_ROOT}/src/app/accesseventgate.cpp ${APP_ROOT}/src/app/voiceannouncer.cpp
        ${APP_ROOT}/src/ui/facebackendservice.h)
    target_include_directories(face_access_integration_tests PRIVATE ${APP_ROOT}/src/app ${APP_ROOT}/src/ui ${APP_ROOT}/include ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(face_access_integration_tests PRIVATE FACE_TEST_PROJECT_ROOT="${APP_ROOT}" FACE_TEST_ACCESS_PYTHON="${Python3_EXECUTABLE}")
    target_link_libraries(face_access_integration_tests PRIVATE Qt5::Widgets Qt5::Network Qt5::Test ${FACEAPP_CORE_BUILD}/libfacecore.a ${OpenCV_LIBS} ${CMAKE_DL_LIBS})
    add_test(NAME face.access_integration COMMAND face_access_integration_tests)
    add_executable(face_backend_tests feature_backend_tests.cpp ${ADMIN_UI_SOURCES}
        ${APP_ROOT}/src/app/opencvfacebackend.cpp
        ${APP_ROOT}/src/app/opencvfacebackend_access.cpp
        ${APP_ROOT}/src/app/accesseventgate.cpp
        ${APP_ROOT}/src/app/voiceannouncer.cpp
        ${APP_ROOT}/src/ui/mainwindow.cpp
        ${APP_ROOT}/src/ui/frameview.cpp
        ${APP_ROOT}/src/ui/facedatabasedialog.cpp
        ${APP_ROOT}/src/ui/facebackendservice.h)
    target_include_directories(face_backend_tests PRIVATE ${APP_ROOT}/src/app ${APP_ROOT}/src/ui ${APP_ROOT}/include ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(face_backend_tests PRIVATE FACE_TEST_PROJECT_ROOT="${APP_ROOT}")
    target_link_libraries(face_backend_tests PRIVATE Qt5::Widgets Qt5::Network Qt5::Test
        ${FACEAPP_CORE_BUILD}/libfacecore.a ${OpenCV_LIBS} ${CMAKE_DL_LIBS})
    find_library(TEST_RKNN_RUNTIME NAMES rknnrt PATHS /lib)
    if(TEST_RKNN_RUNTIME)
        target_link_libraries(face_access_integration_tests PRIVATE ${TEST_RKNN_RUNTIME})
        target_link_libraries(face_backend_tests PRIVATE ${TEST_RKNN_RUNTIME})
    endif()
    add_test(NAME face.backend COMMAND face_backend_tests)
    set_tests_properties(face.backend PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
endif()
