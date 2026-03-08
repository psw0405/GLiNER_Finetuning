cmake_minimum_required(VERSION 3.16)
project(gliner_tizen_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of ONNX Runtime installation")

find_package(sentencepiece REQUIRED)
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

find_path(
    ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
        ${ONNXRUNTIME_ROOT}/include
        $ENV{ONNXRUNTIME_ROOT}/include
)

find_library(
    ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS
        ${ONNXRUNTIME_ROOT}/lib
        ${ONNXRUNTIME_ROOT}/lib64
        $ENV{ONNXRUNTIME_ROOT}/lib
        $ENV{ONNXRUNTIME_ROOT}/lib64
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found. Set ONNXRUNTIME_ROOT or provide include/lib paths.")
endif()

add_library(gliner_tizen_infer STATIC
    src/gliner_tizen_infer.cpp
)

target_include_directories(gliner_tizen_infer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(gliner_tizen_infer
    PUBLIC
        ${ONNXRUNTIME_LIBRARY}
        sentencepiece::sentencepiece
        nlohmann_json::nlohmann_json
)

add_executable(gliner_tizen_demo
    src/main.cpp
)

target_link_libraries(gliner_tizen_demo
    PRIVATE
        gliner_tizen_infer
)

if(WIN32)
    set(ONNXRUNTIME_DLL "" CACHE FILEPATH "Path to onnxruntime.dll for runtime copy")
    if(ONNXRUNTIME_DLL)
        add_custom_command(
            TARGET gliner_tizen_demo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_DLL}
                $<TARGET_FILE_DIR:gliner_tizen_demo>
        )
    endif()
endif()
