cmake_minimum_required(VERSION 3.22)
project(moonglide_linux LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# glslang >= 16 is required: the NR kernels use GL_NV_cooperative_matrix2 / float8 syntax.
set(GLSLANG "glslang" CACHE FILEPATH "glslang standalone binary (>= 16.x)")

set(NR_DIR ${CMAKE_SOURCE_DIR}/third_party/opendlss-nr)
set(VK_HEADERS ${CMAKE_SOURCE_DIR}/third_party/vulkan-headers/include)
set(VOLK_DIR ${CMAKE_SOURCE_DIR}/third_party/volk)

add_compile_definitions(VK_ENABLE_BETA_EXTENSIONS VK_USE_PLATFORM_XLIB_KHR)

# --- volk (C) ---------------------------------------------------------------
add_library(volk STATIC ${VOLK_DIR}/volk.c)
target_include_directories(volk PUBLIC ${VOLK_DIR} ${VK_HEADERS})
target_link_libraries(volk PUBLIC dl)

# --- OpenDLSS-NR as a library -------------------------------------------------
add_library(opendlss_nr STATIC
  ${NR_DIR}/src/kernels.cpp
  ${NR_DIR}/src/nr_graph.cpp
  ${NR_DIR}/src/nr_model.cpp
  ${NR_DIR}/src/nr_schedule.cpp
  ${NR_DIR}/src/reference.cpp
  ${NR_DIR}/src/vk_context.cpp)
target_include_directories(opendlss_nr PUBLIC ${NR_DIR}/src)
target_link_libraries(opendlss_nr PUBLIC volk Threads::Threads)
target_compile_options(opendlss_nr PRIVATE -O2 -w)
find_package(Threads REQUIRED)

# --- shaders ------------------------------------------------------------------
# NR kernels go to build/nr/shaders (Kernels looks for PTX at <shaders>/../ptx).
set(NR_SPV_DIR ${CMAKE_BINARY_DIR}/nr/shaders)
set(NR_PTX_DIR ${CMAKE_BINARY_DIR}/nr/ptx)
set(APP_SPV_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${NR_SPV_DIR} ${NR_PTX_DIR} ${APP_SPV_DIR})

set(SPV_OUTPUTS)
file(GLOB NR_SHADERS ${NR_DIR}/shaders/*.comp)
file(GLOB NR_SHADER_INCLUDES ${NR_DIR}/shaders/*.glsl)
foreach(src ${NR_SHADERS})
  get_filename_component(name ${src} NAME_WE)
  set(out ${NR_SPV_DIR}/${name}.spv)
  add_custom_command(OUTPUT ${out}
    COMMAND ${GLSLANG} -V --target-env vulkan1.3 -I${NR_DIR}/shaders ${src} -o ${out} > /dev/null
    DEPENDS ${src} ${NR_SHADER_INCLUDES} VERBATIM)
  list(APPEND SPV_OUTPUTS ${out})
endforeach()

file(GLOB APP_SHADERS ${CMAKE_SOURCE_DIR}/shaders/*.comp)
file(GLOB APP_SHADER_INCLUDES ${CMAKE_SOURCE_DIR}/shaders/*.glsl)
foreach(src ${APP_SHADERS})
  get_filename_component(name ${src} NAME_WE)
  set(out ${APP_SPV_DIR}/${name}.spv)
  add_custom_command(OUTPUT ${out}
    COMMAND ${GLSLANG} -V --target-env vulkan1.3 -I${CMAKE_SOURCE_DIR}/shaders ${src} -o ${out} > /dev/null
    DEPENDS ${src} ${APP_SHADER_INCLUDES} VERBATIM)
  list(APPEND SPV_OUTPUTS ${out})
endforeach()

# PTX is generated code (scripts/ptx/*.py emitters, see third_party/opendlss-nr/VENDORED.md);
# it is vendored so the build needs no Python.
file(GLOB NR_PTX ${NR_DIR}/ptx/*.ptx)
set(PTX_STAMP ${NR_PTX_DIR}/.stamp)
add_custom_command(OUTPUT ${PTX_STAMP}
  COMMAND ${CMAKE_COMMAND} -E copy ${NR_PTX} ${NR_PTX_DIR}
  COMMAND ${CMAKE_COMMAND} -E touch ${PTX_STAMP}
  DEPENDS ${NR_PTX})
add_custom_target(gpu_code ALL DEPENDS ${SPV_OUTPUTS} ${PTX_STAMP})

# --- Dear ImGui (MIT, vendored): settings overlay -----------------------------------
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui)
add_library(imgui STATIC
  ${IMGUI_DIR}/imgui.cpp ${IMGUI_DIR}/imgui_draw.cpp ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/imgui_tables.cpp ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp)
target_include_directories(imgui PUBLIC ${IMGUI_DIR})
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK IMGUI_DISABLE_OBSOLETE_FUNCTIONS)
target_link_libraries(imgui PUBLIC volk)
target_compile_options(imgui PRIVATE -O2 -w)

# --- the app --------------------------------------------------------------------
find_package(X11 REQUIRED)
find_library(JPEG_LIB NAMES jpeg REQUIRED)
# PS5 Remote Play source: decodes the tap's HEVC / H.264 (NVDEC through libavcodec's CUDA hwaccel).
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET libavcodec libavutil)

add_executable(moonglide
  src/main.cpp
  src/gpu.cpp
  src/window_x11.cpp
  src/capture_v4l2.cpp
  src/jpeg_decode.cpp
  src/nr_stage.cpp
  src/overlay.cpp
  src/stats.cpp
  src/optical_flow.cpp
  src/hud_zones.cpp
  src/nvvfx.cpp
  src/rife.cpp
  src/ms2109.cpp
  src/remoteplay.cpp
  src/config.cpp
  src/input_setup.cpp
  src/mcp.cpp
  src/ui.cpp
  src/i18n.cpp)
target_include_directories(moonglide PRIVATE src ${X11_INCLUDE_DIR})
target_link_libraries(moonglide PRIVATE opendlss_nr imgui volk ${X11_LIBRARIES} ${JPEG_LIB} PkgConfig::LIBAV Threads::Threads)
target_compile_options(moonglide PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
target_compile_definitions(moonglide PRIVATE
  MOONGLIDE_SHADER_DIR=moonglide::dataDir\(\"${APP_SPV_DIR}\",\"shaders\"\)
  MOONGLIDE_NR_SHADER_DIR=moonglide::dataDir\(\"${NR_SPV_DIR}\",\"nr/shaders\"\)
  MOONGLIDE_SCRIPTS_DIR=moonglide::dataDir\(\"${CMAKE_SOURCE_DIR}/scripts\",\"scripts\"\))
# paths.h: shaders / tap are looked up next to the executable first (AppImage, prefix installs).
target_compile_options(moonglide PRIVATE -include ${CMAKE_SOURCE_DIR}/src/paths.h)
add_dependencies(moonglide gpu_code)

# --- optional NVIDIA Video Effects backend ----------------------------------------
# Built against the user's own SDK install (headers only; the libraries are dlopen()ed at run
# time). Nothing from the SDK is part of this repository.
set(MOONGLIDE_NVVFX_SDK "/usr/local/VideoFX" CACHE PATH "NVIDIA Video Effects SDK root (optional)")
find_path(NVVFX_INCLUDE nvVideoEffects.h PATHS ${MOONGLIDE_NVVFX_SDK}/include NO_DEFAULT_PATH)
find_path(NVVFX_CUDA_INCLUDE cuda.h PATHS ${MOONGLIDE_NVVFX_SDK}/external/cuda/include /usr/local/cuda/include)
if(NVVFX_INCLUDE AND NVVFX_CUDA_INCLUDE)
  message(STATUS "nvvfx: building the optional backend against ${MOONGLIDE_NVVFX_SDK}")
  set_source_files_properties(src/nvvfx.cpp PROPERTIES
    COMPILE_DEFINITIONS MOONGLIDE_HAVE_NVVFX
    INCLUDE_DIRECTORIES "${NVVFX_INCLUDE};${NVVFX_CUDA_INCLUDE}")
else()
  message(STATUS "nvvfx: SDK headers not found at ${MOONGLIDE_NVVFX_SDK}; backend stubbed")
endif()

# --- optional RIFE frame generation (TensorRT) -----------------------------------
# Headers only at build time; libnvinfer / libnvonnxparser are dlopen()ed from the user's own
# TensorRT 10 install at run time (see src/rife.cpp). Nothing NVIDIA is part of this repository.
set(MOONGLIDE_TRT_INCLUDE "${MOONGLIDE_NVVFX_SDK}/external/tensorrt/include" CACHE PATH "TensorRT 10 headers (optional)")
find_path(RIFE_TRT_INCLUDE NvInfer.h PATHS ${MOONGLIDE_TRT_INCLUDE} /usr/include/x86_64-linux-gnu NO_DEFAULT_PATH)
if(RIFE_TRT_INCLUDE AND NVVFX_CUDA_INCLUDE)
  message(STATUS "rife: building the TensorRT backend against ${RIFE_TRT_INCLUDE}")
  set_source_files_properties(src/rife.cpp PROPERTIES
    COMPILE_DEFINITIONS MOONGLIDE_HAVE_RIFE_TRT
    COMPILE_OPTIONS "-Wno-deprecated-declarations"
    INCLUDE_DIRECTORIES "${RIFE_TRT_INCLUDE};${NVVFX_CUDA_INCLUDE}")
else()
  message(STATUS "rife: TensorRT headers not found; backend stubbed")
endif()

# A stand-in PS5 for the Remote Play source (tests without a console).
add_executable(moonglide-rp-fake src/tools/rp_fake.cpp)
target_include_directories(moonglide-rp-fake PRIVATE src)
