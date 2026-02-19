# MODULE 12: Build Systems
# File: CMakeLists.txt
#
# TOPIC: CMake Build Configuration
#
# USAGE:
#   mkdir build && cd build
#   cmake ..
#   make
#   ./myapp

cmake_minimum_required(VERSION 3.10)
project(GPUProgramming VERSION 1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
file(GLOB SOURCES "src/*.cpp")

# Executable
add_executable(myapp ${SOURCES})

# Optional: Find OpenGL
find_package(OpenGL)
if(OPENGL_FOUND)
    target_link_libraries(myapp ${OPENGL_LIBRARIES})
    target_include_directories(myapp PRIVATE ${OPENGL_INCLUDE_DIR})
endif()

# Optional: Find CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    file(GLOB CUDA_SOURCES "src/*.cu")
    add_executable(gpu_app ${CUDA_SOURCES})
    set_target_properties(gpu_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Install target
install(TARGETS myapp DESTINATION bin)

# TRY THIS:
# 1. Add multiple executables
# 2. Create library target
# 3. Add unit tests with CTest
# 4. Add external dependencies (glm, glfw)
# 5. Add custom CUDA compilation flags

# Example for GLFW:
# find_package(glfw3 REQUIRED)
# target_link_libraries(myapp glfw)

# Example for GLM:
# find_package(glm REQUIRED)
# target_include_directories(myapp PRIVATE ${GLM_INCLUDE_DIRS})

# Example CUDA flags:
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75 --expt-relaxed-constexpr")

# Build instructions:
# mkdir build
# cd build
# cmake ..
# make
# ./myapp
