# Phase 4 (Weeks 9-10): high-performance Monte Carlo engine.
# Builds the `mc_engine` Python module via pybind11.
#
# Build (once sources exist):
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#
cmake_minimum_required(VERSION 3.18)
project(mc_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# pybind11 (vendored via `pip install pybind11` then `pybind11-config`, or
# add_subdirectory of a git submodule). Uncomment when sources are added:
# find_package(pybind11 CONFIG REQUIRED)

# OpenMP for path parallelism (Phase 4).
find_package(OpenMP)

# pybind11_add_module(mc_engine
#   src/mc_engine.cpp
#   src/swaption_pricer.cpp
#   src/bindings.cpp
# )
# if(OpenMP_CXX_FOUND)
#   target_link_libraries(mc_engine PRIVATE OpenMP::OpenMP_CXX)
# endif()
