cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(viam-mlmodelservice-triton
  DESCRIPTION "Viam Robotics Triton MLModelService"
  HOMEPAGE_URL https://github.com/viamrobotics/viam-mlmodelservice-triton
  LANGUAGES CXX
)

# Use of the C++ SDK requires C++14 or later. We make use of std::filesystem, which requires C++17 or later.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Produce a compilation database when generating for a build
# system that is able to make one.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Produce GNU-style variables for installation, and default the
# installation directory to be local to the build. If you intend to
# install elsewhere, pass an explicit argument to CMAKE_INSTALL_PREFIX
# on the command line:
#
# cmake ... -DCMAKE_INSTALL_PREFIX=$HOME/opt
#
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "..." FORCE)
endif()
include(GNUInstallDirs)


set(CMAKE_BUILD_WITH_INSTALL_RPATH false)
set(CMAKE_BUILD_RPATH_USE_ORIGIN true)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)
set(CMAKE_MACOSX_RPATH true)
set(CMAKE_SKIP_BUILD_RPATH false)

list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  # TODO: Is there a way to do this with generator expressions?
  file(RELATIVE_PATH BINTOPREFIXRELPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR} ${CMAKE_INSTALL_PREFIX})
  file(RELATIVE_PATH PREFIXTOLIBRELPATH ${CMAKE_INSTALL_PREFIX} ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
  if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(CMAKE_INSTALL_NAME_DIR "@rpath/${PREFIXTOLIBRELPATH}")
    list(PREPEND CMAKE_INSTALL_RPATH "@loader_path/${BINTOPREFIXRELPATH}")
  else()
    list(PREPEND CMAKE_INSTALL_RPATH "$ORIGIN/${BINTOPREFIXRELPATH}/${PREFIXTOLIBRELPATH}")
  endif()
endif()

# Everything needs threads, and prefer -pthread if available
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(viam-cpp-sdk CONFIG REQUIRED viamsdk)
find_package(CUDAToolkit REQUIRED)
find_package(RapidJSON REQUIRED)

add_subdirectory(src)
