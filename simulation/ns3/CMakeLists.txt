cmake_minimum_required(VERSION 3.16)
project(uav-olsr-dataset CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate the Homebrew ns-3 installation. The shipped pkg-config files are
# broken (-linterface_libs-NOTFOUND), so link the ns-3 libraries directly.
if(NOT DEFINED NS3_PREFIX)
    execute_process(
        COMMAND brew --prefix ns-3
        OUTPUT_VARIABLE NS3_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE BREW_RESULT
    )
    if(NOT BREW_RESULT EQUAL 0)
        message(FATAL_ERROR "ns-3 not found. Install with: brew install ns-3 "
                            "(or pass -DNS3_PREFIX=/path/to/ns3)")
    endif()
endif()

file(GLOB NS3_LIBS "${NS3_PREFIX}/lib/libns3*.dylib" "${NS3_PREFIX}/lib/libns3*.so")
if(NS3_LIBS STREQUAL "")
    message(FATAL_ERROR "No ns-3 libraries found under ${NS3_PREFIX}/lib")
endif()

add_executable(uav-olsr-dataset uav-olsr-dataset.cc)
target_include_directories(uav-olsr-dataset PRIVATE "${NS3_PREFIX}/include")
target_link_libraries(uav-olsr-dataset PRIVATE ${NS3_LIBS})
