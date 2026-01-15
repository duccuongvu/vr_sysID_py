cmake_minimum_required(VERSION 3.12)
project(validate_regressor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Aggressive optimization flags for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -march=native -flto -funroll-loops -finline-functions -fipa-pta -fomit-frame-pointer -mfpmath=sse -ftree-vectorize")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native -flto -funroll-loops -finline-functions -fipa-pta -fomit-frame-pointer -mfpmath=sse -ftree-vectorize")

# Set output directory to build/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/include
)

# Regressor source files (in sysid/py/autogen)
set(RIGHT_ARM_REGRESSOR_SRCS
    autogen/VRM2RightArm/right_arm_regressor.c
    autogen/VRM2RightArm/right_arm_tau_from_regressor.c
    autogen/VRM2RightArm/right_arm_params_to_X.c
)

set(LEFT_ARM_REGRESSOR_SRCS
    autogen/VRM2LeftArm/left_arm_regressor.c
    autogen/VRM2LeftArm/left_arm_tau_from_regressor.c
    autogen/VRM2LeftArm/left_arm_params_to_X.c
)

# Dynamics source files (in lib/include)
set(RIGHT_ARM_DYNAMICS_SRCS
    ../../lib/include/VRM2RightArm/right_arm_massMatrix.c
    ../../lib/include/VRM2RightArm/right_arm_coriolisVector.c
    ../../lib/include/VRM2RightArm/right_arm_gravityVector.c
)

set(LEFT_ARM_DYNAMICS_SRCS
    ../../lib/include/VRM2LeftArm/left_arm_massMatrix.c
    ../../lib/include/VRM2LeftArm/left_arm_coriolisVector.c
    ../../lib/include/VRM2LeftArm/left_arm_gravityVector.c
)

# All source files
set(ALL_SRCS
    validate_regressor.cpp
    ${RIGHT_ARM_REGRESSOR_SRCS}
    ${LEFT_ARM_REGRESSOR_SRCS}
    ${RIGHT_ARM_DYNAMICS_SRCS}
    ${LEFT_ARM_DYNAMICS_SRCS}
)

# Create executable
add_executable(validate_regressor ${ALL_SRCS})

# Compiler flags
target_compile_options(validate_regressor PRIVATE
    -Wall
    -Wextra
    -O2
)

# Find CasADi for sysID_right_arm
list(APPEND CMAKE_PREFIX_PATH $ENV{HOME}/.local)
find_package(casadi REQUIRED)

# Find cJSON (needed for sysID_right_arm_with_friction and plot_robot_data)
# Try pkg-config first (most reliable)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(CJSON libcjson)
endif()

# Fallback to manual search if pkg-config didn't work
if(NOT CJSON_FOUND)
    find_library(CJSON_LIBRARY cjson PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
    find_path(CJSON_INCLUDE_DIR cJSON.h 
        PATHS 
        /usr/include/cjson
        /usr/local/include/cjson
        /usr/include
        /usr/local/include
    )
    
    if(CJSON_LIBRARY AND CJSON_INCLUDE_DIR)
        set(CJSON_LIBRARIES ${CJSON_LIBRARY})
        set(CJSON_INCLUDE_DIRS ${CJSON_INCLUDE_DIR})
        set(CJSON_FOUND TRUE)
    endif()
endif()

if(CJSON_FOUND)
    message(STATUS "Found cJSON")
    message(STATUS "  Include dirs: ${CJSON_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${CJSON_LIBRARIES}")
else()
    message(WARNING "cJSON not found. sysID_right_arm_with_friction and plot_robot_data will not be built.")
    message(STATUS "To install cJSON: sudo apt-get install libcjson-dev")
endif()

# Create executable for sysID_right_arm
add_executable(sysID_right_arm sysID_right_arm.cpp)

# Link CasADi
target_link_libraries(sysID_right_arm PRIVATE casadi::casadi)

# Get library directory for RPATH
get_target_property(CASADI_TARGET_LOCATION casadi::casadi IMPORTED_LOCATION)
get_filename_component(CASADI_RPATH_DIR ${CASADI_TARGET_LOCATION} DIRECTORY)

# Set RPATH so CasADi plugin libraries can be found at runtime
# Include both ~/.local/lib and /usr/local/lib for plugins
set_target_properties(sysID_right_arm PROPERTIES
    BUILD_RPATH "${CASADI_RPATH_DIR}:/usr/local/lib"
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH "${CASADI_RPATH_DIR}:/usr/local/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Compiler flags
target_compile_options(sysID_right_arm PRIVATE
    -Wall
    -Wextra
    -O2
)

# Create executable for sysID_right_arm_with_friction
add_executable(sysID_right_arm_with_friction sysID_right_arm_with_friction.cpp ${RIGHT_ARM_REGRESSOR_SRCS})

# Link CasADi and cJSON
target_link_libraries(sysID_right_arm_with_friction PRIVATE casadi::casadi)
if(CJSON_FOUND)
    target_include_directories(sysID_right_arm_with_friction PRIVATE ${CJSON_INCLUDE_DIRS})
    target_link_libraries(sysID_right_arm_with_friction PRIVATE ${CJSON_LIBRARIES})
else()
    message(FATAL_ERROR "cJSON is required for sysID_right_arm_with_friction. Please install libcjson-dev.")
endif()

# Set RPATH for CasADi plugins
set_target_properties(sysID_right_arm_with_friction PROPERTIES
    BUILD_RPATH "${CASADI_RPATH_DIR}:/usr/local/lib"
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH "${CASADI_RPATH_DIR}:/usr/local/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Compiler flags (Release builds will use CMAKE_CXX_FLAGS_RELEASE with aggressive optimizations)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(sysID_right_arm_with_friction PRIVATE
        -Wall
        -Wextra
    )
else()
    target_compile_options(sysID_right_arm_with_friction PRIVATE
        -Wall
        -Wextra
        -O2
    )
endif()

# Create executable for plot_robot_data (cJSON already found above)
if(CJSON_FOUND)
    add_executable(plot_robot_data plot_robot_data.c)
    target_include_directories(plot_robot_data PRIVATE ${CJSON_INCLUDE_DIRS})
    target_link_libraries(plot_robot_data PRIVATE ${CJSON_LIBRARIES} m)
    target_compile_options(plot_robot_data PRIVATE
        -Wall
        -Wextra
        -O2
    )
    message(STATUS "Building plot_robot_data")
else()
    message(WARNING "cJSON not found. plot_robot_data will not be built.")
    message(STATUS "To install cJSON: sudo apt-get install libcjson-dev")
endif()

# Print build information
message(STATUS "Building validate_regressor")
message(STATUS "  CXX: ${CMAKE_CXX_COMPILER}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

