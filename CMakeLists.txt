# CMakeLists.txt

# Set the minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(fraud-detection)

# Set the C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)

# Add the source files for the project (main.cpp and data_load.cxx)
add_executable(fraud_detection src/main.cpp src/data_load.cxx)

# Optionally, find Python3 and set it up if needed for direct invocation
find_package(Python3 REQUIRED)

# Link Python libraries if needed (optional, based on your use case)
target_include_directories(fraud_detection PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(fraud_detection PRIVATE ${Python3_LIBRARIES})

# Add a custom target to run the Python script after building the C++ project
add_custom_target(run_python
    COMMAND ${Python3_EXECUTABLE} python/model.py
    DEPENDS fraud_detection
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

