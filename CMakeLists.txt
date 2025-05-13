cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_subdirectory(extern/pybind11)


add_subdirectory(bungee-core)


pybind11_add_module(bungee src/bindings.cpp)

target_link_libraries(bungee PRIVATE bungee_library)
target_include_directories(bungee PRIVATE bungee-core)


# Install the Python extension module
install(TARGETS bungee
        LIBRARY DESTINATION bungee_python
        RUNTIME DESTINATION bungee_python
        COMPONENT bungee_python)

# Install the bungee_library (assuming it's a shared library bungee.so depends on)
# The RPATH settings in pyproject.toml and CMAKE_INSTALL_RPATH should help bungee.so find it
# if they are in the same directory within the wheel.
install(TARGETS bungee_library
        LIBRARY DESTINATION bungee_python
        RUNTIME DESTINATION bungee_python
        COMPONENT bungee_python)