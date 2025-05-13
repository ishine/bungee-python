cmake_minimum_required(VERSION 3.15)
project(bungee-python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_subdirectory(extern/pybind11)


add_subdirectory(bungee-core)


pybind11_add_module(bungee src/bindings.cpp)

target_link_libraries(bungee PRIVATE bungee_library)
target_include_directories(bungee PRIVATE bungee-core)


set_target_properties(bungee PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH ON
)

if(APPLE)
  set_target_properties(bungee PROPERTIES
      INSTALL_NAME_DIR "@rpath"
  )
endif()



install(TARGETS bungee bungee_library
        LIBRARY DESTINATION bungee_python
        RUNTIME DESTINATION bungee_python
        ARCHIVE DESTINATION bungee_python
)

# 复制.so库到bungee_python目录
install(CODE "
    file(GLOB BUNGEE_LIB_FILES \${CMAKE_BINARY_DIR}/*bungee*.so* \${CMAKE_BINARY_DIR}/*bungee*.dll* \${CMAKE_BINARY_DIR}/*bungee*.dylib*)
    foreach(LIB_FILE \${BUNGEE_LIB_FILES})
        file(COPY \${LIB_FILE} DESTINATION \${CMAKE_INSTALL_PREFIX}/bungee_python/
             FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
        message(STATUS \"Copied library: \${LIB_FILE} to \${CMAKE_INSTALL_PREFIX}/bungee_python/\")
    endforeach()
")

install(FILES
    src/bungee_python/bungee.pyi
    src/bungee_python/__init__.py
    src/bungee_python/py.typed
    DESTINATION bungee_python
)