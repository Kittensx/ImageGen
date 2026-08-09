cmake_minimum_required(VERSION 3.20)
project(imagegen_discord_helper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED DISCORD_SDK_ROOT)
  message(FATAL_ERROR "Set DISCORD_SDK_ROOT to the extracted Discord Social SDK directory.")
endif()

set(DISCORD_SDK_INCLUDE_DIR "${DISCORD_SDK_ROOT}/include")
set(DISCORD_SDK_LIB_DIR "${DISCORD_SDK_ROOT}/lib/release")
set(DISCORD_SDK_BIN_DIR "${DISCORD_SDK_ROOT}/bin/release")

if(NOT EXISTS "${DISCORD_SDK_INCLUDE_DIR}/discordpp.h")
  message(FATAL_ERROR "discordpp.h was not found under ${DISCORD_SDK_INCLUDE_DIR}.")
endif()

if(WIN32)
  set(DISCORD_SDK_LIBRARY "${DISCORD_SDK_LIB_DIR}/discord_partner_sdk.lib")
  set(DISCORD_SDK_SHARED "${DISCORD_SDK_BIN_DIR}/discord_partner_sdk.dll")
elseif(APPLE)
  set(DISCORD_SDK_LIBRARY "${DISCORD_SDK_LIB_DIR}/libdiscord_partner_sdk.dylib")
  set(DISCORD_SDK_SHARED "${DISCORD_SDK_LIB_DIR}/libdiscord_partner_sdk.dylib")
else()
  set(DISCORD_SDK_LIBRARY "${DISCORD_SDK_LIB_DIR}/libdiscord_partner_sdk.so")
  set(DISCORD_SDK_SHARED "${DISCORD_SDK_LIB_DIR}/libdiscord_partner_sdk.so")
endif()

if(NOT EXISTS "${DISCORD_SDK_LIBRARY}")
  message(FATAL_ERROR "Discord Social SDK library was not found at ${DISCORD_SDK_LIBRARY}.")
endif()
if(NOT EXISTS "${DISCORD_SDK_SHARED}")
  message(FATAL_ERROR "Discord Social SDK runtime was not found at ${DISCORD_SDK_SHARED}.")
endif()

add_executable(imagegen_discord_helper imagegen_discord_helper.cpp)
target_include_directories(imagegen_discord_helper PRIVATE "${DISCORD_SDK_INCLUDE_DIR}")
target_link_libraries(imagegen_discord_helper PRIVATE "${DISCORD_SDK_LIBRARY}")

if(UNIX)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  if(APPLE)
    set(CMAKE_INSTALL_RPATH "@executable_path")
  else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
  endif()
endif()

if(MSVC)
  target_compile_options(imagegen_discord_helper PRIVATE /W4 /EHsc)
else()
  target_compile_options(imagegen_discord_helper PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_custom_command(TARGET imagegen_discord_helper POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${DISCORD_SDK_SHARED}"
          "$<TARGET_FILE_DIR:imagegen_discord_helper>"
)
