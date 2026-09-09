cmake_minimum_required(VERSION 4.2)
project(BOSSTransactionLogEngine)

############################## Custom build modes ###############################

set(CMAKE_CXX_FLAGS_SANITIZE "-fno-omit-frame-pointer -fsanitize=address,signed-integer-overflow,null,alignment,bounds,function,return,vla-bound -O0 -g"
  CACHE STRING
  "Flags used by the C++ compiler during Sanitize builds."
  FORCE)
set(CMAKE_C_FLAGS_SANITIZE "-fno-omit-frame-pointer -fsanitize=address,signed-integer-overflow,null,alignment,bounds,function,return,vla-bound -O0 -g"
  CACHE STRING
  "Flags used by the C compiler during Sanitize builds."
  FORCE)
set(CMAKE_EXE_LINKER_FLAGS_SANITIZE
  ${CMAKE_EXE_LINKER_FLAGS_DEBUG} CACHE STRING
  "Flags used for linking binaries during Sanitize builds."
  FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_SANITIZE
  ${CMAKE_SHARED_LINKER_FLAGS_DEBUG} CACHE STRING
  "Flags used by the shared libraries linker during Sanitize builds."
  FORCE)
mark_as_advanced(
  CMAKE_CXX_FLAGS_SANITIZE		  CMAKE_EXE_LINKER_FLAGS_SANITIZE
  CMAKE_C_FLAGS_SANITIZE		  CMAKE_SHARED_LINKER_FLAGS_SANITIZE
  )

set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
  "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Sanitize."
  FORCE)

############################### External Projects ###############################

include(ExternalProject)
set(BOSS_SOURCE_BRANCH new-table-format CACHE STRING "if boss core is built from source, which branch should be fetched")
set(BOSS_SOURCE_REPOSITORY https://github.com/symbol-store/BOSS.git CACHE STRING "url for the boss repository")
ExternalProject_Add(BOSS
  GIT_REPOSITORY ${BOSS_SOURCE_REPOSITORY}
  DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/downloads/${CMAKE_PROJECT_NAME}
  GIT_TAG ${BOSS_SOURCE_BRANCH}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${BOSSTransactionLogEngine_BINARY_DIR}/deps $<IF:$<CONFIG:>,,-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}> -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
  )


#################################### Targets ####################################
if(MSVC)
  # not making any difference on Windows
  # and helps Visual Studio to parse it correctly as a target
  set(LibraryType SHARED)
else()
  set(LibraryType MODULE)
endif(MSVC)
add_library(BOSSTransactionLogEngine ${LibraryType} Source/BOSSTransactionLogEngine.cpp)

set_property(TARGET BOSSTransactionLogEngine PROPERTY CXX_STANDARD 23)
if(MSVC)
  target_compile_options(BOSSTransactionLogEngine PUBLIC "/Zc:__cplusplus")
  target_compile_options(BOSSTransactionLogEngine PUBLIC "/EHsc")
endif(MSVC)
target_include_directories(BOSSTransactionLogEngine SYSTEM PUBLIC ${BOSSTransactionLogEngine_BINARY_DIR}/deps/include)
add_dependencies(BOSSTransactionLogEngine BOSS)

set_target_properties(BOSSTransactionLogEngine PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS BOSSTransactionLogEngine LIBRARY DESTINATION lib)

# Interactive BOSS-expression client. It is optional so the existing engine-only
# configure/build path remains unchanged when a demonstration is not required.
option(BUILD_BOSS_DEMO "Build the interactive BOSS transaction demonstration" OFF)
if(BUILD_BOSS_DEMO)
  set(BOSS_DEMO_CORE_DIR "${BOSSTransactionLogEngine_BINARY_DIR}/deps" CACHE PATH
    "Installed BOSS core used to compile the interactive demonstration")
  set(BOSS_DEMO_CORE_LIBRARY
    "${BOSS_DEMO_CORE_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}BOSS${CMAKE_SHARED_LIBRARY_SUFFIX}")

  add_executable(BOSSDemo Source/BOSSDemo.cpp)
  set_property(TARGET BOSSDemo PROPERTY CXX_STANDARD 23)
  target_include_directories(BOSSDemo SYSTEM PRIVATE ${BOSS_DEMO_CORE_DIR}/include)
  target_link_libraries(BOSSDemo PRIVATE ${BOSS_DEMO_CORE_LIBRARY} ${CMAKE_DL_LIBS})
  if(BOSS_DEMO_CORE_DIR STREQUAL "${BOSSTransactionLogEngine_BINARY_DIR}/deps")
    add_dependencies(BOSSDemo BOSS)
  endif()
  set_target_properties(BOSSDemo PROPERTIES
    BUILD_RPATH ${BOSS_DEMO_CORE_DIR}/lib
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
  install(TARGETS BOSSDemo RUNTIME DESTINATION bin)
endif()
