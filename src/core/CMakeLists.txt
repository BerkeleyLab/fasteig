# ------------------------------------------------------------------------------
# Core sources - compiled directly into the fasteig target
# ------------------------------------------------------------------------------
file(GLOB FASTEIG_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
)

if(NOT FASTEIG_CORE_SOURCES)
    message(FATAL_ERROR "No source files found in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

target_sources(fasteig PRIVATE ${FASTEIG_CORE_SOURCES})

# Only needed if core/ contains headers that are included as "foo.hpp"
# rather than "core/foo.hpp" from include/. Drop this otherwise.
target_include_directories(fasteig PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
