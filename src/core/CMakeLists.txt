# Core executor infrastructure and CPU implementations are always part of the
# fasteig library.
file(GLOB FASTEIG_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

target_sources(fasteig PRIVATE ${FASTEIG_CORE_SOURCES})

