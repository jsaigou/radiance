# A test-only architecture plugin. Its kind directory is `archtest`, which the engine does not
# scan, so a normal radiance run never loads it; tests/duplex_plugin_test.cpp copies it into a
# temporary RADIANCE_HOME/architectures.
set(RAD_PLUGIN_INSTALL OFF)
rad_add_plugin(radtest_duplex archtest SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/radtest_duplex.cpp)
