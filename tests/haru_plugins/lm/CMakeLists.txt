# A test-only architecture plugin. Its kind directory is `archtest`, which the engine does not
# scan, so a normal radiance run never loads it; tests/haru_lm_test.cpp gets it copied into a
# temporary RADIANCE_HOME/architectures by tests/haru_harness.h. The block's sources are compiled
# in by path from the plugin source (the hyv3_common convention), not listed here.
set(RAD_PLUGIN_INSTALL OFF)
rad_add_plugin(radtest_haru_lm archtest SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/radtest_haru_lm.cpp)
