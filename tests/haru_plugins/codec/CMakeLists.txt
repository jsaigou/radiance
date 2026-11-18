# The codec block's test-only architecture plugin, built into archtest so a normal run never loads
# it. The block itself lives in arch/haru_common and is compiled into this target by path.
set(RAD_PLUGIN_INSTALL OFF)
rad_add_plugin(radtest_haru_codec archtest
  SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/radtest_haru_codec.cpp
    ${CMAKE_SOURCE_DIR}/arch/haru_common/haru_codec.cpp
  INCLUDES ${CMAKE_SOURCE_DIR}/arch)
