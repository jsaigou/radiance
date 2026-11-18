# A test-only duplex tenant for the Haru MeanFlow head and its audio projectors. Its kind
# directory is `archtest`, which the engine does not scan, so a normal radiance run never loads it;
# tests/haru_head_test.cpp copies it into a temporary RADIANCE_HOME/architectures.
#
# The block's own sources live in arch/haru_common and are compiled in by path, as arch/hyv3_common
# is by its users.
set(RAD_PLUGIN_INSTALL OFF)
rad_add_plugin(radtest_haru_head archtest SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/radtest_haru_head.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../arch/haru_common/haru_head.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../arch/haru_common/haru_proj.cpp)
