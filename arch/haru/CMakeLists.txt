# The assembled Haru duplex tenant. Its blocks live in arch/haru_common and are compiled in by
# path, as arch/hyv3_common is by its users.
rad_add_plugin(haru architectures SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/haru.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../haru_common/haru_codec.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../haru_common/haru_head.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../haru_common/haru_lm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../haru_common/haru_proj.cpp)
