# Test-only plugins for the Haru building blocks (arch/haru_common). One directory per block; a block's
# directory is added only when it has a CMakeLists.txt, so an unfinished block adds nothing. Like
# tests/duplex_plugin, these build into a kind directory the engine does not scan.
foreach(block codec head lm)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${block}/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${block} ${CMAKE_CURRENT_BINARY_DIR}/${block})
  endif()
endforeach()
