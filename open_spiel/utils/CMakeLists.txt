add_library (utils OBJECT
  circular_buffer.h
  combinatorics.h
  combinatorics.cc
  data_logger.h
  data_logger.cc
  file.h
  file.cc
  functional.h
  init.h
  init.cc
  json.h
  json.cc
  logger.h
  lru_cache.h
  random.h
  random.cc
  run_python.h
  run_python.cc
  serializable_circular_buffer.h
  serialization.h
  stats.h
  tensor_view.h
  thread.h
  thread.cc
  threaded_queue.h
)
target_include_directories (utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


if (OPEN_SPIEL_BUILD_WITH_PYTHON)
  add_executable(run_python_test run_python_test.cc ${OPEN_SPIEL_OBJECTS}
                 $<TARGET_OBJECTS:tests>)
  add_test(run_python_test run_python_test
           --python_command=${Python3_EXECUTABLE})
  set_property(TEST run_python_test
      PROPERTY ENVIRONMENT
      PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../..;
      TEST_SRCDIR=${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

if (OPEN_SPIEL_BUILD_WITH_LIBNOP)
  add_executable(serializable_circular_buffer_test
                 serializable_circular_buffer_test.cc ${OPEN_SPIEL_OBJECTS}
                 $<TARGET_OBJECTS:tests>)
  add_test(serializable_circular_buffer_test serializable_circular_buffer_test)
endif()

# Failing on Ubuntu 18.04 since upgrade of abseil version (2021-05-17).
# Disabling while we look into it.
# add_executable(thread_test thread_test.cc ${OPEN_SPIEL_OBJECTS}
#                $<TARGET_OBJECTS:tests>)
# add_test(thread_test thread_test)