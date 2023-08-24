add_library(viam_mlmodelservice_triton_impl SHARED)

target_sources(viam_mlmodelservice_triton_impl
  PRIVATE viam_mlmodelservice_triton_impl.cpp
)

target_include_directories(viam_mlmodelservice_triton_impl
  # TODO: TritonServer installation location variable
  PRIVATE /opt/tritonserver/include/tritonserver
  PRIVATE ${RAPIDJSON_INCLUDE_DIRS}
)

target_link_libraries(viam_mlmodelservice_triton_impl
  PRIVATE Threads::Threads
  PRIVATE viam-cpp-sdk::viamsdk
  PRIVATE CUDA::toolkit
)

install(
  TARGETS viam_mlmodelservice_triton_impl
)

add_executable(viam_mlmodelservice_triton)

target_sources(viam_mlmodelservice_triton
  PRIVATE viam_mlmodelservice_triton.cpp
)

target_include_directories(viam_mlmodelservice_triton
  PRIVATE /opt/tritonserver/include/tritonserver
)

target_link_directories(viam_mlmodelservice_triton
  PRIVATE /opt/tritonserver/lib
)

target_link_libraries(viam_mlmodelservice_triton
  PRIVATE Threads::Threads
  PRIVATE ${CMAKE_DL_LIBS}
  PRIVATE tritonserver
)

install(
  TARGETS viam_mlmodelservice_triton
  COMPONENT runtime
)
