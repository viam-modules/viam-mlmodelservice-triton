// Copyright 2023 Viam Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "viam_mlmodelservice_triton.hpp"

#include <dlfcn.h>

#include <iostream>
#include <string>

namespace viam {
namespace mlmodelservice {
namespace triton {
namespace {

constexpr char service_name[] = "viam_mlmodelservice_triton";
const std::string usage = "usage: viam_mlmodelservice_triton /path/to/unix/socket";

}  // namespace
}  // namespace triton
}  // namespace mlmodelservice
}  // namespace viam

int main(int argc, char* argv[]) {
    using namespace viam::mlmodelservice::triton;

    if (argc < 2) {
        std::cout << service_name << "ERROR: insufficient arguments\n";
        std::cout << usage << std::endl;
        return EXIT_FAILURE;
    }

    // Please see
    // https://forums.developer.nvidia.com/t/symbol-resolution-conflicts-with-triton-server-for-jetpack-tensorflow-backend-grpc-protobuf-absl-etc/262468/4
    // for background on the problem being solved here. By binding to
    // Triton's symbols here and dependency injecting them into our
    // `impl` library, we make it so we can dlopen our implementation
    // RTLD_LOCAL. That ensures that symbols that we use are not made
    // available to the backend implementations like TensorFlow that
    // Triton Server will later dlopen itself.

    cxxapi::the_shim.ApiVersion = &TRITONSERVER_ApiVersion;

    cxxapi::the_shim.ErrorNew = &TRITONSERVER_ErrorNew;
    cxxapi::the_shim.ErrorCodeString = &TRITONSERVER_ErrorCodeString;
    cxxapi::the_shim.ErrorMessage = &TRITONSERVER_ErrorMessage;
    cxxapi::the_shim.ErrorDelete = &TRITONSERVER_ErrorDelete;

    cxxapi::the_shim.ServerOptionsNew = &TRITONSERVER_ServerOptionsNew;
    cxxapi::the_shim.ServerOptionsSetModelControlMode =
        &TRITONSERVER_ServerOptionsSetModelControlMode;
    cxxapi::the_shim.ServerOptionsSetBackendDirectory =
        &TRITONSERVER_ServerOptionsSetBackendDirectory;
    cxxapi::the_shim.ServerOptionsSetLogWarn = &TRITONSERVER_ServerOptionsSetLogWarn;
    cxxapi::the_shim.ServerOptionsSetLogError = &TRITONSERVER_ServerOptionsSetLogError;
    cxxapi::the_shim.ServerOptionsSetLogVerbose = &TRITONSERVER_ServerOptionsSetLogVerbose;
    cxxapi::the_shim.ServerOptionsSetMinSupportedComputeCapability =
        &TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability;
    cxxapi::the_shim.ServerOptionsSetModelRepositoryPath =
        &TRITONSERVER_ServerOptionsSetModelRepositoryPath;
    cxxapi::the_shim.ServerOptionsSetStrictModelConfig =
        &TRITONSERVER_ServerOptionsSetStrictModelConfig;
    cxxapi::the_shim.ServerOptionsDelete = &TRITONSERVER_ServerOptionsDelete;

    cxxapi::the_shim.ParameterNew = &TRITONSERVER_ParameterNew;
    cxxapi::the_shim.ParameterDelete = &TRITONSERVER_ParameterDelete;

    cxxapi::the_shim.ServerNew = &TRITONSERVER_ServerNew;
    cxxapi::the_shim.ServerIsLive = &TRITONSERVER_ServerIsLive;
    cxxapi::the_shim.ServerIsReady = &TRITONSERVER_ServerIsReady;
    cxxapi::the_shim.ServerLoadModel = &TRITONSERVER_ServerLoadModel;
    cxxapi::the_shim.ServerLoadModelWithParameters = &TRITONSERVER_ServerLoadModelWithParameters;
    cxxapi::the_shim.ServerModelIsReady = &TRITONSERVER_ServerModelIsReady;
    cxxapi::the_shim.ServerInferAsync = &TRITONSERVER_ServerInferAsync;
    cxxapi::the_shim.ServerUnloadModel = &TRITONSERVER_ServerUnloadModel;
    cxxapi::the_shim.ServerDelete = &TRITONSERVER_ServerDelete;

    cxxapi::the_shim.ServerModelMetadata = &TRITONSERVER_ServerModelMetadata;
    cxxapi::the_shim.MessageSerializeToJson = &TRITONSERVER_MessageSerializeToJson;
    cxxapi::the_shim.MessageDelete = &TRITONSERVER_MessageDelete;

    cxxapi::the_shim.ResponseAllocatorNew = &TRITONSERVER_ResponseAllocatorNew;
    cxxapi::the_shim.ResponseAllocatorSetQueryFunction =
        &TRITONSERVER_ResponseAllocatorSetQueryFunction;
    cxxapi::the_shim.ResponseAllocatorDelete = &TRITONSERVER_ResponseAllocatorDelete;

    cxxapi::the_shim.InferenceRequestNew = &TRITONSERVER_InferenceRequestNew;
    cxxapi::the_shim.InferenceRequestSetReleaseCallback =
        &TRITONSERVER_InferenceRequestSetReleaseCallback;
    cxxapi::the_shim.InferenceRequestRemoveAllInputs =
        &TRITONSERVER_InferenceRequestRemoveAllInputs;
    cxxapi::the_shim.InferenceRequestRemoveAllRequestedOutputs =
        &TRITONSERVER_InferenceRequestRemoveAllRequestedOutputs;
    cxxapi::the_shim.InferenceRequestAddInput = &TRITONSERVER_InferenceRequestAddInput;
    cxxapi::the_shim.InferenceRequestAppendInputData =
        &TRITONSERVER_InferenceRequestAppendInputData;
    cxxapi::the_shim.InferenceRequestAddRequestedOutput =
        &TRITONSERVER_InferenceRequestAddRequestedOutput;
    cxxapi::the_shim.InferenceRequestSetResponseCallback =
        &TRITONSERVER_InferenceRequestSetResponseCallback;
    cxxapi::the_shim.InferenceRequestDelete = &TRITONSERVER_InferenceRequestDelete;

    cxxapi::the_shim.InferenceResponseError = &TRITONSERVER_InferenceResponseError;
    cxxapi::the_shim.InferenceResponseOutputCount = &TRITONSERVER_InferenceResponseOutputCount;
    cxxapi::the_shim.InferenceResponseOutput = &TRITONSERVER_InferenceResponseOutput;
    cxxapi::the_shim.InferenceResponseDelete = &TRITONSERVER_InferenceResponseDelete;

    // TODO: This should probably account for windows/macOS someday.
    auto lvmti = dlopen("libviam_mlmodelservice_triton_impl.so", RTLD_LOCAL | RTLD_NOW);
    if (!lvmti) {
        const auto err = dlerror();
        std::cout << service_name
                  << ": Failed to open libviam_mlmodelservice_triton_impl.so Library: " << err;
        return EXIT_FAILURE;
    }

    const auto lvmti_serve = dlsym(lvmti, "viam_mlmodelservice_triton_serve");
    if (!lvmti_serve) {
        const auto err = dlerror();
        std::cout << service_name
                  << ": Failed to find entry point in implementation libray: " << err;
        return EXIT_FAILURE;
    }

    return reinterpret_cast<int (*)(cxxapi::shim*, int, char*[])>(lvmti_serve)(
        &cxxapi::the_shim, argc, argv);
}
