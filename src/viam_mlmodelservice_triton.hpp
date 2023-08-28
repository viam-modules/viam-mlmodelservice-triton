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

#pragma once

#include <memory>
#include <sstream>
#include <stdexcept>

#include <triton/core/tritonserver.h>

namespace viam {
namespace mlmodelservice {
namespace triton {
namespace cxxapi {

struct shim {
    decltype(TRITONSERVER_ApiVersion)* ApiVersion = nullptr;

    decltype(TRITONSERVER_ErrorNew)* ErrorNew = nullptr;
    decltype(TRITONSERVER_ErrorCodeString)* ErrorCodeString = nullptr;
    decltype(TRITONSERVER_ErrorMessage)* ErrorMessage = nullptr;
    decltype(TRITONSERVER_ErrorDelete)* ErrorDelete = nullptr;

    decltype(TRITONSERVER_ServerOptionsNew)* ServerOptionsNew = nullptr;
    decltype(TRITONSERVER_ServerOptionsSetBackendDirectory)* ServerOptionsSetBackendDirectory =
        nullptr;
    decltype(TRITONSERVER_ServerOptionsSetLogWarn)* ServerOptionsSetLogWarn = nullptr;
    decltype(TRITONSERVER_ServerOptionsSetLogError)* ServerOptionsSetLogError = nullptr;
    decltype(TRITONSERVER_ServerOptionsSetLogVerbose)* ServerOptionsSetLogVerbose = nullptr;
    decltype(TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability)*
        ServerOptionsSetMinSupportedComputeCapability = nullptr;
    decltype(TRITONSERVER_ServerOptionsSetModelRepositoryPath)*
        ServerOptionsSetModelRepositoryPath = nullptr;
    decltype(TRITONSERVER_ServerOptionsSetStrictModelConfig)* ServerOptionsSetStrictModelConfig =
        nullptr;
    decltype(TRITONSERVER_ServerOptionsDelete)* ServerOptionsDelete = nullptr;

    decltype(TRITONSERVER_ServerNew)* ServerNew = nullptr;
    decltype(TRITONSERVER_ServerIsLive)* ServerIsLive = nullptr;
    decltype(TRITONSERVER_ServerIsReady)* ServerIsReady = nullptr;
    decltype(TRITONSERVER_ServerModelIsReady)* ServerModelIsReady = nullptr;
    decltype(TRITONSERVER_ServerInferAsync)* ServerInferAsync = nullptr;
    decltype(TRITONSERVER_ServerDelete)* ServerDelete = nullptr;

    decltype(TRITONSERVER_ServerModelMetadata)* ServerModelMetadata = nullptr;
    decltype(TRITONSERVER_MessageSerializeToJson)* MessageSerializeToJson = nullptr;
    decltype(TRITONSERVER_MessageDelete)* MessageDelete = nullptr;

    decltype(TRITONSERVER_ResponseAllocatorNew)* ResponseAllocatorNew = nullptr;
    decltype(TRITONSERVER_ResponseAllocatorSetQueryFunction)* ResponseAllocatorSetQueryFunction =
        nullptr;
    decltype(TRITONSERVER_ResponseAllocatorDelete)* ResponseAllocatorDelete = nullptr;

    decltype(TRITONSERVER_InferenceRequestNew)* InferenceRequestNew = nullptr;
    decltype(TRITONSERVER_InferenceRequestSetReleaseCallback)* InferenceRequestSetReleaseCallback =
        nullptr;
    decltype(TRITONSERVER_InferenceRequestRemoveAllInputs)* InferenceRequestRemoveAllInputs =
        nullptr;
    decltype(TRITONSERVER_InferenceRequestRemoveAllRequestedOutputs)*
        InferenceRequestRemoveAllRequestedOutputs = nullptr;
    decltype(TRITONSERVER_InferenceRequestAddInput)* InferenceRequestAddInput = nullptr;
    decltype(TRITONSERVER_InferenceRequestAppendInputData)* InferenceRequestAppendInputData =
        nullptr;
    decltype(TRITONSERVER_InferenceRequestSetResponseCallback)*
        InferenceRequestSetResponseCallback = nullptr;
    decltype(TRITONSERVER_InferenceRequestDelete)* InferenceRequestDelete = nullptr;

    decltype(TRITONSERVER_InferenceResponseError)* InferenceResponseError = nullptr;
    decltype(TRITONSERVER_InferenceResponseOutputCount)* InferenceResponseOutputCount = nullptr;
    decltype(TRITONSERVER_InferenceResponseOutput)* InferenceResponseOutput = nullptr;
    decltype(TRITONSERVER_InferenceResponseDelete)* InferenceResponseDelete = nullptr;
};

shim the_shim;

// Declare this here so we can use it, but the implementation relies on subsequent specializaitons.
template <typename Stdex = std::runtime_error, typename... Args>
[[gnu::warn_unused_result]] constexpr auto call(TRITONSERVER_Error* (*fn)(Args... args)) noexcept;

template <typename T>
struct lifecycle_traits;

template <typename T>
struct traits {
    using lifecycle = lifecycle_traits<T>;

    struct deleter_type {
        void operator()(T* t) const {
            lifecycle::dtor(t);
        }
    };

    using unique_ptr = std::unique_ptr<typename lifecycle::value_type, deleter_type>;
};

template <typename T>
using unique_ptr = typename traits<T>::unique_ptr;

template <typename T, class... Args>
unique_ptr<T> make_unique(Args&&... args) {
    using lifecycle = typename traits<T>::lifecycle;
    using unique_ptr = unique_ptr<T>;
    return unique_ptr(lifecycle::ctor(std::forward<Args>(args)...));
}

template <typename T>
unique_ptr<T> take_unique(T* t) {
    using lifecycle = typename traits<T>::lifecycle;
    using unique_ptr = unique_ptr<T>;
    return unique_ptr(t);
}

template <typename T, class... Args>
std::shared_ptr<T> make_shared(Args&&... args) {
    using lifecycle = typename traits<T>::lifecycle;
    return std::shared_ptr<T>(lifecycle::ctor(std::forward<Args>(args)...), lifecycle::dtor);
}

template <>
struct lifecycle_traits<TRITONSERVER_Error> {
    using value_type = TRITONSERVER_Error;
    template <class... Args>
    static auto ctor(Args&&... args) {
        return the_shim.ErrorNew(std::forward<Args>(args)...);
    }
    template <class... Args>
    static auto dtor(Args&&... args) {
        return the_shim.ErrorDelete(std::forward<Args>(args)...);
    }
};

template <>
struct lifecycle_traits<TRITONSERVER_ServerOptions> {
    using value_type = TRITONSERVER_ServerOptions;
    template <class... Args>
    static auto ctor(Args&&... args) {
        TRITONSERVER_ServerOptions* opts = nullptr;
        call(the_shim.ServerOptionsNew)(&opts, std::forward<Args>(args)...);
        return opts;
    }
    template <class... Args>
    static auto dtor(Args&&... args) {
        call(the_shim.ServerOptionsDelete)(std::forward<Args>(args)...);
    }
};

template <>
struct lifecycle_traits<TRITONSERVER_Server> {
    using value_type = TRITONSERVER_Server;
    template <class... Args>
    static auto ctor(Args&&... args) {
        TRITONSERVER_Server* server = nullptr;
        call(the_shim.ServerNew)(&server, std::forward<Args>(args)...);
        return server;
    }
    template <class... Args>
    static auto dtor(Args&&... args) {
        call(the_shim.ServerDelete)(std::forward<Args>(args)...);
    }
};

template <>
struct lifecycle_traits<TRITONSERVER_ResponseAllocator> {
    using value_type = TRITONSERVER_ResponseAllocator;
    template <class... Args>
    static auto ctor(Args&&... args) {
        value_type* pself;
        call(the_shim.ResponseAllocatorNew)(&pself, std::forward<Args>(args)...);
        return pself;
    }
    template <class... Args>
    static auto dtor(Args&&... args) {
        call(the_shim.ResponseAllocatorDelete)(std::forward<Args>(args)...);
    }
};

template <>
struct lifecycle_traits<TRITONSERVER_InferenceRequest> {
    using value_type = TRITONSERVER_InferenceRequest;
    template <class... Args>
    static auto ctor(Args&&... args) {
        value_type* pself;
        call(the_shim.InferenceRequestNew)(&pself, std::forward<Args>(args)...);
        return pself;
    };
    template <class... Args>
    static auto dtor(Args&&... args) {
        call(the_shim.InferenceRequestDelete)(std::forward<Args>(args)...);
    };
};

template <>
struct lifecycle_traits<TRITONSERVER_InferenceResponse> {
    using value_type = TRITONSERVER_InferenceResponse;
    template <class... Args>
    static auto dtor(Args&&... args) {
        call(the_shim.InferenceResponseDelete)(std::forward<Args>(args)...);
    };
};

template <>
struct lifecycle_traits<TRITONSERVER_Message> {
    using value_type = TRITONSERVER_Message;
    template <class... Args>
    static auto dtor(Args&&... args) {
        call(the_shim.MessageDelete)(std::forward<Args>(args)...);
    };
};

template <typename Stdex = std::runtime_error, typename... Args>
[[gnu::warn_unused_result]] constexpr auto call(TRITONSERVER_Error* (*fn)(Args... args)) noexcept {
    // NOTE: The lack of perfect forwarding here is deliberate.
    return [=](Args... args) {
        const auto error = take_unique(fn(args...));
        if (error) {
            std::ostringstream buffer;
            buffer << ": Triton Server Error: " << the_shim.ErrorCodeString(error.get()) << " - "
                   << the_shim.ErrorMessage(error.get());
            throw Stdex(buffer.str());
        }
    };
}

}  // namespace cxxapi
}  // namespace triton
}  // namespace mlmodelservice
}  // namespace viam
