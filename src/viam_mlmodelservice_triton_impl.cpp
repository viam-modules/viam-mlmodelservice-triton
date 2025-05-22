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

#include "viam_mlmodelservice_triton_impl.hpp"

#include <pthread.h>
#include <signal.h>

#include <cuda_runtime_api.h>

#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <stdexcept>

#include <grpcpp/channel.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include <viam/sdk/common/instance.hpp>
#include <viam/sdk/components/component.hpp>
#include <viam/sdk/config/resource.hpp>
#include <viam/sdk/module/service.hpp>
#include <viam/sdk/registry/registry.hpp>
#include <viam/sdk/resource/reconfigurable.hpp>
#include <viam/sdk/resource/stoppable.hpp>
#include <viam/sdk/rpc/server.hpp>
#include <viam/sdk/services/mlmodel.hpp>

#include "viam_mlmodelservice_triton.hpp"

namespace viam {
namespace mlmodelservice {
namespace triton {
namespace {

namespace vsdk = ::viam::sdk;
constexpr char service_name[] = "viam_mlmodelservice_triton";

template <typename Stdex = std::runtime_error, typename... Args>
auto call_cuda(cudaError_t (*fn)(Args... args)) noexcept {
    return [=](Args... args) {
        const auto error = fn(args...);
        if (error != cudaSuccess) {
            std::ostringstream buffer;
            buffer << cudaGetErrorString(error);
            throw Stdex(buffer.str());
        }
    };
}

// An MLModelService instance which runs many models via the NVidia
// Triton Server API.
//
// Please see
// https://github.com/viamrobotics/viam-mlmodelservice-triton/blob/main/README.md
// for configuration parameters and deployment guidelines.
//
// NOTE: At one time, `MLModelService` required implementation of
// `Stoppable`/`Reconfigurable`, so this class implemented the
// necessary behaviors. Later, the C++ SDK was refactored such that
// `MLModelService` implementations no longer automatically derived
// from `Stoppable` and `Reconfigurable`. It would have been possible
// to remove the supporting code from `Server`. However, is seems
// better to retain it, in case support is ever needed again. So,
// `Service` here explicitly derives from `Stoppable` and
// `Reconfigurable`, but in practice those methods are unreachable.
class Service : public vsdk::MLModelService, public vsdk::Stoppable, public vsdk::Reconfigurable {
   public:
    explicit Service(vsdk::Dependencies dependencies, vsdk::ResourceConfig configuration)
        : MLModelService(configuration.name()),
          state_(configure_(std::move(dependencies), std::move(configuration))) {}

    ~Service() final {
        stop();
    }

    void stop(const vsdk::ProtoStruct& extra) noexcept final {
        // Before we change the state of the class (especially by modifying the model repository
        // used by Triton itself), wait until all old inferences are finished.
        const std::unique_lock<std::shared_mutex> state_wlock(state_lock_);
        stop_locked_(state_wlock);
    }

    using Stoppable::stop;

    void reconfigure(const vsdk::Dependencies& dependencies,
                     const vsdk::ResourceConfig& configuration) final try {
        // Before we change the state of the class (especially by modifying the model repository
        // used by Triton itself), wait until all old inferences are finished.
        const std::unique_lock<std::shared_mutex> state_wlock(state_lock_);
        stop_locked_(state_wlock);
        state_ = configure_(std::move(dependencies), std::move(configuration));
    } catch (...) {
        // If reconfiguration fails for any reason, become stopped and rethrow.
        stop();
        throw;
    }

    std::shared_ptr<named_tensor_views> infer(const named_tensor_views& inputs,
                                              const vsdk::ProtoStruct& extra) final {
        // Acquire a shared lock on the mutex. This allows multiple `infer` calls to run in
        // parallel, but does not allow `reconfigure` to run in parallel with any other call.
        std::shared_lock<std::shared_mutex> state_rlock(state_lock_);

        if (!state_) {
            std::ostringstream buffer;
            buffer << service_name
                   << ": cannot call `infer` on a stopped or unconfigured triton service";
            throw std::runtime_error(buffer.str());
        }

        auto inference_request = get_inference_request_();

        // Attach inputs to the inference request
        for (const auto& kv : inputs) {
            const std::string* input_name = &kv.first;
            const auto where = state_->input_name_remappings_reversed.find(*input_name);
            if (where != state_->input_name_remappings_reversed.end()) {
                input_name = &where->second;
            }
            inference_request_input_visitor_ visitor(input_name, inference_request.get());
            boost::apply_visitor(visitor, kv.second);
        }

        std::promise<TRITONSERVER_InferenceResponse*> inference_promise;
        auto inference_future = inference_promise.get_future();

        cxxapi::call(cxxapi::the_shim.InferenceRequestSetResponseCallback)(
            inference_request.get(),
            state_->allocator.get(),
            state_.get(),
            [](TRITONSERVER_InferenceResponse* response,
               const uint32_t flags,
               void* userp) noexcept {
                auto* promise =
                    reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
                promise->set_value(response);
            },
            &inference_promise);

        cxxapi::call(cxxapi::the_shim.ServerInferAsync)(
            state_->server.get(), inference_request.release(), nullptr);
        auto inference_response = cxxapi::take_unique(inference_future.get());
        auto error = cxxapi::the_shim.InferenceResponseError(inference_response.get());
        if (error) {
            std::ostringstream buffer;
            buffer << ": Triton Server Inference Error: " << cxxapi::the_shim.ErrorCodeString(error)
                   << " - " << cxxapi::the_shim.ErrorMessage(error);
            throw std::runtime_error(buffer.str());
        }

        // The result object we return to our parent. We keep our
        // lease on the state to anchor the existence of everything
        // else. We will take ownership of the inference response so
        // it doesn't return to the pool, which would clear the output
        // state. We have a slot for an accumulated buffers we need to
        // own for copies out of the GPU. And then we have the tensor
        // views that cover buffers either in the inference response
        // or in the bufs.
        struct inference_result_type {
            std::shared_lock<std::shared_mutex> state_rlock;
            decltype(inference_response) ir;
            std::vector<std::unique_ptr<unsigned char[]>> bufs;
            named_tensor_views ntvs;
        };
        auto inference_result = std::make_shared<inference_result_type>();

        std::uint32_t outputs = 0;
        cxxapi::call(cxxapi::the_shim.InferenceResponseOutputCount)(inference_response.get(),
                                                                    &outputs);

        for (decltype(outputs) output = 0; output != outputs; ++output) {
            const char* output_name_cstr;
            TRITONSERVER_DataType data_type;
            const std::int64_t* shape;
            std::uint64_t shape_size;
            const void* data;
            std::size_t data_bytes;
            TRITONSERVER_MemoryType memory_type;
            std::int64_t memory_type_id;
            void* userp;

            cxxapi::call(cxxapi::the_shim.InferenceResponseOutput)(inference_response.get(),
                                                                   output,
                                                                   &output_name_cstr,
                                                                   &data_type,
                                                                   &shape,
                                                                   &shape_size,
                                                                   &data,
                                                                   &data_bytes,
                                                                   &memory_type,
                                                                   &memory_type_id,
                                                                   &userp);

            if (!output_name_cstr) {
                std::ostringstream buffer;
                buffer << service_name
                       << ": Model returned anonymous output tensor which cannot be returned ";
                throw std::runtime_error(buffer.str());
            }

            std::string output_name(output_name_cstr);
            const auto where = state_->output_name_remappings.find(output_name);
            if (where != state_->output_name_remappings.end()) {
                output_name = where->second;
            }

            // If the memory is on the GPU we need to copy it out, since the higher
            // level doesn't know that it can't just memcpy.
            //
            // TODO: We could save a copy here if we got smarter. Maybe. We would need a way
            // to tunnel back to the upper layer (mlmodel/server.cpp) that it needs to use
            // `cudaMemcpy` when writing to the protobuf.
            if (memory_type == TRITONSERVER_MEMORY_GPU) {
                inference_result->bufs.push_back(std::make_unique<unsigned char[]>(data_bytes));
                auto allocated = reinterpret_cast<void*>(inference_result->bufs.back().get());
                call_cuda(cudaMemcpy)(allocated, data, data_bytes, cudaMemcpyDeviceToHost);
                data = allocated;
            }

            std::vector<std::size_t> shape_vector;
            shape_vector.reserve(shape_size);
            for (size_t i = 0; i != shape_size; ++i) {
                auto val = shape[i];
                if (val < 0) {
                    std::ostringstream buffer;
                    buffer << service_name << ": Model returned negative value " << val
                           << " in shape for output " << output_name;
                    throw std::runtime_error(buffer.str());
                }
                shape_vector.push_back(static_cast<std::size_t>(val));
            }

            auto tv = make_tensor_view_(data_type, data, data_bytes, std::move(shape_vector));

            inference_result->ntvs.emplace(std::move(output_name), std::move(tv));
        }

        // Move the read lock on `state` and ownership of `inference_response` into
        // `inference_result`. Otherwise, the result would return to the pool and our
        // views would no longer be valid.
        inference_result->state_rlock = std::move(state_rlock);
        inference_result->ir = std::move(inference_response);

        // Finally, construct an aliasing shared_ptr which appears to
        // the caller as a shared_ptr to views, but in fact manages
        // the lifetime of the inference_result. When the
        // inference_result object is destroyed, we will return
        // the response to the pool.
        auto* const ntvs = &inference_result->ntvs;
        return {std::move(inference_result), ntvs};
    }

    struct metadata metadata(const vsdk::ProtoStruct& extra) final {
        // Acquire a shared lock on the mutex. This allows multiple `metadata` and `infer` calls to
        // run in parallel, but does not allow `reconfigure` to run in parallel with any other
        // call.
        const std::shared_lock<std::shared_mutex> state_rlock(state_lock_);

        if (!state_) {
            std::ostringstream buffer;
            buffer << service_name
                   << ": cannot call `metadata` on a stopped or unconfigured triton service";
            throw std::runtime_error(buffer.str());
        }

        return state_->metadata;
    }

   private:
    struct state_;

    void stop_locked_(const std::unique_lock<std::shared_mutex>&) noexcept {
        try {
            // Take ownership of the current state so it will be
            // destroyed when we leave the block.
            const std::unique_ptr<struct state_> state = std::move(state_);

            if (state) {
                // Remove any symlinks we've added to our module workspace.
                // TODO: Can this be managed by `struct state` now?
                remove_repo_symlink_(*state);
            }
        } catch (...) {
        }
    }

    static std::filesystem::path get_module_data_path_(const struct state_& state) {
        // The overall Viam config might have multiple Triton components that each run on separate
        // GPUs. Each one gets its own subdirectory within our module data to avoid hitting the
        // others.
        return std::filesystem::path(std::getenv("VIAM_MODULE_DATA")) / state.configuration.name();
    }

    static void symlink_mlmodel_(struct state_& state) {
        const auto& attributes = state.configuration.attributes();

        auto model_path = attributes.find("model_path");
        if (model_path == attributes.end()) {
            std::ostringstream buffer;
            buffer << service_name
                   << ": Required parameter `model_path` not found in configuration";
            throw std::invalid_argument(buffer.str());
        }

        const std::string* model_path_string = model_path->second.get<std::string>();
        if (!model_path_string || model_path_string->empty()) {
            std::ostringstream buffer;
            buffer << service_name
                   << ": Required non-empty string parameter `model_path` is either not "
                      "a string or is an empty string";
            throw std::invalid_argument(buffer.str());
        }

        // The user doesn't have a way to set the version number: they've downloaded the only
        // version available to them. So, set the version to 1.
        const std::string model_version = "1";
        state.model_version = 1;

        // If there exists a `saved_model.pb` file in the model path, this is a TensorFlow model.
        // In that case, Triton uses a different directory structure compared to all other models.
        // For details, see
        // https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html#model-files
        auto saved_model_pb_path = std::filesystem::path(*model_path_string) / "saved_model.pb";
        const bool is_tf = std::filesystem::exists(saved_model_pb_path);

        std::filesystem::path directory_name = get_module_data_path_(state);
        state.model_repo_path = directory_name.string();

        directory_name /= state.model_name;
        if (is_tf) {
            directory_name /= model_version;
        }
        std::filesystem::create_directories(directory_name);

        if (is_tf) {
            directory_name /= "model.savedmodel";
        } else {
            directory_name /= model_version;
        }
        const std::string triton_name = directory_name.string();

        if (std::filesystem::exists(triton_name)) {
            // TODO: make a backup copy instead of deleting
            const bool success = std::filesystem::remove(triton_name);
            if (!success) {
                std::ostringstream buffer;
                buffer << service_name << ": Unable to delete old model symlink";
                throw std::invalid_argument(buffer.str());
            }
        }
        std::filesystem::create_directory_symlink(*model_path_string, triton_name);
    }

    static void remove_repo_symlink_(const struct state_& state) {
        // There might be old models left over from a previous run in which our component was
        // killed without exiting smoothly. Remove everything in our directory.
        // Note that remove_all succeeds even if the directory it was given doesn't exist!
        std::filesystem::remove_all(get_module_data_path_(state));
    }

    static std::unique_ptr<struct state_> configure_(vsdk::Dependencies dependencies,
                                                     vsdk::ResourceConfig configuration) {
        auto state =
            std::make_unique<struct state_>(std::move(dependencies), std::move(configuration));

        // Validate that our dependencies (if any - we don't actually
        // expect any for this service) exist. If we did have
        // Dependencies this is where we would have an opportunity to
        // downcast them to the right thing and store them in our
        // state so we could use them as needed.
        //
        // TODO(RSDK-3601): Validating that dependencies are present
        // should be handled by the ModuleService automatically,
        // rather than requiring each component to validate the
        // presence of dependencies.
        for (const auto& kv : state->dependencies) {
            if (!kv.second) {
                std::ostringstream buffer;
                buffer << service_name << ": Dependency "
                       << "`" << kv.first.to_string() << "` was not found during (re)configuration";
                throw std::invalid_argument(buffer.str());
            }
        }

        const auto& attributes = state->configuration.attributes();

        // Pull the model name out of the configuration.
        auto model_name = attributes.find("model_name");
        if (model_name == attributes.end()) {
            std::ostringstream buffer;
            buffer << service_name
                   << ": Required parameter `model_name` not found in configuration";
            throw std::invalid_argument(buffer.str());
        }

        auto* const model_name_string = model_name->second.get<std::string>();
        if (!model_name_string || model_name_string->empty()) {
            std::ostringstream buffer;
            buffer << service_name
                   << ": Required non-empty string parameter `model_name` is either not a "
                      "string "
                      "or is an empty string";
            throw std::invalid_argument(buffer.str());
        }
        state->model_name = std::move(*model_name_string);

        // Pull the model repository path out of the configuration.
        auto model_repo_path = attributes.find("model_repository_path");
        if (model_repo_path == attributes.end()) {
            // With no model repository path, we try to construct our own by symlinking a single
            // model path. This line also sets state.model_repo_path and state.model_version.
            symlink_mlmodel_(*state.get());
        } else {
            // If the model_repository_path is specified, forbid specifying the model_path.
            if (attributes.find("model_path") != attributes.end()) {
                std::ostringstream buffer;
                buffer << service_name
                       << ": Both the `model_repository_path` and `model_path` are set, "
                          "but we expect only one or the other.";
                throw std::invalid_argument(buffer.str());
            }

            auto* const model_repo_path_string = model_repo_path->second.get<std::string>();
            if (!model_repo_path_string || model_repo_path_string->empty()) {
                std::ostringstream buffer;
                buffer << service_name
                       << ": Non-empty string parameter `model_repository_path` is either not "
                          "a string or is an empty string";
                throw std::invalid_argument(buffer.str());
            }
            state->model_repo_path = std::move(*model_repo_path_string);

            // If you specify your own model repo path, you may specify your own model version
            // number, too.
            auto model_version = attributes.find("model_version");
            if (model_version != attributes.end()) {
                auto* const model_version_value = model_version->second.get<double>();
                if (!model_version_value || (*model_version_value < 1) ||
                    (std::nearbyint(*model_version_value) != *model_version_value)) {
                    std::ostringstream buffer;
                    buffer << service_name
                           << ": Optional parameter `model_version` was provided, but is not a "
                              "natural number";
                    throw std::invalid_argument(buffer.str());
                }
                state->model_version = static_cast<std::int64_t>(*model_version_value);
            }
        }

        // Pull the backend directory out of the configuration, if provided.
        auto backend_directory = attributes.find("backend_directory");
        if (backend_directory != attributes.end()) {
            auto* const backend_directory_string = backend_directory->second.get<std::string>();
            if (!backend_directory_string || backend_directory_string->empty()) {
                std::ostringstream buffer;
                buffer << service_name
                       << ": Configuration parameter `backend_directory` is either not a"
                          "string "
                          "or is an empty string";
                throw std::invalid_argument(buffer.str());
            }
            state->backend_directory = std::move(*backend_directory_string);
        }

        // Process any tensor name remappings provided in the config.
        auto remappings = attributes.find("tensor_name_remappings");
        if (remappings != attributes.end()) {
            const auto remappings_attributes = remappings->second.get<vsdk::ProtoStruct>();
            if (!remappings_attributes) {
                std::ostringstream buffer;
                buffer << service_name
                       << ": Optional parameter `tensor_name_remappings` must be a dictionary";
                throw std::invalid_argument(buffer.str());
            }

            const auto populate_remappings = [](const vsdk::ProtoValue& source,
                                                auto& target,
                                                auto& inv_target) {
                const auto source_attributes = source.get<vsdk::ProtoStruct>();
                if (!source_attributes) {
                    std::ostringstream buffer;
                    buffer << service_name
                           << ": Fields `inputs` and `outputs` of `tensor_name_remappings` must be "
                              "dictionaries";
                    throw std::invalid_argument(buffer.str());
                }
                for (const auto& kv : *source_attributes) {
                    const auto& k = kv.first;
                    const auto* const kv_string = kv.second.get<std::string>();
                    if (!kv_string) {
                        std::ostringstream buffer;
                        buffer
                            << service_name
                            << ": Fields `inputs` and `outputs` of `tensor_name_remappings` must "
                               "be dictionaries with string values";
                        throw std::invalid_argument(buffer.str());
                    }
                    target[kv.first] = *kv_string;
                    inv_target[*kv_string] = kv.first;
                }
            };

            const auto inputs_where = remappings_attributes->find("inputs");
            if (inputs_where != remappings_attributes->end()) {
                populate_remappings(inputs_where->second,
                                    state->input_name_remappings,
                                    state->input_name_remappings_reversed);
            }
            const auto outputs_where = remappings_attributes->find("outputs");
            if (outputs_where != remappings_attributes->end()) {
                populate_remappings(outputs_where->second,
                                    state->output_name_remappings,
                                    state->output_name_remappings_reversed);
            }
        }

        auto allocator = cxxapi::make_unique<TRITONSERVER_ResponseAllocator>(
            allocate_response_, deallocate_response_, nullptr);

        auto server_options = cxxapi::make_unique<TRITONSERVER_ServerOptions>();

        // TODO: We should probably pool servers based on repo path
        // and backend directory.
        cxxapi::call(cxxapi::the_shim.ServerOptionsSetModelRepositoryPath)(
            server_options.get(), state->model_repo_path.c_str());

        cxxapi::call(cxxapi::the_shim.ServerOptionsSetBackendDirectory)(
            server_options.get(), state->backend_directory.c_str());

        cxxapi::call(cxxapi::the_shim.ServerOptionsSetLogWarn)(server_options.get(), true);
        cxxapi::call(cxxapi::the_shim.ServerOptionsSetLogError)(server_options.get(), true);

        // Needed so we can load a tensorflow model without a config file
        cxxapi::call(cxxapi::the_shim.ServerOptionsSetStrictModelConfig)(server_options.get(),
                                                                         false);

        auto server = cxxapi::make_unique<TRITONSERVER_Server>(server_options.get());

        // TODO(RSDK-4663): For now, we are hardcoding a wait that is
        // a subset of the RDK default timeout.
        const size_t timeout_seconds = 30;
        bool result = false;
        for (size_t tries = 0; !result && (tries != timeout_seconds); ++tries) {
            cxxapi::call(cxxapi::the_shim.ServerIsLive)(server.get(), &result);
            if (result) {
                cxxapi::call(cxxapi::the_shim.ServerModelIsReady)(
                    server.get(), state->model_name.c_str(), state->model_version, &result);
                if (!result) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                }
            }
        }
        if (!result) {
            std::ostringstream buffer;
            buffer << service_name << "Triton Server did not become live and ready within "
                   << timeout_seconds << " seconds.";
            throw std::runtime_error(buffer.str());
        }

        cxxapi::unique_ptr<TRITONSERVER_Message> model_metadata_message;
        {
            TRITONSERVER_Message* out = nullptr;
            cxxapi::call(cxxapi::the_shim.ServerModelMetadata)(
                server.get(), state->model_name.c_str(), state->model_version, &out);
            model_metadata_message = cxxapi::take_unique(out);
        }

        const char* model_metadata_json_bytes;
        size_t model_metadata_json_size;
        cxxapi::call(cxxapi::the_shim.MessageSerializeToJson)(
            model_metadata_message.get(), &model_metadata_json_bytes, &model_metadata_json_size);

        rapidjson::Document model_metadata_json;
        model_metadata_json.Parse(model_metadata_json_bytes, model_metadata_json_size);
        if (model_metadata_json.HasParseError()) {
            std::ostringstream buffer;
            buffer << service_name
                   << ": Failed parsing model metadata returned by triton server at offset "
                   << model_metadata_json.GetErrorOffset() << ": "
                   << rapidjson::GetParseError_En(model_metadata_json.GetParseError());
            throw std::runtime_error(buffer.str());
        }

        const auto populate_tensor_infos = [&model_metadata_json](const auto& array,
                                                                  const auto& name_remappings,
                                                                  auto* tensor_infos) {
            static const std::map<std::string, MLModelService::tensor_info::data_types>
                datatype_map = {{"UINT8", MLModelService::tensor_info::data_types::k_uint8},
                                {"UINT16", MLModelService::tensor_info::data_types::k_uint16},
                                {"UINT32", MLModelService::tensor_info::data_types::k_uint32},
                                {"UINT64", MLModelService::tensor_info::data_types::k_uint64},
                                {"INT8", MLModelService::tensor_info::data_types::k_int8},
                                {"INT16", MLModelService::tensor_info::data_types::k_int16},
                                {"INT32", MLModelService::tensor_info::data_types::k_int32},
                                {"INT64", MLModelService::tensor_info::data_types::k_int64},
                                {"FP32", MLModelService::tensor_info::data_types::k_float32},
                                {"FP64", MLModelService::tensor_info::data_types::k_float64}};

            for (const auto& element : array.GetArray()) {
                if (!element.IsObject()) {
                    std::ostringstream buffer;
                    buffer << service_name
                           << ": Model metadata array is expected to contain object fields";
                    throw std::runtime_error(buffer.str());
                }
                if (!element.HasMember("name")) {
                    std::ostringstream buffer;
                    buffer << service_name << ": Model metadata entry has no `name` field";
                    throw std::runtime_error(buffer.str());
                }
                const auto& name_element = element["name"];
                if (!name_element.IsString()) {
                    std::ostringstream buffer;
                    buffer << service_name << ": Model metadata entry `name` field is not a string";
                    throw std::runtime_error(buffer.str());
                }
                const auto name = name_element.GetString();
                auto viam_name = name;
                const auto name_remappings_where = name_remappings.find(name);
                if (name_remappings_where != name_remappings.end()) {
                    viam_name = name_remappings_where->second.c_str();
                }

                if (!element.HasMember("datatype")) {
                    std::ostringstream buffer;
                    buffer << service_name << ": Model metadata entry for tensor `" << name
                           << "` does not have a `datatype` field";
                    throw std::runtime_error(buffer.str());
                }
                const auto& datatype_element = element["datatype"];
                if (!datatype_element.IsString()) {
                    std::ostringstream buffer;
                    buffer << service_name << ": Model metadata entry `datatype` field for tensor `"
                           << name << "` is not a string";
                    throw std::runtime_error(buffer.str());
                }
                const auto& triton_datatype = datatype_element.GetString();
                const auto datatype_map_where = datatype_map.find(triton_datatype);
                if (datatype_map_where == datatype_map.end()) {
                    std::ostringstream buffer;
                    buffer << service_name << ": Model metadata entry `datatype` field for tensor `"
                           << name << "` contains unsupported data type `" << triton_datatype
                           << "`";
                    throw std::runtime_error(buffer.str());
                }
                const auto viam_datatype = datatype_map_where->second;

                if (!element.HasMember("shape")) {
                    std::ostringstream buffer;
                    buffer << service_name << ": Model metadata entry for tensor `" << name
                           << "` does not have a `shape` field";
                    throw std::runtime_error(buffer.str());
                }
                const auto& shape_element = element["shape"];
                if (!shape_element.IsArray()) {
                    std::ostringstream buffer;
                    buffer << service_name << ": Model metadata entry `shape` field for tensor `"
                           << name << "` is not an array";
                    throw std::runtime_error(buffer.str());
                }

                std::vector<int> shape;
                for (const auto& shape_element_entry : shape_element.GetArray()) {
                    if (!shape_element_entry.IsInt()) {
                        std::ostringstream buffer;
                        buffer << service_name
                               << ": Model metadata entry `shape` field for tensor `" << name
                               << "` contained a non-integer value";
                    }
                    shape.push_back(shape_element_entry.GetInt());
                }

                tensor_infos->push_back({
                    // `name`
                    viam_name,

                    // `description`
                    "",

                    // `data_type`
                    viam_datatype,

                    // `shape`
                    std::move(shape),
                });
            }
        };

        if (!model_metadata_json.HasMember("inputs")) {
            std::ostringstream buffer;
            buffer << service_name << ": Model metadata does not include an `inputs` field";
            throw std::runtime_error(buffer.str());
        }
        const auto& inputs = model_metadata_json["inputs"];
        if (!inputs.IsArray()) {
            std::ostringstream buffer;
            buffer << service_name << ": Model metadata `inputs` field is not an array";
            throw std::runtime_error(buffer.str());
        }
        populate_tensor_infos(inputs, state->input_name_remappings, &state->metadata.inputs);

        if (!model_metadata_json.HasMember("outputs")) {
            std::ostringstream buffer;
            buffer << service_name << ": Model metadata does not include an `outputs` field";
            throw std::runtime_error(buffer.str());
        }
        const auto& outputs = model_metadata_json["outputs"];
        if (!outputs.IsArray()) {
            std::ostringstream buffer;
            buffer << service_name << ": Model metadata `outputs` field is not an array";
            throw std::runtime_error(buffer.str());
        }
        populate_tensor_infos(outputs, state->output_name_remappings, &state->metadata.outputs);

        state->allocator = std::move(allocator);
        state->server = std::move(server);

        return state;
    }

    static TRITONSERVER_Error* allocate_response_(TRITONSERVER_ResponseAllocator* allocator,
                                                  const char* tensor_name,
                                                  std::size_t byte_size,
                                                  TRITONSERVER_MemoryType memory_type,
                                                  std::int64_t memory_type_id,
                                                  void* userp,
                                                  void** buffer,
                                                  void** buffer_userp,
                                                  TRITONSERVER_MemoryType* actual_memory_type,
                                                  std::int64_t* actual_memory_type_id) noexcept {
        auto* const state = reinterpret_cast<struct state_*>(userp);
        *buffer_userp = state;

        if (!byte_size) {
            *buffer = nullptr;
            return nullptr;
        }

        switch (memory_type) {
            case TRITONSERVER_MEMORY_CPU_PINNED:  // Fallthrough
            case TRITONSERVER_MEMORY_GPU: {
                auto cuda_error = cudaSetDevice(memory_type_id);
                if (cuda_error != cudaSuccess) {
                    std::ostringstream buffer;
                    buffer << service_name
                           << ": Failed in `cudaSetDevice`` while allocating for response: "
                           << cudaGetErrorString(cuda_error);
                    return cxxapi::the_shim.ErrorNew(TRITONSERVER_ERROR_UNAVAILABLE,
                                                     buffer.str().c_str());
                }
                cuda_error = (memory_type == TRITONSERVER_MEMORY_CPU_PINNED)
                                 ? cudaHostAlloc(buffer, byte_size, cudaHostAllocPortable)
                                 : cudaMalloc(buffer, byte_size);
                if (cuda_error != cudaSuccess) {
                    std::ostringstream buffer;
                    buffer << service_name
                           << ": Failed in `cuda[HostA|Ma]lloc` while allocating for response: "
                           << cudaGetErrorString(cuda_error);
                    return cxxapi::the_shim.ErrorNew(TRITONSERVER_ERROR_UNAVAILABLE,
                                                     buffer.str().c_str());
                }
                break;
            }
            case TRITONSERVER_MEMORY_CPU:  // Fallthrough
            default: {
                memory_type = TRITONSERVER_MEMORY_CPU;
                *buffer = std::malloc(byte_size);
                if (!*buffer) {
                    std::ostringstream buffer;
                    buffer << service_name
                           << ": Failed in `std::malloc` while allocating for response";
                    return cxxapi::the_shim.ErrorNew(TRITONSERVER_ERROR_UNAVAILABLE,
                                                     buffer.str().c_str());
                }
                break;
            }
        }

        // We made it!
        *buffer_userp = state;
        *actual_memory_type = memory_type;
        *actual_memory_type_id = memory_type_id;

        return nullptr;
    }

    static TRITONSERVER_Error* deallocate_response_(TRITONSERVER_ResponseAllocator* allocator,
                                                    void* buffer,
                                                    void* buffer_userp,
                                                    std::size_t byte_size,
                                                    TRITONSERVER_MemoryType memory_type,
                                                    std::int64_t memory_type_id) noexcept {
        auto* const state = reinterpret_cast<struct state_*>(buffer_userp);

        switch (memory_type) {
            case TRITONSERVER_MEMORY_CPU_PINNED:  // Fallthrough
            case TRITONSERVER_MEMORY_GPU: {
                auto cuda_error = cudaSetDevice(memory_type_id);
                if (cuda_error != cudaSuccess) {
                    std::cerr << service_name
                              << ": Failed to obtain cuda device when deallocating response data: `"
                              << cudaGetErrorString(cuda_error) << "` - terminating" << std::endl;
                    std::abort();
                }
                auto cudaFreeFn =
                    (memory_type == TRITONSERVER_MEMORY_CPU_PINNED) ? cudaFreeHost : cudaFree;
                cuda_error = cudaFreeFn(buffer);
                if (cuda_error != cudaSuccess) {
                    std::cerr << service_name
                              << ": Failed cudaFree[host] when deallocating response data: `"
                              << cudaGetErrorString(cuda_error) << "` - terminating" << std::endl;
                    std::abort();
                }
                break;
            }
            case TRITONSERVER_MEMORY_CPU: {
                std::free(buffer);
                break;
            }
            default: {
                std::cerr << service_name
                          << ": Cannot honor request to deallocate unknown MemoryType "
                          << memory_type << " - terminating" << std::endl;
                std::abort();
            }
        }

        return nullptr;
    }

    cxxapi::unique_ptr<TRITONSERVER_InferenceRequest> get_inference_request_() {
        cxxapi::unique_ptr<TRITONSERVER_InferenceRequest> result;
        {
            const std::unique_lock<std::mutex> lock(state_->mutex);
            if (!state_->inference_requests.empty()) {
                result = std::move(state_->inference_requests.top());
                state_->inference_requests.pop();
                return result;
            }
        }

        result = cxxapi::make_unique<TRITONSERVER_InferenceRequest>(
            state_->server.get(), state_->model_name.c_str(), state_->model_version);

        cxxapi::call(cxxapi::the_shim.InferenceRequestSetReleaseCallback)(
            result.get(), &release_inference_request_, state_.get());

        return result;
    }

    static void release_inference_request_(TRITONSERVER_InferenceRequest* request,
                                           const uint32_t flags,
                                           void* userp) noexcept {
        if (flags != TRITONSERVER_REQUEST_RELEASE_ALL) {
            std::cerr << service_name
                      << ": Unhandled request to release inference request without RELEASE_ALL - "
                         "terminating"
                      << std::endl;
            std::abort();
        }

        try {
            auto taken = cxxapi::take_unique(request);
            cxxapi::call(cxxapi::the_shim.InferenceRequestRemoveAllInputs)(taken.get());
            cxxapi::call(cxxapi::the_shim.InferenceRequestRemoveAllRequestedOutputs)(taken.get());
            auto* const state = reinterpret_cast<struct state_*>(userp);
            const std::unique_lock<std::mutex> lock(state->mutex);
            // TODO: Should there be a maximum pool size?
            state->inference_requests.push(std::move(taken));
        } catch (...) {
            std::cerr << service_name
                      << ": Attempt to return an InferenceRequest to the pool failed" << std::endl;
        }
    }

    class inference_request_input_visitor_ : public boost::static_visitor<void> {
       public:
        inference_request_input_visitor_(const std::string* name,
                                         TRITONSERVER_InferenceRequest* request)
            : name_(name), request_(request) {}

        template <typename T>
        void operator()(const T& mlmodel_tensor) const {
            // TODO: Should we just eat the copy rather than reinterpreting?
            cxxapi::call(cxxapi::the_shim.InferenceRequestAddInput)(
                request_,
                name_->c_str(),
                triton_datatype_for_(mlmodel_tensor),
                reinterpret_cast<const int64_t*>(mlmodel_tensor.shape().data()),
                mlmodel_tensor.shape().size());

            const auto* const mlmodel_data_begin =
                reinterpret_cast<const unsigned char*>(mlmodel_tensor.data());
            const auto* const mlmodel_data_end = reinterpret_cast<const unsigned char*>(
                mlmodel_tensor.data() + mlmodel_tensor.size());
            const auto mlmodel_data_size =
                static_cast<size_t>(mlmodel_data_end - mlmodel_data_begin);

            cxxapi::call(cxxapi::the_shim.InferenceRequestAppendInputData)(request_,
                                                                           name_->c_str(),
                                                                           mlmodel_data_begin,
                                                                           mlmodel_data_size,
                                                                           TRITONSERVER_MEMORY_CPU,
                                                                           0);
        }

       private:
        template <typename T>
        using tv = vsdk::MLModelService::tensor_view<T>;

        static TRITONSERVER_DataType triton_datatype_for_(const tv<std::int8_t>& t) {
            return TRITONSERVER_TYPE_INT8;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<std::uint8_t>& t) {
            return TRITONSERVER_TYPE_UINT8;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<std::int16_t>& t) {
            return TRITONSERVER_TYPE_INT16;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<std::uint16_t>& t) {
            return TRITONSERVER_TYPE_UINT16;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<std::int32_t>& t) {
            return TRITONSERVER_TYPE_INT32;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<std::uint32_t>& t) {
            return TRITONSERVER_TYPE_UINT32;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<std::int64_t>& t) {
            return TRITONSERVER_TYPE_INT64;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<std::uint64_t>& t) {
            return TRITONSERVER_TYPE_UINT64;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<float>& t) {
            return TRITONSERVER_TYPE_FP32;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<double>& t) {
            return TRITONSERVER_TYPE_FP64;
        }

        const std::string* name_;
        TRITONSERVER_InferenceRequest* request_;
    };

    static MLModelService::tensor_views make_tensor_view_(TRITONSERVER_DataType data_type,
                                                          const void* data,
                                                          size_t data_bytes,
                                                          std::vector<std::size_t>&& shape_vector) {
        switch (data_type) {
            case TRITONSERVER_TYPE_INT8: {
                return make_tensor_view_t_<std::int8_t>(data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_UINT8: {
                return make_tensor_view_t_<std::uint8_t>(data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_INT16: {
                return make_tensor_view_t_<std::int16_t>(data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_UINT16: {
                return make_tensor_view_t_<std::uint16_t>(
                    data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_INT32: {
                return make_tensor_view_t_<std::int32_t>(data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_UINT32: {
                return make_tensor_view_t_<std::uint32_t>(
                    data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_INT64: {
                return make_tensor_view_t_<std::int64_t>(data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_UINT64: {
                return make_tensor_view_t_<std::uint64_t>(
                    data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_FP32: {
                return make_tensor_view_t_<float>(data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_FP64: {
                return make_tensor_view_t_<double>(data, data_bytes, std::move(shape_vector));
            }
            default: {
                std::ostringstream buffer;
                buffer << service_name
                       << ": Model returned unsupported tflite data type: " << data_type;
                throw std::invalid_argument(buffer.str());
            }
        }
    }

    template <typename T>
    static MLModelService::tensor_views make_tensor_view_t_(
        const void* data, size_t data_bytes, std::vector<std::size_t>&& shape_vector) {
        const auto* const typed_data = reinterpret_cast<const T*>(data);
        const auto typed_size = data_bytes / sizeof(*typed_data);
        return MLModelService::make_tensor_view(typed_data, typed_size, std::move(shape_vector));
    }

    // All of the meaningful internal state of the service is held in
    // a separate state object so we can keep our current state alive
    // while building a new one during reconfiguration, and then
    // atomically swap it in on success. Existing invocations will
    // continue to work against the old state, and new invocations
    // will pick up the new state.
    struct state_ {
        explicit state_(vsdk::Dependencies dependencies, vsdk::ResourceConfig configuration)
            : dependencies{std::move(dependencies)}, configuration{std::move(configuration)} {}

        // The dependencies and configuration we were given at
        // construction / reconfiguration.
        vsdk::Dependencies dependencies;
        vsdk::ResourceConfig configuration;

        // The path to the model repository. The provided directory must
        // meet the layout requirements for a triton model repository. See
        //
        // https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md
        std::string model_repo_path;

        // The path to the backend directory containing execution backends.
        std::string backend_directory = kDefaultBackendDirectory;

        // The name of the specific model that this instance of the
        // triton service will bind to.
        std::string model_name;

        // The model version. Defaults to `-1` to mean the latest deployed
        // version of the named model, but can be set explicitly in the configuration
        // to bind to an older version.
        std::int64_t model_version = -1;

        // Metadata about input and output tensors that was extracted
        // during configuration. Callers need this in order to know
        // how to interact with the service.
        struct MLModelService::metadata metadata;

        // Tensor renamings as extracted from our configuration. The
        // keys are the names of the tensors per the model, the values
        // are the names of the tensors clients expect to see / use
        // (e.g. a vision service component expecting a tensor named
        // `image`).
        std::unordered_map<std::string, std::string> input_name_remappings;
        std::unordered_map<std::string, std::string> output_name_remappings;

        // As above, but inverted.
        std::unordered_map<std::string, std::string> input_name_remappings_reversed;
        std::unordered_map<std::string, std::string> output_name_remappings_reversed;

        // The response allocator and server this state will use.
        //
        // TODO: Pooling?
        cxxapi::unique_ptr<TRITONSERVER_ResponseAllocator> allocator;
        cxxapi::unique_ptr<TRITONSERVER_Server> server;

        // Inference requests are pooled and reused.
        std::mutex mutex;
        std::stack<cxxapi::unique_ptr<TRITONSERVER_InferenceRequest>> inference_requests;
    };

    // The shared mutex allows concurrent access to `state_` by
    // readers in `infer` and `metadata` but grants exclusive access
    // in `close` and `reconfigure`.
    std::shared_mutex state_lock_;

    // The current state, if we have one. Held by `unique_ptr` rather than `optional` because
    // the `mutex` in `struct state` inhibits mobility.
    std::unique_ptr<struct state_> state_;
};

int serve(int argc, char* argv[]) noexcept try {
    const vsdk::Instance instance;

    // Validate that the version of the triton server that we are
    // running against is sufficient w.r.t. the version we were built
    // against.
    std::uint32_t triton_version_major;
    std::uint32_t triton_version_minor;
    cxxapi::call(cxxapi::the_shim.ApiVersion)(&triton_version_major, &triton_version_minor);

    if ((TRITONSERVER_API_VERSION_MAJOR != triton_version_major) ||
        (TRITONSERVER_API_VERSION_MINOR > triton_version_minor)) {
        std::ostringstream buffer;
        buffer << service_name << ": Triton server API version mismatch: need "
               << TRITONSERVER_API_VERSION_MAJOR << "." << TRITONSERVER_API_VERSION_MINOR
               << " but have " << triton_version_major << "." << triton_version_minor << ".";
        throw std::domain_error(buffer.str());
    }
    std::cout << service_name << ": Running Triton API: " << triton_version_major << "."
              << triton_version_minor << std::endl;

    // Create a new model registration for the service.
    auto module_registration = std::make_shared<vsdk::ModelRegistration>(
        // Identify that this resource offers the MLModelService API
        vsdk::API::get<vsdk::MLModelService>(),

        // Declare a model triple for this service.
        vsdk::Model{"viam", "mlmodelservice", "triton"},

        // Define the factory for instances of the resource.
        [](vsdk::Dependencies deps, vsdk::ResourceConfig config) {
            return std::make_shared<Service>(std::move(deps), std::move(config));
        });

    // Construct the module service and tell it where to place the socket path.
    std::vector<std::shared_ptr<vsdk::ModelRegistration>> mrs = {module_registration};
    auto module_service = std::make_shared<vsdk::ModuleService>(argc, argv, mrs);
    module_service->serve();

    return EXIT_SUCCESS;
} catch (const std::exception& ex) {
    std::cerr << "ERROR: A std::exception was thrown from `serve`: " << ex.what() << std::endl;
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "ERROR: An unknown exception was thrown from `serve`" << std::endl;
    return EXIT_FAILURE;
}

}  // namespace
}  // namespace triton
}  // namespace mlmodelservice
}  // namespace viam

namespace {
namespace vmt = viam::mlmodelservice::triton;
}  // namespace

extern "C" int viam_mlmodelservice_triton_serve(vmt::cxxapi::shim* shim, int argc, char* argv[]) {
    vmt::cxxapi::the_shim = *shim;
    return vmt::serve(argc, argv);
}
