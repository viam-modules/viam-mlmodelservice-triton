# Viam Triton MLModel Service

## Intro

A Viam MLModelService resource backed by Nvidia's [Triton
Server](https://developer.nvidia.com/triton-inference-server). You can read a [tutorial on how to set it up](https://docs.viam.com/modular-resources/examples/triton/) on the Viam docs site.

## (In)-stability Notice

This module is still under active development.

## Prerequisites

- An NVidia Jetson Orin board with Jetpack 5 installed. Note that use of an NVMe SSD is *strongly* recommended over using an SD card.

## Install

- Ensure that the NVidia Container Runtime is installed: `sudo apt-get
  install nvidia-container`. Note that `nvidia-container` is part of
  `nvidia-jetpack` so if you have Jetpack installed on the board, you
  probably already have this. But it is worth running the above
  command to make sure!

## Registering the Module with a Robot

Follow the instructions to [add a modular service to your robot](https://docs.viam.com/extend/modular-resources/configure/#add-a-modular-service-from-the-viam-registry), 
and search for "triton", then select the version from the Registry.

## Creating a Model Respository

This module currently requires manual setup of a [Model
Repository](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)
under the `~/.viam` directory of the user who will run
`viam-server`. Here, we place the model repository under
`~/.viam/triton/repository`, but the exact subpath under `~/.viam`
does not matter.

For instance, to add the [EfficientDet-Lite4 Object
Detection](https://tfhub.dev/tensorflow/efficientdet/lite4/detection/2)
model, you should have a layout like this after unpacking the module:

```
$ tree ~/.viam
~/.viam
├── cached_cloud_config_05536cf6-f8a6-464f-b05c-bf1e57e9d1d9.json
└── triton
    └── repository
        └── efficientdet-lite4-detection
            ├── 1
            │   └── model.savedmodel
            │       ├── saved_model.pb
            │       └── variables
            │           ├── variables.data-00000-of-00001
            │           └── variables.index
            └── config.pbext
```

The `config.pbext` file must exist, but at least for TensorFlow models
it can be empty. The version here is `1` but it can be any positive
integer. Newer versions will be preferred by default.

## Instantiating the module for a deployed model

The next step is to create an instance of the resource this module
serves. This will go in the `services` section of your robot's JSON
configuration:

A minimal configuration looks like:

```
  ...
  "services": [
    ...
    {
      "type": "mlmodel",
      "attributes": {
        "model_name": "efficientdet-lite4-detection",
        "model_repository_path": "/path/to/.viam/triton/repository",
      },
      "model": "viam:mlmodelservice:triton",
      "name": "mlmodel-effdet-triton"
    },
    ...
  ],
  ...
```

A complete configuration, specifying many optional parameters, might look like:


```
  ...
  "services": [
    ...
    {
      "type": "mlmodel",
      "attributes": {
        "backend_directory": "/opt/tritonserver/backends",
        "model_name": "efficientdet-lite4-detection",
        "model_version": 1,
        "model_repository_path": "/path/to/.viam/triton/repository",
        "preferred_input_memory_type_id": 0,
        "preferred_input_memory_type": "gpu",
        "tensor_name_remappings": {
          "outputs": {
            "output_3": "n_detections",
            "output_0": "location",
            "output_1": "score",
            "output_2": "category"
          },
          "inputs": {
            "images": "image"
          }
        }
      },
      "model": "viam:mlmodelservice:triton",
      "name": "mlmodel-effdet-triton"
    },
    ...
  ],
  ...
```

The `type` field must be `mlmodel`, and the `model` field must use the
`viam:mlmodelservice:triton` tag, but the `name` of this module is up
to you. The following `attribute` level configurations are available:

- `model_name` [required]: The model to be loaded from the repository.

- `model_repository_path` [required]: The (container side) path to a
  model repository. Note that this must be a subdirectory of the
  `$HOME/.viam` directory of the user running `viam-server`.

- `backend_directory` [optional, default determined at build time]: A
  container side path to the TritonServer "backend" directory. You
  normally do not need to override this; the build will set it to the
  backend directory of the Triton Server installation in the
  container. You may set it if you wish to use a different set of
  backends.

- `model_version` [optional, defaults to -1, meaning 'newest']: The
  version of the model to be loaded. If not specified, the module will
  use the newest version of the model named by `model_name`.

- `preferred_input_memory_type` [optional, see below for default]: One
  of `cpu`, `cpu-pinned`, or `gpu`. This controlls the type of memory
  that will be allocated by the module for input tensors. If not
  specified, this will default to `cpu` if no CUDA-capable devices are
  detected at runtime, or to `gpu` if CUDA-capable devices are found.

- `preferred_input_memory_type_id` [optional, defaults to `0`]: CUDA
  identifier on which to allocate `gpu` or `cpu-pinned` input
  tensors. This defaults to `0`, meaning the first device. You
  probably don't need to change this unless you have multiple GPUs.

- `tensor_name_remappings` [optional, defaults to `{}`]: Provides two
  dictionaries under the `input` and `output` keys that rename the
  models tensors. Higher level services may expect tensors with
  particular names (e.g. the Viam Vision services). Use this map to
  rename the tensors from the loaded model as needed to meet those
  requirements.

## Connecting the Viam Vision Service

If all has gone right, you can now create a Viam vision service with a
configuration like the following:

```
  ...
  "services": [
    ...
    {
      "attributes": {
        "mlmodel_name": "mlmodel-effdet-triton"
      },
      "model": "mlmodel",
      "name": "vision-effdet-triton",
      "type": "vision"
    }
```

You can now connect this vision service to a transform camera, or get
detections programatically via any SDK.

## Verifying that the GPU is in use

I recommend using the
[`jtop`](https://github.com/rbonghi/jetson_stats) utility on the
Jetson line in order to monitor GPU usage and validate that Triton is
accelerating inference via the GPU.
