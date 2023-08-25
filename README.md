# Viam Triton MLModel Service

## Intro

A Viam MLModelService resource backed by Nvidia's [Triton
Server](https://developer.nvidia.com/triton-inference-server).

## (In)-stability Notice

This module is still under active development.

## Prerequisites

- An NVidia Jetson Orin board.

## Build and Install

- Ensure that the NVidia Container Runtime is installed: `sudo apt-get
  install nvidia-container`. Note that `nvidia-container` is part of
  `nvidia-jetpack` so if you have Jetpack installed on the board, you
  probably already have this. But it is worth running the above
  command to make sure!

- Clone this repository to any location on the system where you intend
  to run inference.

- Build the docker image `docker build . -t viam-mlmodelservice-triton
  -f etc/docker/Dockerfile.triton-jetpack-focal`
    - NOTE: When this module is available in
   the module registry, this step will be unncessary.
    - NOTE: The image name currently *must* be
   `viam-mlmodelservice-triton`. This restriction will be lifted in
   the future once the Docker image is published to
   [GHCR](https://ghcr.io).

## Registering the Module with a Robot

In your robot configuration on [app.viam.com](app.viam.com), navigate
to the `Config/Modules` card and create a new module. The name of the
module does not particularly matter, but for the purposes of this
README it will be called `viam-mlmodelservice-triton-module`. Set the
`Executable Path` for the module to be result of running `realpath
bin/viam-mlmodelservice-triton.sh` from the root of your checkout.

NOTE: When this module is available in the module registry, this step
will be unnecessary / simplified.

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

The next step is to create an instance of the resource this module serves. This will go in the `services` section of your robot's JSON configuration:

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

- `backend_directory` [required]: This must be
  `/opt/tritonserver/backends` unless you have relocated the backends
  directory somewhere. Note that this is a container side path.

- `model_name` [required]: The model to be loaded from the repository.

- `model_version` [optional]: The version of the model to be
  loaded. If not specified, the module will use the newest version of
  the model named by `model_name`.

- `model_repository_path` [required]: The (container side) path to a
  model repository. Note that this must be a subdirectory of the
  `$HOME/.viam` directory of the user running `viam-server`.

- `preferred_input_memory_type`: One of `cpu`, `cpu-pinned`, or
  `gpu`. This controlls the type of memory that will be allocated by
  the module for input tensors. For most models, `gpu` is probably the
  best choice, but the default is `cpu` since the module cannot assume
  that a GPU is available.

- `preferred_input_memory_type_id`: CUDA identifier on which to
  allocate `gpu` or `cpu-pinned` input tensors. This defaults to `0`,
  meaning the first device. You probably don't need to change this
  unless you have multiple GPUs.

- `tensor_name_remappings`: Provides two dictionaries under the
  `input` and `output` keys that rename the models tensors. Higher
  level services may expect tensors with particular names (e.g. the
  Viam Vision services). Use this map to rename the tensors from the
  loaded model as needed to meet those requirements.

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
