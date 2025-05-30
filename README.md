# Viam Triton ML Model Service

A Viam provided ML Model service backed by NVIDIA's [Triton Inference Server](https://developer.nvidia.com/triton-inference-server).
Configure this ML Model service as a modular resource on your machine with a [Jetson board](https://docs.viam.com/build/configure/components/board/jetson/) or another supported machine to deploy [supported models](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/backend/docs/backend_platform_support_matrix.html) to your machine faster while consuming less computer power.

> [!Note]
> For more information, see the [ML Model service documentation](/ml/deploy/).

## Requirements

An NVIDIA Jetson Orin board or other machine with an NVIDIA GPU with the following installed:

1. The NVIDIA Container Runtime (which can be installed with `sudo apt-get install nvidia-container`)
2. On machines that support it, [Jetpack 5 or Jetpack 6](https://developer.nvidia.com/embedded/jetpack) (which can be installed with `sudo apt-get install nvidia-jetpack`)

Then, if you haven't done so already, create a new robot in [the Viam app](https://app.viam.com).
[Install `viam-server` on the board](https://docs.viam.com/get-started/installation/prepare/jetson-agx-orin-setup/) and connect to the robot.

## Install from Viam registry

The module is named `viam:mlmodelservice-triton-jetpack`. Once you've got the module on your smart machine, the service is named `viam:mlmodelservice:triton`. It implements the [`MLModelService` interface](https://github.com/viamrobotics/api/blob/main/proto/viam/service/mlmodel/v1/mlmodel.proto): the main way to interact with it is with the `Infer` RPC, though you can also get info about the service with the `Metadata` RPC. You probably don't need to send RPCs to it directly, though: instead, have a Vision Service send things to it, and you interact with the Vision Service.

## Build and Run Locally

To build this as a local module on your machine, run one of these three lines:
```sh { class="command-line" data-prompt="$"}
# Only run one of these! Which one to use depends on what platform you're on.
make -f Makefile.module image-jetpack5 module-jetpack5
make -f Makefile.module image-jetpack6 module-jetpack6
make -f Makefile.module image-cuda     module-cuda
```
Then, set up your robot so that the local module points to the `module.tar.gz` file in the root
directory of this repo.

## Configure your Triton ML Model Service

> [!Note]
> Before configuring your ML Model service module, you must [create a robot](https://docs.viam.com/manage/fleet/robots/#add-a-new-robot).

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/).
Click on the **Services** subtab and click **Create service**.
Select **ML Model**, then search for the `mlmodelservice:triton` model.
Click **Add module**.
Give your resource a name of your choice and click **Create**.

First, make sure your module version is correct.
Select **Raw JSON** mode.
Your `"modules"` array should have an entry like the following:

```json
{
  "type": "registry",
  "name": "viam_mlmodelservice-triton-jetpack",
  "module_id": "viam:mlmodelservice-triton-jetpack",
  "version": "0.4.0"
}
```
(The version used might differ from this example.)
Save your config.

> [!NOTE]
> For more information, see [Configure a Robot](https://docs.viam.com/manage/configuration/).

Now, to configure the service's **Attributes**, you have two options.
You can use a TensorFlow or PyTorch model from the [Registry](https://app.viam.com/registry), or you can load an existing TensorFlow or PyTorch model on your machine.
To deploy a model from the Registry, navigate back to the **Config** tab of your machine in the Viam app and switch back to **Builder** mode.
Click on the **Select ML model** field and select a model from the dropdown that appears.
Your ML model service will automatically be configured with this model.
You can explore the available models in the [Registry](https://app.viam.com/registry).

To deploy an existing model on your machine, you can either 1) specify the name of a model downloaded from the Viam registry, or 2) [create your own local model repository](#create-a-repository-to-store-the-ml-model-to-deploy) on your machine.

### Option 1: use a model from the Viam registry

You can download an ML model from Viam's registry. Head to
https://app.viam.com/registry?type=ML+Model to see the available models. Choose a model that uses a [supported model framework](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/backend/docs/backend_platform_support_matrix.html). Once you've added this
Triton MLModel service to your robot's config, you can select the model you want to use from its
section in the builder tab. To clarify: you don't add the model with the `+` symbol in the top-left
of the builder page; you add it with the `select model` button in the RC card for the Triton
service. Your `model_path` should end up looking something like this:
```
"model_path": "${packages.ml_model.TF2-EfficientDetD0-COCO}",
```
where the `TF2-EfficientDetD0-COCO` is replaced with the name of the model you want to use instead.
You should also name the model, and add in the name remappings mentioned in the registry entry for
your model.

#### Minimal example
```
{
  "model_path": "${packages.ml_model.TF2-EfficientDetD0-COCO}",
  "model_name": "coco",
  "tensor_name_remappings": {
    "outputs": {
      "output_1": "score",
      "output_2": "category",
      "output_3": "n_detections",
      "output_0": "location"
    },
    "inputs": {
      "images": "image"
    }
  }
}
```
Remember to select/download the model, so the `model_path` actually exists!

### Option 2: create a repository to store the ML model to deploy

You can manually create a Triton [model repository](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html).
On your robot's Jetson computer, create a [structured repository](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html) under the `~/.viam` directory.
The exact subpath under `~/.viam` does not matter.

> [!NOTE]
> Where you should place your model repository depends on the `.viam` directory where the cloud config file is located.
>
> You may need to place the model repository in the `/root/.viam` directory for it to work properly, depending on the `.viam` directory you are running from. If you encounter any issues, consider trying the `/root/.viam` directory as an alternative location.

For example, after unpacking the module, to add the [EfficientDet-Lite4 Object Detection](https://tfhub.dev/tensorflow/efficientdet/lite4/detection/2) model, place the model repository under `~/.viam/triton/repository`:

```sh { class="command-line" data-prompt="$"}
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
            └── config.pbtxt
```

The `config.pbtxt` is usually optional; see the [Triton model configuration documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html) and [Triton Server model repository documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html) for details on the correct contents of the `config.pbtxt` file.
The version here is `1` but it can be any positive integer.
Newer versions will be preferred by default.

After creating your model repository, configure the required attributes to deploy your model on your robot.
Navigate back to the **Config** tab of your machine in the Viam app.
Fill in the required **Attributes** to deploy the model:

### Attributes

The following attributes are available for the MLModel service `viam:mlmodelservice:triton`:

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `model_name` | string | **Required** | The model to be loaded from the model repository. |
| `model_repository_path` | string | **Semi-Required** | The container-side path to a model repository. Note that this must be a subdirectory of the `$HOME/.viam` directory of the user running `viam-server`. Exactly one of the `model_repository_path` or the `model_path` is required. |
| `model_path` | string | **Semi-Required** | The directory in which the model to use is stored. You can use strings like `${packages.ml_model.MyModel}`, too. Exactly one of the `model_repository_path` or the `model_path` is required. |
| `backend_directory` | string | Optional | A container side path to the TritonServer "backend" directory. You normally do not need to override this; the build will set it to the backend directory of the Triton Server installation in the container. You may set it if you wish to use a different set of backends. |
| `model_version` | int | Optional | The version of the model to be loaded from `model_repository_path`. If not specified, the module will use the newest version of the model named by model_name.<br><br>Default: `-1` (newest) |
| `tensor_name_remappings` | obj | Optional | Provides two dictionaries under the `inputs` and `outputs` keys that rename the models' tensors. Other Viam services, like the [vision service]([/ml/vision/](https://docs.viam.com/registry/advanced/mlmodel-design/)) may expect to work with tensors with particular names. Use this map to rename the tensors from the loaded model to what the vision service expects as needed to meet those requirements.<br><br>Default: `{}` |
| `model_config` | obj | Optional | [Triton Model Configuration Parameters](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html). If provided, this setting will override any configuration in the configured repository. |

### Example configurations

An example minimal configuration would look like this, within your robot’s "services" array:

```json {class="line-numbers linkable-line-numbers"}
{
  "type": "mlmodel",
  "attributes": {
    "model_name": "efficientdet-lite4-detection",
    "model_path": "${packages.ml_model.FaceDetector}"
  },
  "model": "viam:mlmodelservice:triton",
  "name": "mlmodel-effdet-triton"
}
```

An example detailed configuration with optional parameters specified would look like this:

```json {class="line-numbers linkable-line-numbers"}
{
  "type": "mlmodel",
  "attributes": {
    "backend_directory": "/opt/tritonserver/backends",
    "model_name": "efficientdet-lite4-detection",
    "model_version": 1,
    "model_path": "${packages.ml_model.FaceDetector}",
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
}
```

If you have your own Triton model repository, you could use it like this:

```json {class="line-numbers linkable-line-numbers"}
{
  "type": "mlmodel",
  "attributes": {
    "model_name": "efficientdet-lite4-detection",
    "model_repository_path": "/path/to/.viam/triton/repository"
  },
  "model": "viam:mlmodelservice:triton",
  "name": "mlmodel-effdet-triton"
}
```

## Next Steps

- Use the [`jtop`](https://github.com/rbonghi/jetson_stats) utility on the Jetson line to monitor GPU usage to validate that Triton is accelerating inference through the GPU.

- Use the `viam:mlmodelservice:triton` modular service to perform inference with the machine learning models available in the Triton service's model repository on your robot.

- Create a [vision service](https://docs.viam.com/ml/vision/) with a configuration in your `"services"` array like the following:

  ```json {class="line-numbers linkable-line-numbers"}
  {
    "attributes": {
      "mlmodel_name": "mlmodel-effdet-triton"
    },
    "model": "mlmodel",
    "name": "vision-effdet-triton",
    "type": "vision"
  }
  ```

    You can now connect this vision service to a [transform camera](https://docs.viam.com/build/configure/components/camera/transform/), or get detections programmatically through one of Viam's [client SDKs](https://docs.viam.com/build/program/).
