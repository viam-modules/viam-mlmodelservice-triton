#!/bin/bash

# TODO: Docker pull
# TODO: Validate presence of NVidia container runtime
# TODO: Use ghcr name of image to run

exec docker run \
     --rm \
     --runtime=nvidia --gpus=all \
     -v /etc/passwd:/etc/passwd \
     -v /etc/group:/etc/group \
     -v $(dirname $1):$(dirname $1) \
     -v $(realpath ~/.viam):$(realpath ~/.viam) \
     viam-mlmodelservice-triton \
     sudo -u \#$(id -u) LD_PRELOAD=libjemalloc.so.2 /opt/viam/bin/viam_mlmodelservice_triton $1
