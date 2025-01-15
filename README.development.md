Initial Setup - Outside container

The first part of the process is to establish a container in which you can do iterative development. This is a one time setup cost, so, while there are a few steps, you shouldn't need to repeat them too often.


1) Start by building the `build-deps` target against the baseline you want (jp5, jp6, cuda). In my case, I'm on JP5, so the example will use that. Give the container image some useful name to indicate that is a development base:

```
$ docker build -f ./etc/docker/Dockerfile.triton-jetpack-focal --target build-deps . -t acm-triton-jp5-iterative
```

If you were going to work on a JP6 board, you would probalby build that more like:

```
$ docker build -f ./etc/docker/Dockerfile.nvcr-triton-containers --target build-deps --build-arg JETPACK=1 . -t acm-triton-jp6-iterative
```


2) Start a docker container based on that image, with the Viam Triton Module sources mounted into it.

```
$ docker run -id --name acm-triton-jp5-dev -v <path-to-triton-module-sources>:/root/opt/src/viam-mlmodelservice-triton acm-triton-jp5-iterative
```


3) Install the packages required by the `runtime` layer (if any) from the Dockerfile you built from above in step 1 into the container, along with one of `tmux` or `screen` if you like to use those, then checkpoint the container image:

Following the example here where things are using the JP5 container, the `runtime` layer has quite a number of things, so this will look like:


```
$ docker exec -it acm-triton-jp5-dev apt-get install \
    tmux \
    \
    libc-ares2 \
    libre2-5 \
    libssl1.1 \
    zlib1g \
    \
    cuda-nvtx-11-4 \
    nvidia-cuda \
    nvidia-cudnn8 \
    nvidia-cupva \
    nvidia-opencv \
    nvidia-tensorrt \
    nvidia-vpi \
    \
    libarchive13 \
    libb64-0d \
    libopenblas0 \
    python3 \
    python3-pip \
    \
    libboost-log1.71.0 \
    \
    libjemalloc2
```

But if you used the NVCR containers it might be more like the following, since the `runtime` layer has fewer additional packages.

```
$ docker exec --it acm-triton-jp5-dev apt-get install \
    tmux \
    \
    libboost-log1.74.0 \
    libjemalloc2 \
    libabsl20210324 \
    libgrpc++1 \
    libprotobuf23
```

Finally, update the container image with these installs, so that if you need to re-create the container from the image for some reason, you can. But for now, you can just keep using the container you made.

```
docker container commit acm-triton-jp5-dev acm-triton-jp5-iterative:latest
```


4) Establish a persistent session inside the container with tmux or screen

```
$ docker exec --it acm-triton-jp5-dev tmux -u -CC new-session -A -s dev
```


Iterarative Development - Inside Container

Within the tmux or screen session inside the docker container, you can now do ordinary interative development. Changes to the triton module sources made outside the container will be reflected inside the container due to the bind mount. Personally, I like to go a step further and have mutagen sync my trition module sources from my development machine over to the machine where the container is running, but if you are using an IDE that supports remote work trees that's another way to go about it. That may be advantageous, as you will have a complete stack there, so things like LSP are more likely to work.

1) If not currently connected, reconnect to the persistent session in the dev container:

```
$ docker exec --it acm-triton-jp5-dev tmux -u -CC new-session -A -s dev
```


2) Build the Viam Triton module inside the container with cmake and Ninja. Note that the build commands here reflect the container build step for the `build` stage, adjust as necessary for your situation:


```
$ cd ~/opt/src/viam-mlmodelservice-triton
$ cmake -S . -B build -G Ninja -DVIAM_MLMODELSERVICE_TRITON_TRITONSERVER_ROOT=/opt/tritonserver/tritonserver -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=/opt/viam
$ cmake --build build --target install -- -v
$ cmake --install build --prefix /opt/viam
```


3) Make edits as needed (either inside our outside the container) and rebuild:

```
$ cmake --build build --target install -- -v
```

Repeat until you arrive at a state you want to try to run. If you have unit tests, run them here until you are happy with the results.


4) Reinstall the modified build to `/opt/viam`:
```
$ cmake --install build --prefix /opt/viam
```


Deployment - Outside Container

This section assumes that you already have a robot built and running on the device and using a stock copy of the Viam Triton Module. What we are going to do is replace the container image that the Triton Module startup script (i.e. `viam-mlmodelservice-triton.sh`) uses with a snapshot produced with the development state we currently have in our container.


1) Find the name of the image used for the currently deployed module. This will likely look something like this:

```
$ docker image list | grep 'viamrobotics.*triton'
ghcr.io/viamrobotics/viam-mlmodelservice-triton   0.6.0     9fc8aef5b007   5 months ago     7.71GB
```

2) Overwrite that container image with a snapshot of your container contents

```
$ docker container commit acm-triton-jp5-dev ghcr.io/viamrobotics/viam-mlmodelservice-triton:0.6.0
```

3) Restart the robot

You should find that your changes are now reflected in the running module.
