# Viam Triton Module Development README

This document describes a way to do iterative development for the Viam Triton Module without requiring full container
builds on each change.
In essence, it builds up two containers, one for development, and another for deployment, and then copies the development
results into the deployment results and replaces the running module with a snapshot of the deployment container.

## Initial Setup - Outside container

The first part of the process is to establish a container in which you can do iterative development.
This is a one time setup cost, so, while there are a few steps, you shouldn't need to repeat them too often.

1. Start by building the `build-deps` target against the baseline you want (jp5, jp6, cuda).
In my case, I'm on JP5, so the example will use that.
Give the container image some useful name to indicate that is a development base:

```sh
$ docker build -f ./etc/docker/Dockerfile.triton-jetpack-focal --target build-deps . -t triton-jp5-iterative-dev
```

If you were going to work on a JP6 board, you would probably build that more like:

```sh
$ docker build -f ./etc/docker/Dockerfile.nvcr-triton-containers --target build-deps --build-arg JETPACK=1 . -t triton-jp6-iterative-dev
```


2. Start a docker container based on that image, with the Viam Triton Module sources mounted into it.

```sh
$ docker run -id --name triton-jp5-dev -v <path-to-triton-module-sources>:/root/opt/src/viam-mlmodelservice-triton triton-jp5-iterative-dev
```

3. Next, create a deployment base by building the `runtime` target against the baseline you want (jp5, jp6, cuda)

```sh
$ docker build -f ./etc/docker/Dockerfile.triton-jetpack-focal --target runtime . -t triton-jp5-iterative-deploy
```

If you were going to work on a JP6 board, you would probably build that more like:

```sh
$ docker build -f ./etc/docker/Dockerfile.nvcr-triton-containers --target runtime --build-arg JETPACK=1 . -t triton-jp6-iterative-deploy
```

4. Start a docker container based on the deploy image (no need to mount sources)

```sh
$ docker run -id --name triton-jp5-deploy triton-jp5-iterative-deploy
```

or

```sh
$ docker run -id --name triton-jp6-deploy triton-jp6-iterative-deploy
```

5. Establish a shell session inside the build container

```sh
$ docker exec -it triton-jp5-dev /bin/bash
```

Alternatively, consider using `tmux` to make the session persistent, so it isn't tied to your
current terminal session.

```sh
$ docker exec -it triton-jp5-dev apt-get update
$ docker exec -it triton-jp5-dev apt-get install tmux
$ docker exec -it triton-jp5-dev tmux -u new-session -A -s dev
```

You could also install and use `screen` instead of `tmux`.

## Iterative Development - Inside Container

Within a shell session inside the docker container, you can now do ordinary interative development.
Changes to the triton module sources made outside the container will be reflected inside the container due to the bind mount.
Personally, I like to go a step further and have mutagen sync my trition module sources from my development machine over to the machine where the container is running, but if you are using an IDE that supports remote work trees that's another way to go about it.
That may be advantageous, as you will have a complete stack there, so things like LSP are more likely to work.

1. If not currently connected, reconnect to the dev container:

```sh
$ docker exec -it triton-jp5-dev /bin/bash
```

Alternatively, if using `tmux` (or `screen`):

```sh
$ docker exec -it triton-jp5-dev tmux -u new-session -A -s dev
```

2. Do an initial build of the Viam Triton module inside the container with cmake and Ninja.
Note that the build commands here reflect the container build step for the `build` stage, adjust as necessary for your situation:

```sh
$ cd ~/opt/src/viam-mlmodelservice-triton
$ cmake -S . -B build -G Ninja -DVIAM_MLMODELSERVICE_TRITON_TRITONSERVER_ROOT=/opt/tritonserver/tritonserver -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=/opt/viam -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
$ cmake --build build --target install -- -v
$ cmake --install build --prefix /opt/viam
```

Note that for NVCR containers, the `VIAM_MLMODELSERVICE_TRITON_TRITONSERVER_ROOT` is just `/opt/tritonserver`.

3. Make edits as needed (either inside our outside the container) and rebuild:

```sh
$ cmake --build build --target install -- -v
```

Repeat until you arrive at a state you want to try to run.
If you have unit tests, run them here until you are happy with the results.
You can return to this step after testing the deployment to continue iterating, unless you need to regenerate the build system with different CMake parameters, in which case return to step 2 above.


4. Reinstall the modified build to `/opt/viam`:
```sh
$ cmake --install build --prefix /opt/viam
```

## Deployment - Outside Container

This section assumes that you already have a robot built and running on the device and using a stock copy of the Viam Triton Module.
What we are going to do is replace the container image that the Triton Module startup script (i.e. `viam-mlmodelservice-triton.sh`) uses with a snapshot produced with the development state we currently have in our container.

1. Copy the installation from `/opt/viam` in the development container into the deployment container

```sh
$ docker cp -a triton-jp5-dev:/opt/viam - | docker cp -a - triton-jp5-deploy:/opt
```

If you are using the JP5 container, you also need to copy in the Triton Server libraries.
Note that you only need to do this step once for the deployment container.
If you are on a JP6 or CUDA container (nvcr base), you don't need to do it at all.

```sh
$ docker cp -a triton-jp5-dev:/opt/tritonserver - | docker cp -a - triton-jp5-deploy:/opt
```

NOTE: Unfortunately, the above command may erroneously report that it copied zero bytes. It is safe to disregard this message.

2. Find the name of the image used for the currently deployed module.
This will likely look something like this:

```sh
$ docker image list | grep 'viamrobotics.*triton'
ghcr.io/viamrobotics/viam-mlmodelservice-triton   0.6.0     9fc8aef5b007   5 months ago     7.71GB
```

NOTE: I'd like to eliminate this step.
Potentially, there may be a way to mount the `/opt/viam` (and, when needed for JP5, the `/opt/tritionserver`) directories from the `-dev` container into the `-deploy` container, such that writes in the `-dev` container immediately affect the running `-deploy` container, which would then get saved via the `docker commit`.

3. Overwrite that container image with a snapshot of your deployment container contents

```sh
$ docker container commit triton-jp5-deploy ghcr.io/viamrobotics/viam-mlmodelservice-triton:0.6.0
```

4. Restart the robot

You should find that your changes are now reflected in the running module.

5. Iterate

To iterate, make changes to the Viam Triton Server Module sources, rerun the cmake commands to build and install (steps 3 and 4 under `Iterative Development - Inside Container`), copy the results from the `-dev` container to the `-deploy` container, `commit` the results, and restart `viam-server`.
