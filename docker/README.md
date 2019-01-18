# Docker for CapsNet training

## Build Docker image

You can build Docker image by following: 

```
# PROJ_DIR=/path/to/capsnet_pytorch
$ cd $PROJ_DIR/docker
$ bash build.sh
```

## Run Docker container

You can launch a container from the Docker image by following:

```
$ cd $PROJ_DIR/docker
$ bash run.sh
# Now you will be inside the container
```

Root of this repository is mounted to `/workspace` of the container. 
You can start training inside the container by following.

```
$(docker) cd /workspace
$(docker) python main.py
```

## Run a command in the container

You can run a command in the container already running. 
This is useful to execute `tensorboard` to monitor training status:

```
$ cd $PROJ_DIR/docker
$ bash exec.sh
# Now you should be inside the container already running

$(docker) cd /workspace
$(docker) tensorboard --logdir runs
# Then, open "http://localhost:6006" from your browser. 
```
