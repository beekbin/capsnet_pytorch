# Docker for CapsNet training

## Build Docker image

You can build Docker image by following: 

```
$ cd docker
$ bash build.sh
```

## Run Docker container

You can launch a container from the Docker image by following:

```
$ cd docker
$ bash run.sh
# Now you will be inside the container
```

Root of this repository is mounted to `/workspace` of the container. 
You can start training inside the container by following.

```
# After executing run.sh, you should be in /workspace inside the container.
$(docker) python main.py
```

## Run a command in the container

You can run a command in the container already running. 
This is useful to execute `tensorboard` to monitor training status:

```
$ cd docker
$ bash exec.sh
# Now you should be in /workspace inside the container

$(docker) tensorboard --logdir runs
# Then, open "http://localhost:6006" from your browser. 
```
