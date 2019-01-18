#!/bin/bash

# Set image name
IMAGE="capsnet-pytorch:latest"
if [ $# -eq 1 ]; then
    IMAGE=$1
fi

# Set project root dicrectory to map to docker 
THIS_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=`dirname ${THIS_DIR}`

# Run container
CONTAINER="capsnet-pytorch"

nvidia-docker run -it --rm --ipc=host \
	-p 6006:6006 \
	-v ${PROJ_DIR}:/workspace \
	--name ${CONTAINER} \
	${IMAGE}
