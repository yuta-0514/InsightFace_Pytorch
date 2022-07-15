~#!/bin/bash

CONTAINER_NAME=fr
IMAGES=yuta0514/fr 
TAGS=1.0
PORT=8888

docker run --rm -it --gpus all --ipc host -v ~/dataset:/mnt -v $PWD:$PWD -p ${PORT}:${PORT} --name ${CONTAINER_NAME} ${IMAGES}:${TAGS}

#run "umask 000" after this script
#docker makes file or dir by root, therefore make permission 777
