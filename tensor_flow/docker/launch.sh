#! /bin/bash
IMAGE="tf"
CONTAINER=${IMAGE}
sudo docker run -tid --privileged -v /data:/data --name ${CONTAINER} ${IMAGE} /sbin/init
