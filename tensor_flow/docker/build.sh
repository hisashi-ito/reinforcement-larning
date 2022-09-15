#! /bin/bash
IMAGE="tf"
cd ..
sudo docker build -t ${IMAGE} --no-cache -f docker/Dockerfile .
