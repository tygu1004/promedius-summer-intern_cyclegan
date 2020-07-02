#!/bin/bash

docker image load -i /home/gaeun.lee/CycleganTest/cycle.gaeun_1.8.0.tar.gz

nvidia-docker run -it --gpus all --rm  \
	-v "$PWD"/logs:/logs               \
	-v "$PWD"/src:/src                 \
	-v "$PWD"/data:/data           	   \
	--name cont.gaeun.cyclegan.test    \
	-p 8889:8889                       \
	-p 6006:6006                       \
	cycle.gaeun:1.8.0                  \
	sh -c 'jupyter notebook --allow-root --port 8889'
