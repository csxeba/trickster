UID = $(shell id -u)
notebook_port ?= 8888

help:
	echo "make {build|run|lab} [notebook_port=8888]"

build:
	docker build \
	--tag trickster/trickster:latest \
	--build-arg UID=${UID} \
	--build-arg USER=${USER} \
	docker

run: build
	docker run -it --rm \
	-u ${USER} \
	-v $(shell pwd):/workspace \
	trickster/trickster:latest \
	/bin/bash

lab: build
	docker run --rm \
	-u ${USER} \
	-v $(shell pwd):/workspace \
	--hostname $(shell cat /etc/hostname) \
	--network host \
	trickster/trickster:latest \
	jupyter notebook \
	--notebook-dir /workspace \
	--ip 0.0.0.0 \
	--no-browser
