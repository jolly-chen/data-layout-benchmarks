#!/bin/bash

docker load < artifact-image.tar.gz

if [[ "$*" == *"--full"* ]]; then
    docker run --rm -v $(pwd)/results:/root/src/results artifact-image:latest /root/src/run_in_container.sh --full
else
    docker run --rm -v $(pwd)/results:/root/src/results artifact-image:latest /root/src/run_in_container.sh --quick
fi
