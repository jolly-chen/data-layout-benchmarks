#!/bin/bash

docker load < artifact-image.tar.gz

rm -rf results

if [[ "$*" == *"--full"* ]]; then
    docker run --cap-add=PERFMON --rm -v $(pwd)/results:/root/src/results artifact-image:latest /root/src/run_in_container.sh --full
else
    docker run --cap-add=PERFMON --rm -v $(pwd)/results:/root/src/results artifact-image:latest /root/src/run_in_container.sh --quick
fi
