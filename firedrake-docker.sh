#!/bin/bash
docker run -it --rm -e DISPLAY=host.docker.internal:0 -v $(pwd):/src nicknytko/firedrake /bin/zsh
