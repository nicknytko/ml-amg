#!/bin/bash
docker run -it --rm -e DISPLAY=$DISPLAY -v $(pwd):/src -v /tmp/.X11-unix:/tmp/.X11-unix nicknytko/firedrake /bin/zsh
