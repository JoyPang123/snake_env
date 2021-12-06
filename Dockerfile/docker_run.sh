#!/usr/bin/env bash

XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]; then
    xauth_list=$(xauth nlist $DISPLAY)
    xauth_list=$(sed -e 's/^..../ffff/' <<<"$xauth_list")
    if [ ! -z "$xauth_list" ]; then
        echo "$xauth_list" | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

xhost +

docker run -it --rm \
    --device=/dev/dri:/dev/dri \
    -e DISPLAY=unix${DISPLAY} \
    -v "$XAUTH:$XAUTH" \
    -v "/etc/localtime:/etc/localtime:ro" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix" \
    snake