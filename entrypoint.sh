#!/bin/sh

if [ "$1" = "tests" ]; then
    shift
    pytest "$@"
else
    spotter_wave_process "$@"
fi
