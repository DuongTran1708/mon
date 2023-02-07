#!/bin/bash

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
DIR_CURRENT=$(dirname "${THIS}")
DIR_PARRENT=$(dirname $PWD)
DIR_SOURCE=$DIR_PARRENT"/src"

# Add python path
export PYTHONPATH=$PYTHONPATH:$PWD
export PYTHONPATH=$PYTHONPATH:$DIR_SOURCE
export CUDA_LAUNCH_BLOCKING=1

echo $PYTHONPATH

START_TIME="$(date -u +%s.%N)"
###########################################################################################################



###########################################################################################################
END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
