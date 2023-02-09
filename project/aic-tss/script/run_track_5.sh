#!/bin/bash

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
DIR_CURRENT=$(dirname "${THIS}")                    # mon/project/aic-tss/script
export DIR_TSS=$(dirname $DIR_CURRENT)              # mon/project/aic-tss
export DIR_SOURCE=$DIR_TSS"/src"                    # mon/project/aic-tss/scr
export DIR_MON=$(dirname $(dirname $DIR_TSS))"/src" # mon/scr/

# Add data dir
export DIR_DATA="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/"

# Add python path
export PYTHONPATH=$PYTHONPATH:$PWD        # .
export PYTHONPATH=$PYTHONPATH:$DIR_SOURCE # mon/project/aic-tss/scr
export PYTHONPATH=$PYTHONPATH:$DIR_MON    # mon/scr/


export CUDA_LAUNCH_BLOCKING=1

START_TIME="$(date -u +%s.%N)"
###########################################################################################################

# NOTE: DETECTION
python $DIR_SOURCE/main.py  \
    --detection  \
    --config $DIR_TSS"/configs/default.yaml"

###########################################################################################################
END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
