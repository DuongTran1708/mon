#!/bin/bash

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
DIR_CURRENT=$(dirname "${THIS}")                    # mon/project/aic-tss/script
export DIR_TSS=$(dirname $DIR_CURRENT)              # mon/project/aic-tss
export DIR_SOURCE=$DIR_TSS"/src"                    # mon/project/aic-tss/scr
export DIR_MON=$(dirname $(dirname $DIR_TSS))"/src" # mon/scr/

# Add data dir
#export DIR_DATA="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/"
export DIR_DATA="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5_test/"

# Add python path
export PYTHONPATH=$PYTHONPATH:$PWD        # .
export PYTHONPATH=$PYTHONPATH:$DIR_SOURCE # mon/project/aic-tss/scr
export PYTHONPATH=$PYTHONPATH:$DIR_MON    # mon/scr/


export CUDA_LAUNCH_BLOCKING=1

START_TIME="$(date -u +%s.%N)"
###########################################################################################################

# NOTE: DETECTION
#echo "*********"
#echo "DETECTION"
#echo "*********"
#python $DIR_SOURCE/main_s1.py  \
#    --detection  \
#    --config $DIR_TSS"/configs/solution_1.yaml"

# NOTE: DRAW DETECTION RESULT
#echo "*********************"
#echo "DRAW DETECTION RESULT"
#echo "*********************"
#python $DIR_SOURCE/utils/drawing_result.py \
#    --draw_pickle  \
#    --path_pickle_in "${DIR_DATA}/outputs_s1/dets_crop_pkl/yolov8x6/"  \
#    --path_video_out "${DIR_DATA}/outputs_s1/dets_crop_pkl_debug/"  \
#    --path_video_in "${DIR_DATA}/videos/"

# NOTE: DRAW HEURISTIC DETECTION
#echo "*******************"
#echo "HEURISTIC DETECTION"
#echo "*******************"
#python $DIR_SOURCE/main_s1.py  \
#    --heuristic  \
#    --config $DIR_TSS"/configs/solution_1.yaml"

# NOTE: WRITE FINAL RESULT
#echo "*****************"
#echo "WRITE FINAL RESULT"
#echo "*****************"
#python $DIR_SOURCE/main_s1.py  \
#    --write_final  \
#    --config $DIR_TSS"/configs/solution_1.yaml"

# NOTE: DRAW FINAL RESULT
echo "*****************"
echo "DRAW FINAL RESULT"
echo "*****************"
python $DIR_SOURCE/utils/drawing_result.py \
    --draw_final  \
    --path_final "${DIR_DATA}/outputs_s1/final_result_s1.txt"  \
    --path_video_out "${DIR_DATA}/outputs_s1/dets_full_debug/"  \
    --path_video_in "${DIR_DATA}/videos/"

# NOTE: DRAW COMPARISON
#echo "***************"
#echo "DRAW COMPARISON"
#echo "***************"
#python /media/sugarubuntu/Data/2_Workspace/_MAIN_CODE/utils/imagesorvideo/images_video.py

###########################################################################################################
END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
