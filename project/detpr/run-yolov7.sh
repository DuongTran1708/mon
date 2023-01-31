#!/bin/bash

echo "$HOSTNAME"

task=$1
machine=$HOSTNAME
read -e -i "$task" -p "Task [install, train, test, predict]: " task
# read -e -i "$machine" -p "Machine [pc, server]: " machine

# Initialization
cd "yolov7" || exit

# Install
if [ "$task" == "install" ]; then
  echo -e "\nInstalling YOLOv7"
  # pip install ultralytics
  if [ "$machine" == "pc" ]; then
    sudo nvidia-docker run --name yolov7 -it \
    -v "/mnt/workspace/mon/data/":/data/ \
    -v "/mnt/workspace/mon/project/detpr/":/detpr \
    --shm-size=64g nvcr.io/nvidia/pytorch:21.08-py3
    apt update
    apt install -y zip htop screen libgl1-mesa-glx
    pip install seaborn thop
  elif [ "$machine" == "server" ]; then
    sudo nvidia-docker run --name yolov7 -it \
    -v "/home/longpham/workspace/mon/data/":/data/ \
    -v "/home/longpham/workspace/mon/project/detpr/":/detpr \
    --shm-size=64g nvcr.io/nvidia/pytorch:21.08-py3
    apt update
    apt install -y zip htop screen libgl1-mesa-glx
    pip install seaborn thop
  fi
fi

# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  if [ "$machine" == "LP-LabDesktop01-Ubuntu" ]; then
    python train_aux.py \
      --weights "weight/yolov7-e6e-training.pt" \
      --cfg "cfg/training/yolov7-e6e.yaml" \
      --data "data/visdrone-a2i2-of.yaml" \
      --hyp "data/hyp.scratch.custom.yaml" \
      --epochs 100 \
      --batch-size 4 \
      --img-size 1280 \
      --workers 4 \
      --device 0 \
      --sync-bn \
      --exist-ok \
      --project "../run/train" \
      --name "yolov7-e6e-visdrone-a2i2-of-1280" \
      # --resume
  elif [ "$machine" == "VSW-WS02" ]; then
    python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train_aux.py \
      --weights "weight/yolov7-e6e-training.pt" \
      --cfg "cfg/training/yolov7-e6e.yaml" \
      --data "data/visdrone-a2i2-of.yaml" \
      --hyp "data/hyp.scratch.custom.yaml" \
      --epochs 100 \
      --batch-size 4 \
      --img-size 2160 \
      --workers 4 \
      --device 0,1 \
      --sync-bn \
      --exist-ok \
      --project "../run/train" \
      --name "yolov7-e6e-visdrone-a2i2-of-2160" \
      # --resume
  elif [ "$machine" == "VSW-WS03" ]; then
    python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train_aux.py \
      --weights "weight/yolov7-e6e-training.pt" \
      --cfg "cfg/training/yolov7-e6e.yaml" \
      --data "data/visdrone-a2i2-of.yaml" \
      --hyp "data/hyp.scratch.custom.yaml" \
      --epochs 100 \
      --batch-size 4 \
      --img-size 2160 \
      --workers 4 \
      --device 0,1 \
      --sync-bn \
      --exist-ok \
      --project "../run/train" \
      --name "yolov7-e6e-visdrone-a2i2-of-2160" \
      # --resume
  fi
fi

# Test
if [ "$task" == "test" ]; then
  echo -e "\nTesting"
  python test.py \
    --weights "run/train/yolov7-e6e-visdrone-1280/weights/best.pt" \
    --data "data/a2i2.yaml" \
    --batch-size 8 \
    --img-size 1280 \
    --conf-thres 0.00001 \
    --iou-thres 0.5 \
    --device 0 \
    --augment \
    --project "../run/test" \
    --name yolov7-e6e-visdrone-1280 \
fi

# Predict
if [ "$task" == "predict" ]; then
  echo -e "\nPredicting"
  python detect.py \
    --weights "run/train/yolov7-e6e-visdrone-1280/weights/best.pt" \
    --source "../data/ai2i-haze/dry-run/2023/images/" \
    --img-size 1280 \
    --conf-thres 0.00001 \
    --iou-thres 0.5 \
    --agnostic-nms \
    --augment \
    --project "../run/predict" \
    --name yolov7-e6e-visdrone-1280 \
fi

cd ..
