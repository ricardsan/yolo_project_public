#!/bin/bash

for i in {1..5}
do
    yolo detect train \
    model=yolov8n.pt \
    data=/home/rica/Documents/dataset700/dataset_5fold/fold$i/data.yaml \
    epochs=100 \
    batch=16 \
    imgsz=640 \
    name=train_fold$i \
    deterministic=True
done
