#!/usr/bin/env bash

data=scripts/configs/coco.yaml
#data=scripts/configs/coco128.yaml
CUDA_VISIBLE_DEVICES=6,7 python train.py --data $data --cfg yolov5s.yaml --weights '' --batch-size 64

# 本地测试：
# python train.py  --data scripts/configs/coco128.yaml --cfg yolov5s.yaml --weights '' --batch-size 16