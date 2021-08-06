#!/usr/bin/env bash

data=enhance/configs/voc.yaml
CUDA_VISIBLE_DEVICES=4,5 python train.py --data $data --cfg yolov5s.yaml --weights '' --batch-size 64

# 本地测试：
# python train.py  --data  enhance/configs/voc.yaml --cfg yolov5s.yaml --weights '' --batch-size 16
