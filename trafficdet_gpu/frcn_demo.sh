#!/usr/bin/env bash
#fuser -v /dev/nvidia*
gpu=2
WORK_DIR=$(cd "$(dirname "$0")"; pwd)
export PYTHONPATH=${WORK_DIR}/

# train
python tools/train.py -n ${gpu} -b 3 \
  -f configs/faster_rcnn_res50_800size_trafficdet_demo.py -d . \
  -w logs/faster_rcnn_res50_800size_trafficdet_demo_gpus2/epoch_20.pkl

# test
# 1X
#python3 tools/test.py -n ${gpu} -se 11 \
#  -f configs/faster_rcnn_res50_800size_trafficdet_demo.py -d .

# 2X
#python3 tools/test.py -n ${gpu} -se 23 \
#  -f configs/faster_rcnn_res50_800size_trafficdet_demo.py -d .
