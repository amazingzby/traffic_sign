export PYTHONPATH=/home/megstudio/workspace/trafficdet_gpu:$PYTHONPATH
python tools/test.py -n 1 -se 0 -f configs/faster_rcnn_res50_800size_trafficdet_demo.py -d ./       -w logs/faster_rcnn_res50_800size_trafficdet_demo_gpus2/epoch_20.pkl
python tools/test_final.py -n 1 -se 0 -f configs/faster_rcnn_res50_800size_trafficdet_demo.py -d ./ -w logs/faster_rcnn_res50_800size_trafficdet_demo_gpus2/epoch_20.pkl

