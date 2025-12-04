computeEvaltool visioneval --model resnet18 yolov10-s vitlarge\
  --time 15 \
  --adaptive \
  --rps-threshold 0.05 \
  --latency-threshold 0.1 \
  --hosts 10.1.18.121 \
  --ports 9000 9001 9002 9003 9004 9005 9006 9007 \
  --image-folder /root/cocodataset/val2017
