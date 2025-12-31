computeEvaltool visioneval --model resnet18 yolov10-s vitlarge \
  --adaptive --time 30 \
  --hosts 9.0.2.60 9.0.3.19 9.0.3.31 9.0.3.32\
  --ports 9000 9001 \
  --image-folder /root/cocodataset/val2017
