import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
img = "/app/Desktop/Dataset/1/frames/108.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

# 提取检测的物体信息：边框位置、类别、置信度等
detections = results.pandas().xyxy[0]

# 显示提取的检测信息
print(detections)

# 保存带有物体边框的图片（默认路径：`runs/detect/exp`）
results.save()