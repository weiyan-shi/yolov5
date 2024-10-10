import torch

# 从 PyTorch Hub 加载 YOLOv5-seg 模型
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s-seg', pretrained=True)
model = torch.hub.load(
    "ultralytics/yolov5", "custom", "yolov5m-seg.pt"
)  # load from PyTorch Hub (WARNING: inference not yet supported)

# 加载图片
img = '/app/Desktop/Dataset/1/frames/9.jpg'

# 进行推理，包括分割和检测
results = model(img)

# 打印检测到的结果
results.print()

# 显示分割结果
results.show()

# 保存带有分割掩膜的图片（默认保存到 runs/segment/exp 目录）
results.save()
