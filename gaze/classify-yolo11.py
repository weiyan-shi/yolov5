import os
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolo11n-cls.pt")  # 使用官方模型

# 输入文件夹路径
input_folder = "/app/Desktop/segment/segmented_objects"  # 替换为你自己的文件夹路径
output_folder = "/app/Desktop/segment/output_results"  # 保存结果的文件夹

# 如果输出文件夹不存在，则创建它
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 只处理图像文件
        image_path = os.path.join(input_folder, filename)
        
        # 对图像进行推理
        results = model(image_path)  # 对每个图像进行推理
        
        # 处理每个推理结果
        for i, result in enumerate(results):
            # 为每个推理结果设置输出文件路径
            output_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_result_{i}.jpg")
            
            # 保存推理结果
            result.save(filename=output_filename)

        print(f"处理完成：{filename}，结果保存到 {output_folder}")
