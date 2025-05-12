import os
import cv2
import numpy as np
import torch
from CNN import CNN  # 请确保此处能导入你的模型定义

def preprocess_image(img_path: str) -> torch.Tensor:
    # 验证输入
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # 读取灰度图
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    
    # 自适应二值化
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 轮廓检测与裁剪
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        roi = img[y:y+h, x:x+w]
    else:
        print(f"Warning: No contours found in {img_path}, using original image")
        roi = img
    
    # 等比例缩放
    h, w = roi.shape
    if h > w:
        new_h = 20
        new_w = int(w * 20 / h)
    else:
        new_w = 20
        new_h = int(h * 20 / w)
    roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 居中填充到 28×28
    top = (28 - new_h) // 2
    bottom = 28 - new_h - top
    left = (28 - new_w) // 2
    right = 28 - new_w - left
    img28 = cv2.copyMakeBorder(roi, top, bottom, left, right,
                               borderType=cv2.BORDER_CONSTANT, value=0)
    
    # 归一化到 [-1, 1]
    img28 = img28.astype(np.float32) / 255.0
    
    # 转为 Tensor
    tensor = torch.from_numpy(img28).unsqueeze(0).unsqueeze(0)
    return tensor

def process_and_infer(img_path: str,
                      model_path: str,
                      input_size: int = 28) -> int:
    """
    完整流程：预处理 → 加载模型 → 推理 → 返回预测标签
    """
    # 1. 设备 & 模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(input_size=input_size)
    try:
        state_dict = torch.load(model_path, map_location=device)
        print("Model state_dict keys:", state_dict.keys())
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
    model.to(device)
    model.eval()

    # 2. 预处理
    img_tensor = preprocess_image(img_path).to(device)

    # 3. 推理
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()

    return pred
