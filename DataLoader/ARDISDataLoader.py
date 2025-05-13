import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from typing import Tuple, List

class ARDISDataLoader:
    def __init__(self, data_dir: str = 'ARDIS_DATASET_3'):
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, 'ardis_data.npy')
        self.target_file = os.path.join(data_dir, 'ardis_target.npy')

    def preprocess_image(self, img_gray: np.ndarray) -> np.ndarray:
        # 输入：28×28 灰度图像前的 ROI
        # 自适应二值化
        _, img = cv2.threshold(img_gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # 轮廓裁剪
        contours, _ = cv2.findContours(img.copy(),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            roi = img[y:y+h, x:x+w]
        else:
            roi = img
        # 等比例缩放最长边到20
        h, w = roi.shape
        if h > w:
            new_h, new_w = 20, int(w * 20 / h)
        else:
            new_w, new_h = 20, int(h * 20 / w)
        roi = cv2.resize(roi, (new_w, new_h),
                         interpolation=cv2.INTER_LINEAR)
        # 填充至28×28
        top = (28 - new_h) // 2
        bottom = 28 - new_h - top
        left = (28 - new_w) // 2
        right = 28 - new_w - left
        img28 = cv2.copyMakeBorder(roi, top, bottom, left, right,
                                   borderType=cv2.BORDER_CONSTANT,
                                   value=0)
        # 归一化到 [0,1]
        return img28.astype(np.float32) / 255.0

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        # 如果已有 .npy，直接加载
        if os.path.exists(self.data_file) and os.path.exists(self.target_file):
            X = np.load(self.data_file, allow_pickle=False)
            y = np.load(self.target_file, allow_pickle=False)
            return X, y

        # 否则遍历目录，做预处理并保存
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        for label in sorted(os.listdir(self.data_dir)):
            label_dir = os.path.join(self.data_dir, label)
            if not os.path.isdir(label_dir) or not label.isdigit():
                continue
            for fname in os.listdir(label_dir):
                if os.path.splitext(fname)[1].lower() not in exts:
                    continue
                path = os.path.join(label_dir, fname)
                img_rgb = cv2.imread(path, cv2.IMREAD_COLOR)
                if img_rgb is None:
                    print(f"Warning: failed to load {path}")
                    continue
                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                img28 = self.preprocess_image(img_gray)
                X_list.append(img28.flatten())  # 保存为 28*28 = 784 向量
                y_list.append(int(label))

        X = np.stack(X_list, axis=0)  # [N, 784]
        y = np.array(y_list, dtype=np.int8)  # [N]

        # 保存到磁盘
        np.save(self.data_file, X)
        np.save(self.target_file, y)
        return X, y

if __name__ == '__main__':
    # 演示
    dataloader = ARDISDataLoader(data_dir='/home/disk2/cba/recognition/ARDIS_DATASET_3')
    X, y = dataloader.load_data()
    # 划分并 reshape，跟 MNIST 一样的流程
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = X_train.reshape(-1, 1, 28, 28).astype(np.float32)
    X_test  = X_test.reshape(-1, 1, 28, 28).astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test  = y_test.astype(np.int64)

    print("Train:", X_train.shape, y_train.shape)
    print("Test: ", X_test.shape, y_test.shape)
