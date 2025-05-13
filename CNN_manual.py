import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset  # Subset is useful here
import numpy as np




def manual_conv2d_forward_numpy(X_numpy, W_numpy, b_numpy, stride, padding):
    """
    手动卷积前向传播 (NumPy) - 遵循 (N, C_in, H_in, W_in) 输入格式
    W_numpy: (C_out, C_in, KH, KW)
    b_numpy: (C_out,)
    """
    N, C_in, H_in, W_in = X_numpy.shape
    C_out, _, KH, KW = W_numpy.shape

    if padding > 0:
        X_padded_numpy = np.pad(X_numpy, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant',
                                constant_values=0)
    else:
        X_padded_numpy = X_numpy

    H_padded, W_padded = X_padded_numpy.shape[2], X_padded_numpy.shape[3]

    H_out = (H_padded - KH) // stride + 1
    W_out = (W_padded - KW) // stride + 1

    Z_numpy = np.zeros((N, C_out, H_out, W_out))

    for n in range(N):
        for c_out_idx in range(C_out):
            for h_out_idx in range(H_out):
                for w_out_idx in range(W_out):
                    h_start = h_out_idx * stride
                    h_end = h_start + KH
                    w_start = w_out_idx * stride
                    w_end = w_start + KW

                    X_slice = X_padded_numpy[n, :, h_start:h_end, w_start:w_end]
                    conv_sum = np.sum(X_slice * W_numpy[c_out_idx, :, :, :])
                    Z_numpy[n, c_out_idx, h_out_idx, w_out_idx] = conv_sum + b_numpy[c_out_idx]
    return Z_numpy


def manual_conv2d_backward_numpy(dZ_numpy, X_numpy, W_numpy, stride, padding):
    """
    手动卷积反向传播 (NumPy)
    dZ_numpy: (N, C_out, H_out, W_out)
    X_numpy (原始输入，未填充): (N, C_in, H_in, W_in)
    W_numpy: (C_out, C_in, KH, KW)
    """
    N, C_in, H_in, W_in = X_numpy.shape
    C_out, _, KH, KW = W_numpy.shape
    _, _, H_out, W_out = dZ_numpy.shape

    if padding > 0:
        X_padded_numpy = np.pad(X_numpy, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant',
                                constant_values=0)
    else:
        X_padded_numpy = X_numpy

    dX_padded_numpy = np.zeros_like(X_padded_numpy)
    dW_numpy = np.zeros_like(W_numpy)
    db_numpy = np.sum(dZ_numpy, axis=(0, 2, 3))

    for n in range(N):
        for c_out_idx in range(C_out):
            for h_out_idx in range(H_out):
                for w_out_idx in range(W_out):
                    h_start = h_out_idx * stride
                    h_end = h_start + KH
                    w_start = w_out_idx * stride
                    w_end = w_start + KW

                    X_slice = X_padded_numpy[n, :, h_start:h_end, w_start:w_end]
                    grad_val = dZ_numpy[n, c_out_idx, h_out_idx, w_out_idx]

                    dW_numpy[c_out_idx, :, :, :] += X_slice * grad_val
                    dX_padded_numpy[n, :, h_start:h_end, w_start:w_end] += W_numpy[c_out_idx, :, :, :] * grad_val

    if padding > 0:
        dX_numpy = dX_padded_numpy[:, :, padding:-padding, padding:-padding]
    else:
        dX_numpy = dX_padded_numpy

    return dX_numpy, dW_numpy, db_numpy


def manual_maxpool2d_forward_numpy(X_numpy, pool_h, pool_w, stride):
    """
    手动最大池化前向传播 (NumPy) - 遵循 (N, C, H_in, W_in) 输入格式
    """
    N, C, H_in, W_in = X_numpy.shape

    H_out = (H_in - pool_h) // stride + 1
    W_out = (W_in - pool_w) // stride + 1

    A_numpy = np.zeros((N, C, H_out, W_out))
    mask_indices_h = np.zeros_like(A_numpy, dtype=int)
    mask_indices_w = np.zeros_like(A_numpy, dtype=int)

    for n in range(N):
        for c_idx in range(C):
            for h_out_idx in range(H_out):
                for w_out_idx in range(W_out):
                    h_start = h_out_idx * stride
                    h_end = h_start + pool_h
                    w_start = w_out_idx * stride
                    w_end = w_start + pool_w

                    X_slice = X_numpy[n, c_idx, h_start:h_end, w_start:w_end]
                    max_val = np.max(X_slice)
                    A_numpy[n, c_idx, h_out_idx, w_out_idx] = max_val

                    r_idx, c_idx_slice = np.unravel_index(np.argmax(X_slice), X_slice.shape)

                    mask_indices_h[n, c_idx, h_out_idx, w_out_idx] = h_start + r_idx
                    mask_indices_w[n, c_idx, h_out_idx, w_out_idx] = w_start + c_idx_slice

    return A_numpy, (mask_indices_h, mask_indices_w)


def manual_maxpool2d_backward_numpy(dA_numpy, X_shape_original, mask_indices, pool_h, pool_w, stride):
    """
    手动最大池化反向传播 (NumPy)
    dA_numpy: (N, C, H_out, W_out)
    """
    N, C, H_out, W_out = dA_numpy.shape
    mask_indices_h, mask_indices_w = mask_indices

    dX_numpy = np.zeros(X_shape_original)

    for n in range(N):
        for c_idx in range(C):
            for h_out_idx in range(H_out):
                for w_out_idx in range(W_out):
                    orig_h_idx = mask_indices_h[n, c_idx, h_out_idx, w_out_idx]
                    orig_w_idx = mask_indices_w[n, c_idx, h_out_idx, w_out_idx]
                    dX_numpy[n, c_idx, orig_h_idx, orig_w_idx] += dA_numpy[n, c_idx, h_out_idx, w_out_idx]
    return dX_numpy


# --- 将NumPy逻辑封装到 torch.autograd.Function ---

class ManualConv2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_tensor, W_tensor, b_tensor, stride, padding):
        X_numpy = X_tensor.detach().cpu().numpy()
        W_numpy = W_tensor.detach().cpu().numpy()
        b_numpy = b_tensor.detach().cpu().numpy()
        Z_numpy = manual_conv2d_forward_numpy(X_numpy, W_numpy, b_numpy, stride, padding)
        ctx.save_for_backward(X_tensor, W_tensor)
        ctx.stride = stride
        ctx.padding = padding
        return torch.from_numpy(Z_numpy).float().to(X_tensor.device)

    @staticmethod
    def backward(ctx, dZ_tensor):
        X_tensor, W_tensor = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dZ_numpy = dZ_tensor.detach().cpu().numpy()
        X_numpy = X_tensor.detach().cpu().numpy()
        W_numpy = W_tensor.detach().cpu().numpy()
        dX_numpy, dW_numpy, db_numpy = manual_conv2d_backward_numpy(dZ_numpy, X_numpy, W_numpy, stride, padding)
        dX_tensor = torch.from_numpy(dX_numpy).float().to(dZ_tensor.device)
        dW_tensor = torch.from_numpy(dW_numpy).float().to(dZ_tensor.device)
        db_tensor = torch.from_numpy(db_numpy).float().to(dZ_tensor.device)
        return dX_tensor, dW_tensor, db_tensor, None, None


class ManualMaxPool2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_tensor, pool_h, pool_w, stride):
        X_numpy = X_tensor.detach().cpu().numpy()
        A_numpy, mask_indices_numpy = manual_maxpool2d_forward_numpy(X_numpy, pool_h, pool_w, stride)
        ctx.X_shape_original = X_numpy.shape
        ctx.mask_indices_numpy = mask_indices_numpy
        ctx.pool_h = pool_h
        ctx.pool_w = pool_w
        ctx.stride = stride
        return torch.from_numpy(A_numpy).float().to(X_tensor.device)

    @staticmethod
    def backward(ctx, dA_tensor):
        dA_numpy = dA_tensor.detach().cpu().numpy()
        dX_numpy = manual_maxpool2d_backward_numpy(dA_numpy,
                                                   ctx.X_shape_original,
                                                   ctx.mask_indices_numpy,
                                                   ctx.pool_h,
                                                   ctx.pool_w,
                                                   ctx.stride)
        dX_tensor = torch.from_numpy(dX_numpy).float().to(dA_tensor.device)
        return dX_tensor, None, None, None


# --- 创建 nn.Module 封装自定义 Function ---

class ManualConv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.bias = nn.Parameter(torch.randn(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return ManualConv2DFunction.apply(x, self.weight, self.bias, self.stride, self.padding)


class ManualMaxPool2DLayer(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size[0]

    def forward(self, x):
        return ManualMaxPool2DFunction.apply(x, self.kernel_size[0], self.kernel_size[1], self.stride)


# --- 构建完整的CNN模型 ---
class TinyMNIST_CNN_PyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ManualConv2DLayer(1, 4, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = ManualMaxPool2DLayer(2, 2)

        self.conv2 = ManualConv2DLayer(4, 8, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = ManualMaxPool2DLayer(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 5 * 5, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x



# --- 训练和评估 ---
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=3):  # 默认epochs改为3
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx > 0 and batch_idx % 20 == 0:  # 调整打印频率以适应小数据集
                print(
                    f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / 20:.4f}')
                running_loss = 0.0

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        accuracy = 100 * correct / total if total > 0 else 0
        print(f'Epoch {epoch + 1} Summary: Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%')


import sys
from PIL import Image,ImageOps
import matplotlib.pyplot  as plt
import cv2

def predict_single_image(image_path, model, device):
    # 加载并转换为灰度图
    img = Image.open(image_path).convert('L')
    np_img = np.array(img)

    # 如果背景偏亮（白底黑字），进行反色
    mean_pixel = np.mean(np_img)
    if mean_pixel < 127:
        img = ImageOps.invert(img)
        np_img = np.array(img)

    # 二值化 (Otsu) 并取反，使数字为白色(255)，背景为黑色(0)
    _, binary = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)

    # 查找外部轮廓，获取最大轮廓区域
    #contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #if contours:
        #c = max(contours, key=cv2.contourArea)
        #x, y, w, h = cv2.boundingRect(c)
        #digit = binary[y:y+h, x:x+w]
    #else:
        #raise ValueError("未找到数字轮廓")

    # 缩放到20x20并嵌入28x28黑色画布中心
    resized = cv2.resize(binary, (20, 20), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    canvas[y_offset:y_offset+20, x_offset:x_offset+20] = resized

    # 归一化到0-1
    normalized = canvas.astype('float32') / 255.0

    # 可视化中间结果
    plt.imshow(canvas, cmap='gray')
    plt.title("MNIST Style Input")
    plt.axis('off')
    plt.show()

    # 转换为Tensor并标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(normalized).unsqueeze(0).to(device)

    # 推理
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    print(f"识别结果：{predicted.item()}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if len(sys.argv) == 2:
        # 如果传入图片路径，则加载模型并识别图片
        model = TinyMNIST_CNN_PyTorch()
        model.load_state_dict(torch.load("mnist_manual_cnn.pth", map_location=device))
        model.to(device)
        predict_single_image(sys.argv[1], model, device)
    else:
        # 否则进行训练
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, Subset
        import torch.optim as optim

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)

        num_train_samples = 6000
        num_val_samples = 500

        indices = torch.randperm(len(full_train_dataset)).tolist()
        train_indices = indices[:num_train_samples]
        val_indices = indices[num_train_samples: num_train_samples + num_val_samples]

        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(full_train_dataset, val_indices)

        print(f"Using {len(train_subset)} samples for training, {len(val_subset)} samples for validation.")

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        model = TinyMNIST_CNN_PyTorch()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("Starting training with reduced data and epochs...")
        train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)

        # 保存模型
        torch.save(model.state_dict(), "mnist_manual_cnn.pth")
        print("模型已保存为 mnist_manual_cnn.pth")

        print("Training finished. Evaluating on test set...")
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
        accuracy = 100 * correct / total if total > 0 else 0
        print(f'Test Set: Average Loss: {avg_test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
