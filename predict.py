import torch
import matplotlib.pyplot as plt
import os
from CVAE import CVAE  # 假设你把主训练文件命名为 cvae_model.py

def generate_digit(digit=5, model_path="./_pycache_/result/cvae_model.pt", output_path=None, num_samples=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = CVAE(input_dim=784, hidden_dim=512, latent_dim=40, num_classes=10, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 指定生成数字
    labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)
    with torch.no_grad():
        samples = model.sample(labels).cpu().numpy().reshape(num_samples, 28, 28)

    # 显示或保存图像
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(samples[i], cmap='gray')
        plt.axis('off')

    if output_path:
        file_name = f"digit_{digit}_samples.png"
        save_path = os.path.join(output_path, file_name)
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--digit', type=int, required=True, help='Digit (0-9) to generate')
    parser.add_argument('--model', type=str, default='./result/cvae_model.pt', help='Path to model file')
    parser.add_argument('--output', type=str, default=None, help='Path to save image (optional)')
    parser.add_argument('--num_samples', type=int, default=6, help='Number of samples to generate')
    args = parser.parse_args()

    generate_digit(digit=args.digit, model_path=args.model, output_path=args.output, num_samples=args.num_samples)
