import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --------------------------
# Generator 定义
# --------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Linear(1024, 784),  # 28x28 = 784
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# --------------------------
# Discriminator 定义
# --------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# --------------------------
# GAN Trainer 类
# --------------------------
class GANTrainer:
    def __init__(self, generator, discriminator, train_loader, device, latent_dim, num_epochs):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.device = device
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs

        self.criterion = nn.BCELoss()
        self.g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.g_losses = []
        self.d_losses = []

    def train(self):
        for epoch in tqdm(range(self.num_epochs), desc="Epoch"):
            for i, (real_images, _) in enumerate(tqdm(self.train_loader, desc="Batch", leave=False)):
                batch_size = real_images.size(0)
                real_images = real_images.view(-1, 784).to(self.device)

                real_label = torch.ones(batch_size, 1).to(self.device)
                fake_label = torch.zeros(batch_size, 1).to(self.device)

                # ---- 训练 Discriminator ----
                self.d_optimizer.zero_grad()
                d_real = self.discriminator(real_images)
                d_real_loss = self.criterion(d_real, real_label)

                noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_images = self.generator(noise)
                d_fake = self.discriminator(fake_images.detach())
                d_fake_loss = self.criterion(d_fake, fake_label)

                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.d_optimizer.step()

                # ---- 训练 Generator ----
                self.g_optimizer.zero_grad()
                fake_images = self.generator(noise)
                d_fake = self.discriminator(fake_images)
                g_loss = self.criterion(d_fake, real_label)
                g_loss.backward()
                self.g_optimizer.step()

                self.g_losses.append(g_loss.item())
                self.d_losses.append(d_loss.item())

            print(f"Epoch [{epoch+1}/{self.num_epochs}]  G_loss: {g_loss.item():.4f}  D_loss: {d_loss.item():.4f}")

    def plot_losses(self):
        if not os.path.exists('./result'):
            os.makedirs('./result')
        plt.figure(figsize=(10, 5))
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('GAN Training Losses')
        plt.legend()
        plt.savefig('./result/gan_losses.png')
        plt.close()

    def generate_samples(self, num_samples=36):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim).to(self.device)
            fake_images = self.generator(noise)
            fake_images = fake_images.cpu().view(num_samples, 28, 28)

            plt.figure(figsize=(8, 8))
            for i in range(6):
                for j in range(6):
                    plt.subplot(6, 6, i * 6 + j + 1)
                    plt.imshow(fake_images[i * 6 + j], cmap='gray')
                    plt.axis('off')

            plt.suptitle('Generated MNIST Images', fontsize=20)
            plt.savefig('./result/gan_generated_samples.png')
            plt.close()

# --------------------------
# 主程序入口
# --------------------------
if __name__ == '__main__':
    # 加载 MNIST 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # 展平为784维
    ])
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

    # 设置超参数和设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 100
    num_epochs = 50

    # 初始化模型与训练器
    generator = Generator(latent_dim)
    discriminator = Discriminator()
    trainer = GANTrainer(generator, discriminator, train_loader, device, latent_dim, num_epochs)

    # 执行训练和结果保存
    trainer.train()
    trainer.plot_losses()
    trainer.generate_samples()
