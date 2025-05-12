import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms
import numpy as np


class MNISTDataLoader:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])

    def load_data(self):
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        X = torch.stack([data[0] for data in dataset])
        y = torch.tensor([data[1] for data in dataset])
        return X, y


class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes, device):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def one_hot(self, labels):
        labels = labels.long()
        return torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()

    def encode(self, x, labels):
        label_onehot = self.one_hot(labels).to(self.device)
        x = torch.cat([x, label_onehot], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        labels = labels.to(self.device)
        z = z.to(self.device)
        if labels.dim() > 1:
            labels = labels.view(-1)
        label_onehot = self.one_hot(labels).to(self.device)
        label_onehot = label_onehot.view(z.size(0), -1)
        z = torch.cat([z, label_onehot], dim=1)
        return self.decoder(z)

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

    def sample(self, labels, num_samples=None):
        labels = labels.to(self.device)
        if num_samples is None:
            num_samples = labels.shape[0]
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        return self.decode(z, labels)


class Trainer:
    def __init__(self, model, optimizer, train_loader, device, num_epochs):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.num_epochs = num_epochs
        self.train_losses = []

    def loss_function(self, recon_x, x, mu, logvar):
        reconstruction_loss = nn.MSELoss(reduction='sum')(recon_x, x)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (reconstruction_loss + kld_loss) / x.size(0)

    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.num_epochs), desc="Epoch"):
            total_loss = 0.0
            for x, labels in tqdm(self.train_loader, desc="Train"):
                x = x.view(-1, 28 * 28).to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                recon_x, mu, logvar = self.model(x, labels)
                loss = self.loss_function(recon_x, x, mu, logvar)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            total_loss /= len(self.train_loader)
            self.train_losses.append(total_loss)
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    def plot_losses(self):
        if not os.path.exists('./result'):
            os.makedirs('./result')
        plt.plot(self.train_losses, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('./result/cvae_loss.png')
        plt.close()

    def save_model(self):
        torch.save(self.model.state_dict(), './result/cvae_model.pt')

    def generate_given_label(self, digit, num_samples=6):
        self.model.eval()
        with torch.no_grad():
            labels = torch.full((num_samples,), digit, dtype=torch.long).to(self.device)
            samples = self.model.sample(labels).cpu().numpy().reshape(num_samples, 28, 28)

            plt.figure(figsize=(10, 2))
            for i in range(num_samples):
                plt.subplot(1, num_samples, i + 1)
                plt.imshow(samples[i], cmap="gray")
                plt.title(f"{digit}")
                plt.axis("off")

            plt.tight_layout()
            plt.savefig(f'./result/cvae_digit_{digit}.png')
            plt.close()


if __name__ == '__main__':
    dataloader = MNISTDataLoader()
    X, y = dataloader.load_data()
    X = X.float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dim = 784
    hidden_dim = 512
    latent_dim = 40
    batch_size = 256
    num_epochs = 60
    lr = 0.001

    train_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    model = CVAE(input_dim, hidden_dim, latent_dim, num_classes=10, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(model, optimizer, train_loader, device, num_epochs)
    trainer.train()
    trainer.plot_losses()
    trainer.save_model()

    for digit in range(10):
        trainer.generate_given_label(digit, num_samples=6)
