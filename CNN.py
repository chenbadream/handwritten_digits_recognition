import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from DataLoader import MNISTDataLoader
from ARDISDataLoader import ARDISDataLoader

# class CNN
# define the Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self, input_size, output_size=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, output_size)
        self.input_size = input_size
        self.act_func = nn.ReLU()


    def forward(self, x):
        x = x.reshape(x.shape[0], 1, self.input_size, self.input_size)
        out = self.act_func(self.conv1(x))
        out = self.pool(out)
        out = self.act_func(self.conv2(out))
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.act_func(self.fc1(out))
        out = self.fc2(out)

        return out
    
class Trainer():
    def __init__(self, model, train_loader, valid_loader, num_epochs, lr):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs
        self.lr = lr
        self.label = 'CNN'
        
    def training(self):
        # The cross entropy loss for classification
        loss_func = nn.CrossEntropyLoss()
        # The simple gradient descent optimizer
        # You can try different hyperparameters if you want
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        
        # record the loss and accuracy in each epoch
        loss_train = []
        loss_valid = []
        acc_train = []
        acc_valid = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Model is running on: " + str(device) + "\n")

        self.model.to(device)

        for epoch in tqdm(range(self.num_epochs), desc="Epoch"):
            # Training loop
            self.model.train()
            epoch_train_loss = 0
            epoch_train_acc_num = 0
            for inputs, labels in tqdm(self.train_loader, desc="Train"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass
                # Outputs is the prediction of the network
                outputs = self.model.forward(inputs)
                loss = loss_func(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                predict = torch.argmax(outputs, 1)
                epoch_train_acc_num += (predict == labels).sum().item()
                epoch_train_loss += loss.item()

            epoch_train_loss = epoch_train_loss / (len(self.train_loader) * self.train_loader.batch_size)
            loss_train.append(epoch_train_loss)
            epoch_train_acc = epoch_train_acc_num / (len(self.train_loader) * self.train_loader.batch_size)
            acc_train.append(epoch_train_acc)

            # Evluation on the validation set
            self.model.eval()
            epoch_valid_loss = 0
            epoch_valid_acc_num = 0
            for inputs, labels in tqdm(self.valid_loader, desc="Valid"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model.forward(inputs)
                loss = loss_func(outputs, labels)
                predict = torch.argmax(outputs, 1)
                epoch_valid_acc_num += (predict == labels).sum().item()
                epoch_valid_loss += loss.item()
                
            epoch_valid_loss = epoch_valid_loss / (len(self.valid_loader) * self.valid_loader.batch_size)
            loss_valid.append(epoch_valid_loss)
            epoch_valid_acc = epoch_valid_acc_num / (len(self.valid_loader) * self.valid_loader.batch_size)
            acc_valid.append(epoch_valid_acc)

            print('Epoch: ' + str(epoch) + ',\n Train Loss: ' + str(epoch_train_loss) + ',\n Train Acc: ' + str(epoch_train_acc) + ',\n Valid Loss: ' + str(epoch_valid_loss) + ',\n Valid Acc: ' + str(epoch_valid_acc))
            self.visualize(loss_train, loss_valid, acc_train, acc_valid)

        # 保存训练好的模型
        torch.save(self.model.state_dict(), './result/cnn_model_big.pth')  # 保存模型
        print("Model saved at './result/cnn_model.pth'")

    def visualize(self, loss_train, loss_valid, acc_train, acc_valid):
        # Plot the loss and accuracy
        epochs = np.array(range(len(loss_train)))
        loss_train = np.array(loss_train)
        loss_valid = np.array(loss_valid)
        acc_train = np.array(acc_train)
        acc_valid = np.array(acc_valid)

        # check the result folder
        if not os.path.exists('./result'):
            os.makedirs('./result')
        
        plt.figure()
        plt.plot(epochs, loss_train)
        plt.plot(epochs, loss_valid)
        plt.title(self.label + ' Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Valid'])
        plt.savefig('./result/' + self.label + '_loss.png')
        
        plt.figure()
        plt.plot(epochs, acc_train)
        plt.plot(epochs, acc_valid)
        plt.title(self.label + ' Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Valid'])
        plt.savefig('./result/' + self.label + '_acc.png')

        plt.close('all')

if __name__ == '__main__':
    # Load the MNIST dataset
    dataloader_mnist = MNISTDataLoader(data_dir='data')
    dataloader_ardis = ARDISDataLoader(data_dir='ARDIS_DATASET_3')

    # 分别加载
    X_mnist, y_mnist = dataloader_mnist.load_data()    # 形状 [N1, 784]
    X_ardis, y_ardis = dataloader_ardis.load_data()    # 形状 [N2, 784]

    # 合并成一个更大的数据集
    X_all = np.concatenate([X_mnist, X_ardis], axis=0)  # [N1+N2, 784]
    y_all = np.concatenate([y_mnist, y_ardis], axis=0)  # [N1+N2, ]

    # 打乱并划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, 
    test_size=0.2, 
    random_state=42,      # 保证可复现
    shuffle=True, 
    stratify=y_all        # 保证各类别比例一致
    )

    # 变形并转换类型，以适配 PyTorch
    X_train = X_train.reshape(-1, 1, 28, 28).astype(np.float32)
    X_test  = X_test.reshape(-1, 1, 28, 28).astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test  = y_test.astype(np.int64)
    # Hyperparameters of training
    BATCH_SIZE = 32
    EPOCH = 16
    HIDDEN_SIZE = 1024
    LEARNING_RATE = 0.1

    train_loader = torch.utils.data.DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=BATCH_SIZE, shuffle=True)
    
    # Class number
    OUTPUT_SIZE = 10
    print(X_train.shape)
    # Train the MLP model
    mlp = CNN(input_size=X_train.shape[2], output_size=OUTPUT_SIZE)
    trainer = Trainer(model=mlp, 
             train_loader=train_loader, 
             valid_loader=test_loader, 
             num_epochs=EPOCH, lr=LEARNING_RATE)
    trainer.training()