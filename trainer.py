from argparse import ArgumentParser

import os
import lightning as L
import pandas as pd
import seaborn as sn
import torch
from IPython.display import display
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


class LitMNIST(L.LightningModule):
  #⚡闪电模型
    def __init__(self, data_dir=PATH_DATASETS, layer_1_dim=32, hidden_size=64, learning_rate=2e-4):
        super().__init__()

        # 将初始化参数设置为类属性
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.layer_1_dim = layer_1_dim

        # 硬编码一些数据集特定的属性
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # 定义PyTorch模型
        self.model = nn.Sequential(
            nn.Flatten(), #展开
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),  #激活函数
            nn.Dropout(0.1), #暂退法，防止过拟合
            nn.Linear(hidden_size, hidden_size), #隐藏层
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # 调用self.log将为你在TensorBoard中显示标量
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # 调用self.log将为你在TensorBoard中显示标量
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # 下载
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # 为数据加载器分配train/val数据集
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # 为数据加载器分配test数据集
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)

parser = ArgumentParser()

# 训练器参数
parser.add_argument("--devices", type=int, default=2)

# 最大训练次数
parser.add_argument("--max_epochs", type=int, default=5)

# 模型超参数
parser.add_argument("--layer_1_dim", type=int, default=128)

# 解析用户输入和默认值(返回argparse.Namespace)
args = parser.parse_args()

# 在程序中使用已解析的参数
trainer = L.Trainer(devices=args.devices, max_epochs=args.max_epochs)
model = LitMNIST(layer_1_dim=args.layer_1_dim)

# 开始训练
trainer.fit(model)