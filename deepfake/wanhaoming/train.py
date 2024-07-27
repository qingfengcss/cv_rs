import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import math
import argparse

import sys
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model.convnext import convnext_base

class ImageDataset(Dataset):
    def __init__(self, txt_file, img_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(txt_file, header=0, names=['img_name', 'target'])
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = int(self.annotations.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def main():
    train_txt_file = "E:/DataSet/waitan2024_deepfake_challenge__赛道1对外发布数据集_phase1/waitan2024_deepfake_challenge__赛道1对外发布数据集_phase1/phase1/trainset_label.txt"
    train_img_dir = "E:/DataSet/waitan2024_deepfake_challenge__赛道1对外发布数据集_phase1/waitan2024_deepfake_challenge__赛道1对外发布数据集_phase1/phase1/trainset"

    val_txt_file = "E:/DataSet/waitan2024_deepfake_challenge__赛道1对外发布数据集_phase1/waitan2024_deepfake_challenge__赛道1对外发布数据集_phase1/phase1/valset_label.txt"
    val_img_dir = "E:/DataSet/waitan2024_deepfake_challenge__赛道1对外发布数据集_phase1/waitan2024_deepfake_challenge__赛道1对外发布数据集_phase1/phase1/valset"

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = ImageDataset(txt_file=train_txt_file, img_dir=train_img_dir, transform=transform)
    val_dataset = ImageDataset(txt_file=val_txt_file, img_dir=val_img_dir, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=0)

    # 加载权重，删除需要微调的权重
    device="cuda:0"
    device = "cuda:0"
    model = model = convnext_base(100).to(device)


    # 设置学习率、优化器
    lr = 0.001
    lrf = 0.01
    epochs = 100
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5)

    lf = lambda x: (1 + math.cos(x * math.pi / epochs) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    for epoch in range(epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_dataloader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                        data_loader=val_dataloader,
                                        device=device,
                                        epoch=epoch)
        if epoch%5 == 0:
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))