# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:18:13 2021

@author: Yun He
"""

import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class RestNet18_bilinear(nn.Module):
    def __init__(self):
        super(RestNet18_bilinear,self).__init__()
        self.features = nn.Sequential(resnet18().conv1,
                                     resnet18().bn1,
                                     resnet18().relu,
                                     resnet18().maxpool,
                                     resnet18().layer1,
                                     resnet18().layer2,
                                     resnet18().layer3,
                                     resnet18().layer4)
        self.classifiers = nn.Sequential(nn.Linear(512**2,22))


    def forward(self, x):
        x=self.features(x)
        batch_size = x.size(0)
        feature_size = x.size(2)*x.size(3)
        x = x.view(batch_size , 512, feature_size)
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / feature_size).view(batch_size, -1)
        x = torch.nn.functional.normalize(torch.sign(x)*torch.sqrt(torch.abs(x)+1e-10))
        x = self.classifiers(x)
        return x


def main():
    batchsz = 128

    # cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ]), download=True)
    # cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32,32)), #修改张量大小
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    cifar_train = ImageFolder("./DATA/All/train/",transform = data_transform)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)
    
    

    # cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ]), download=True)
    # cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)
    cifar_test = ImageFolder("./DATA/All/validation/",transform = data_transform)
    cifar_test = Data.DataLoader(cifar_test, batch_size=batchsz, shuffle=True)
    

    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')
    # model = Lenet5().to(device)
    model = RestNet18_bilinear().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    for epoch in range(110):

        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)

            logits = model(x)
            # logits: [b, 10]
            # label:  [b]
            # loss: tensor scalar
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss:', loss.item())

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # [b, 3, 32, 32]
                # [b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                # print(correct)

            acc = total_correct / total_num
            print(epoch, 'test acc:', acc)


main()
