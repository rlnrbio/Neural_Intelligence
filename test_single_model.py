# -*- coding: utf-8 -*-
"""
Created on Sun May 23 19:00:03 2021

@author: rapha
"""

import os
import time

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST, CIFAR10
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F


class one_pass(nn.Module):
    def __init__(self, input_size, output_size, activation = nn.ReLU(True), criterion = nn.MSELoss(), lr = 0.0001):
        super(one_pass, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, output_size),
            activation)
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.parameters(), lr)

    def forward(self, inp):
        x = inp.detach()
        y = self.linear(x)
        return y.detach()
    
    def forward_train(self, inp, target):
        x = inp.detach()
        y = self.linear(x)        
        #print(target.shape, y.shape)

        if self.training:
            loss = self.criterion(target, Variable(y.data, requires_grad = True))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return y.detach(), loss.item()

class le_model(nn.Module):
    def __init__(self):
        super(le_model, self).__init__()
        self.f1 = one_pass(28*28,128)
        self.f2 = one_pass(128,64)
        self.f3 = one_pass(64,10, activation = nn.Softmax())
        
        self.g3 = one_pass(10,64)
        self.g2 = one_pass(64,128)
        
    def f_forward(self, inp):
        a1 = self.f1(inp)
        a2 = self.f2(a1)
        a3 = self.f3(a2)
        return a1, a2, a3
    
    def g_backward(self, target):
        a2_ = self.g3(target)
        a1_ = self.g2(a2_)
        return a2_, a1_
    
    def train(self, inp, target):        
        self.g2.eval()
        self.g3.eval()
        self.f1.train()
        self.f2.train()
        self.f3.train()     
        ga2, ga1 = self.g_backward(target)
        # print(ga1.shape, a1.shape)
   
        a1, loss_a1 = self.f1.forward_train(inp, ga1)
        a2, loss_a2 = self.f2.forward_train(a1, ga2)
        a3, loss_a3 = self.f3.forward_train(a2, target)
        self.g2.train()
        self.g3.train()
        ga2, loss_ga1 = self.g3.forward_train(target, a2)
        ga1, loss_ga2 = self.g2.forward_train(ga2, a1)
        
        return [loss_a1, loss_a2, loss_a3, loss_ga1, loss_ga2]
    

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = FashionMNIST('./data',train = True, transform=img_transform)
dataset_test = FashionMNIST('./data', train = False, transform=img_transform)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

model = le_model().cuda()

for epoch in range(num_epochs):
    losses = np.zeros((num_epochs, 5))
    accuracy = np.zeros((num_epochs, 2))
    la1, la2, la3, lga1, lga2 = [], [], [], [], []
    correct = 0
    total = 0
    for data in dataloader:
        img, target_full = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        target = Variable(F.one_hot(target_full)).float().cuda()
        losses = model.train(img, target)
        
        la1.append(losses[0])
        la2.append(losses[1])
        la3.append(losses[2])
        lga1.append(losses[3])
        lga2.append(losses[4])

        
    print('epoch [{}/{}], a1_loss: {:.3f}, a2_loss: {:.3f}, a3_loss: {:.3f}, ga2_loss: {:.3f}, ga1_loss: {:.3f}'.format(
            epoch + 1, num_epochs,  np.mean(la1), np.mean(la2), np.mean(la3), np.mean(lga1), np.mean(lga2)))

