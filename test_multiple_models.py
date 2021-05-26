# -*- coding: utf-8 -*-
"""
Created on Sun May 23 19:43:27 2021

@author: rapha
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision.utils import save_image

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-2
layer_criterion = nn.MSELoss()
target_criterion = nn.MSELoss()


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = FashionMNIST('./data',train = True, transform=img_transform)
dataset_test = FashionMNIST('./data', train = False, transform=img_transform)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

class forward_pass(nn.Module):
    def __init__(self, input_size, output_size, activation = nn.ReLU(True)):
        super(forward_pass, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, output_size),
            activation)
    def forward(self, x):
        #x = x.detach()
        y = self.encoder(x)
        return y #y.detach()
    
    
class backward_pass(nn.Module):
    def __init__(self, input_size, output_size, activation = nn.ReLU(True)):
        super(backward_pass, self).__init__()
        #here input and output size should be switched from forward pass
        self.decoder = nn.Sequential(
            nn.Linear(input_size, output_size),
            activation)
    def forward(self, x):
        #x = x.detach()
        y = self.decoder(x)
        return y #y.detach()


f1 = forward_pass(28*28,128).cuda()
f2 = forward_pass(128,64).cuda()
f3 = forward_pass(64,10, activation = nn.Softmax()).cuda()

g2 = backward_pass(64,128).cuda()
g3 = backward_pass(10,64).cuda()

of1 = torch.optim.Adam(f1.parameters(), lr=learning_rate, weight_decay=1e-5)
of2 = torch.optim.Adam(f2.parameters(), lr=learning_rate, weight_decay=1e-5)
of3 = torch.optim.Adam(f3.parameters(), lr=learning_rate, weight_decay=1e-5)

og2 = torch.optim.Adam(g3.parameters(), lr=learning_rate, weight_decay=1e-5)
og3 = torch.optim.Adam(g2.parameters(), lr=learning_rate, weight_decay=1e-5)

passes_forward = [f1, f2, f3] # in order of later training
passes_backward = [g3, g2]
optimizers_forward = [of1, of2, of3] # in order of later training
optimizers_backward = [og3, og2]

#a2_ = torch.rand((128, 64))
#a1_ = torch.rand((128, 128))


for epoch in range(num_epochs):
    losses = np.zeros((num_epochs, 5))
    accuracy = np.zeros((num_epochs, 2))
    la1, la2, la3, lga1, lga2 = [], [], [], [], []
    correct = 0
    total = 0
    for data in dataloader:
        img, target_full = data
        if target_full.shape[0] != batch_size:
            continue
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        target = Variable(F.one_hot(target_full)).float().cuda()
        
        #a2_ = Variable(a2_).cuda()
        #a1_ = Variable(a1_).cuda()
        
        #-------------------forward_training---------------------------
        # set backward passes to evaluation:
        # do backward passes without training
        g2.eval()
        g3.eval()
        f1.train()
        f2.train()
        f3.train()
        a2_ = g3(target)
        
        a1_ = g2(a2_)
        
        # training f1
        
        a1_ = Variable(a1_.data, requires_grad = True)
        of1.zero_grad()
        a1 = f1(img)
        loss_a1 = layer_criterion(a1_, a1)
        loss_a1.backward()
        of1.step()

        
        of2.zero_grad()
        a1 = Variable(a1.data, requires_grad=True)        
        a2_ = Variable(a2_.data, requires_grad = True)
        a2 = f2(a1)
        loss_a2 = layer_criterion(a2_, a2)
        loss_a2.backward()
        of2.step()
                
        of3.zero_grad()
        a2 = Variable(a2.data, requires_grad=True)
        target = Variable(target.data, requires_grad=True)
        a3 = f3(a2)
        loss_a3 = target_criterion(target, a3)
        loss_a3.backward()
        of3.step()           
        
        #-------------------backward training--------------------------
        g2.train()
        g3.train()
        f1.eval()
        f2.eval()
        f3.eval()
        
        a2 = Variable(a2.data, requires_grad = True)
        og3.zero_grad()
        a2_ = g3(target)
        loss_ga2 = layer_criterion(a2, a2_) # needs to be changed for difference target propagation
        loss_ga2.backward()
        og3.step()

        a2_ = Variable(a2_.data, requires_grad=True)
        a1 = Variable(a1.data, requires_grad = True)        
        og2.zero_grad()
        a1_ = g2(a2_)
        loss_ga1 = layer_criterion(a1, a1_) # needs to be changed for difference target propagation
        loss_ga1.backward()
        og2.step()        

        #-----------------log - housekeeping------------------
        la1.append(loss_a1.item())
        la2.append(loss_a2.item())
        la3.append(loss_a3.item())
        lga1.append(loss_ga1.item())
        lga2.append(loss_ga1.item())
    
        # training accuracy:
        _, predicted = torch.max(a3.data, 1)
        total += target.size(0)
        correct += (predicted == target_full.cuda()).sum().item()
    train_acc = 100*correct/total
    tcorrect = 0
    ttotal = 0
    for data in testloader:
        img, target_full = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        target = Variable(F.one_hot(target_full)).float().cuda()
        ta3 = f3(f2(f1(img)))
        _, predicted = torch.max(ta3.data, 1)
        ttotal += target.size(0)
        tcorrect += (predicted == target_full.cuda()).sum().item()    
    test_acc = 100*tcorrect/ttotal

    print('epoch [{}/{}], a1_loss: {:.3f}, a2_loss: {:.3f}, a3_loss: {:.3f}, ga2_loss: {:.3f}, ga1_loss: {:.3f}; training acc: {:,.3f}, test acc: {:.3f}'.format(
            epoch + 1, num_epochs,  np.mean(la1), np.mean(la2), np.mean(la3), np.mean(lga1), np.mean(lga2), train_acc, test_acc))

    losses[epoch] = np.mean(la1), np.mean(la2), np.mean(la3), np.mean(lga1), np.mean(lga2)
    accuracy[epoch] = train_acc, test_acc



