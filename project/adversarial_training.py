#!/usr/bin/env python3 
import os
import argparse
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from pgd_attack import pgd_attack


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def adversarial_training(net, train_loader, pth_filename, num_epochs, epsilon=0.03, alpha=0.01, num_iter=10):

    print("Starting adversarial training")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            adv_inputs = pgd_attack(net, inputs, labels, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
            optimizer.zero_grad()
            outputs = net(adv_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499:  
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0


    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))