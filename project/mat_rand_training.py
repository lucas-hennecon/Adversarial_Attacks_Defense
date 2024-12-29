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
from pgd_attack import pgd_linf_attack, pgd_l2_attack


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



def mat_rand_training(net, train_loader, pth_filename, num_epochs, epsilon_linf=0.03, alpha_linf=0.01, 
                      epsilon_l2=0.5, alpha_l2=0.1, num_iter=40):
    print("Starting MAT-Rand training")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            if torch.rand(1).item() < 0.5:
                adv_inputs = pgd_linf_attack(net, inputs, labels, epsilon=epsilon_linf, alpha=alpha_linf, num_iter=num_iter)
            else:
                adv_inputs = pgd_l2_attack(net, inputs, labels, epsilon=epsilon_l2, alpha=alpha_l2, num_iter=num_iter)

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