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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def fgsm_attack(net, images, labels, epsilon=0.03):
    images = images.clone().detach().to(device)
    images.requires_grad = True 
    outputs = net(images)
    loss = F.cross_entropy(outputs, labels)
    net.zero_grad()
    loss.backward()
    grad = images.grad.data
    perturbed_images = images + epsilon * grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images


def test_fgsm(net, test_loader, epsilon=0.03):
    correct = 0
    total = 0
    for i, data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        perturbed_images = fgsm_attack(net, images, labels, epsilon=epsilon)
        outputs = net(perturbed_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total
