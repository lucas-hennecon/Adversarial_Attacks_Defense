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

def pgd_attack(net, images, labels, epsilon=0.03, alpha=0.01, num_iter=40):
    net.train()
    perturbed_images = images.clone().detach().requires_grad_(True)

    for _ in range(num_iter):
        
        perturbed_images.requires_grad = True
        outputs = net(perturbed_images)
        loss = F.cross_entropy(outputs, labels)
        net.zero_grad()
        loss.backward()
        grad = perturbed_images.grad.data
        perturbed_images = perturbed_images + alpha * grad.sign()
        perturbed_images = torch.clamp(perturbed_images, images - epsilon, images + epsilon)
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        perturbed_images = perturbed_images.clone().detach().requires_grad_(True)
        # perturbed_images.requires_grad = True

    return perturbed_images

def pgd_linf_attack(net, images, labels, epsilon=0.03, alpha=0.01, num_iter=40):

    perturbed_images = images.clone().detach().to(device)
    perturbed_images.requires_grad = True

    for _ in range(num_iter):
        outputs = net(perturbed_images)
        loss = F.cross_entropy(outputs, labels)
        net.zero_grad()
        loss.backward()

        grad = perturbed_images.grad.data
        perturbed_images = perturbed_images + alpha * grad.sign()
        perturbed_images = torch.clamp(perturbed_images, images - epsilon, images + epsilon)
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        perturbed_images = perturbed_images.detach().clone()
        perturbed_images.requires_grad = True

    return perturbed_images

def pgd_l2_attack(net, images, labels, epsilon=0.5, alpha=0.1, num_iter=40):
    perturbed_images = images.clone().detach().to(device)
    perturbed_images.requires_grad = True

    for _ in range(num_iter):
        outputs = net(perturbed_images)
        loss = F.cross_entropy(outputs, labels)
        net.zero_grad()
        loss.backward()

        grad = perturbed_images.grad.data
        grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1, keepdim=True)
        grad_normalized = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)

        perturbed_images = perturbed_images + alpha * grad_normalized
        delta = perturbed_images - images
        delta_norm = torch.norm(delta.view(delta.shape[0], -1), dim=1, keepdim=True)
        delta = delta / torch.clamp(delta_norm / epsilon, min=1.0).view(-1, 1, 1, 1)
        perturbed_images = images + delta
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        perturbed_images = perturbed_images.detach().clone()
        perturbed_images.requires_grad = True

    return perturbed_images


def test_pgd(net, test_loader, epsilon=0.03, alpha=0.01, num_iter=40):

    correct = 0
    total = 0
    
    for i, data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        perturbed_images = pgd_attack(net, images, labels, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
        outputs = net(perturbed_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total
