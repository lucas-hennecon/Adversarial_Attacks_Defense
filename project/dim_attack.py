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

def dim_attack(net, images, labels, epsilon=0.03, alpha=0.01, num_iter=40, decay=1.0, p=0.5):
    """
    Diversity Input Momentum Iterative Method (DI-MIM) attack.

    Args:
        net (torch.nn.Module): Model to attack.
        images (torch.Tensor): Input images.
        labels (torch.Tensor): True labels for the input images.
        epsilon (float): Maximum allowed perturbation (L_infinity constraint).
        alpha (float): Step size for each iteration.
        num_iter (int): Number of attack iterations.
        decay (float): Decay factor for momentum.
        p (float): Probability of applying input transformation.

    Returns:
        torch.Tensor: Adversarial examples.
    """
    perturbed_images = images.clone().detach().to(device)
    perturbed_images.requires_grad = True
    momentum = torch.zeros_like(images).to(device)

    def input_diversity(input_tensor):
        """Applies random resizing and padding with probability p."""
        if torch.rand(1).item() > p:
            return input_tensor
        batch_size, channels, height, width = input_tensor.size()
        resize_scale = torch.randint(int(0.9 * height), int(1.1 * height), (1,)).item()
        input_resized = F.interpolate(input_tensor, size=(resize_scale, resize_scale), mode='bilinear', align_corners=False)
        padded = F.pad(input_resized, (0, width - resize_scale, 0, height - resize_scale), mode='constant', value=0)
        return padded

    for _ in range(num_iter):
        # Apply input diversity
        diversified_inputs = input_diversity(perturbed_images)

        # Forward pass and compute gradients
        outputs = net(diversified_inputs)
        loss = F.cross_entropy(outputs, labels)
        net.zero_grad()
        loss.backward()

        # Normalize the gradient
        grad = perturbed_images.grad.data
        grad_norm = grad / (grad.abs().view(grad.shape[0], -1).sum(dim=1, keepdim=True) + 1e-8)

        # Update momentum
        momentum = decay * momentum + grad_norm

        # Update adversarial example
        perturbed_images = perturbed_images + alpha * momentum.sign()

        # Project to L_infinity ball
        perturbed_images = torch.clamp(perturbed_images, images - epsilon, images + epsilon)

        # Clamp to valid pixel range
        perturbed_images = torch.clamp(perturbed_images, 0, 1)

        # Detach for next iteration
        perturbed_images = perturbed_images.detach().clone()
        perturbed_images.requires_grad = True

    return perturbed_images

def test_dim(net, test_loader, epsilon=0.03, alpha=0.01, num_iter=40, decay=1.0, p=0.5):
    correct = 0
    total = 0
    for i, data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        adv_images = dim_attack(net, images, labels, epsilon=epsilon, alpha=alpha, num_iter=num_iter, decay=decay, p=p)
        outputs = net(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total
