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

class RandomizedEnsemble(nn.Module):
    def __init__(self, models):
        super(RandomizedEnsemble, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):

        random_idx = torch.randint(0, len(self.models), (1,)).item()
        chosen_model = self.models[random_idx]
        return chosen_model(x)
    def save(self, model_file):
        torch.save(self.state_dict(), model_file)

    def load(self, model_file, device="cpu"):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

def train_model(net, train_loader, pth_filename, num_epochs):
    '''Basic training function (from pytorch doc.)'''
    print("Starting training")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))


def train_ensemble(ensemble, train_loader, pth_filename, num_epochs):
    print("Starting training for randomized ensemble")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ensemble.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  
        ensemble.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            random_idx = torch.randint(0, len(ensemble.models), (1,)).item()
            chosen_model = ensemble.models[random_idx]

            optimizer.zero_grad()
            outputs = chosen_model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499: 
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    torch.save(ensemble.state_dict(), pth_filename)
    print('Randomized ensemble model saved in {}'.format(pth_filename))

def test_natural(net, test_loader):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def get_train_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the training part.'''

    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    train = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return train

def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)

    return valid
