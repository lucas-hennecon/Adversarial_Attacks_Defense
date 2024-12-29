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
from dim_attack import dim_attack, test_dim
from fgsm_attack import fgsm_attack, test_fgsm
from pgd_attack import pgd_l2_attack, pgd_linf_attack, test_pgd
from mim_attack import mim_attack, test_mim
from mat_rand_training import *
from utils import *
from model import *


def main():

    #### Parse command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file,
                        help="Name of the file used to load or to sore the model weights."\
                        "If the file exists, the weights will be load from it."\
                        "If the file doesn't exists, or if --force-train is set, training will be performed, "\
                        "and the model weights will be stored in this file."\
                        "Warning: "+Net.model_file+" will be used for testing (see load_for_testing()).")
    parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists"\
                             "Warning: previous model file will be erased!).")
    parser.add_argument('-e', '--num-epochs', type=int, default=10,
                        help="Set the number of epochs during training")
    args = parser.parse_args()

    #### Create model and move it to whatever device is available (gpu/cpu)
    net = Net()
    net.to(device)
    # models = [Net().to(device) for _ in range(5)]
    # net = RandomizedEnsemble(models).to(device)
    #### Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model")
        print(args.model_file)

        train_transform = transforms.Compose([transforms.ToTensor()]) 
        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        # train_model(net, train_loader, args.model_file, args.num_epochs)
        # train_ensemble(net, train_loader,args.model_file, num_epochs=args.num_epochs)
        # adversarial_training(net, train_loader, args.model_file, num_epochs=args.num_epochs,
        #                      epsilon=0.03, alpha=0.01, num_iter=40)
        mat_rand_training(net, train_loader,args.model_file, num_epochs=args.num_epochs)
        print("Model save to '{}'.".format(args.model_file))

    #### Model testing
    print("Testing with model from '{}'. ".format(args.model_file))
    # net=ensemble
    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
    valid_loader = get_validation_loader(cifar, valid_size)

    net.load(args.model_file)
    acc_pgd = test_pgd(net, valid_loader, epsilon=0.03, alpha=0.01, num_iter=40)
    print("Model accuracy under PGD attack (test): {}".format(acc_pgd))
    acc_fgsm = test_fgsm(net, valid_loader, epsilon=0.03)
    print("Model accuracy under FGSM attack (test): {}".format(acc_fgsm))
    acc = test_natural(net, valid_loader)
    print("Model natural accuracy (valid): {}".format(acc))
    acc_mim = test_mim(net, valid_loader, epsilon=0.03, alpha=0.01, num_iter=40, decay=1.0)
    print("Model accuracy under MIM attack (test): {}".format(acc_mim))
    acc_dim = test_dim(net, valid_loader, epsilon=0.03, alpha=0.01, num_iter=40, decay=1.0, p=0.5)
    print("Model accuracy under DI-MIM attack (test): {}".format(acc_dim))

    if args.model_file != Net.model_file:
        print("Warning: '{0}' is not the default model file, "\
              "it will not be the one used for testing your project. "\
              "If this is your best model, "\
              "you should rename/link '{0}' to '{1}'.".format(args.model_file, Net.model_file))

if __name__ == "__main__":
    main()