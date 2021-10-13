import os
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
import sys
from datetime import datetime
import argparse

device = "cpu"
torch.set_num_threads(4)

batch_size = 128 # batch for one node
def train_model(model, train_loader, optimizer, criterion, epoch, rank, dist):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    prev_time = datetime.now()
    group = dist.new_group([0, 1])
    model.train()
    # remember to exit the train loop at end of the epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        # Your code goes here!
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.grad = p.grad / 4.0
            dist.all_reduce(p.grad, op=dist.reduce_op.SUM, group=group, async_op=False)
        optimizer.step()

        output = output.float()
        loss = loss.float()

        if batch_idx % 20 == 0:
            print(batch_idx, "loss: ", loss.item())
        if batch_idx < 40:
            later = datetime.now()
            print("average time: ", (later - prev_time).total_seconds())
            prev_time = later

    return None

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            

def main():
    torch.manual_seed(5000)
    np.random.seed(5000)

    parser = argparse.ArgumentParser()
    parser.add_argument('--master-ip', dest='master_ip', type=str, help='master ip, 10.10.1.1')
    parser.add_argument('--num-nodes', dest='size', type=int, help='number of nodes, 4')
    parser.add_argument('--rank', dest='rank', type=int, help='rank, 0')
    args = parser.parse_args()

    dist.init_process_group(backend="gloo",
                            init_method="tcp://"+args.master_ip+":1234",
                            rank=args.rank,
                            world_size=args.size)

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)
    sampler_d = DistributedSampler(training_set,num_replicas=args.size,rank=args.rank) if torch.distributed.is_available() else None
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=sampler_d,
                                                    #shuffle=True,
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    

    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch, args.rank, dist)
        test_model(model, test_loader, training_criterion)

if __name__ == "__main__":
    main()
