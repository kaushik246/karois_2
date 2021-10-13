import sys
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
import argparse
from datetime import datetime
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

device = "cpu"
torch.set_num_threads(4)

batch_size = 64 # batch for one node

def train_model(model, train_loader, optimizer, criterion, epoch, rank):
    model.train()
    prev_time = datetime.now()
    # remember to exit the train loop at end of the epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # run forward pass, backward pass and optimizer step on the DDP model
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        loss = loss.float()

        if batch_idx % 20 == 0:    # print every 2000 mini-batches
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

# initialise the process group which blocks until all processes have joined
def initialise(master_ip, rank, size, vgg_model, backend='gloo'):
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = '27810'
    dist.init_process_group(backend="gloo",
                            init_method="tcp://"+args.master_ip+":1234",
                            rank=args.rank,
                            world_size=args.size)

    vgg_model(rank, size)

def vgg_model(rank,size):
    torch.manual_seed(5000)
    np.random.seed(5000)
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
    
    train_sampler = DistributedSampler(training_set,num_replicas=size,rank=rank) 
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=train_sampler,
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

    # wrap the model with DDP to leverage multi-process parallelism
    distDataPrl = DDP(model)
    optimizer = optim.SGD(distDataPrl.parameters(), lr=0.1,
                            momentum=0.9, weight_decay=0.0001)

    # run the training and testing model for 1 epoch
    for epoch in range(1):
        train_model(distDataPrl, train_loader, optimizer,
                    training_criterion, epoch, rank)
        test_model(distDataPrl, test_loader, training_criterion)

if __name__ == "__main__":
    parseObj = argparse.ArgumentParser()
    parseObj.add_argument('--master-ip', dest='master_ip', type=str, help='master ip, 10.10.1.1')
    parseObj.add_argument('--num-nodes', dest='size', type=int, help='number of nodes, 4')
    parseObj.add_argument('--rank', dest='rank', type=int, help='rank, 0')
    args = parseObj.parse_args()

    # parse the input parameters
    initialise(args.master_ip, args.rank, args.size, vgg_model)
