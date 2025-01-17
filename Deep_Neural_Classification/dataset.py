import torch
import os
import torchvision
from transform import build_transforms

def prepare_data():
    transform_train = build_transforms(is_train=True)
    transform_test = build_transforms(is_train=False)


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    return trainloader, testloader