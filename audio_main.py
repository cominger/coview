'''Train audio with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from dataset import DatasetFolder, numpy_loader, RepeatTensor
from model.resnet import *
from run import *

import pdb

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--use_cuda', action='store_true', help='resume from checkpoint', default=True)
filename = 'coview_ver1_resnet_50_with_transform_pretrained_imagenet_SGD_audio'
parser.add_argument('--filename', default=filename)
parser.add_argument('--debug','-d',action='store_true', help ='pdb enable')
args = parser.parse_args()


def main():
    use_cuda = torch.cuda.is_available()
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    end_epoch = 25
    lr_step = [5, 10, 15, 20]
    # lr_step = [60,90]
    
    t_batch_size=50 #limit 100

    if args.debug:
        pdb.set_trace()

    # Data
    print('==> Preparing data..')
    #prepare data

    transform_train = transforms.Compose([
        RepeatTensor(3, 1, 1),
        transforms.ToPILImage(),
        # transforms.RandomResizedCrop(224),
        transforms.Resize((224, 224)),
        #transforms.Resize((256,256)),
        #transforms.RandomCrop(224),
        transforms.ToTensor(),

    ])
    
    transform_test = transforms.Compose([
        RepeatTensor(3, 1, 1),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

    ])


    #coview_val = DatasetFolder(root='./val', loader=numpy_loader, extensions='npz', transform=transform_train)
    #dataloader = DataLoader(coview_val, batch_size=5, shuffle=True)

    coview_train = DatasetFolder(root='../coview_data/train_ver1/', loader=numpy_loader, extensions='npz', transform=transform_train)
    trainloader = DataLoader(coview_train, batch_size=t_batch_size, shuffle=True, num_workers=8)

    coview_val = DatasetFolder(root='../coview_data/val_ver1/', loader=numpy_loader, extensions='npz', transform=transform_test)
    valloader = DataLoader(coview_val, batch_size=t_batch_size, shuffle=False, num_workers=8)


    # Model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        print('==> Building model..')
        # net = VGG('VGG19')
        net = resnet50(conv1_channel=3, pretrained = True)
        # net = ResNet18_DNC()
        # net = PreActResNet18()
        # net = GoogLeNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()

    if use_cuda:
        net.cuda(1)
        # net = torch.nn.DataParallel(net, device_ids=[1])
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MultiLabelMarginLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=lr_step, gamma=0.1)
    
    start_time = time.time()
    for epoch in range(start_epoch, end_epoch):
        # pdb.set_trace()
        #test(epoch, net,valloader,criterion, args)
        train_audio(epoch, net, optimizer, trainloader, criterion, args)
        scheduler.step()
        test_audio(epoch, net,valloader,criterion, args)
    
    time_cost = time.time() - start_time
    print("total_time_cost",int(time_cost/60), "min")

if __name__ == '__main__':
    main()
