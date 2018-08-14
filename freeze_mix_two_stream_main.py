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
from model.weight_fusion import *
from model.resnet import *
from run import *

import pdb

parser = argparse.ArgumentParser(description='Coview Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--use_cuda', action='store_true', help='resume from checkpoint', default=True)
filename = 'coview_resnet_50_pretrained_imagenet_SGD_mix_two_stream'
parser.add_argument('--filename', default=filename)
parser.add_argument('--debug','-d',action='store_true', help ='pdb enable')
args = parser.parse_args()


def main():
    use_cuda = torch.cuda.is_available()
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    end_epoch = 120
    #lr_step = [5, 10, 15, 20]
    lr_step = [60,90]
    
    t_batch_size=150 #limit 100

    if args.debug:
        pdb.set_trace()

    # Data
    print('==> Preparing data..')
    #prepare data

    transform_train = transforms.Compose([
        RepeatTensor(3, 1, 1),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

    ])
    
    transform_test = transforms.Compose([
        RepeatTensor(3, 1, 1),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

    ])

    num_workers = 4

    coview_train = DatasetFolder(root='../coview_data/train_ver1/', loader=numpy_loader, extensions='npz', transform=transform_train)
    trainloader = DataLoader(coview_train, batch_size=t_batch_size, shuffle=True, num_workers= num_workers)

    coview_val = DatasetFolder(root='../coview_data/val_ver1/', loader=numpy_loader, extensions='npz', transform=transform_test)
    valloader = DataLoader(coview_val, batch_size=t_batch_size, shuffle=False, num_workers= num_workers)

    #coview_test = DatasetFolder(root='../coview_data/test_ver1/', loader=numpy_loader, extensions='npz', transform=transform_test)
    #testloader = DataLoader(coview_test, batch_size=t_batch_size, shuffle=False, num_workers= num_workers)


    # Model
    # Load checkpoint.
    pre_trained_audio_net = resnet50(conv1_channel=3)
    pre_trained_rgb_net   = resnet50(conv1_channel=3)
    pre_trained_audio_net = torch.nn.DataParallel(pre_trained_audio_net, device_ids=[0,1])
    pre_trained_rgb_net   = torch.nn.DataParallel(pre_trained_rgb_net, device_ids=[0,1])
   
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/coview_resnet_50_pretrained_imagenet_SGD_audio_.pt')          #ver_1
    #checkpoint = torch.load('./checkpoint/coview_ver2_resnet_50_pretrained_imagenet_SGD_audio_.pt')      #ver_2
    pre_trained_audio_net.load_state_dict(checkpoint['net'])
    pre_trained_audio_best_acc = checkpoint['acc']
    pre_trained_autio_start_epoch = checkpoint['epoch']
    
    checkpoint = torch.load('./checkpoint/coview_resnet_50_pretrained_imagenet_SGD_rgb_.pt')            #ver_1
    #checkpoint = torch.load('./checkpoint/coview_ver2_resnet_50_pretrained_imagenet_SGD_audio_.pt')      #ver_2
    pre_trained_rgb_net.load_state_dict(checkpoint['net'])
    pre_trained_rgb_best_acc = checkpoint['acc']
    pre_trained_rgb_start_epoch = checkpoint['epoch']

    audio_net = weight_fusion(29, 285, pre_trained_audio_net)
    rgb_net   = weight_fusion(29, 285, pre_trained_rgb_net)
    
    if use_cuda:
        audio_net.cuda()
        rgb_net.cuda()
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(audio_net.parameters())+list(rgb_net.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=lr_step, gamma=0.1)
    
    start_time = time.time()
    for epoch in range(start_epoch,end_epoch):
        train_two_stream(epoch, audio_net, rgb_net, optimizer, trainloader, criterion, 1, args)
        train_two_stream(epoch, audio_net, rgb_net, optimizer, trainloader, criterion, 2, args)
        scheduler.step()
        test_two_stream(epoch, audio_net, rgb_net, optimizer, valloader, criterion, args)

    time_cost = time.time() - start_time
    print("total_time_cost",int(time_cost/60), "min")

if __name__ == '__main__':
    main()
