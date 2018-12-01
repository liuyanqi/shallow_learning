from __future__ import print_function

import os
import time

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import save_image

import numpy as np
from model import ShallowNetwork_k1
from model import ShallowBlock
from model import psi

from utils import progress_bar


import tensorflow as tf
import time

# import model
batch_size = 128
max_epoch = 1
test=False

print('==> Preparing data..')
transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def debug_layer_parameters(filename, all_block):
	with open(filename, "a") as text_file:
		for block in all_block:
			net_cpu_dict = block.state_dict()
			for param in net_cpu_dict.keys():
				net_cpu_dict[param]=net_cpu_dict[param].cpu().numpy()
				print(param, file=text_file)
				print("parameter stored on cpu as numpy: %s  "%(net_cpu_dict[param]),file=text_file)

def freezeGradient(blocks):
	for k in range(len(blocks)):
		for p in blocks[k].parameters():
			p.requires_grad = False

def train_layer(all_block, train_block, data_loader, downsampling=False):
	freezeGradient(all_block)
	for e in range(max_epoch):
		for batch_idx, (inputs, targets) in enumerate(data_loader):
			inputs = Variable(inputs).cuda()
			targets = Variable(targets).cuda()
			for block in all_block:
				inputs = block(inputs, targets)
			if downsampling:
				down = psi(2)
				inputs = down(inputs, targets)
			output = train_block(inputs, targets)
	if downsampling:
		all_block.append(psi(2))

def test_layer(all_block, data_loader):
	total = 0
	correct = 0
	test_loss = 0
	criterion = nn.CrossEntropyLoss()
	for batch_idx, (inputs, targets) in enumerate(data_loader):
		inputs = Variable(inputs).cuda()
		targets = Variable(targets).cuda()
		for block in all_block:
			inputs = block(inputs, targets)
		outputs = block.auxiliary_classifier(inputs)
		_, predicted = torch.max(outputs.data, 1)

		loss = criterion(outputs, targets)
		test_loss += loss.data[0]
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()
		progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (test_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))


trainset_class = torchvision.datasets.CIFAR10(root='./data/', train=True, download=True,transform=transform_train)
trainloader_classifier = torch.utils.data.DataLoader(trainset_class, batch_size=128, shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR10(root='./data/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

all_block = []

start_time = time.time()
block1 = ShallowBlock(3, 256, avg_size=16, in_size=32).cuda()
block1.train()
train_layer(all_block, block1, trainloader_classifier, downsampling=False)
block1.eval()
all_block.append(block1)
print("done block1 with {:f}".format(time.time()-start_time))

if test:
	test_layer(all_block, testloader)

start_time = time.time()
block2 = ShallowBlock(1024, 512, aux_feature=512, avg_size=8, in_size=16).cuda()
block2.train()
train_layer(all_block, block2, trainloader_classifier, downsampling=True)
block2.eval()
all_block.append(block2)
print("done block2 with {:f}".format(time.time()-start_time))

if test:
	test_layer(all_block, testloader)



start_time = time.time()
block3 = ShallowBlock(512, 512, aux_feature=512, avg_size=8, in_size=16).cuda()
block3.train()
train_layer(all_block, block3, trainloader_classifier, downsampling=False)
block3.eval()
all_block.append(block3)
print("done block3 with {:f}".format(time.time()-start_time))

if test:
	test_layer(all_block, testloader)


start_time = time.time()
block4 = ShallowBlock(2048, 1024, aux_feature=1024, avg_size=4, in_size=8).cuda()
block4.train()
train_layer(all_block, block4, trainloader_classifier, downsampling=True)
block4.eval()
all_block.append(block4)
print("done block4 with {:f}".format(time.time()-start_time))

if test:
	test_layer(all_block, testloader)

start_time = time.time()
block5 = ShallowBlock(1024, 1024, aux_feature=1024, avg_size=4, in_size=8).cuda()
block5.train()
train_layer(all_block, block5, trainloader_classifier, downsampling=False)
block5.eval()
all_block.append(block5)
print("done block5 with {:f}".format(time.time()-start_time))

if test:
	test_layer(all_block, testloader)

# print(all_block)

model = nn.Sequential(*all_block)

torch.save(model.state_dict(), "shallowNet_k1.pkl")
