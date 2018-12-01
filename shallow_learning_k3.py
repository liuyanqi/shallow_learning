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
from model import ShallowNetwork_k3
from model import ShallowBlock
from model import psi

from utils import progress_bar


import tensorflow as tf
import time
# import model
batch_size = 128
max_epoch = 50
test = True
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

class psi(nn.Module):
    def __init__(self, block_size):
        super(psi, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()


def freezeGradient(blocks):
	for k in range(len(blocks)):
		for p in blocks[k].parameters():
			p.requires_grad = False

def train_layer(all_block, train_block, data_loader, test_loader, downsampling=False):
	freezeGradient(all_block)
	if downsampling:
		all_block.append(psi(2))
	for e in range(max_epoch):
		total = 0
		test_loss = 0
		correct = 0
		print("epoch: %d" % e)
		for batch_idx, (inputs, targets) in enumerate(data_loader):
			inputs = Variable(inputs).cuda()
			targets = Variable(targets).cuda()
			for block in all_block:
				inputs = block(inputs)
			# if downsampling:
			# 	down = psi(2)
			# 	inputs = down(inputs)
			output, loss = train_block(inputs, targets)
			_, predicted = torch.max(output.data, 1)

			test_loss += loss.data[0]
			total += targets.size(0)
			correct += predicted.eq(targets.data).cpu().sum()
			progress_bar(batch_idx, len(data_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
				% (test_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))


		if test:
			test_layer(all_block, train_block, testloader)
	all_block.append(train_block.feature)



def test_layer(all_block, train_block, data_loader):
	total = 0
	correct = 0
	test_loss = 0
	criterion = nn.CrossEntropyLoss()
	for batch_idx, (inputs, targets) in enumerate(data_loader):
		inputs = Variable(inputs, volatile=True).cuda()
		targets = Variable(targets).cuda()
		for block in all_block:
			inputs = block(inputs)
		outputs = train_block.feature(inputs)
		outputs = train_block.auxiliary_classifier(outputs)
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

model = ShallowNetwork_k3(k=3).cuda()

block1 = model.block1.cuda()
start_time = time.time()
block1.train()
all_block = []
train_layer(all_block, block1, trainloader_classifier, testloader)
print(all_block)

print("done block1 with {:f}".format(time.time()-start_time))


start_time = time.time()
block2 = model.block2.cuda()
block2.train()
train_layer(all_block, block2, trainloader_classifier, testloader)
print(all_block)

print("done block2 with {:f}".format(time.time()-start_time))



start_time = time.time()
block3 = model.block3.cuda()
block3.train()
train_layer(all_block, block3, trainloader_classifier, testloader,downsampling=True)
print(all_block)

print("done block3 with {:f}".format(time.time()-start_time))



start_time = time.time()
block4 = model.block4.cuda()
block4.train()
train_layer(all_block, block4, trainloader_classifier,testloader)
print(all_block)

print("done block4 with {:f}".format(time.time()-start_time))

# # print(all_block)
# model = nn.Sequential(*all_block)

# torch.save(model.state_dict(), "shallowNet_k3.pkl")
