import os
import torch

import torchvision

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import torch.nn.functional as F

from model_shallow import CDAutoEncoder

pretrain_epoch = 0
supervised_epoch = 0

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

class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()

    def forward(self, inputs):
        return inputs

class auxiliary_classifier(nn.Module):
	def __init__(self, input_size, output_size, avg_size, in_size, k=1):
		super(auxiliary_classifier, self).__init__()
		self.avg_size = avg_size
		self.block = []
		if k == 1:
			self.bn = identity()
		else:
			self.bn = nn.BatchNorm2d(output_size)
		for n in range(k-1):
			if n == 0:
				input_feature = input_size
			else:
				input_feature = output_size
			self.block.append(nn.Sequential(nn.Conv2d(input_feature, output_size, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(output_size), nn.ReLU()))
		
		self.blocks = nn.ModuleList(self.block)
		self.linear = nn.Linear(output_size * (in_size//avg_size) * (in_size//avg_size), 10)
	def forward(self, inputs):
		for layer in self.blocks:
			inputs = layer(inputs)
		out = F.avg_pool2d(inputs, self.avg_size)
		out = self.bn(out)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out

def freezeGradient(blocks):
	for k in range(len(blocks)):
		for p in blocks[k].parameters():
			p.requires_grad = False

def unfreezeGradient(blocks, idx):
	for p in blocks[idx].parameters():
		p.requires_grad = True


def train_autoencoder(model_block, model, dataloader):
	freezeGradient(model_block)
	model.train()
	for epoch in range(pretrain_epoch):
		for i, data in enumerate(dataloader):
			img, _ = data
			inputs = Variable(img).cuda()
			for block in model_block:
				inputs = block(inputs)
			loss = model(inputs)

		print("pretrain loss: ", loss)





num_epochs = 10
batch_size = 128
learning_rate = 0.1

transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

ae1 = CDAutoEncoder(3, 256, 0.1).cuda()
ae2 = CDAutoEncoder(1024, 512, 0.1).cuda()
ae3 = CDAutoEncoder(512, 512, 0.5).cuda()
ae4 = CDAutoEncoder(2048, 1024, 0.5).cuda()
ae5 = CDAutoEncoder(1024, 1024, 0.5).cuda()

ae_list = [ae1, ae2, ae3, ae4, ae5]
#pretrain
model_block = []
for idx, ae in enumerate(ae_list):
	print("pretrain", idx, model_block)
	train_autoencoder(model_block, ae,dataloader)
	ae.eval()
	model_block.append(ae.forward_pass)
	if idx ==0 or idx ==2:
		model_block.append(psi(2))
print(model_block)
#supervised train
criterion = nn.CrossEntropyLoss()
classifier1 = auxiliary_classifier(256, 256, avg_size=16, in_size=32)
classifier2 = auxiliary_classifier(512, 512, avg_size=8, in_size=16)
classifier3 = auxiliary_classifier(512, 512, avg_size=8, in_size=16)
classifier4 = auxiliary_classifier(1024, 1024, avg_size=4, in_size=8)
classifier5 = auxiliary_classifier(1024, 1024, avg_size=4, in_size=8)

classifier_list = [classifier1, classifier2, classifier3, classifier4, classifier5]

feature_idx = 0
for idx in range(len(classifier_list)):
	print("supervised training: ", idx)
	freezeGradient(model_block)
	unfreezeGradient(model_block, feature_idx)
	if idx ==1 or idx ==3:
		feature_layer = nn.Sequential(*model_block[:feature_idx+2])
		feature_idx = feature_idx+2
	else:
		feature_layer = nn.Sequential(*model_block[:feature_idx+1])
		feature_idx = feature_idx+1

	model = torch.nn.DataParallel(nn.Sequential(feature_layer, classifier_list[idx])).cuda()
	# print(model)
	optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1, momentum=0.9, weight_decay=5e-4)
	print(optimizer.state_dict())
	for epoch in range(supervised_epoch):
		for batch_idx, (inputs, targets) in enumerate(dataloader):
			img = Variable(inputs).cuda()
			targets = Variable(targets).cuda()
			output = model.forward(img)

			loss = criterion(output, targets)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print("supervised loss: ", loss.data[0])

