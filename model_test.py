import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

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
		for n in range(k-1):
			if n == 0:
				input_feature = input_size
			else:
				input_feature = output_size
			self.block.append(nn.Sequential(nn.Conv2d(input_feature, output_size, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(output_size), nn.ReLU()))
		
		self.blocks = nn.ModuleList(self.block)
		if k == 1:
			self.bn = identity()
		else:
			self.bn = nn.BatchNorm2d(output_size)
		self.linear = nn.Linear(output_size * (in_size//avg_size) * (in_size//avg_size), 10)
	def forward(self, inputs):
		for layer in self.blocks:
			inputs = layer(inputs)

		out = F.avg_pool2d(inputs, self.avg_size)
		out = self.bn(out)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out


class ShallowBlock(nn.Module):
	def __init__(self, input_size, output_size, aux_feature=256, avg_size=16, in_size=32, bn=False, k=1):
		super(ShallowBlock, self).__init__()
		self.feature_blocks = []
		if bn:
			self.bn = nn.BatchNorm2d(output_size)
		else:
			self.bn = identity()
		self.feature=nn.Sequential(
			nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1, bias=False),
			self.bn,
			nn.ReLU()
			)

		self.auxiliary_classifier= auxiliary_classifier(output_size, aux_feature, avg_size, in_size, k)
		self.criterion = nn.CrossEntropyLoss()
		# self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

	# def forward(self, inputs, targets):
	# 	feature = self.feature(inputs)
	# 	if self.training:
	# 		output = self.auxiliary_classifier(feature)
	# 		loss = self.criterion(output, targets)
	# 		self.optimizer.zero_grad()
	# 		loss.backward()
	# 		self.optimizer.step()

	# 		return output, loss

	# 	return feature

	def forward(self, inputs, targets):
		feature = self.feature(inputs)
		return feature



class psi(nn.Module):
    def __init__(self, block_size):
        super(psi, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input, target):
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


class ShallowNetwork_k1(nn.Module):
	def __init__(self):
		super(ShallowNetwork_k1, self).__init__()
		self.block1 = ShallowBlock(3, 256, avg_size=16, in_size=32)
 		self.block2 = ShallowBlock(256, 256, 16, 32)
 		self.block3 = ShallowBlock(1024, 512, 8, 16)
 		self.block4 = ShallowBlock(2048, 1024, 4, 8)
 		self.block5 = ShallowBlock(4096, 2048, 2, 4)
 		self.down = psi(2)

 	def forward(self, inputs, target):
 		out = self.block1(inputs, target)
 		out = self.block2(out, target)
 		out = self.down(out, target)
 		out = self.block3(out, target)
 		out = self.down(out, target)
 		out = self.block4(out, target)
 		out = self.down(out, target)
 		out = self.block5(out, target)

 		return out

class ShallowNetwork_k3(nn.Module):
	def __init__(self, k):
		super(ShallowNetwork_k3, self).__init__()
		self.block1 = ShallowBlock(3, 128, aux_feature=128, avg_size=16, in_size=32, bn=True, k=k)
		self.block2 = ShallowBlock(128, 128, aux_feature=128, avg_size=16, in_size=32, bn=True, k=k)
		self.block3 = ShallowBlock(512, 256, avg_size=8, in_size=16, bn=True, k=k)
		self.block4 = ShallowBlock(256, 256, avg_size=8, in_size=16, bn=True, k=k)
		self.down = psi(2)

	def forward(self, inputs, target):
		out = self.block1(inputs, target)
		out = self.block2(out, target)
		out = self.down(out, target)
		out = self.block3(out, target)
		out = self.block4(out, target)

		return out






		