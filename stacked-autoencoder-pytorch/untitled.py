import os

import torch
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
import torchvision

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from model3 import VAE


if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = x.view(x.size(0), 3, 32, 32)
    return x

num_epochs = 10
batch_size = 128
learning_rate = 1e-3


transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)


model = VAE().cuda()

model.train()
for epoch in range(20):
    for i, data in enumerate(dataloader):
        img, _ = data
        # noisy_img = theano_rng.binomial(size=img.shape, n=1, p=0.1, dtype=theano.config.floatX) * img
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img, epoch)
    # ===================log========================
    # print("sparsity:", torch.sum(output.data > 0.0)*100 / output.data.numel())
    x_reconstructed = model.reconstruct(output)
    orig = to_img(img.cpu().data)
    save_image(orig, './imgs_cifar/orig_1_{}.png'.format(epoch))
    pic = to_img(x_reconstructed.cpu().data)
    save_image(pic, './imgs_cifar/reconstruction_1_{}.png'.format(epoch))



##fine tuning
model.eval()
classifier = nn.Sequential(nn.Linear(8*8*200, 324), nn.ReLU(), nn.Linear(324, 10), nn.Softmax())
criterion = nn.CrossEntropyLoss()
params = list(VAE.encoder.parameters()) + list(classifier.parameters())
optimizer = torch.optim.SGD(params, lr=0.1)
for epoch in range(30):
    for i, data in enumerate(dataloader):
        img, target = data
        img = Variable(img).cuda()
        target = Variable(target).cuda()
        feature = VAE(img)
        feature = feature.view(feature.size(0), -1)
        prediction = classifier(feature)

        loss = criterion(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = prediction.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()




    # if epoch % 10 == 0:
        # x = to_img(img.cpu().data)
        # x_hat = to_img(output.cpu().data)
        # x_noisy = to_img(noisy_img.cpu().data)
        # weights = to_img(model.encoder[0].weight.cpu().data)
        # save_image(x, './mlp_img/x_{}.png'.format(epoch))
        # save_image(x_hat, './mlp_img/x_hat_{}.png'.format(epoch))
        # save_image(x_noisy, './mlp_img/x_noisy_{}.png'.format(epoch))
        # save_image(weights, './filters/epoch_{}.png'.format(epoch))

# torch.save(model.state_dict(), './sim_autoencoder.pth')