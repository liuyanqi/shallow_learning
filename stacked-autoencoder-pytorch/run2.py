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
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from models2 import CDAutoEncoder


if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 25
batch_size = 128
learning_rate = 1e-3



def plot_sample_img(img, name):
    img = img.view(1, 28, 28)
    save_image(img, './sample_{}.png'.format(name))


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def tensor_round(tensor):
    return torch.round(tensor)


transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)



ae1 = CDAutoEncoder(784, 1000, 5e-2).cuda()
ae2 = CDAutoEncoder(1000, 2000,1e-1).cuda()
ae3 = CDAutoEncoder(2000, 3000, 5e-1).cuda()


ae1.train()
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        img, _ = data
        img = img.view(img.size(0), -1)
        # noisy_img = theano_rng.binomial(size=img.shape, n=1, p=0.1, dtype=theano.config.floatX) * img
        img = Variable(img).cuda()
        # ===================forward=====================
        output = ae1(img, epoch)
    # ===================log========================
    # print("sparsity:", torch.sum(output.data > 0.0)*100 / output.data.numel())
    x_reconstructed = ae1.reconstruct(output)
    orig = to_img(img.cpu().data)
    save_image(orig, './imgs/orig_1_{}.png'.format(epoch))
    pic = to_img(x_reconstructed.cpu().data)
    save_image(pic, './imgs/reconstruction_1_{}.png'.format(epoch))


ae1.eval()
ae2.train()
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        # ===================forward=====================
        output = ae2(ae1(img, epoch), epoch)
    # print("sparsity:", torch.sum(output.data > 0.0)*100 / output.data.numel())
    x_reconstructed = ae1.reconstruct(ae2.reconstruct(output))
    orig = to_img(img.cpu().data)
    save_image(orig, './imgs/orig_2_{}.png'.format(epoch))
    pic = to_img(x_reconstructed.cpu().data)
    save_image(pic, './imgs/reconstruction_2_{}.png'.format(epoch))

ae2.eval()
ae3.train()

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        output = ae3(ae2(ae1(img, epoch), epoch), epoch)
    
    x_reconstructed = ae1.reconstruct(ae2.reconstruct(ae3.reconstruct(output)))
    orig = to_img(img.cpu().data)
    save_image(orig, './imgs/orig_3_{}.png'.format(epoch))
    pic = to_img(x_reconstructed.cpu().data)
    save_image(pic, './imgs/reconstruction_3_{}.png'.format(epoch))

    # print("sparsity:", torch.sum(output.data > 0.0)*100 / output.data.numel())

#finetune:
ae3.eval()
my_model = nn.Sequential(ae1.forward_pass, ae2.forward_pass, ae3.forward_pass, nn.Linear(3000, 10))
my_model.cuda()
torch.save(my_model.state_dict(), './CDAE_MNIST.pth')
# classifier = nn.Linear(3000, 10).cuda()
criterion = nn.CrossEntropyLoss()
# params = list(ae1.forward_pass.parameters()) + list(ae2.forward_pass.parameters()) + list(ae3.forward_pass.parameters()) + list(classifier.parameters())
optimizer = torch.optim.SGD(my_model.parameters(), lr=0.1, momentum=0.5)
for epoch in range(10):
    correct =0
    for i ,data in enumerate(dataloader):
        img, target = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        target = Variable(target).cuda()

        prediction= my_model(img)

        loss = criterion(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = prediction.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print("Linear classifier performance: {:.4f}%".format(100.0 * float(correct) / (len(dataloader)*batch_size)))


#test
correct = 0
for i, data in enumerate(testloader):
    img, target = data
    img = img.view(img.size(0), -1)
    img = Variable(img).cuda()
    target = Variable(target).cuda()

    prediction= my_model(img)

    pred = prediction.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

print("Test Linear classifier performance: {:.4f}%".format(100.0 * float(correct) / (len(testloader)*batch_size)))



    # if epoch % 10 == 0:
        # x = to_img(img.cpu().data)
        # x_hat = to_img(output.cpu().data)
        # x_noisy = to_img(noisy_img.cpu().data)
        # weights = to_img(model.encoder[0].weight.cpu().data)
        # save_image(x, './mlp_img/x_{}.png'.format(epoch))
        # save_image(x_hat, './mlp_img/x_hat_{}.png'.format(epoch))
        # save_image(x_noisy, './mlp_img/x_noisy_{}.png'.format(epoch))
        # save_image(weights, './filters/epoch_{}.png'.format(epoch))

torch.save(my_model.state_dict(), './sim_autoencoder.pth')