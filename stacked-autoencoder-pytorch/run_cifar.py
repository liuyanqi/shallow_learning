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

from model import CDAutoEncoder
from model import MaxPool

pretrain_epoch = 0
finetune_epoch = 10

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = x.view(x.size(0), 3, 32, 32)
    return x

batch_size = 128
learning_rate = 1e-3


transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)



# ae1 = CDAutoEncoder(3, 32, kernel=2, 2, 0.1).cuda()
# ae2 = CDAutoEncoder(32, 64, kernel=2, 2, 0.5).cuda()
# ae3 = CDAutoEncoder(64, 128, kernel=2, 2, 1).cuda()
# ae4 = CDAutoEncoder(128, 512, kernel=2, 2, 1).cuda()

ae1 = CDAutoEncoder(3, 100, 5, 1, 2,  0.1).cuda()
pool = torch.nn.MaxPool2d(kernel_size=2)
ae2 = CDAutoEncoder(100, 150, 5, 1, 2, 0.1).cuda()
ae3 = CDAutoEncoder(150, 200, 3, 1, 1,  0.1).cuda()
classifier = nn.Sequential(
    torch.nn.Linear(8*8*200, 300),
    torch.nn.ReLU(),
    torch.nn.Linear(300, 10)
).cuda()



ae1.train()
for epoch in range(pretrain_epoch):
    for i, data in enumerate(dataloader):
        img, _ = data
        # noisy_img = theano_rng.binomial(size=img.shape, n=1, p=0.1, dtype=theano.config.floatX) * img
        img = Variable(img).cuda()
        # ===================forward=====================
        output = ae1(img, epoch)
    # ===================log========================
    # print("sparsity:", torch.sum(output.data > 0.0)*100 / output.data.numel())
    x_reconstructed = ae1.reconstruct(output)
    orig = to_img(img.cpu().data)
    save_image(orig, './imgs_cifar/orig_1_{}.png'.format(epoch))
    pic = to_img(x_reconstructed.cpu().data)
    save_image(pic, './imgs_cifar/reconstruction_1_{}.png'.format(epoch))


ae1.eval()
ae2.train()
for epoch in range(pretrain_epoch):
    for i, data in enumerate(dataloader):
        img, _ = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = ae2(pool(ae1(img, epoch)), epoch)
    # print("sparsity:", torch.sum(output.data > 0.0)*100 / output.data.numel())
    # x_reconstructed = ae1.reconstruct(ae2.reconstruct(output))
    # orig = to_img(img.cpu().data)
    # save_image(orig, './imgs_cifar/orig_2_{}.png'.format(epoch))
    # pic = to_img(x_reconstructed.cpu().data)
    # save_image(pic, './imgs_cifar/reconstruction_2_{}.png'.format(epoch))

ae2.eval()
ae3.train()

for epoch in range(pretrain_epoch):
    for i, data in enumerate(dataloader):
        img, _ = data
        img = Variable(img).cuda()
        output = ae3(pool(ae2(pool(ae1(img, epoch)), epoch)), epoch)
    
    # x_reconstructed = ae1.reconstruct(ae2.reconstruct(ae3.reconstruct(output)))
    # orig = to_img(img.cpu().data)
    # save_image(orig, './imgs_cifar/orig_3_{}.png'.format(epoch))
    # pic = to_img(x_reconstructed.cpu().data)
    # save_image(pic, './imgs_cifar/reconstruction_3_{}.png'.format(epoch))

    # print("sparsity:", torch.sum(output.data > 0.0)*100 / output.data.numel())

ae3.eval()
# ae4.train()

# for epoch in range(pretrain_epoch):
#     for i, data in enumerate(dataloader):
#         img, _ = data
#         img = Variable(img).cuda()
#         output = ae4(ae3(ae2(ae1(img, epoch), epoch), epoch), epoch)
    
#     x_reconstructed = ae1.reconstruct(ae2.reconstruct(ae3.reconstruct(ae4.reconstruct(output))))
#     orig = to_img(img.cpu().data)
#     save_image(orig, './imgs_cifar/orig_4_{}.png'.format(epoch))
#     pic = to_img(x_reconstructed.cpu().data)
#     save_image(pic, './imgs_cifar/reconstruction_4_{}.png'.format(epoch))


    # print("sparsity:", torch.sum(output.data > 0.0)*100 / output.data.numel())


#finetune:
# ae4.eval()
my_model = nn.Sequential(ae1.forward_pass, pool, ae2.forward_pass, pool, ae3.forward_pass)
# my_model.cuda()
torch.save(my_model.state_dict(), './CDAE_4_1.pth')

# my_model.load_state_dict(torch.load('./CDAE_4.pth'))
my_model.cuda()

criterion = nn.CrossEntropyLoss()
params = list(my_model.parameters()) + list(classifier.parameters())
optimizer = torch.optim.SGD(params, lr=0.1)
relu = torch.nn.ReLU()
for epoch in range(finetune_epoch):
    correct =0
    for i ,data in enumerate(dataloader):
        img, target = data
        img = Variable(img).cuda()
        target = Variable(target).cuda()

        output= my_model(img)
        output = output.view(output.size(0),-1)

        prediction = classifier(output)

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
    img = Variable(img).cuda()
    target = Variable(target).cuda()

    output= my_model(img)
    output = output.view(output.size(0), -1)
    # print(output.size())
    prediction = classifier(output)

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

# torch.save(model.state_dict(), './sim_autoencoder.pth')