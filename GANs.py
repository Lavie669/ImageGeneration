from __future__ import print_function
import argparse
import os
import random
from PIL import Image
import skimage
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=4, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='./GANs_samples2/netG_epoch_0_256.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='./GANs_samples2/netD_epoch_0_256.pth', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./GANs_samples2_continue', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

opt = parser.parse_args()
# print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
# print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Apply transformations on images to make them become a tensor also with normalization
transform = transforms.Compose([transforms.Resize(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])


# Convert npy dataset to PyTorch dataset for loading into DataLoader
class QUICKDRAW(torch.utils.data.Dataset):
    def __init__(self, path, reshape=False):
        if reshape is True:
            self.data = np.load(path).reshape((-1, 28, 28))
        self.data = np.load(path)
        self.transforms = transform

    def __getitem__(self, index):
        hdct = self.data[index, :, :]
        hdct = np.squeeze(hdct)
        ldct = 2.5 * skimage.util.random_noise(hdct * (0.4 / 255), mode='poisson', seed=None) * 255  # add poisson noise
        hdct = Image.fromarray(np.uint8(hdct))  # convert to image format
        ldct = Image.fromarray(np.uint8(ldct))  # convert to image format
        hdct = self.transforms(hdct)  # transform to tensor
        ldct = self.transforms(ldct)  # transform to tensor
        return ldct, hdct

    def __len__(self):
        return self.data.shape[0]  # total number of data


# Define data loaders to iterate over datasets
dataset = QUICKDRAW("./data_npy/panda.npy")
dataset = torch.utils.data.Subset(dataset, np.arange(32000))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
# the number of channels
nc = 1

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


def multi_plot_data(results, names, ylable='Reconstruction error', title1='', title2=''):
    x = np.arange(len(results[0]))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i, y in enumerate(results):
        plt.plot(x + 1, y, '-', markersize=2, label=names[i])
    plt.legend(loc='upper right', prop={'size': 16}, numpoints=10, title=title1)
    plt.xlabel('Epochs')
    plt.ylabel(ylable)
    ax.set_title(title2)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)
        return output


# if opt.netG != '':
#     netG.load_state_dict(torch.load(opt.netG))
# print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)

        return output.view(-1, 1).squeeze(1)


# if opt.netD != '':
#     netD.load_state_dict(torch.load(opt.netD))
# print(netD)

# use Binary Cross Entropy (BCE) between target and output
criterion = nn.BCELoss()

# for single run
if opt.dry_run:
    opt.niter = 1


def train():
    D_error = []
    G_error = []
    D_e = 0
    G_e = 0

    # set up noise and create labels for real data and generated (fake) data
    fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # training
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label,
                               dtype=real_cpu.dtype, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                                  '%s/real_samples.png' % opt.outf,
                                  normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                                  '%s/fake_samples_epoch_%03d_%d.png' % (opt.outf, epoch, nz),
                                  normalize=True)

            D_e += errD.item()
            G_e += errG.item()

            if i == len(dataloader) - 1:
                D_error.append(D_e / len(dataloader))
                G_error.append(G_e / len(dataloader))

            if opt.dry_run:
                break
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d_%d.pth' % (opt.outf, epoch, nz))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d_%d.pth' % (opt.outf, epoch, nz))

    return D_error, G_error


# training and save trained generator and discriminator
if __name__ == '__main__':
    D_errors = []
    G_errors = []
    names = []
    for nz_size in [256]:
        nz = nz_size
        netD = Discriminator(ngpu).to(device)
        netD.apply(weights_init)
        if opt.netD != '':
            netD.load_state_dict(torch.load(opt.netD))
        netG = Generator(ngpu).to(device)
        netG.apply(weights_init)
        if opt.netG != '':
            netG.load_state_dict(torch.load(opt.netG))
        D_error, G_error = train()
        D_errors.append(D_error)
        G_errors.append(G_error)
        names.append(nz_size)

    # plot how the error change over epochs at Discriminator and Generator
    multi_plot_data(D_errors, names, 'Discriminator error', 'The size of input noise', 'Discriminator')

    multi_plot_data(G_errors, names, 'Generator error', 'The size of input noise', 'Generator')
