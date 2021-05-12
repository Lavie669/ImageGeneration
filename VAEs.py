import torch
from matplotlib.ticker import MaxNLocator
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import skimage
from tqdm.notebook import tqdm
import random

# set random seeds
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)

# setup device cuda vs. cpu
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

# Apply transformations on images to make them become a tensor also with normalization
transform = transforms.Compose([transforms.ToTensor()])


# Convert npy dataset to PyTorch dataset for loading into DataLoader
class QUICKDRAW(data.Dataset):
    def __init__(self, path):
        self.data = np.load(path)
        self.transforms = transform

    def __getitem__(self, index):
        hdct = self.data[index, :]
        hdct = np.squeeze(hdct)  # delete the data from channel number's dimension
        ldct = 2.5 * skimage.util.random_noise(hdct * (0.4 / 255), mode='poisson', seed=None) * 255  # add poisson noise
        hdct = Image.fromarray(np.uint8(hdct))  # convert to image format
        ldct = Image.fromarray(np.uint8(ldct))  # convert to image format
        hdct = self.transforms(hdct)  # transform to tensor
        ldct = self.transforms(ldct)  # transform to tensor
        return ldct, hdct

    def __len__(self):
        return self.data.shape[0]  # total number of data


def loadData(path="./data_npy/panda.npy"):
    dataset = QUICKDRAW("./data_npy/panda.npy")
    #  Create a smaller subset of the original dataset
    dataset = data.Subset(dataset, np.arange(50000))

    #  Split the data : 70% for training, 15% for validation, and 15% for testing
    train_set, val_test = data.random_split(dataset, [35000, 15000])
    val_set, test_set = data.random_split(val_test, [7500, 7500])

    # Define data loaders to iterate over datasets
    train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True,
                                   num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)
    return train_loader, val_loader, test_loader


def multi_plot_data(results, names, title1='', title2=''):
    x = np.arange(len(results[0]))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i, y in enumerate(results):
        plt.plot(x + 1, y, '-', markersize=2, label=names[i])
    plt.legend(loc='upper right', prop={'size': 16}, numpoints=10, title=title1)
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction error')
    ax.set_title(title2)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


def compare_imgs(results, title=None, stride=1):
    count = 0
    num_plots = 2 * (num_epochs - stride + 1)
    plt.figure(figsize=(9, num_plots))
    plt.gray()
    for k in range(0, num_epochs, stride):
        imgs = results[k][1].detach().numpy()
        recon = results[k][2].detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9: break
            count += 1
            plt.subplot(num_plots, 9, count)
            item = item.reshape(-1, 28, 28)
            # item: 1, 28, 28
            plt.imshow(item[0])
            plt.axis('off')

        for i, item in enumerate(recon):
            if i >= 9: break
            count += 1
            plt.subplot(num_plots, 9, count)
            item = item.reshape(-1, 28, 28)
            # item: 1, 28, 28
            plt.imshow(item[0])
            plt.axis('off')
    if title is not None:
        plt.suptitle("Latent dimensionality: %d" % title)
    plt.show()


def plot_samples_grid(in_, n_rows=8, n_cols=8, fig_size=8, img_dim=28, title=None):
    f, axarr = plt.subplots(n_rows, n_cols, figsize=(fig_size, fig_size))

    in_pic = in_.data.cpu().view(-1, 28, 28)
    for i, ax in enumerate(axarr.flat):
        ax.imshow(in_pic[i])
        ax.axis('off')

    plt.suptitle(title)
    plt.show()


class VAE(nn.Module):
    def __init__(self, latent_size: int, vtype: str):
        super().__init__()
        self.latent_size = latent_size
        self.type = vtype

        if self.type == 'Linear':
            self.encoder = nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, latent_size * 2)
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_size, 128),
                nn.ReLU(),
                nn.Linear(128, 784),
                nn.Sigmoid(),
            )

        if self.type == 'Conv':
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2),  # 1x28x28 => 32x14x14
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # 32x14x14 => 64x7x7
                nn.GELU(),
                nn.Flatten(),  # 64x7x7 => (7x7x64)x1x1
                nn.Linear(49 * 64, latent_size * 2)  # (7x7x64)x1x1 => (latent_size * 2)x1x1
            )

            self.linear = nn.Sequential(
                nn.Linear(latent_size, 49 * 64),  # (latent_size)x1x1 => (7x7x64)x1x1
                nn.GELU()
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, output_padding=1, padding=1, stride=2),  # 64x7x7 => 32x14x14
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.GELU(),
                # 32x14x14=> 1x28x28
                nn.ConvTranspose2d(32, 1, 3, output_padding=1, padding=1, stride=2),
                nn.Sigmoid()
            )

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, x):
        if self.type == 'Linear':
            x = x.view(-1, 784)
        if self.type == 'Conv':
            x = x.view(-1, 1, 28, 28)
        mu_logvar = self.encoder(x).view(-1, 2, self.latent_size)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        return mu, logvar

    def decode(self, z):
        if self.type == 'Conv':
            z = self.linear(z)
            z = z.reshape(z.shape[0], -1, 7, 7)  # (7x7x64)x1x1 => 64x7x7
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def sample(self, n_samples):
        z = torch.randn((n_samples, self.latent_size))
        return self.decode(z)


# Reconstruction + KL divergence losses
def loss_function(x_hat, x, mu, logvar):
    # reconstruction loss
    BCE = nn.functional.binary_cross_entropy(
        x_hat.view(-1, 784), x.view(-1, 784), reduction='sum'
    )

    # regularization term using KL-divergence
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

    return BCE + KLD  # we can use a beta parameter here (BCE + beta * KLD)


def train(latent_size, vtype):
    model = VAE(latent_size, vtype).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    outputs = []
    for epoch in range(num_epochs):
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            # ===================forward=====================
            x_hat, mu, logvar = model(x)
            loss = loss_function(x_hat, x, mu, logvar)
            train_loss += loss.item()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = train_loss / len(train_loader.dataset)
        print(f'====> Epoch: {epoch + 1} Average train loss: {loss:.4f}')
        test_loss, test_x, test_x_hat = test(model)
        print(f'====> Epoch: {epoch + 1} Average test loss: {test_loss:.4f}')
        outputs.append((epoch, test_x, test_x_hat, test_loss))
    torch.save(model.state_dict(), 'vae-%s%d-model%d.pt' % (vtype, latent_size, num_epochs))
    return outputs


def test(model):
    loss = 0
    with torch.no_grad():
        model.eval()
        for x, y in test_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss += loss_function(x_hat, x, mu, logvar).item()
    loss = loss / len(test_loader.dataset)
    return loss, x, x_hat


def experiment3():
    losses = []
    sizes = []
    for latent_space in [64, 128, 256, 358]:
        loss = []
        result = train(latent_space, 'Conv')
        if latent_space >= 128:
            # compare the first image and last image during training
            compare_imgs(result, latent_space, num_epochs - 1)
        for k in range(num_epochs):
            loss.append(result[k][3])
        losses.append(loss)
        sizes.append(latent_space)
    multi_plot_data(losses, sizes, 'Latent dimensionality', 'VAEs with different latent space')


def experiment4():
    losses = []
    names = []
    for nntype in ['Linear', 'Conv']:
        loss = []
        result = train(256, nntype)
        for k in range(num_epochs):
            loss.append(result[k][3])
        losses.append(loss)
        names.append(nntype)
    multi_plot_data(losses, names, 'Network Architecture', 'VAEs with different network architecture')


if __name__ == '__main__':
    # freeze_support()
    train_loader, _, test_loader = loadData()
    num_epochs = 50
    model = VAE(128, 'Linear').to(device)
    # train(128, 'Linear')
    model.load_state_dict(torch.load('./VAEs_models/vae-Linear128-model50.pt'))
    samples = model.sample(64)
    plot_samples_grid(samples)
