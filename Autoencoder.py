import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skimage
# Progress bar
from tqdm.notebook import tqdm
# PyTorch
import torch
import torch.nn as nn
import torch.utils.data as data
from matplotlib.ticker import MaxNLocator
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F


# Apply transformations on images to make them become a tensor also with normalization
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])


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

        for i, item in enumerate(recon):
            if i >= 9: break
            count += 1
            plt.subplot(num_plots, 9, count)
            item = item.reshape(-1, 28, 28)
            # item: 1, 28, 28
            plt.imshow(item[0])
    if title is not None:
        plt.suptitle("Latent dimensionality: %d" % title)
    plt.show()


def display_images(in_, out, label, n=1, count=False):
    title_set = False
    for N in range(n):
        if in_ is not None:
            in_pic = in_.data.cpu().view(-1, 28, 28)
            plt.figure(figsize=(18, 4))
            if not title_set:
                plt.suptitle(label, color='w', fontsize=20)
                title_set = True
            for i in range(4):
                plt.subplot(1, 4, i + 1)
                plt.imshow(in_pic[i + 4 * N])
                plt.axis('off')
        if out is not None:
            out_pic = out.data.cpu().view(-1, 28, 28)
            plt.figure(figsize=(18, 6))
            if not title_set:
                plt.suptitle(label, color='w', fontsize=20)
                title_set = True
            for i in range(4):
                plt.subplot(1, 4, i + 1)
                plt.imshow(out_pic[i + 4 * N])
                plt.axis('off')
                if count: plt.title(str(4 * N + i), color='w')
    plt.show()


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int, atype: int, act_fn=nn.GELU):
        super().__init__()
        input_channels = 1
        c_hid = 32
        self.latent_dim = latent_dim
        self.type = atype

        # Autoencoder 1
        if self.type == 1:
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 1x28x28 => 32x14x14
                act_fn(),
                nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 32x14x14 => 64x7x7
                act_fn(),
                nn.Flatten(),  # 64x7x7 => (7x7x64)x1x1
                nn.Linear(2 * 49 * c_hid, latent_dim)  # (7x7x64)x1x1 => (latent_dim)x1x1
            )

            self.linear = nn.Sequential(
                nn.Linear(latent_dim, 2 * 49 * c_hid),  # (latent_dim)x1x1 => (7x7x64)x1x1
                act_fn()
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(2 * c_hid, c_hid, 3, output_padding=1, padding=1, stride=2),  # 64x7x7 => 32x14x14
                act_fn(),
                nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                act_fn(),
                # 32x14x14=> 1x28x28
                nn.ConvTranspose2d(c_hid, input_channels, 3, output_padding=1, padding=1, stride=2),
                nn.Tanh()  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
            )

        # Autoencoder 2
        if self.type == 2:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2),  # 1x28x28 => 32x14x14
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # 32x14x14 => 64x7x7
                nn.ReLU(),
                nn.Conv2d(64, latent_dim, kernel_size=7))  # 64x7x7 => (latent_dim)x1x1

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, 64, kernel_size=7),  # (latent_dim)x1x1 => 64x7x7
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),  # 64x7x7 => 32x14x14
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1, stride=2, output_padding=1),  # 32x14x14 => 1x28x28
                nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        if self.type == 1:
            x = self.linear(x)
            x = x.reshape(x.shape[0], -1, 7, 7)  # (7x7x64)x1x1 => 64x7x7
        x_hat = self.decoder(x)
        return x_hat

    def sample(self, n_samples):
        z = torch.randn((n_samples, self.latent_dim)).cpu()
        if self.type == 1:
            z = self.linear(z)
            z = z.reshape(z.shape[0], -1, 7, 7)  # (7x7x64)x1x1 => 64x7x7
        return self.decoder(z)


def train(latent_dim, atype):
    outputs = []
    model = Autoencoder(latent_dim, atype).cpu()
    model.train()
    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    for epoch in range(num_epochs):
        for trains in train_loader:
            img, _ = trains
            img = torch.reshape(img, (256, 1, 28, 28))
            img = Variable(img).cpu()
            # print(np.shape(img))
            # ===================forward=====================
            output = model(img)
            loss = distance(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        outputs.append((epoch, img, output, loss.item()))
    torch.save(model.state_dict(), 'ae-model.pt')
    return outputs


def plot_samples_grid(in_, n_rows=8, n_cols=8, fig_size=8, img_dim=28, title=None):
    f, axarr = plt.subplots(n_rows, n_cols, figsize=(fig_size, fig_size))

    in_pic = in_.data.cpu().view(-1, 28, 28)
    for i, ax in enumerate(axarr.flat):
        ax.imshow(in_pic[i])
        ax.axis('off')

    plt.suptitle(title)
    plt.show()


def experiment1():
    losses = []
    dims = []
    for latent_space in [64, 128, 256, 384]:
        loss = []
        result = train(latent_space, 1)
        compare_imgs(result, stride=4)
        for k in range(num_epochs):
            loss.append(result[k][3])
        losses.append(loss)
        dims.append(latent_space)
    multi_plot_data(losses, dims, 'Latent dimensionality', 'Autoencoders with different latent space')


def experiment2():
    losses = []
    names = []
    for actF in [1, 2]:
        loss = []
        result = train(256, actF)
        for k in range(num_epochs):
            loss.append(result[k][3])
        losses.append(loss)
        names.append(actF)
    multi_plot_data(losses, names, 'Network Architecture', 'Autoencoders with different network architecture')


def embed_imgs(model, data_loader):
    # Encode all images in the data_laoder using model, and return both images and encodings
    img_list, embed_list = [], []
    model.eval()
    for imgs, _ in tqdm(data_loader, desc="Encoding images", leave=False):
        imgs = torch.reshape(imgs, (256, 1, 28, 28))
        with torch.no_grad():
            z = model.encoder(imgs.to(model.device))
        img_list.append(imgs)
        embed_list.append(z)
    return torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0)


if __name__ == '__main__':
    num_epochs = 5
    train_loader, val_loader, test_loader = loadData()
    model = Autoencoder(256, 1).cpu()
    train_img_embeds = embed_imgs(model, train_loader)
    test_img_embeds = embed_imgs(model, test_loader)
    # train(256, 1)
    # # loading a trained module
    # model.load_state_dict(torch.load('ae-model.pt'))
    # samples = model.sample(64)
    # plot_samples_grid(samples)

