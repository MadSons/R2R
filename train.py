import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from model import DnCNN
import sys
import math

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import peak_signal_noise_ratio

def test_PSNR(img, clean):
    PSNR = 0
    for i in range(img.shape[0]):
        PSNR += peak_signal_noise_ratio(clean[i,:,:,:], img[i,:,:,:], data_range=1)
    return PSNR/img.shape[0]
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        clean = self.data[index]
        noisy = self.targets[index]

        return torch.Tensor(clean), torch.Tensor(noisy)

# Load the data
train_files = os.listdir('data/train_noise')
images = []
for file in train_files:
    img = plt.imread(os.path.join('data/train_noise', file))
    img = img[:, :, 0] # remove channels
    images.append(img)

images = np.array(images)

patches_aug_noisy = np.load('data/patches_aug_noisy.npy')
patches_aug = np.load('data/patches_aug.npy')

clean = np.reshape(patches_aug, (patches_aug.shape[0] * patches_aug.shape[1], 1,  40, 40))
noisy = np.reshape(patches_aug_noisy, (patches_aug_noisy.shape[0] * patches_aug_noisy.shape[1], 1, 40, 40))

alpha = 0.5 # D= alpha * I
epochs = 3
batch_size = 128
lr = 0.001
sigma = 25
e = sigma/255.

train_dataset = CustomDataset(clean, noisy)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# create the R2R network
model = DnCNN(channels=1, num_of_layers=17, kernel_size=3, padding=1, features=64)

loss_func = nn.MSELoss(size_average=False)
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        # get data for each batch
        clean, y = data

        # training step
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        # y calculation
        z = e * torch.FloatTensor(y.size()).normal_(0, 1)
        y_hat = y + alpha*z
        y_tilde = y - z/alpha

        # loss
        y_hat = model(y_hat)
        loss = loss_func(y_hat, y_tilde) / (y_tilde.shape[0]*2)

        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        y_hat = torch.clamp(model(y), 0, 1)


        psnr = test_PSNR(y_hat.data.numpy(), clean.data.numpy())

        print(f'Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, PSNR: {psnr:.4f}')


torch.save(model.state_dict(), 'model.pth')
print('Model saved')



