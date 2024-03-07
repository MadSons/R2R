import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from model import DnCNN
from train import test_PSNR

print('Testing...\n')

# load model
model = DnCNN(channels=1, num_of_layers=17, kernel_size=3, padding=1, features=64)
model.load_state_dict(torch.load('model.pth'))
model.eval()

files = os.listdir('data/test')

# set parameters
alpha = 0.5
T = 50
sigma = 25
e = sigma/255

# load and test noisy test images
psnr_tot = 0
for file in files:
    print(f'Testing {file}')

    # prepate clean data
    clean = plt.imread(os.path.join('data/test', file))
    clean = np.reshape(clean, (1, 1, clean.shape[0], clean.shape[1]))
    clean = torch.Tensor(clean)

    # Add noise
    noise = torch.FloatTensor(clean.size()).normal_(0, sigma/255)
    noise = clean + noise
    out = None

    # Monte Carlo approximation
    for _ in range(T):
        z = e * torch.FloatTensor(noise.size()).normal_(0, 1)
        star = noise + alpha * z
        with torch.no_grad():
            out_s = model(star)
        if out is None:
            out = out_s.detach()
        else:
            out += out_s.detach()  
    
    # normalize output
    out = torch.clamp(out/T, 0, 1)
    plt.imsave(f'data/test_output/{file}', out.data.numpy()[0, 0], cmap='gray')

    # calculate
    psnr = test_PSNR(out.data.numpy(), clean.data.numpy())
    psnr_tot += psnr

    print(f'{file} PSNR: {psnr}')

print(f"\nAverage PSNR: {psnr_tot / len(files)}")

print('End of testing')
                
