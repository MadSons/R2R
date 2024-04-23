import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from model import DnCNN
from train import test_PSNR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Testing...\n')

# load model
model = DnCNN(channels=1, num_of_layers=17, kernel_size=3, padding=1, features=64)
model.load_state_dict(torch.load('model.pth'))
model.to(device)
model.eval()

PATH = 'data/real_world_test'
files = os.listdir(PATH)

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
    data = os.path.join(PATH, file)
    clean = plt.imread(data)
    if len(clean.shape) > 2:
        from PIL import Image
        img = Image.open(data).convert('L')
        os.remove(data)
        img.save(data)
        clean = plt.imread(data)
    if clean[0, 0] > 1:
        clean = clean / 255
    clean = np.reshape(clean, (1, 1, clean.shape[0], clean.shape[1]))
    clean = torch.Tensor(clean).to(device)


    # Add noise
    noise = torch.FloatTensor(clean.size()).normal_(0, sigma/255).to(device)
    noise = clean + noise
    plt.imsave(f'data/real_world_output_noise/{file}', noise.cpu().data.numpy()[0, 0], cmap='gray')

    out = None

    # Monte Carlo approximation
    for _ in range(T):
        z = e * torch.FloatTensor(noise.size()).normal_(0, 1).to(device)
        star = noise + alpha * z
        with torch.no_grad():
            out_s = model(star)
        if out is None:
            out = out_s.detach()
        else:
            out += out_s.detach()  
    
    # normalize output
    out = torch.clamp(out/T, 0, 1)
    plt.imsave(f'data/real_world_output/{file}', out.cpu().data.numpy()[0, 0], cmap='gray')

    # calculate
    psnr = test_PSNR(out.data.numpy(), clean.data.numpy())
    psnr_tot += psnr

    print(f'{file} PSNR: {psnr}')

print(f"\nAverage PSNR: {psnr_tot / len(files)}")

print('End of testing')
                
