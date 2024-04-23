import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from model import DnCNN

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
for file in files:
    print(f'Testing {file}')

    # load data
    data = os.path.join(PATH, file)
    noise = plt.imread(data)

    # convert to greyscale if not
    if len(noise.shape) > 2:
        from PIL import Image
        img = Image.open(data).convert('L')
        os.remove(data)
        img.save(data)
        noise = plt.imread(data)
    
    if noise[0, 0] > 1:
        noise = noise / 255

    # reformat data
    noise = np.reshape(noise, (1, 1, noise.shape[0], noise.shape[1]))
    noise = torch.Tensor(noise).to(device)
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
    out = out.cpu()
    plt.imsave(f'data/real_world_output_no_added_noise/{file}', out.data.numpy()[0, 0], cmap='gray')

print('End of testing')
                
