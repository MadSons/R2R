import os
import numpy as np
import random
import matplotlib.pyplot as plt

# Add noise to image
def noise(sigma, img):
    return img + np.random.normal(0, sigma / 255., img.shape)

# Create patches from image
def create_patches(img, patch_size, stride):
    patches = []
    for i in range(0, img.shape[0]-patch_size, stride):
        for j in range(0, img.shape[1]-patch_size, stride):
            patch = img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    patches = np.array(patches)
    return patches

# Data augmentation (flips and rotates)
def augment(img):
    val = random.randint(1, 8)
    if val == 1:
        pass
    elif val == 2:
        img =  np.flipud(img)
    elif val == 3:
        img = np.rot90(img)
    elif val == 4:
        img =  np.flipud(np.rot90(img))
    elif val == 5:
        img =  np.rot90(img, k=2)
    elif val == 6:
        img =  np.flipud(np.rot90(img, k=2))
    elif val == 7:
        img = np.rot90(img, k=3)
    elif val == 8:
        img = np.flipud(np.rot90(img, k=3))
    
    return img

# Script to prepare data
if __name__ == '__main__':
    train_files = os.listdir('data/train')

    sigma = 25
    patch = 40
    stride = 5 # paper stride = 1

    all_patches_aug_noisy = []
    all_patches_aug = []
    for file in train_files:
        img = plt.imread(os.path.join('data/train', file))

        # Add noise to image
        img_noise = noise(sigma, img)
        plt.imsave(os.path.join('data/train_noise', file), img_noise, cmap='gray')

        # Create patches
        patches_aug_noisy = [augment(patch) for patch in create_patches(img_noise, patch, stride)]
        patches_aug = [augment(patch) for patch in create_patches(img, patch, stride)]

        # Convert to numpy array
        patches_aug_noisy = np.array(patches_aug_noisy)
        patches_aug = np.array(patches_aug)

        # Append patches to list
        all_patches_aug_noisy.append(patches_aug_noisy)
        all_patches_aug.append(patches_aug)

    # Convert to numpy array
    all_patches_aug_noisy = np.array(all_patches_aug_noisy)
    all_patches_aug = np.array(all_patches_aug)

    # Save data
    np.save('data/patches_aug_noisy.npy', all_patches_aug_noisy)
    np.save('data/patches_aug.npy', all_patches_aug)

    print('Data prepared')

