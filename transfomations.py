from PIL import Image, ImageEnhance
import skimage
from load_data import load_data 
import numpy as np
import matplotlib.pyplot as plt

"""
Transforms an image into new images with all 7 types of noise
"""
def all_noise(img_arr):
    #library of noise types
    noises = ['gaussian', 'localvar', 'salt', 'pepper', 's&p', 'speckle', 'poisson']

    loud= np.ones((7, 784)) #space to put transformations
    for i in range(len(noises)):
        noisier = skimage.util.random_noise(img_arr, noises[i])
        loud[i]= noisier.reshape((1, 784))
    return loud


def all_rot(img_arr, num_trans):
    all_rotations = np.ones((num_trans, 784))
    degree = np.random.uniform(-10, 10, size=num_trans)
    
    for i in range(0, num_trans):
        rotated = skimage.transform.rotate(img_arr, degree[i])
        all_rotations[i] = rotated.reshape((1, 784))
    return all_rotations


"""
Transforms a 2D numpy array into new arrays with different exposure levels.
"""
def all_expo(img_arr, num_trans, dir):
    all_exposures = np.ones((num_trans, 784))
    
    if dir == 'light':
        up = np.random.uniform(0.1, 0.9, size=num_trans)

        for i in range(num_trans):
            lighter = skimage.exposure.adjust_gamma(img_arr, up[i])
            all_exposures[i]= lighter.reshape((1, 784))

    elif dir == 'dark': 
        dwn = np.random.uniform(2, 3, size=num_trans)

        for i in range(num_trans):
            darker = np.asarray(skimage.exposure.adjust_gamma(img_arr, dwn[i]))
            all_exposures[i]= (darker.reshape((1, 784)))

    return all_exposures


"""
function just to see ONE instance to make sure picture looks as intended.
"""
def show_image(x_set, index, newImg):
    fig, axgrid = plt.subplots(1, 2, figsize=(8, 4))

    ax1 = axgrid[0]
    ax2 = axgrid[1]
    x_SS2 = x_set[index].reshape((28,28))

    #shows the original image
    ax1.imshow(x_SS2, vmin=0, vmax=1, cmap='gray')
    ax1.set_xticks([]); ax1.set_yticks([]);

    #display new image.
    ax2.imshow(newImg, vmin=0, vmax=1, cmap='gray')
    ax2.set_xticks([]); ax2.set_yticks([]);

    plt.tight_layout();
    plt.show();


def noise_exp(img_arr, num_trans, dir):
    #library of noise types
    noises = ['gaussian', 'localvar', 'salt', 'pepper', 's&p', 'speckle', 'poisson']

    noisy_exp= np.ones((num_trans, 784)) #space to put transformations
       
    for i in range(num_trans):
        sound = np.random.choice(noises)   #picks random sound 
        noisy_img = skimage.util.random_noise(img_arr, sound)
        
        if dir == 'light':
            up = np.random.uniform(0.1, 0.9, size=1)
            lighter = skimage.exposure.adjust_gamma(noisy_img, up)
            noisy_exp[i]= lighter.reshape((1, 784))
        elif dir == 'dark': 
            dwn = np.random.uniform(2, 3, size=1)
            darker = np.asarray(skimage.exposure.adjust_gamma(noisy_img, dwn))
            noisy_exp[i]= (darker.reshape((1, 784)))

    return noisy_exp


"""
    adds noise and flips ONE IMAGE and returns (z, 782) array. Where z is the
    number of noisy and flipped images we want. 
    Must give (28, 28 array)
"""
def noise_flip(img_arr, num_trans):
    flipped = (np.fliplr(img_arr))

    #library of noise types
    noises = ['gaussian', 'localvar', 'salt', 'pepper', 's&p', 'speckle', 'poisson']

    noisy_flips = np.ones((num_trans, 784)) #space to put transformations
    sound = np.random.choice(noises, size=num_trans)   #picks random sounds 

    # Randomly chooses noise and puts it into the array.
    for i in range(num_trans):
        noisy_img = skimage.util.random_noise(flipped, sound[i])
        noisy_flips[i] = (noisy_img.reshape((1, 784)))

    return noisy_flips


"""
Randomly rotates and lighten/darken ONE IMAGE and returns (z, 782) array.
where z is the number of noisy and flipped images we want. 
Must give (28, 28 array) and dir = 'light' or 'dark'
"""
def rot_expose(img_arr, num_trans, dir):
    rotated_exp = np.ones((num_trans, 784))
    degree = np.random.uniform(-10, 10, num_trans)  #choose random degrees rot.

    for i in range(num_trans):
        rotated = skimage.transform.rotate(img_arr, degree[i])
        
        if dir == 'light':
            up = np.random.uniform(0.1, 0.9, size=1)
            lighter = skimage.exposure.adjust_gamma(rotated, up)
            rotated_exp[i]= lighter.reshape((1, 784))
        elif dir == 'dark': 
            dwn = np.random.uniform(2, 3, size=1)
            darker = np.asarray(skimage.exposure.adjust_gamma(rotated, dwn))
            rotated_exp[i]= (darker.reshape((1, 784)))

    return rotated_exp