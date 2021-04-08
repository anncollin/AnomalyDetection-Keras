#from keras import backend as K
import os, sys
from pathlib import Path

import numpy as np
import json
from skimage.io import imread, imsave
from skimage.color import gray2rgb
from skimage.util import random_noise
from skimage.exposure import equalize_hist
from skimage.transform import resize

from scipy import misc
from collections import OrderedDict

""" COLLIN Anne-Sophie """ 

""" -----------------------------------------------------------------------------------------
Write a dictionnary in a json file in file_path location
INPUT:
    - dict: the dictionnary to write
    - file_path: path to the folder containing the images
OUTPUT:
----------------------------------------------------------------------------------------- """ 
def write_json(dict, file_path, sort_keys=True): 
    if not os.path.exists( str(Path(file_path).parent )):
        os.makedirs( str(Path(file_path).parent))
    with open(file_path, 'w') as fp:
        json.dump(dict, fp, sort_keys=sort_keys, indent=4)

""" -----------------------------------------------------------------------------------------
Read a dictionnary in a json file in file_path location
INPUT:
    - file_path: path to the folder containing the images
OUTPUT:
    - dictionnary
----------------------------------------------------------------------------------------- """ 
def read_json(file_path): 
    with open(file_path, 'r') as fp:
        return json.load(fp, object_pairs_hook=OrderedDict)

""" -----------------------------------------------------------------------------------------
return (grayscale) images with image in file_path
INPUT:
    - file_path: path to the image
    - gray_scale: read_png in grayscale or not
OUTPUT:
    - grayscale image
----------------------------------------------------------------------------------------- """ 
def read_png(file_path, rescale_factor= 'Default'): 
    img = imread(file_path)
    if rescale_factor != 'Default':
        img = resize(img, ( int(img.shape[0]*rescale_factor), int(img.shape[1]*rescale_factor) ))
    
    if img is not None:
        if img.ndim == 2 :
            img = gray2rgb(img)
        if np.max(img) <= 1: 
            img *= 255
        return img.astype(np.uint8)
    
""" -----------------------------------------------------------------------------------------
create an array of (grayscale) images with all images contained in folder_path
INPUT:
    - folder_path: path to the folder containing the images
    (-max_size: max number of images to read)
OUTPUT:
    - array with all grayscale images
----------------------------------------------------------------------------------------- """ 
def read_folder(folder_path, max_size=None):
    list_file_path, images, nb_imgs  = [], [], 0
    for filename in os.listdir(folder_path):
        if 'png' in filename or 'jpg' in filename: 
            list_file_path.append(os.path.join(folder_path, filename))
    list_file_path.sort(key = lambda x : int( (x.split('/')[-1]).split('.')[0]))
    for path in list_file_path:
        images.append( read_png(path) )
        nb_imgs += 1
        if nb_imgs == max_size:
            break
    return np.asarray(images)

""" -----------------------------------------------------------------------------------------
Print the content of an array of image in folder_path
INPUT:
    - array: array of images
    - folder_path: path to the folder containing the images
    (- suffix: eventually add suffix to paths)
    (- suffix: eventually add prefix to paths)
OUTPUT:
----------------------------------------------------------------------------------------- """ 
def print_array(array, folder_path, suffix=None, prefix=None): 

    if array.shape[-1] == 1: 
        array = np.squeeze(array, -1)

    if not os.path.exists( folder_path ):
        os.makedirs( folder_path)
    for idx,img in enumerate(array): 
        if suffix and prefix: 
            imsave(folder_path + '/' + prefix + str(idx) + suffix + '.png', img)
        elif suffix: 
            imsave(folder_path + '/' + str(idx) + suffix + '.png', img)
        elif prefix: 
            imsave(folder_path + '/' + prefix + str(idx) + '.png', img)
        else: 
            imsave(folder_path + '/' + str(idx) + '.png', img)

""" -----------------------------------------------------------------------------------------
Print the content of an array of image in folder_path
INPUT:
    - array: array of images
    - folder_path: path to the folder containing the images
    (- overwrite: delete previous folder content)
    (- suffix_iter: second iterator to identify images)
OUTPUT:
----------------------------------------------------------------------------------------- """ 
def drop_array(array, folder_path, overwrite=False, suffix_iter=False, iter=None): 
    if not os.path.exists( folder_path ):
        os.makedirs( folder_path)
    if overwrite : 
        filesToRemove = [folder_path + '/' + x for x in os.listdir(folder_path)]
        for f in filesToRemove:
            os.remove(f) 

    if iter is None:
        shift = len(os.listdir(folder_path))
    else: 
        shift = int(iter)
    iter = 0
    for idx,img in enumerate(array): 
        if not suffix_iter :    
            imsave(folder_path + '/' + str(shift+idx) + '.png', img)
        else: 
            imsave(folder_path + '/' + str(shift) + '_' + str(iter) + '.png', img)
            iter += 1 

""" -----------------------------------------------------------------------------------------
Absolute difference of two images in uint8 format 
INPUT:
    -im1, im2: the two images
OUTPUT:
    - array with element-wise abosulte differences
----------------------------------------------------------------------------------------- """ 
def diff(img1, img2):
    return(np.uint8(np.abs(np.int16(img1)-img2)))

""" -----------------------------------------------------------------------------------------
Add two images in uint8 format (no overflow)
INPUT:
    -im1, im2: the two images
OUTPUT:
    - array with element-wise addition
----------------------------------------------------------------------------------------- """ 
def add(img1, img2):
    res = np.int16(img1)+np.int16(img2)
    res[res > 256] = 256
    return np.uint8(res) 

""" -----------------------------------------------------------------------------------------
 Convert an array of grayscale images into a 4-dimensional array as required by keras 
 INPUT : 
    - im_array: initial array of size (nb_images x row x col)
    - im_row: number of rows in images 
    - im_col: number of colums in images
OUTPUT : 
    - array of images with shape (nb_images x 1 x row x col) or (nb_images x row x col x 1) 
    depending on keras version 
----------------------------------------------------------------------------------------- """ 
def process_im(im_array, img_row, img_col):
    im_array = im_array.astype('float32') / 255.
    channels = im_array.shape[-1]
    if K.image_dim_ordering() == "th":
        new_array = np.reshape(im_array, (len(im_array), channels, img_row, img_col))
    else:
        new_array = np.reshape(im_array, (len(im_array), img_row, img_col, channels))
    return np.array(new_array)

""" -----------------------------------------------------------------------------------------
Inverse function of process_im 
 INPUT : 
    - im_array: initial array of size (nb_images x row x col)
    - im_row: number of rows in images 
    - im_col: number of colums in images
OUTPUT : 
    - array of images with shape (nb_images x 1 x row x col) or (nb_images x row x col x 1) 
    depending on keras version 
----------------------------------------------------------------------------------------- """ 
def deprocess_im(im_array):
    im_array = im_array * 255.
    im_array = im_array.astype(np.uint8) 
    if K.image_dim_ordering() == "th":
        new_array = np.squeeze(im_array, axis=1)
    else:
        new_array = np.squeeze(im_array, axis=3)
    return np.array(new_array)

""" -----------------------------------------------------------------------------------------
Generates labels needed for GAN training
INPUT : 
    - shape : shape of the output
    (- var : variance around the true value)
    (- corr : integer between 0 and 100)
OUTPUT : 
    - array of labels of shape given in argument
----------------------------------------------------------------------------------------- """ 
def label_generator(size, var=0, corr=0): 
    if var == 0: 
        label_real = np.ones(size)
        label_fake = np.zeros(size)
    else:
        label_real = np.random.randint(low=101-var*100, high=101, size=size)/100.
        label_fake = np.random.randint(low=0, high=var*100+1, size=size)/100.
    
    if corr != 0:
        chance = np.random.randint(1,100, size)
        for idx, this_chance in enumerate(chance): 
            if this_chance < corr: 
                if var == 0:
                    label_real[idx] = 0
                else:
                    label_real[idx] = np.random.randint(low=0, high=var*100+1)/100.

        chance = np.random.randint(1,100, size)
        for idx, this_chance in enumerate(chance): 
            if this_chance < corr: 
                if var == 0:
                    label_fake[idx] = 1
                else:
                    label_fake[idx] = np.random.randint(low=101-var*100, high=101)/100.

    return label_real, label_fake

""" -----------------------------------------------------------------------------------------
Return all entries of nested dictionnaries
INPUT : 
    - d : dictionnary over which to iterate
OUTPUT : 
    - dict: new "flatten" version of the dictionary
----------------------------------------------------------------------------------------- """ 
def iterate_dict(d, new_dict):
    for k, v in d.items():
        if isinstance(v, dict):
            iterate_dict(v, new_dict)
        else:
            new_dict[k] = v
    return new_dict