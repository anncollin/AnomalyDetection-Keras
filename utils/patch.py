import numpy as np
import math

""" -----------------------------------------------------------------------------------------
Create patches of size (patch_rowxpatch_col) out of an image
 INPUT : 
    - im: initial image
    - patch_row : number of rows of the patches
    - patch_col : number of colums of the patches
    - stride_row : horizontal distance between two consecutive patches
    - stride_col : vertical distance between two consecutive patches
OUTPUT : 
    - array of patches
----------------------------------------------------------------------------------------- """ 

def patchify(im, patch_row, patch_col, stride_row, stride_col): 
    im_row, im_col = im.shape[0], im.shape[1]
    if patch_row > im_row:
        raise ValueError('Patch height is higher than the height of the original image.')
    if patch_col > im_col:
        raise ValueError('Patch width is higher than the width of the original image.')
    if patch_row < 0 or patch_col < 0:
        raise ValueError('Width and height must be greater than zero.')
    if patch_row == im_row and patch_col == im_col:
        return im
    nb_row = math.ceil( (im_row-patch_row)/stride_row )
    nb_col = math.ceil( (im_col-patch_col)/stride_col )

    patches        = None
    up_left_corner = []

    for idx_row in range(0,nb_row): 
        for idx_col in range(0,nb_col): 
            if idx_row == 0 and idx_col == 0: 
                patches = np.array([im[0:patch_row,0:patch_col,:]])
            else: 
                temp =  im[idx_row*stride_row:idx_row*stride_row+patch_row, idx_col*stride_col:idx_col*stride_col+patch_col,:] 
                patches = np.vstack( (patches,  np.array([temp])) )

            up_left_corner.append((idx_row*stride_row, idx_col*stride_col)) 
            
            if idx_col == nb_col-1 and idx_col*stride_col+patch_col < im_col:
                temp =  im[idx_row*stride_row:idx_row*stride_row+patch_row, im_col-(patch_col+1):im_col-1,:]
                patches = np.vstack( (patches,  np.array([temp])) )
                up_left_corner.append((idx_row*stride_row, im_col-(patch_col+1)))
        if idx_row == nb_row-1 and idx_row*stride_row+patch_row < im_row:
            for idx_col in range(0,nb_col): 
                temp =  im[im_row-(patch_row+1):im_row-1, idx_col*stride_col:idx_col*stride_col+patch_col,:]
                patches = np.vstack( (patches,  np.array([temp])) )
                up_left_corner.append((im_row-(patch_row+1), idx_col*stride_col))
                if idx_col == nb_col-1 and idx_col*stride_col+patch_col < im_col:
                    temp =  im[im_row-(patch_row+1):im_row-1, im_col-(patch_col+1):im_col-1,:]
                    patches = np.vstack( (patches,  np.array([temp])) )
                    up_left_corner.append((im_row-(patch_row+1), im_col-(patch_col+1)))
                         
    return patches, up_left_corner


    
""" -----------------------------------------------------------------------------------------
Combines patches with coordinates given by up_left_corner into a single image
 INPUT : 
    - patches: all the patches
    - im_row : number of rows of the patches
    - im_col : number of colums of the patches
    - up_left_corner : coordinates of up left corner of each patch
OUTPUT : 
    - reconstructed image
----------------------------------------------------------------------------------------- """ 

def depatchify(patches, im_row, im_col, up_left_corner): 
    patch_row, patch_col, channel = patches.shape[1], patches.shape[2], patches.shape[3]
    result               = np.zeros((im_row, im_col, channel))
    normalization        = np.zeros((im_row, im_col, channel))
    for this_corner, this_patch in zip(up_left_corner, patches): 
        result[this_corner[0]:this_corner[0]+patch_row, this_corner[1]:this_corner[1]+patch_col,:] += this_patch
        normalization[this_corner[0]:this_corner[0]+patch_row, this_corner[1]:this_corner[1]+patch_col,:] += 1 

    normalization[ normalization == 0 ] = 1
    return (np.divide(result, normalization)).astype(np.uint8) 