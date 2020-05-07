from keras import backend as K
import keras_contrib.backend as KC

import tensorflow as tf


""" -----------------------------------------------------------------------------------------
Mean Squarred Error
----------------------------------------------------------------------------------------- """ 
def MSE(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

""" -----------------------------------------------------------------------------------------
SSIM
----------------------------------------------------------------------------------------- """ 
def DSSIM_base(y_true, y_pred, kernel_size):
    kernel = [kernel_size,kernel_size]
    y_true = K.reshape(y_true, [-1] + list(K.int_shape(y_pred)[1:]))
    y_pred = K.reshape(y_pred, [-1] + list(K.int_shape(y_pred)[1:]))

    patches_pred = KC.extract_image_patches(y_pred, kernel, kernel, 'valid',
                                            K.image_data_format())
    patches_true = KC.extract_image_patches(y_true, kernel, kernel, 'valid',
                                            K.image_data_format())

    # Reshape to get the var in the cells
    bs, w, h, c1, c2, c3 = K.int_shape(patches_pred)
    patches_pred = K.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
    patches_true = K.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
    # Get mean
    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)
    # Get variance
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    # Get std dev
    covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

    ssim = (2 * u_true * u_pred + 0.01**2) * (2 * covar_true_pred + 0.03**2)
    denom = ((K.square(u_true)
                + K.square(u_pred)
                + 0.01**2) * (var_pred + var_true +  0.03**2))
    ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
    print('----------------------', K.mean((1.0 - ssim) / 2.0))
    return K.mean((1.0 - ssim) / 2.0)

def DSSIM(y_true, y_pred): 
    return DSSIM_base(y_true, y_pred, 11)

def DSSIM5(y_true, y_pred): 
    return DSSIM_base(y_true, y_pred, 5)

def DSSIM7(y_true, y_pred): 
    return DSSIM_base(y_true, y_pred, 7)


""" -----------------------------------------------------------------------------------------
Mean Squarred Error for network with uncertainty prediction
----------------------------------------------------------------------------------------- """ 
def uncertainty_MSE(y_true, y_pred):
    return K.mean((y_pred[:,:,:,0] - y_true[:,:,:,0])**2. * K.exp(-y_pred[:,:,:,1]) + y_pred[:,:,:,1])

""" -----------------------------------------------------------------------------------------
 PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
----------------------------------------------------------------------------------------- """ 
def PSNRLoss(y_true, y_pred):
    print('PSNR', -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.))
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

""" -----------------------------------------------------------------------------------------
PSNR for network with uncertainty prediction
----------------------------------------------------------------------------------------- """ 
def uncertainty_PSNR(y_true, y_pred): 
    return -10. * K.log(K.mean(K.square(y_pred[:,:,:,0] - y_true[:,:,:,0]))) / K.log(10.)

""" -----------------------------------------------------------------------------------------
Sum of residual
----------------------------------------------------------------------------------------- """ 
def sum_of_residual(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred))


