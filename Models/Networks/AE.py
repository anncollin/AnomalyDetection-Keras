from utils.import_lib import *
from importlib import import_module

import tensorflow as tf
import numpy as np
import math
from Models.Losses import *

''' COLLIN Anne-Sophie '''

""" -----------------------------------------------------------------------------------------
Constructor for a Convolutional Auto-encoders 
- with increasing number of features with depth (=butterfly)
- with Batch normalization 
- with strided convolutions (Encoder) and convolutions (Decoder)
- without regularizer (L2-Norm or Dropout)
----------------------------------------------------------------------------------------- """ 

def build_model(net, init):
    def resize_layer(conv, deconv): 
        if tf.keras.backend.image_data_format() == "channels_first":
            if deconv.get_shape().as_list()[2] > conv.get_shape().as_list()[2]:
                deconv = Cropping2D(cropping=((0, 1), (0, 0)))(deconv)
            if deconv.get_shape().as_list()[3] > conv.get_shape().as_list()[3]:
                deconv = Cropping2D(cropping=((0, 0), (0, 1)))(deconv)
        else:
            if deconv.get_shape().as_list()[1] > conv.get_shape().as_list()[1]:
                deconv = Cropping2D(cropping=((0, 1), (0, 0)))(deconv)
            if deconv.get_shape().as_list()[2] > conv.get_shape().as_list()[2]:
                deconv = Cropping2D(cropping=((0, 0), (0, 1)))(deconv)
        return deconv

    def get_name(type, prev_layer=None): 
        if 'CONV' in  type : 
            get_name.iter += 1
            name = str(get_name.iter).zfill(3) + '-layer_' + type + '_' + str(int(prev_layer.get_shape().as_list()[1]/2)) + 'x' + str(int(prev_layer.get_shape().as_list()[2]/2))
        else: 
            name = str(get_name.iter).zfill(3) + '-layer_' + type
        return name
    get_name.iter = 0

    drop = [0, 0, 0.0, 0.0, 0.1, 0.2]

    # ------------
    # Encoder part
    # ------------
    conv_layers = [None]*net.Depth
    for idx in range(net.Depth): 
        conv_layers[idx] = Conv2D(net.Nb_feature_maps*(2**idx), (net.Filter_size, net.Filter_size), strides=(2, 2), padding='same', 
            name=get_name('strided_CONV', init if idx==0 else conv_layers[idx-1]))(init if idx==0 else conv_layers[idx-1])
        conv_layers[idx] = BatchNormalization(name=get_name('BN'))(conv_layers[idx])
        conv_layers[idx] = LeakyReLU(name=get_name('LeakyReLU'))(conv_layers[idx])
        conv_layers[idx] = Dropout(net.Dropout_rate, name=get_name('Dropout'), seed=2019)(conv_layers[idx])
        if net.MCDropout:
            conv_layers[idx] = Dropout(drop[idx], name=get_name('MCDropout_encoder'))(conv_layers[idx], training=True)

    # ------------
    # Decoder part
    # ------------
    deconv_layers = [None]*net.Depth
    for idx in range(net.Depth): 
        if idx == 0: 
            deconv_layers[idx] = UpSampling2D(name=get_name('UpSampling'))(conv_layers[-1])
        else :
            deconv_layers[idx] = Conv2D(net.Nb_feature_maps*(2**(net.Depth-1-idx)), (net.Filter_size, net.Filter_size), padding='same', 
                name=get_name('deCONV', conv_layers[-1] if idx==0 else deconv_layers[idx-1]))(conv_layers[-1] if idx==0 else deconv_layers[idx-1])
            deconv_layers[idx] = BatchNormalization(name=get_name('BN'))(deconv_layers[idx])  
            deconv_layers[idx] = LeakyReLU(name=get_name('LeakyReLU'))(deconv_layers[idx])
            deconv_layers[idx] = resize_layer(conv_layers[net.Depth-1-idx], deconv_layers[idx])
            if net.MCDropout:
                deconv_layers[idx] = Dropout(drop[net.Depth-1-idx], name=get_name('MCDropout_decoder'))(deconv_layers[idx], training=True)

            deconv_layers[idx] = UpSampling2D(name=get_name('UpSampling'))(deconv_layers[idx])
            deconv_layers[idx] = Dropout(net.Dropout_rate, name=get_name('Dropout'), seed=2019)(deconv_layers[idx])

    
    out = Conv2D(1 if net.Grayscale else 3, (net.Filter_size, net.Filter_size), padding='same', activation='sigmoid', name=get_name('OUT', None))(deconv_layers[-1])
    out = resize_layer(init, out)
    model = tf.keras.Model(init, out)

    module = import_module('Models.Losses')
    loss_fct = getattr(module, net.Loss)
    model.compile(optimizer=net.Optimizer, loss=loss_fct, metrics=[PSNRLoss])
    #model.summary()

    return model


