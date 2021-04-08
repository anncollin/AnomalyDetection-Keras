from utils.import_lib import *
from utils.helper import *
from utils.make_graph import *
from utils.instantiate_net import *

import tensorflow as tf

from random import randint
import importlib
from Models.SuperClass import Network_Class
from utils.patch import patchify, depatchify
from utils.ROC import compute_ROC, compute_combinedROC, compute_2DcombinedROC
from copy import deepcopy

from skimage.color import rgb2gray

import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
plt.rcParams.update({'font.size': 16})
colors = {'train': 'darkcyan', 'val':'crimson'}

''' COLLIN Anne-Sophie '''

""" -----------------------------------------------------------------------------------------
Super class describing principal methods of "ImtoIm" network
    - taking a corrupted image as input (+ knowledge of the GT)
    - returning a clean image as output 
The network can be learned on the entire image/ on patches
----------------------------------------------------------------------------------------- """ 

class ImtoIm(Network_Class):

    def __init__(self):
        super(ImtoIm, self).__init__("ImtoIm")

    # ---------------------------------------------------------------------------------------
    # Import of the parameters described by the json file (args)
    # ---------------------------------------------------------------------------------------
    def create_model(self, args, dataset):
        super(ImtoIm, self).create_model(args, dataset)

        channels = 1 if self.Grayscale else 3 
        if self.Patch == '' :
            shape = (self.row, self.col, channels)
        else:
            shape = (int(self.Patch.split('x')[0]), int(self.Patch.split('x')[1]), channels)
        init = tf.keras.Input(shape=shape)

        # Create the network 
        net_module  = importlib.import_module('Models.Networks.' + args['3_model_arch'].split('_')[0])
        build_model = getattr(net_module, 'build_model')
        self.model  = build_model(self, init)
    
    # ---------------------------------------------------------------------------------------
    # Training proceudre of the network
    # ---------------------------------------------------------------------------------------
    def fit(self, args, dataset):
        self.tmp = data_path + '/.tmp/' + self.exp_name
        super(ImtoIm, self).augment_data(dataset)
        super(ImtoIm, self).fit(args, dataset)

    # ---------------------------------------------------------------------------------------
    # Infer a single image
    # ---------------------------------------------------------------------------------------
    def make_prediction(self, input_im, GUI=False, row=None, col=None):
        
        im_shape = np.shape(input_im)
        if len(input_im.shape) != 4 : 
            input_im = np.array([input_im])

        if self.Patch == '' : 
            return ((self.model.predict(input_im.astype('float32') / 255))*255).astype(np.uint8) 
           
        else : 
            result_list = None
            for idx,this_im in enumerate(input_im): 
                patches, corner = patchify(this_im, int(self.Patch.split('x')[0]), int(self.Patch.split('x')[1]), 25, 25)
                prediction      = ((self.model.predict(patches.astype('float32') / 255))*255).astype(np.uint8) 
                if idx == 0:
                    result_list = np.array([depatchify(prediction, input_im.shape[1], input_im.shape[2], corner)])
                else: 
                    this_pred   = np.array([depatchify(prediction, input_im.shape[1], input_im.shape[2], corner)])
                    result_list = np.vstack( (result_list, this_pred) )
            return result_list

    # ---------------------------------------------------------------------------------------
    # Infer a all dataset images with Monte Carlo Dropout
    # ---------------------------------------------------------------------------------------
    def make_MCprediction(self, dataset, new_net, print_pred, GUI=False, row=None, col=None):
        layer_dict          = dict([(layer.name, layer) for layer in self.model.layers])
        newlayers           = [layer.name for layer in new_net.model.layers]

        for idx, layer_name in enumerate(newlayers):
            if 'Dropout' not in layer_name: 
                if layer_name in layer_dict and len(layer_dict[layer_name].get_weights())>0 : 
                    new_net.model.layers[idx].set_weights(layer_dict[layer_name].get_weights())

        used     = set()
        subsets  = [x for x in dataset.test_label_name if x not in used and (used.add(x) or True)]
        all_pred = {}
        n_pred = 30
        # Compute the prediction masks
        for this_subset in subsets: 
            input_imgs          = dataset.test[dataset.test_label_name==this_subset]
            if self.Grayscale: 
                input_imgs = rgb2gray(input_imgs)
                input_imgs = np.reshape(input_imgs *255, (len(input_imgs ), self.row, self.col, 1))
            mean_pred, var_pred = np.zeros(input_imgs.shape, dtype='uint8'), np.zeros((len(input_imgs), self.row, self.col, 1))
            for i, input_im in enumerate(input_imgs): 
                these_pred = np.zeros((n_pred, input_im.shape[0], input_im.shape[1], 1 if self.Grayscale else 3), dtype='uint8')
                for j in range(n_pred):
                    these_pred[j] = new_net.make_prediction(input_im)

                mean_pred[i] = np.mean(these_pred, axis=0)
                if self.Grayscale:
                    var_pred[i]  = np.var(these_pred, axis=0)
                else : 
                    temp = np.mean(np.var(these_pred, axis=0), axis=-1)
                    var_pred[i] = np.reshape(temp, (self.row, self.col, 1))

            result_path           = root_path + '/Results_MCDropout/' + new_net.exp_name + '/' + this_subset + '_prediction/'
            all_pred[this_subset] = None
            all_pred[this_subset] = {'input': input_imgs, 'output': mean_pred, 'variance': var_pred}

            # Show the result 
            if print_pred: 
                for input, pred, var, idx in zip(input_imgs, mean_pred, var_pred, range(len(input_imgs))):
                    print_MCevaluation(input, pred, var, result_path + str(idx) +'.png')
        return all_pred

    # ---------------------------------------------------------------------------------------
    # Infer a single image with Monte Carlo Dropout
    # ---------------------------------------------------------------------------------------
    def make_MCpredictionBIS(self, input_img, args):

        im_shape = np.shape(input_img)
        if len(input_img.shape) != 4 : 
            input_img = np.array([input_img])

        args['4_MCdrop']    = True
        my_dataset, new_net = instantiate_net(args, Train=False)
        new_net.MCDropout   = True
        layer_dict          = dict([(layer.name, layer) for layer in self.model.layers])
        newlayers           = [layer.name for layer in new_net.model.layers]
        for idx, layer_name in enumerate(newlayers):
            if 'Dropout' not in layer_name: 
                if layer_name in layer_dict and len(layer_dict[layer_name].get_weights())>0 : 
                    new_net.model.layers[idx].set_weights(layer_dict[layer_name].get_weights())

        n_pred = 30

        mean_pred, var_pred = np.zeros(input_img.shape, dtype='uint8'), np.zeros(input_img.shape)
        for i, input_im in enumerate(input_img): 
            these_pred = np.zeros((n_pred, input_im.shape[0], input_im.shape[1], 1 if self.Grayscale else 3), dtype='uint8')
            for j in range(n_pred):
                these_pred[j] = new_net.make_prediction(input_im)
            mean_pred[i] = np.mean(these_pred, axis=0)
            var_pred[i]  = np.var(these_pred, axis=0)
        return mean_pred, var_pred

