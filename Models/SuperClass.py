from utils.import_lib import *
from utils.helper import *
from utils.make_graph import *
from utils.instantiate_net import *
import shutil

import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import math
import hjson
from utils.patch import patchify, depatchify
from utils.ROC import compute_ROC, compute_ROC_pixel_wise
from datasets.add_corruption import corrupt_dataset, corrupt_image
from utils.callbacks import HistoryCheckpoint

from sklearn.feature_extraction import image
import pickle

import matplotlib
from matplotlib import pyplot as plt
plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
plt.rcParams.update({'font.size': 16})
colors = {'train': 'darkcyan', 'val':'crimson'}

import time
from random import randint, uniform

from skimage.transform import resize
from skimage.exposure import adjust_gamma, equalize_hist
from skimage.color import rgb2gray


import importlib


''' COLLIN Anne-Sophie '''

class Network_Class: 
    # ---------------------------------------------------------------------------------------
    # Initialisation of class variables
    # ---------------------------------------------------------------------------------------
    def __init__(self, class_type):
        self.model       = None 
        self.model_name  = None
        self.exp_name    = None 
        self.class_type  = class_type
        self.row         = None
        self.col         = None

    # ---------------------------------------------------------------------------------------
    # Import of the parameters described by the json file (args)
    # ---------------------------------------------------------------------------------------
    def create_model(self, args, dataset):
        def import_config(type): 
            if type == 'archi': 
                path         = code_path + '/Models/Networks/Architecture_Config/' + args['3_model_arch']
                default_path = code_path + '/Models/Networks/Architecture_Config/00_Default.hjson'
            elif type == 'train': 
                path         = code_path + '/Models/Training_Config/' + args['3_train'] 
                default_path = code_path + '/Models/Training_Config/00_Default.hjson' 
            
            with open(path, 'r') as fp:
                param = hjson.load(fp)
            with open(default_path, 'r') as fp:
                default_param = hjson.load(fp)
            return param, default_param
    
        def init_object(param, default_param): 
            flatten_default_param = iterate_dict(default_param, {})
            flatten_param         = iterate_dict(param, {})
            for this_param in flatten_default_param: 
                if this_param in flatten_param:
                    setattr(self, this_param, flatten_param[this_param])
                else: 
                    setattr(self, this_param, flatten_default_param[this_param])
            for this_param in flatten_param: 
                setattr(self, this_param, flatten_param[this_param])

        archi_param, default_archi_param = import_config('archi')
        train_param, default_train_param = import_config('train')
        init_object(archi_param, default_archi_param)
        init_object(train_param, default_train_param)

        self.row, self.col = dataset.img_row, dataset.img_col
        self.model_name = args['3_model_arch'].split('.')[0]

        if args['4_MCdrop']: 
            self.MCDropout = True
        
        self.exp_name   = dataset.ds_name + str(dataset.img_row) + 'x' + str(dataset.img_col) + '_' \
             + args['3_model_class'] + '_' \
             + args['3_model_arch'].split('.hjson')[0] + '_' \
             + args['3_train'].split('.hjson')[0]


    # ---------------------------------------------------------------------------------------
    # Data Augmentation
    # ---------------------------------------------------------------------------------------
    def augment_data(self, dataset): 

        print('start offline data augmentation')
        overwrite = True 
        # Pure "classical" Data Augmentation 
        if self.Rotation_Translation : 
            dataset.train    = np.vstack((dataset.train, np.flip(dataset.train, 1)))
            dataset.train    = np.vstack((dataset.train, np.flip(dataset.train, 2)))
            dataset.train_GT = dataset.train
        
        if self.Illumination_Changes: 
            new_train_set, new_GT = None, None
            for _ in range(5):
                for im in zip(dataset.train):
                    gamma = uniform(0.5, 2.0)
                    drop_array([adjust_gamma(im, gamma)], self.tmp + '/Train/Imgs', overwrite=overwrite)
                    drop_array([im], self.tmp + '/Train_GT/Imgs', overwrite=overwrite)
                    overwrite = False

        if self.Histogram_Equalization:
            new_train_set, new_GT = None, None
            for _ in range(5):
                for im in zip(dataset.train):
                    drop_array([equalize_hist(im, gamma)], self.tmp + '/Train/Imgs', overwrite=overwrite)
                    drop_array([im], self.tmp + '/Train_GT/Imgs', overwrite=overwrite)
                    overwrite = False

        # Homemade Corruption
        if 'Clean' not in dataset.ds_name and self.Offline_Corruption:
            new_train_set, new_GT = None, None
            GT_images             = np.copy(dataset.train)
            for _ in range(5):
                temp = corrupt_dataset(dataset.train, data_path + '/' + dataset.ds_name.split('_')[0] + '/Train/' + dataset.ds_name.split('_')[1] + '/00_description.json')
                drop_array(temp, self.tmp + '/Train/Imgs', overwrite=overwrite)
                drop_array(dataset.train, self.tmp + '/Train_GT/Imgs', overwrite=overwrite)
                overwrite = False

        # Patch 
        if self.Patch != '': 
            overwrite = False
            idx = 0
            list1 = [x for x in os.listdir(self.tmp + '/Train/Imgs') if '_' not in x]
            list2 = [x for x in os.listdir(self.tmp + '/Train/Imgs') if '_' not in x]
            for im_path, im_GT_path in zip(list1, list2):
                im , im_GT = read_png(self.tmp + '/Train/Imgs/' + im_path), read_png(self.tmp + '/Train/Imgs/' + im_GT_path)
                new_train_set, _    = patchify(im, int(self.Patch.split('x')[0]), int(self.Patch.split('x')[1]), 50, 50)
                new_train_set_GT, _ = patchify(im_GT, int(self.Patch.split('x')[0]), int(self.Patch.split('x')[1]), 50, 50)
                drop_array(new_train_set, self.tmp + '/Train/Imgs', overwrite=overwrite, suffix_iter=True, iter=idx)
                drop_array(new_train_set_GT, self.tmp + '/Train_GT/Imgs', overwrite=overwrite, suffix_iter=True, iter=idx)
                os.remove(self.tmp + '/Train/Imgs/' + im_path) 
                os.remove(self.tmp + '/Train_GT/Imgs/' + im_GT_path) 
                idx += 1 


        if overwrite: 
            drop_array(dataset.train, self.tmp + '/Train/Imgs', overwrite=overwrite)
            drop_array(dataset.train, self.tmp + '/Train_GT/Imgs', overwrite=overwrite)
        
        print('end offline data augmentation')

    # ---------------------------------------------------------------------------------------
    # Training procedure of the network
    # ---------------------------------------------------------------------------------------
    def fit(self, args, dataset):

        # Prepare saving results
        write_json(args, root_path + '/Experiments/' + self.exp_name + '/00_description.json' )
        weights_path   = root_path + '/Experiments/' + self.exp_name + '/weights.h5' 
        history_path   = root_path + '/Experiments/' + self.exp_name + '/history.h5' 

        if not os.path.exists(str( Path(weights_path).parent )):
            os.makedirs(str( Path(weights_path).parent ))
        if not os.path.exists(str( Path(history_path).parent )):
            os.makedirs( str(Path(history_path).parent ))
        
        # Initialize callbacks
        callback_list = [callbacks.ModelCheckpoint(weights_path, monitor='val_PSNRLoss', save_best_only=True, mode='max', save_weights_only=True, verbose=1)]
        if self.Callback == "ReduceonPlateau": 
            callback_list.append(callbacks.ReduceLROnPlateau(monitor='val_PSNRLoss', factor=0.5, patience=30, verbose=0, mode='max', min_delta=1, cooldown=0, min_lr=0))
        if self.Callback == 'Step_decay': 
            def step_down(epoch):
                initial_lrate = self.Learning_rate
                drop = 0.5
                epochs_drop = 30.0
                lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
                return lrate
            callback_list.append(callbacks.LearningRateScheduler(step_decay))
        if self.Callback == "Step_down": 
            def step_down(epoch):
                initial_lrate = self.Learning_rate
                if epoch > 69 and epoch < 149: 
                    return initial_lrate/10
                elif epoch > 149: 
                    return initial_lrate/50
                else:
                    return initial_lrate
            callback_list.append(callbacks.LearningRateScheduler(step_down))
        
        callback_list.append(HistoryCheckpoint(history_path))

        # Training procedure
        seed = 2042
 
        datagen_args = dict(rescale=1.0/255.0, 
                            validation_split=0.2, 
                            rotation_range=self.Rotation,
                            width_shift_range=self.Width_shift,
                            height_shift_range=self.Height_shift,
                            horizontal_flip=self.Horizontal_flip,
                            vertical_flip=self.Vertical_flip,
                            zoom_range=self.Zoom,
                            fill_mode="nearest")
        
        if 'Clean' in dataset.ds_name or self.Offline_Corruption:
            generator_x = ImageDataGenerator(**datagen_args)
        else: 
            corruption_args = read_json(data_path + '/' + dataset.ds_name.split('_')[0] + '/Train/' + dataset.ds_name.split('_')[1] + '/00_description.json')
            def my_corruption(image): 
                st0 = np.random.get_state()
                result =  corrupt_image(image, corruption_args).astype('float32')
                np.random.set_state(st0)
                return result
            generator_x = ImageDataGenerator(preprocessing_function=my_corruption, **datagen_args)
        generator_y = ImageDataGenerator(**datagen_args)
        
        
        dataflow_args = dict(batch_size=self.Batch_size,
                            class_mode=None,
                            target_size=(self.row,self.col),
                            color_mode='grayscale' if self.Grayscale else 'rgb', 
                            shuffle=True,
                            seed=seed
                            )

        train_input = generator_x.flow_from_directory(self.tmp + '/Train', subset='training', **dataflow_args)
        train_masks = generator_y.flow_from_directory(self.tmp + '/Train_GT', subset='training',**dataflow_args)
        val_input   = generator_x.flow_from_directory(self.tmp + '/Train', subset='validation', **dataflow_args)
        val_masks   = generator_y.flow_from_directory(self.tmp + '/Train_GT', subset='validation', **dataflow_args)

        train_generator = zip(train_input, train_masks)
        val_generator   = zip(val_input, val_masks)

        self.model.fit_generator(generator=train_generator,
                epochs = self.Epoch,
                steps_per_epoch = int(2000/self.Batch_size),
                validation_data=val_generator,
                validation_steps = int(500/self.Batch_size), 
                callbacks=callback_list,
                verbose=1, 
                #use_multiprocessing=True,
                #workers=4
                )

        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---------------------------------------------------------------------------------------
    # Print learning curves
    # ---------------------------------------------------------------------------------------
    def print_learning_curves(self, GUI=False, row=None, col=None): 
        history_path  = root_path + '/Experiments/' + self.exp_name + '/history.h5' 
        plot_path     = root_path + '/Experiments/' + self.exp_name + '/Learning' 
        with open(history_path, 'rb') as handle:
            history = pickle.loads(handle.read())

        nb_epoch = len(history['loss'])+1
        if GUI: 
            plt.rc('xtick', labelsize=7) 
            plt.rc('ytick', labelsize=7) 
            plt.rcParams.update({'font.size': 7})
            my_dpi=96
            plt.figure(figsize=(col/my_dpi, row/my_dpi), dpi=my_dpi)

        # summarize history for PSNR
        if GUI: 
            plt.plot(range(1,nb_epoch), history['PSNRLoss'], color=colors['train'], linewidth=1, label='training set')
            plt.plot(range(1,nb_epoch), history['val_PSNRLoss'], color=colors['val'], linewidth=1, label='validation set')
        else: 
            plt.plot(range(1,nb_epoch), history['PSNRLoss'], color=colors['train'], linewidth=2, label='training set')
            plt.plot(range(1,nb_epoch), history['val_PSNRLoss'], color=colors['val'], linewidth=2, label='validation set')
        plt.ylabel('PSNR')
        plt.xlabel('epoch')
        plt.legend(loc='lower right')
        if not GUI:
            plt.tight_layout()
            plt.savefig(plot_path + '_PSNR.png')
            plt.close()
        else :
            plt.legend(loc='lower right')
            plt.tight_layout()
            fig = plt.gcf()
            return fig

        # summarize history for loss
        plt.plot(range(1,nb_epoch), history['loss'], color=colors['train'], linewidth=2, label='training set')
        plt.plot(range(1,nb_epoch), history['val_loss'], color=colors['val'], linewidth=2, label='validation set')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(plot_path + '_loss.png')
        plt.close()
  
    # ---------------------------------------------------------------------------------------
    # Load weights 
    # ---------------------------------------------------------------------------------------
    def load_weights(self, exp_name, weight_file = None):
        self.exp_name = exp_name
        if weight_file is None:
            self.model.load_weights(root_path + '/Experiments/' + self.exp_name + '/weights.h5')
        else: 
            self.model.load_weights(root_path + '/Experiments/' + self.exp_name + '/' + weight_file)

    # ---------------------------------------------------------------------------------------
    # Perform network evaluation and compute ROC curve if ROC is True
    # ---------------------------------------------------------------------------------------
    def evaluate(self, dataset, print_pred=True, image_wise = False, pixel_wise = False, ROC=True, norms=['l0', 'l1', 'l2', 'linf']): 
        # Compute the predictions
        used     = set()
        subsets  = [x for x in dataset.test_label_name if x not in used and (used.add(x) or True)]
        all_pred  = {}
        all_mask = {}

        # Infer all the test images
        print('-------start Inference---------')
        print('xxxxxx', self.exp_name)
        all_inputs = dataset.test
        if self.Grayscale:
            all_inputs = rgb2gray(dataset.test)
            all_inputs = np.reshape(all_inputs*255, (len(all_inputs), self.row, self.col, 1))
        all_preds  = self.make_prediction(np.copy(all_inputs))
        print('-------end Inference---------')

        for this_subset in subsets: 
            prediction = all_preds[dataset.test_label_name==this_subset]
            this_pred             = {'input': all_inputs[dataset.test_label_name==this_subset], 'output' : prediction}
            all_pred[this_subset] = this_pred
            all_mask[this_subset] = dataset.mask[dataset.test_label_name==this_subset]
            
            if print_pred : 
                result_path = root_path + '/Results/' + self.exp_name + '/' + this_subset + '_prediction/'
                for input, pred, idx in zip(all_inputs[dataset.test_label_name==this_subset], prediction, range(len(prediction))):
                    print_evaluation(input, pred, result_path + str(idx) +'.png')

        # Compute the ROC curve for each type of anomaly 
        subsets.remove('clean')
        for this_subset in subsets:
            x = np.concatenate( (all_pred['clean']['input'], all_pred[this_subset]['input']) )
            y = np.concatenate( (all_pred['clean']['output'], all_pred[this_subset]['output']) )
            masks = np.concatenate( (all_mask['clean'], all_mask[this_subset]) )

            binary_test_labels = np.concatenate(( dataset.test_label[dataset.test_label_name=='clean'], dataset.test_label[dataset.test_label_name==this_subset]))
            binary_test_labels[binary_test_labels ==2] = 1

            if image_wise: 
                all_TP, all_FP = compute_ROC(x, y, binary_test_labels)
                result_path = root_path + '/Results/' + self.exp_name + '/ROC_clean_vs_' + this_subset 
                write_json({'all_TP': all_TP, 'all_FP': all_FP}, result_path + '.json') 
                print_ROC(result_path + '.json', result_path + '.png', AUC=True)
            if pixel_wise: 
                TP, FP = compute_ROC_pixel_wise(x, y, masks)
                result_path = root_path + '/Results/' + self.exp_name + '/PIXELWISE_ROC_clean_vs_' + this_subset 
                write_json({'TP': TP, 'FP': FP}, result_path + '.json') 
                print_PIXEL_WISE_ROC(result_path + '.json', result_path + '.png', AUC=True)
        
        # Compute combined ROC curve (all defects categories vs. clean)
        x = np.concatenate( (all_pred['clean']['input'], all_inputs[dataset.test_label==2]) )
        y = np.concatenate( (all_pred['clean']['output'], all_preds[dataset.test_label==2]) )

        binary_test_labels = np.concatenate(( dataset.test_label[dataset.test_label_name=='clean'], dataset.test_label[dataset.test_label==2]))
        binary_test_labels[binary_test_labels ==2] = 1
        masks = np.concatenate(( dataset.mask[dataset.test_label_name=='clean'], dataset.mask[dataset.test_label==2]))

        if image_wise:
            all_TP, all_FP = compute_ROC(x, y, binary_test_labels)
            result_path = root_path + '/Results/' + self.exp_name + '/ROC_clean_vs_realDefaults'
            write_json({'all_TP': all_TP, 'all_FP': all_FP}, result_path + '.json')
            print_ROC(result_path + '.json', result_path + '.png', AUC=True)
        if pixel_wise:
            TP, FP = compute_ROC_pixel_wise(x, y, masks)
            result_path = root_path + '/Results/' + self.exp_name + '/PIXELWISE_ROC_clean_vs_realDefaults'
            write_json({'TP': TP, 'FP': FP}, result_path + '.json') 
            print_PIXEL_WISE_ROC(result_path + '.json', result_path + '.png', AUC=True)
        

    # ---------------------------------------------------------------------------------------
    # Perform network evaluation and compute ROC curve if ROC is True
    # ---------------------------------------------------------------------------------------
    def evaluate_MCDropout(self, args, print_pred=True, image_wise = False, pixel_wise = False, ROC=True, norms=['l0', 'l1', 'l2', 'linf']): 

        args['4_MCdrop']    = True
        my_dataset, new_net = instantiate_net(args, Train=False)
        new_net.MCDropout = True

        # Compute the predictions
        used     = set()
        subsets  = [x for x in my_dataset.test_label_name if x not in used and (used.add(x) or True)]
        all_mask = {}

        for this_subset in subsets: 
            all_mask[this_subset] = my_dataset.mask[my_dataset.test_label_name==this_subset]

        # Infer all the test images
        print('-------start Inference---------')
        print('xxxxxx', self.exp_name)
        all_pred = self.make_MCprediction(my_dataset, new_net, print_pred)
        print('-------end Inference---------')


        # Compute the ROC curve for each type of anomaly 
        subsets.remove('clean')
        for this_subset in subsets:
            y = np.concatenate( (all_pred['clean']['variance'], all_pred[this_subset]['variance']) )
            x = np.zeros(y.shape)
            masks = np.concatenate( (all_mask['clean'], all_mask[this_subset]) )

            binary_test_labels = my_dataset.test_label
            binary_test_labels = np.concatenate(( my_dataset.test_label[my_dataset.test_label_name=='clean'], my_dataset.test_label[my_dataset.test_label_name==this_subset]))
            binary_test_labels[binary_test_labels ==2] = 1
            
            if image_wise: 
                all_TP, all_FP = compute_ROC(x, y, binary_test_labels)
                result_path = root_path + '/Results_MCDropout/' + self.exp_name + '/ROC_clean_vs_' + this_subset 
                write_json({'all_TP': all_TP, 'all_FP': all_FP}, result_path + '.json') 
                print_ROC(result_path + '.json', result_path + '.png', AUC=True)
            if pixel_wise:
                TP, FP = compute_ROC_pixel_wise(x, y, masks)
                result_path = root_path + '/Results_MCDropout/' + self.exp_name + '/PIXELWISE_ROC_clean_vs_' + this_subset 
                write_json({'TP': TP, 'FP': FP}, result_path + '.json') 
                print_PIXEL_WISE_ROC(result_path + '.json', result_path + '.png', AUC=True)
        
        # Compute combined ROC curve (all defects categories vs. clean)
        all_def = my_dataset.test_label_name[my_dataset.test_label ==2] 
        used    = set()
        subsets = [x for x in all_def if x not in used and (used.add(x) or True)]

        prediction = None
        for this_subset in subsets : 
            if prediction is None:
                prediction = all_pred[this_subset]['variance']
            else: 
                prediction = np.vstack( (prediction, all_pred[this_subset]['variance']))
     
        y = np.concatenate( ( all_pred['clean']['variance'], prediction) )
        x = np.zeros(y.shape)
        masks = np.concatenate((my_dataset.mask[my_dataset.test_label_name=='clean'], my_dataset.mask[my_dataset.test_label==2]))

        binary_test_labels = np.concatenate(( my_dataset.test_label[my_dataset.test_label_name=='clean'], my_dataset.test_label[my_dataset.test_label==2]))
        binary_test_labels[binary_test_labels ==2] = 1
        
        if image_wise: 
            all_TP, all_FP = compute_ROC(x, y, binary_test_labels)
            result_path = root_path + '/Results_MCDropout/' + self.exp_name + '/ROC_clean_vs_realDefaults'
            write_json({'all_TP': all_TP, 'all_FP': all_FP}, result_path + '.json')
            print_ROC(result_path + '.json', result_path + '.png', AUC=True)
        if pixel_wise: 
            TP, FP = compute_ROC_pixel_wise(x, y, masks)
            result_path = root_path + '/Results_MCDropout/' + self.exp_name + '/PIXELWISE_ROC_clean_vs_realDefaults'
            write_json({'TP': TP, 'FP': FP}, result_path + '.json') 
            print_PIXEL_WISE_ROC(result_path + '.json', result_path + '.png', AUC=True)
        
    