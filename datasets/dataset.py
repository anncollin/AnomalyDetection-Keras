from utils.import_lib import *
from utils.helper import *

from skimage.color import rgb2gray


""" COLLIN Anne-Sophie """ 

""" -----------------------------------------------------------------------------------------
DATASET CLASS
----------------------------------------------------------------------------------------- """ 
class Dataset():
    def __init__(self):
        self.ds_name = None
        self.train       = None
        self.train_corruption = None
        self.train_GT    = np.empty((0, 0))
        self.test            = None
        self.mask            = None
        self.test_label      = list() # array of 0 (clean),1 (synthetic) and 2 (real)
        self.test_label_name = list() # array of strings
        self.train_size      = 0
        self.validation_size = 0
        self.test_size       = 0
        self.img_col = 0
        self.img_row = 0

    """ -----------------------------------------------------------------------------------------
    Dowload a training set into a dataset object
    INPUT :  
        - dataset : name of the dataset to import 
    ----------------------------------------------------------------------------------------- """
    def import_train_set(self, rescale_factor):
        train_path = data_path + '/'  + self.ds_name.split('_')[0] + '/Train/Clean'
        img_list   = [x for x in os.listdir(train_path) if '.png' in x]
        for img_path in img_list:
            if self.train is None : 
                self.train = np.array([read_png(train_path + '/' + img_path, rescale_factor)])
            else: 
                self.train = np.vstack((self.train, np.array([read_png(train_path + '/' + img_path, rescale_factor)]) )) 
        self.img_row, self.img_col = self.train.shape[1], self.train.shape[2]

    def load_train(self, dataset, scaling='Default'):
        self.ds_name = dataset
        if scaling =='Default': 
            rescale_factor = 1
        else: 
            initial_dim = (read_png(data_path + '/'  + self.ds_name.split('_')[0] + '/Train/Clean/0.png', 1)).shape[0]
            final_dim   = int(scaling)
            rescale_factor = final_dim/initial_dim

        self.train_corruption = read_json(data_path + '/'  + self.ds_name.split('_')[0] + '/Train/' + self.ds_name.split('_')[1] + '/00_description.json' )
        self.import_train_set(rescale_factor)
        

    """ -----------------------------------------------------------------------------------------
    Dowload a test set into a dataset object
    INPUT :  
        - dataset : name of the dataset to import 
    ----------------------------------------------------------------------------------------- """
    def import_test_set(self, subfolder, rescale_factor):
        test_path = data_path + '/'  + self.ds_name.split('_')[0] + '/Test/' + subfolder
        img_list  = [x for x in os.listdir(test_path) if '.png' in x and 'mask' not in x]
        if '_' not in img_list[0]: 
            img_list.sort(key=lambda fname: int(fname.split('.')[0]))

        labels = read_json(test_path + '/00_description.json')
        for img_path in img_list:
            image = np.array([read_png(test_path + '/' + img_path, rescale_factor)])
            if self.test is None : 
                self.test = image
            else: 
                self.test = np.vstack((self.test, image ))

            if subfolder == 'Clean':
                self.test_label.append(0)
                self.test_label_name.append('clean')
                if self.mask is None : 
                    self.mask = np.zeros(rgb2gray(image).shape) 
                else :
                    self.mask = np.vstack(( self.mask, np.zeros(rgb2gray(image).shape)  ))

            elif subfolder == 'Real_corruption':
                self.test_label.append(2)
                self.test_label_name.append(labels[img_path])
                try: 
                    mask = np.array([read_png(test_path + '/mask_' + img_path, rescale_factor)])/255.  
                except: 
                    mask = np.zeros(image.shape) 

                if self.mask is None : 
                    self.mask = rgb2gray(mask)
                else :
                    self.mask = np.vstack(( self.mask, rgb2gray(mask) ))
            else : 
                self.test_label.append(1)
                self.test_label_name.append(subfolder) 
                if self.mask is None : 
                    self.mask = np.zeros(rgb2gray(image).shape) 
                else :
                    self.mask = np.vstack(( self.mask, np.zeros(rgb2gray(image).shape)  ))
        
        self.img_row, self.img_col = self.test.shape[1], self.test.shape[2]
        

    def load_test(self, dataset, scaling='Default'):
        self.ds_name = dataset
        if scaling =='Default': 
            rescale_factor = 1
        else: 
            initial_dim = (read_png(data_path + '/'  + self.ds_name.split('_')[0] + '/Test/Clean/0.png', 1)).shape[0]
            final_dim   = int(scaling)
            rescale_factor = final_dim/initial_dim

        folder_list = ['Clean', 'Real_corruption']
        if dataset != 'Clean': 
            folder_list.append(dataset.split('_')[1])

        for subfolder in folder_list : 
            self.import_test_set(subfolder, rescale_factor)
        self.test_label_name = np.array(self.test_label_name)
        self.test_label      = np.array(self.test_label)







