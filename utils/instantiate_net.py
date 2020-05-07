from utils.import_lib import *
from datasets.dataset import Dataset
from utils.helper import *

import importlib

def instantiate_net(args, Train=True):
    module     = importlib.import_module('Models.Class'+args['3_model_class'])
    class_     = getattr(module, args['3_model_class']) 
    my_net     = class_()

    with_GT = True
    if args['3_dimension'] == '': 
        rescale_factor = 'Default'
    else:
        rescale_factor = args['3_dimension']
    my_dataset = Dataset()
    if Train: 
        my_dataset.load_train(args['3_ds'], rescale_factor)
    my_dataset.load_test(args['3_ds'], rescale_factor)
    my_net.create_model(args, my_dataset)    

    return my_dataset, my_net

