from utils.import_lib import *
from datasets.dataset import Dataset
from utils.helper import *
from utils.instantiate_net import *

from shutil import copyfile
import importlib
from keras import backend as K
from datasets.add_corruption import corrupt


""" COLLIN Anne-Sophie """ 

""" -----------------------------------------------------------------------------------------
IMPORT A NEW DATABASE
----------------------------------------------------------------------------------------- """
def import_new_db(args): 
    ds_name = args["0_ds_name"]

    # Create Empty folders 
    """
    ds_path = data_path + '/' + ds_name
    if not os.path.exists(ds_path):
        os.makedirs(ds_path)
        os.makedirs(ds_path + '/Train/Clean')
        os.makedirs(ds_path + '/Test/Real_corruption')
        os.makedirs(ds_path + '/Test/Clean')
    else: 
        raise ValueError('The %s database already exists' % (ds_name))
    """

    def copy_folder(from_folder, to_folder, mask=False): 
        lst   = sorted(os.listdir(from_folder))
        
        if mask : 
            masks = [x for x in os.listdir(to_folder) if 'mask' in x]
            shift = len(masks)
        else : 
            ims = [x for x in os.listdir(to_folder) if 'mask' not in x]
            shift = len(ims)
        for idx, img_path in enumerate(lst):
            if mask : 
                copyfile(from_folder + '/' + img_path, to_folder + 'mask_' + str(shift+idx) + '.png')
            else : 
                copyfile(from_folder + '/' + img_path, to_folder + str(shift+idx) + '.png')
 
    """
    # Copy train images
    train_path = args['0_train_set']
    copy_folder(train_path, ds_path + '/Train/Clean/', {"1_default":None})

    # Copy test images
    test_clean_path = args['0_test_clean']
    copy_folder(test_clean_path, ds_path + '/Test/Clean/', {"1_default":None})

    all_def_folders = [x for x in args if '0_test_defaul' in x]
    description = {}
    shift = 0
    for this_def in all_def_folders: 
        if args[this_def] is not '': 
            copy_folder(args[this_def], ds_path + '/Test/Real_corruption/')
            for i in range( len(os.listdir(args[this_def])) ): 
                description[str(shift+i) + '.png'] = args['0_test_label'+this_def[-1]]
            shift += (i+1)
    write_json(description, ds_path + '/Test/Real_corruption/00_description.json')
    
    # Copy masks 
    # all_def_folders = [x for x in args if '0_test_defaul' in x] # Change path
    #for this_def in all_def_folders: 
    #    if args[this_def] is not '': 
    #        copy_folder(args[this_def], ds_path + '/Test/Real_corruption/mask_')
    """
    print(data_path)
    description = {}
    ori_ds, new_ds = 'pill', 'Pill'
    ds_path = '/home/anncollin/Desktop/ADRIC/datasets/MVTec/' + ori_ds
    corr = os.listdir(ds_path + '/ground_truth/')
    shift = 0
    for x in sorted(corr): 
        print(ds_path + '/ground_truth/' + x)
        copy_folder(ds_path + '/test/' + x, data_path + '/' +  new_ds + '/Test/Real_corruption/', mask = False)
        copy_folder(ds_path + '/ground_truth/' + x, data_path + '/' +  new_ds + '/Test/Real_corruption/', mask = True)
        for i in range( len(os.listdir(ds_path + '/ground_truth/' + x)) ): 
            description[str(shift+i) + '.png'] = x
        shift += (i+1)
    write_json(description, data_path + '/' +  new_ds + '/Test/Real_corruption//00_description.json', sort_keys=False)

     
""" -----------------------------------------------------------------------------------------
CORRUPT AN EXISTING DATABASE
----------------------------------------------------------------------------------------- """
# Corrupt images with the given corruption type
def get_preview(args): 
    # Get path and names of all clean and corrupted datasetsClean'
    all_clean_ds  = [x for x in os.listdir(data_path)]
    all_clean_ds  = [x for x in all_clean_ds if not '.' in x and not x == '__pycache__' and not x == '00_datasets']
    all_clean_ds.sort(key = lambda x : x)

    ds_to_corrupt = []
    for i in range(len(all_clean_ds)): 
        if args['1_'+str(i)]: 
            ds_to_corrupt.append(all_clean_ds[i])

    for this_ds in ds_to_corrupt: 
        clean_imgs = read_folder(data_path + '/' + this_ds + '/Train/Clean', max_size= 1) 
        res        = corrupt(clean_imgs, args)
    return res

def corrupt_db(args, def_p): 
    # Get path and names of all clean and corrupted datasets
    all_clean_ds  = [x for x in os.listdir(data_path)]
    all_clean_ds  = [x for x in all_clean_ds if not '.' in x and not x == '__pycache__' and not x == '00_datasets']
    all_clean_ds.sort(key = lambda x : x)

    # Compute a string to identify corruption settings 
    corr_name = args['1_cor_name'] 
    full_name = corr_name + args['1_cor_name']
    params    = def_p[args['1_default'][0]]
    dict      = {'1_default': args['1_default'][0]}

    if args['1_HE']: 
        corr_name += 'HE'
    idx = 0
    for key, value in params.items(): 
        corr_name += '_' + key
        corr_name += args['1_p_'+str(idx)]
        dict[key]  = args['1_p_'+str(idx)]
        idx += 1

    ds_to_corrupt = []
    for i in range(len(all_clean_ds)): 
        if args['1_'+str(i)]: 
            ds_to_corrupt.append(all_clean_ds[i])
    
    for this_ds in ds_to_corrupt: 
        clean_imgs = read_folder(data_path + '/' + this_ds + '/Test/Clean') 
        dest_path  = data_path + '/' + this_ds + '/Test/' + args['1_cor_name']

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        else: 
            raise ValueError('The %s database has already a %s corruption folder' % (this_ds, args['1_cor_name']))

        write_json(dict, dest_path + '/00_description.json')
        write_json(dict, data_path + '/' + this_ds + '/Train/' + args['1_cor_name'] + '/00_description.json')

        # For 1 clean image, create 5 corrupted images (only for the test set)
        for repet in range(5): 
            corr_imgs  = corrupt(clean_imgs, args)
            print_array(corr_imgs, data_path + '/' + this_ds + '/Test/' + args['1_cor_name'], '_' + str(repet))

""" -----------------------------------------------------------------------------------------
TRAIN A NETWORK
----------------------------------------------------------------------------------------- """
from multiprocessing import Process
def train_net(args): 
    def call_training(this_exp):
        my_dataset, my_net = instantiate_net(this_exp)
        print(colored('Start to train the network for experiment : %s.', 'red') % (my_net.exp_name))
        my_net.fit(this_exp, my_dataset)
        print(colored('The network is trained.', 'red'))
        my_net.print_learning_curves()
        print('curves printed')
        
        my_net.load_weights(my_net.exp_name)
        my_net.evaluate(my_dataset)
  

    for this_dataset in args['3_ds']: 
        for this_architecture in args['3_model_arch']:
            for this_training in args['3_train']: 

                this_exp = args.copy()
                this_exp['3_ds']         = this_dataset
                this_exp['3_model_arch'] = this_architecture
                this_exp['3_train']      = this_training

                p = Process(target=call_training, args=(this_exp,))
                p.start()
                p.join()


""" -----------------------------------------------------------------------------------------
LIVE DEMO
----------------------------------------------------------------------------------------- """
def live(args): 
    my_dataset, my_net = instantiate_net2(args, args['3_ds'][0])
    my_net.load_weights(my_net.exp_name)
    my_net.live_demo()

""" -----------------------------------------------------------------------------------------
EVALUATE A NETWORK
----------------------------------------------------------------------------------------- """

def evaluate_net(args): 
    if args['4_ROC'] or args['4_ROC_pixel']: 
        for this_exp in args['4_exp']:
            this_args = read_json(root_path + '/Experiments/' + this_exp + '/00_description.json')
            my_dataset, my_net = instantiate_net(this_args)
            my_net.load_weights(this_exp)
            my_net.evaluate(my_dataset, print_pred=args['4_ROC_printpred'], pixel_wise = args['4_ROC_pixel'])

    if args['4_MCdrop'] or args['4_MCdrop_pixel']: 
        for this_exp in args['4_exp']:
            this_args = read_json(root_path + '/Experiments/' + this_exp + '/00_description.json')
            this_args['4_MCdrop'] = True
            _, my_net = instantiate_net(this_args)
            my_net.load_weights(this_exp)
            my_net.evaluate_MCDropout(this_args, print_pred=args['4_MCdrop_printpred'], pixel_wise = args['4_MCdrop_pixel'])

    if args['4_combinedMC']: 
        for this_exp in args['4_exp']:
            this_args = read_json(root_path + '/Experiments/' + this_exp + '/00_description.json')
            this_args['4_MCdrop'] = True
            _, my_net = instantiate_net(this_args)
            my_net.load_weights(this_exp)
            my_net.evaluate_combinedROC(this_args)

    if args['4_OOD']: 
        for this_exp in args['4_exp']:
            this_args = read_json(root_path + '/Experiments/' + this_exp + '/00_description.json')
            dataset, my_net = instantiate_net(this_args)
            my_net.OOD(dataset)


    





        


