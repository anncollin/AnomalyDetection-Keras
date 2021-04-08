from utils.import_lib import *
from utils.helper import *

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import NoNorm
import tkinter as Tk
import pickle


plt.rc('xtick', labelsize=7) 
plt.rc('ytick', labelsize=7) 
plt.rcParams.update({'font.size': 7})
colors = {'train': 'darkcyan', 'val':'crimson'}
colors4 = ['gold','greenyellow','steelblue','midnightblue']

my_dpi=96

""" COLLIN Anne-Sophie """ 

def hjson_parser(path): 
    import hjson
    from collections import OrderedDict
    with open(path, 'r') as fp:
        result = hjson.load(fp)

    return hjson.dumps(result, indent=2, sort_keys=True)

""" -----------------------------------------------------------------------------------------
Find all elements of a list which match a query
INPUT : 
    - list : initial list 
    - query
OUTPUT : 
    - list of matching elements
----------------------------------------------------------------------------------------- """ 
def update_list(list, query): 
    new_list = []
    for this_elem in list: 
        cdt = True
        for this_query in query: 
            if len(this_query)>1: 
                if this_query[0] == '~': 
                    if this_query[1::].lower() in this_elem.lower():
                        cdt = False
                else: 
                    if this_query.lower() not in this_elem.lower():
                        cdt = False
        if cdt: 
            new_list += [this_elem]
    return new_list

""" -----------------------------------------------------------------------------------------
Return a figure with an image in it to be displayed on the GUI
INPUT : 
    - img : a numpy array
    - row,col : size of the image
OUTPUT : 
    - figure
----------------------------------------------------------------------------------------- """ 
def show_im(img, row, col): 
    plt.figure(figsize=(col/my_dpi, row/my_dpi), dpi=my_dpi)
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic', norm=NoNorm())
    plt.axis('off')
    fig = plt.gcf()  
    return fig


""" -----------------------------------------------------------------------------------------
Return a figure with an image in it to be displayed on the GUI
INPUT : 
    - img : a numpy array
    - row,col : size of the image
OUTPUT : 
    - figure
----------------------------------------------------------------------------------------- """ 
def show_im_with_uncertainty(img, uncertainty, row, col): 
    mask = uncertainty
    mask = np.ma.masked_where(mask < 0.1, mask)
    plt.figure(figsize=(col/my_dpi, row/my_dpi), dpi=my_dpi)
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic', norm=NoNorm())
    plt.imshow(mask, cmap = 'jet', interpolation = 'bicubic', norm=NoNorm())
    plt.axis('off')
    fig = plt.gcf()  
    return fig

""" -----------------------------------------------------------------------------------------
Return a figure with an image in it to be displayed on the GUI
INPUT : 
    - input : the input image
    - pred: the prediction of the input image
    - row,col : size of the image
OUTPUT : 
    - figure
----------------------------------------------------------------------------------------- """ 
def show_im_withdef(input, pred, row, col): 
    plt.figure(figsize=(col/my_dpi, row/my_dpi), dpi=my_dpi)
    this_diff = diff(input, pred)
    this_diff = np.ma.masked_where(this_diff < 0.05, this_diff)
    plt.imshow(pred, cmap = 'gray', interpolation = 'bicubic', norm=NoNorm())
    plt.imshow(this_diff, cmap = 'jet', interpolation = 'bicubic')
    plt.axis('off')
    fig = plt.gcf()  
    return fig

""" -----------------------------------------------------------------------------------------
Helper function used to plot an image on a Yk canvas
INPUT : 
    - canvas
    -figure : as returned by show_im
OUTPUT : 
    - photo
----------------------------------------------------------------------------------------- """ 
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


""" -----------------------------------------------------------------------------------------
Print ROC curve 
INPUT : 
    - json_path : path to json file containing informations 
    - row, col: dimension of the figure
    - AUC: True to display AUC values
OUTPUT : 
    - roc figure
----------------------------------------------------------------------------------------- """ 

def print_ROC(json_path, row, col, AUC=False):
    from sklearn.metrics import auc

    labels = ['L0 ', 'L1 ', 'L2 ', 'L$\infty$ ']

    dict = read_json(json_path)
    TP = dict['all_TP']
    FP = dict['all_FP']

    plt.figure(figsize=(col/my_dpi, row/my_dpi), dpi=my_dpi)
    idx, max_auc = 0, 0
    for this_TP, this_FP, this_color, this_label in zip(TP, FP, colors4, labels):
        if AUC : 
            this_label += '('    
            this_auc    = round(auc(this_FP, this_TP),2)
            if this_auc > max_auc: 
                max_auc = this_auc
            this_label += str(this_auc)
            this_label += ')'
        idx += 1
        plt.plot(this_FP, this_TP, color=this_color, linewidth=1, label=this_label)
    plt.xlabel('FP rate')
    plt.ylabel('TP rate')
    plt.legend(bbox_to_anchor=(1.04,1), title= "Max AUC = " + str(max_auc) + "\n \n Norm (AUC) ", loc="upper left", fontsize=6)
    plt.tight_layout()
    fig = plt.gcf()
    return fig

""" -----------------------------------------------------------------------------------------
Print PSNR 
INPUT : 
    - h5_path : path to h5 file containing informations 
    - row, col: dimension of the figure
OUTPUT : 
    - PSNR plot
----------------------------------------------------------------------------------------- """ 
def print_learning_curves(h5_path, row, col): 
    with open(h5_path, 'rb') as handle:
        history = pickle.loads(handle.read())

    plt.figure(figsize=(col/my_dpi, row/my_dpi), dpi=my_dpi)
    nb_epoch = len(history['loss'])+1
    # summarize history for PSNR
    plt.plot(range(1,nb_epoch), history['PSNRLoss'], color=colors['train'], linewidth=1, label='training set')
    plt.plot(range(1,nb_epoch), history['val_PSNRLoss'], color=colors['val'], linewidth=1, label='validation set')
    #plt.plot(range(1,nb_epoch), history['loss'], color=colors['train'], linewidth=2, label='training set')
    #plt.plot(range(1,nb_epoch), history['val_loss'], color=colors['val'], linewidth=2, label='validation set')
    plt.ylabel('PSNR')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.tight_layout()
    fig = plt.gcf()
    return fig


