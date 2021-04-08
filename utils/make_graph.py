from utils.import_lib import *
from utils.helper import *

import pickle
import matplotlib
from matplotlib import pyplot as plt
current_path = os.path.dirname(os.path.abspath(__file__))

from skimage.color import gray2rgb, rgb2gray

import time 
""" COLLIN Anne-Sophie """ 

plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
plt.rcParams.update({'font.size': 16}) 

colors = {'train': 'darkcyan', 'val':'crimson'}
colors5 = ['gold','greenyellow','mediumaquamarine','steelblue','midnightblue']
colors4 = ['gold','greenyellow','steelblue','midnightblue']

""" -----------------------------------------------------------------------------------------
Single image plot
INPUT : 
    im : image to display
OUTPUT : 
----------------------------------------------------------------------------------------- """ 

def show_im(im): 
    plt.imshow(im, cmap = 'gray', interpolation = 'bicubic')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    #plt.close()


""" -----------------------------------------------------------------------------------------
Single image plot
INPUT : 
    - im : image to display
    - file_path: path to store the resulting image
OUTPUT : 
    - png file in filepath location 
----------------------------------------------------------------------------------------- """ 

def show(im, file_path=None, GUI=False, row=None, col=None): 
    plt.close('all')
    if GUI: 
        my_dpi=96
        plt.figure(figsize=(col/my_dpi, row/my_dpi), dpi=my_dpi)
    #plt.imshow(im, cmap = 'gray', interpolation = 'bicubic',  vmin=0, vmax=255)
    plt.imshow(im, cmap = 'gray', interpolation = 'bicubic')
    plt.axis('off')
    if not GUI:
        plt.tight_layout()
        if not os.path.exists( str(Path(file_path).parent )):
            os.makedirs( str(Path(file_path).parent))
        plt.savefig(file_path)
        plt.close()
    else :
        plt.tight_layout(rect=[-0.15, 0.22, 1, 1])
        fig = plt.gcf()
        return fig

""" -----------------------------------------------------------------------------------------
Single image plot - Tight laout
INPUT : 
    - im : image to display
    - file_path: path to store the resulting image
OUTPUT : 
    - png file in filepath location 
----------------------------------------------------------------------------------------- """ 
def show_tight(im, file_path=None, GUI=False, row=None, col=None): 
    plt.figure()
    fig = plt.imshow(im, cmap = 'gray', interpolation = 'bicubic')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(file_path, bbox_inches = 'tight', pad_inches = 0)
    plt.close()
    

""" -----------------------------------------------------------------------------------------
Print training history for a GAN network
INPUT : 
    - path : path to the 'history' file
OUTPUT : 
    - png files in /graphs/  
----------------------------------------------------------------------------------------- """ 

def print_learningcurves(exp_name, GUI=False, row=None, col=None): 

    history_path  = root_path + '/Experiments/' + exp_name + '/history.h5' 
    plot_path     = root_path + '/Experiments/' + exp_name + '/Learning' 
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
  

""" -----------------------------------------------------------------------------------------
Print result of a unsupervised prediction (input, prediction) 
INPUT : 
    - input: input image given to the network 
    - prediction: output of the network (prediction of the GT)
    - file_name: name of the file in prediction folder
OUTPUT : 
    - png files in prediction folder
----------------------------------------------------------------------------------------- """ 

def print_evaluation(input, prediction, file_path, map=False, Vmin=0, Vmax=255):
    if not os.path.exists( str(Path(file_path).parent )):
        os.makedirs( str(Path(file_path).parent))

    if prediction.shape[-1] == 1: 
        prediction = np.squeeze(prediction, axis=-1)
        input      = np.squeeze(input, axis=-1)
    
    #f  = plt.figure(figsize=(8,2.5))
    plt.rcParams['savefig.pad_inches'] = 0
    f= plt.figure()

    fig = f.add_subplot(131)
    fig.set_title('Input', fontsize=16)
    fig.imshow(input, cmap = 'gray', interpolation = 'bicubic')
    fig.axis('off')

    fig = f.add_subplot(132)
    fig.set_title('Prediction', fontsize=16)
    fig.imshow(prediction, cmap = 'gray', interpolation = 'bicubic')
    fig.axis('off')

    fig = f.add_subplot(133)
    diff_im = diff(input,prediction)
    if map: 
        #mask = np.ma.masked_where(diff_im < 0.1, diff_im)
        fig.imshow(input, cmap = 'gray', interpolation = 'bicubic')
        fig.imshow(diff_im, cmap = 'cividis', interpolation = 'bicubic',alpha=.9,  vmin=Vmin, vmax=Vmax)
        #fig.imshow(diff_im, cmap = 'jet', interpolation = 'bicubic',alpha=.8,  vmin=Vmin, vmax=Vmax)
    else: 
        fig.imshow(diff_im, cmap = 'gray', interpolation = 'bicubic')
    fig.set_title('Difference (I-P)', fontsize=16)
    fig.axis('off')

    plt.savefig(file_path)
    plt.savefig(file_path, bbox_inches = 'tight', pad_inches = 0)
    plt.close()


def print_evaluation_individual(input, prediction, file_path, Vmin=0, Vmax=255):
    if not os.path.exists( str(Path(file_path).parent )):
        os.makedirs( str(Path(file_path).parent))

    if prediction.shape[-1] == 1: 
        prediction = np.squeeze(prediction, axis=-1)
        input      = np.squeeze(input, axis=-1)
    
    #plt.rcParams['savefig.pad_inches'] = 0
    plt.figure()
    fig = plt.imshow(input, cmap = 'gray', interpolation = 'bicubic')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(file_path + '_INPUT.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    #plt.rcParams['savefig.pad_inches'] = 0
    plt.figure()
    fig = plt.imshow(prediction, cmap = 'gray', interpolation = 'bicubic')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(file_path + '_OUTPUT.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    #plt.rcParams['savefig.pad_inches'] = 0
    plt.figure()
    plt.imshow(input, cmap = 'gray', interpolation = 'bicubic')
    diff_im = diff(input,prediction)
    fig = plt.imshow(diff_im, cmap = 'cividis', interpolation = 'bicubic',alpha=.9,  vmin=Vmin, vmax=Vmax)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(file_path + '_HEATMAP.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    plt.figure(figsize=(1.5, 3))
    a = np.array([[Vmin, Vmax]])
    fig = plt.imshow(a, cmap='cividis')
    plt.gca().set_visible(False)
    #cax = plt.axes(np.linspace(Vmin, Vmax,5))
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=22) 


    plt.savefig(file_path + '_COLORBAR.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close()




def print_MCevaluation(input, prediction, variance, file_path, Vmin=0, Vmax=255):
    if not os.path.exists( str(Path(file_path).parent )):
        os.makedirs( str(Path(file_path).parent))

    if prediction.shape[-1] == 1: 
        prediction = np.squeeze(prediction, axis=-1)
    if variance.shape[-1] == 1: 
        variance   = np.squeeze(variance, axis=-1)
    if input.shape[-1] == 1: 
        input      = np.squeeze(input, axis=-1)
    
    #f  = plt.figure(figsize=(8,2.5))
    plt.rcParams['savefig.pad_inches'] = 0
    f= plt.figure()

    fig = f.add_subplot(131)
    fig.set_title('Input', fontsize=16)
    fig.imshow(input, cmap = 'gray', interpolation = 'bicubic')
    fig.axis('off')

    fig = f.add_subplot(132)
    fig.set_title('Mean Prediction', fontsize=16)
    fig.imshow(prediction, cmap = 'gray', interpolation = 'bicubic')
    fig.axis('off')

    fig = f.add_subplot(133)
    fig.imshow(input, cmap = 'gray', interpolation = 'bicubic')
    fig.imshow(variance, cmap = 'cividis', interpolation = 'bicubic',alpha=.9,  vmin=Vmin, vmax=Vmax)
    fig.set_title('Variance', fontsize=16)
    fig.axis('off')

    plt.savefig(file_path)
    plt.savefig(file_path, bbox_inches = 'tight', pad_inches = 0)
    plt.close()


def print_MCevaluation_individual(input, prediction, variance, file_path, Vmin=0, Vmax=255):
    if not os.path.exists( str(Path(file_path).parent )):
        os.makedirs( str(Path(file_path).parent))

    if prediction.shape[-1] == 1: 
        prediction = np.squeeze(prediction, axis=-1)
        variance   = np.squeeze(variance, axis=-1)
        input      = np.squeeze(input, axis=-1)
    
    #plt.rcParams['savefig.pad_inches'] = 0
    plt.figure()
    fig = plt.imshow(input, cmap = 'gray', interpolation = 'bicubic')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(file_path + '_INPUT.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    #plt.rcParams['savefig.pad_inches'] = 0
    plt.figure()
    fig = plt.imshow(prediction, cmap = 'gray', interpolation = 'bicubic')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(file_path + '_MEANMC.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    #plt.rcParams['savefig.pad_inches'] = 0
    plt.figure()
    plt.imshow(input, cmap = 'gray', interpolation = 'bicubic')
    fig = plt.imshow(variance, cmap = 'cividis', interpolation = 'bicubic',alpha=.9,  vmin=Vmin, vmax=Vmax)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(file_path + '_VARIANCEMC.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    plt.figure(figsize=(1.5, 3))
    a = np.array([[Vmin, Vmax]])
    fig = plt.imshow(a, cmap='cividis')
    plt.gca().set_visible(False)
    #cax = plt.axes(np.linspace(Vmin, Vmax,5))
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=22) 

    plt.savefig(file_path + 'MC__COLORBAR.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close()


""" -----------------------------------------------------------------------------------------
Print result of a unsupervised prediction (input, prediction) 
INPUT : 
    - input: input image given to the network 
    - prediction: output of the network (prediction of the GT)
    - file_name: name of the file in prediction folder
OUTPUT : 
    - png files in prediction folder
----------------------------------------------------------------------------------------- """ 

def print_evaluationMCDRopout(input, pred, var, file_path):
    if not os.path.exists( str(Path(file_path).parent )):
        os.makedirs( str(Path(file_path).parent))

    #f  = plt.figure(figsize=(8,2.5)

    if pred.shape[-1] == 1: 
        pred  = np.squeeze(pred, axis=-1)
        input = np.squeeze(input, axis=-1)
        var   = np.squeeze(var, axis=-1)

    plt.rcParams['savefig.pad_inches'] = 0
    f= plt.figure()

    fig = f.add_subplot(131)
    fig.set_title('Input', fontsize=16)
    fig.imshow(input, cmap = 'gray', interpolation = 'bicubic')
    fig.axis('off')

    fig = f.add_subplot(132)
    fig.set_title('Mean Pred', fontsize=16)
    fig.imshow(pred, cmap = 'gray', interpolation = 'bicubic')
    fig.axis('off')

    if pred.shape[-1] == 3:
        input = rgb2gray(input)*255
        var = rgb2gray(var)*255
    mask = var
    mask = np.ma.masked_where(mask < 0.1, mask)
    fig = f.add_subplot(133)
    fig.set_title('Uncertainty', fontsize=16)
    fig.imshow(input, cmap = 'gray', interpolation = 'bicubic')
    fig.imshow(mask, cmap = 'jet', interpolation = 'bicubic')
    fig.axis('off')

    plt.savefig(file_path)
    plt.savefig(file_path, bbox_inches = 'tight', pad_inches = 0)
    plt.close()

""" -----------------------------------------------------------------------------------------
Print ROC curve 
INPUT : 
    - TP: true positive rates
    - FP: false positive rates
    - prediction: output of the network (prediction of the GT)
OUTPUT : 
    - png files in prediction folder
----------------------------------------------------------------------------------------- """ 

def print_ROC(json_path, file_path, AUC=False):
    plt.rc('xtick', labelsize=16) 
    plt.rc('ytick', labelsize=16) 
    plt.rcParams.update({'font.size': 16})

    from sklearn.metrics import auc
    if not os.path.exists( str(Path(file_path).parent )):
        os.makedirs( str(Path(file_path).parent))

    labels = ['L0 Norm', 'L1 Norm', 'L2 Norm', 'L$\infty$ Norm']

    dict = read_json(json_path)
    TP = dict['all_TP']
    FP = dict['all_FP']
    f= plt.figure()
    max_auc = 0
    for this_TP, this_FP, this_color, this_label in zip(TP, FP, colors4, labels):
        if len(TP) == 1: 
            this_color = colors4[-1]
            this_label = labels[-1]
        if AUC : 
            this_label += '('    
            this_auc    = round(auc(this_FP, this_TP),2)
            if this_auc > max_auc: 
                max_auc = this_auc
            this_label += str(this_auc)
            this_label += ')'
        plt.plot(this_FP, this_TP, color=this_color, linewidth=2, label=this_label)
    plt.xlabel('FP rate')
    plt.ylabel('TP rate')
    plt.legend(bbox_to_anchor=(1.04,1), title= "Max AUC = " + str(max_auc) + "\n \n Norm (AUC) ", loc="upper left")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()


def print_PIXEL_WISE_ROC(json_path, file_path, AUC=False):
    plt.rc('xtick', labelsize=16) 
    plt.rc('ytick', labelsize=16) 
    plt.rcParams.update({'font.size': 16})

    from sklearn.metrics import auc
    if not os.path.exists( str(Path(file_path).parent )):
        os.makedirs( str(Path(file_path).parent))

    dict = read_json(json_path)
    TP = dict['TP']
    FP = dict['FP']
    f= plt.figure()

    plt.plot(FP, TP, color=colors4[-1], linewidth=2, label='Pixel Difference')
    plt.xlabel('FP rate')
    plt.ylabel('TP rate')
    plt.legend(bbox_to_anchor=(1.04,1), title= "Max AUC = " + str(round(auc(FP, TP),2)), loc="upper left")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()


    """ -----------------------------------------------------------------------------------------
Print ROC curve 
INPUT 2D : 
    - TP: true positive rates
    - FP: false positive rates
    - prediction: output of the network (prediction of the GT)
OUTPUT : 
    - png files in prediction folder
----------------------------------------------------------------------------------------- """ 

def print_2DROC(json_path, file_path):
    def split(u, v, points):
        # return points on left side of UV
        return [p for p in points if np.cross(p - u, v - u) < 0]

    def extend(u, v, points):
        if not points:
            return []

        # find furthest point W, and split search to WV, UW
        w = min(points, key=lambda p: np.cross(p - u, v - u))
        p1, p2 = split(w, v, points), split(u, w, points)
        return extend(w, v, p1) + [w] + extend(u, w, p2)

    def convex_hull(points):
        # find two hull points, U, V, and split to left and right search
        u = min(points, key=lambda p: p[0])
        v = max(points, key=lambda p: p[0])
        left, right = split(u, v, points), split(v, u, points)

        #return [v] + extend(u, v, left) + [u] + extend(v, u, right) + [v]
        return [v] + extend(u, v, left) + [u] 


    plt.rc('xtick', labelsize=14) 
    plt.rc('ytick', labelsize=14) 
    plt.rcParams.update({'font.size': 14})

    from sklearn.metrics import auc
    if not os.path.exists( str(Path(file_path).parent )):
        os.makedirs( str(Path(file_path).parent))

    labels = ['L0 Norm', 'L1 Norm', 'L2 Norm', 'L$\infty$ Norm']
    subplot = {'L0 Norm':'221', 'L1 Norm':'222', 'L2 Norm':'223', 'L$\infty$ Norm':'224'}

    dict = read_json(json_path)
    TP = dict['all_TP']
    FP = dict['all_FP']
    f= plt.figure(figsize=(12,8))

    max_auc = 0
    for this_TP, this_FP, this_color, this_label in zip(TP, FP, colors4, labels):
        if len(TP) == 1: 
            this_color = colors4[-1]
            this_label = labels[-1]

        fig = f.add_subplot(subplot[this_label])

        if this_FP and this_TP:
            flat_fp = [item for sublist in this_FP for item in sublist]
            flat_tp = [item for sublist in this_TP for item in sublist]
            points  = [[x[0], x[1]] for x in zip(flat_fp, flat_tp)]
            hull = convex_hull(np.array(points))

            for fp, tp in zip(this_FP, this_TP):
                fig.plot(fp, tp, 'o', color=this_color, markersize=0.1, alpha=0.8)
                fig.set_xlabel('FP rate')
                fig.set_ylabel('TP rate')

            x, y = [], []
            for point in hull:
                x.append(point[0])
                y.append(point[1])
    
            #fig.plot(point[:,0], point[:,1], 'o', color='red', markersize=0.8)
            fig.plot(x, y, color='red', linewidth=1)
            this_auc    = round(auc(x, y),2)
            plt.legend(title= "AUC = " + str(this_auc), loc="lower right")

            
    
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()