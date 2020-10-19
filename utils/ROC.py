from utils.import_lib import *
from utils.helper import * 

from sklearn.metrics import mean_absolute_error, roc_curve
from operator import itemgetter
import copy

from scipy import ndimage


""" -----------------------------------------------------------------------------------------
 PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
----------------------------------------------------------------------------------------- """ 
def psnr(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                         " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                   str(y_pred.shape))

    return 20 * np.log10(255) -10. * np.log10(np.mean((y_pred - y_true)**2))

""" -----------------------------------------------------------------------------------------
Error metrics between two images 
INPUT: 
    - diff: difference between two images
    - norm: string representing norm of interrest
OUTPUT: 
    - L0, L1 (Mean Absolute Error), L2 (Root Mean Squarred Error) or Linf (Max Absolute Error)
----------------------------------------------------------------------------------------- """
def l_metric(diff, norm): 
    diff = diff.astype(np.float64)
    if diff.shape[-1] != 1: 
        diff = np.sum(diff, axis=-1)
        diff_square = np.sum(diff**2, axis=-1)
    else :
        diff_square = diff**2
    if norm == 'l0': 
        return np.count_nonzero(diff)
    elif norm == 'l1':
        W, H = diff.shape[0], diff.shape[1]
        return np.sum(diff) / (W*H)
    elif norm == 'l2': 
        W, H = diff.shape[0], diff.shape[1]
        return np.sqrt(np.sum(diff_square ) / (W*H))
    elif norm == 'linf': 
        return np.max(diff)
    else: 
        return 0

""" -----------------------------------------------------------------------------------------
Computes FP and TP of a default matrix with threshold sorted in ascending order
INPUT: 
    - x: input images from the network 
    - y: output images 
    - y_labels: binary labels for classification
OUTPUT: 
    - TP and FP arrays for the ROC curve 
----------------------------------------------------------------------------------------- """
def compute_ROC(x, y, y_GT): 
    # Compute all thresholds
    thres_matrix   = {'l0': [], 'l1': [], 'l2': [], 'linf': []} 
    for this_x, this_y in zip(x, y): 
        this_diff    = diff(this_x, this_y)
        for this_norm in ['l0', 'l1', 'l2', 'linf']:
            thres_matrix[this_norm].append(l_metric(this_diff, this_norm))

    # Compute ROC Curves for all norms
    all_TP, all_FP = [], []
    for this_norm in ['l0', 'l1', 'l2', 'linf']: 
        y_score = np.array(thres_matrix[this_norm])
        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score            = y_score[desc_score_indices]
        y_copy             = np.copy(y_GT)
        y_labels           = y_copy[desc_score_indices]

        # Keep transition thresholds
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs         = np.r_[distinct_value_indices, y_labels.size - 1]

        tps = np.cumsum(y_labels)[threshold_idxs]
        fps = 1 + threshold_idxs - tps

        tps = [0] + (tps/tps[-1]).tolist()
        fps = [0] + (fps/fps[-1]).tolist()

        all_TP.append(tps)
        all_FP.append(fps)

    return all_TP, all_FP


""" -----------------------------------------------------------------------------------------
Computes FP and TP of a default matrix with threshold sorted in ascending order
INPUT: 
    - x: input images from the network 
    - y: output images 
    - y_labels: binary labels for classification
OUTPUT: 
    - TP and FP arrays for the ROC curve 
----------------------------------------------------------------------------------------- """
def compute_ROC_pixel_wise(x_in, y_in, mask_in): 
    x, y, mask = copy.deepcopy(x_in), copy.deepcopy(y_in), copy.deepcopy(mask_in) 
    mask = mask > 0.9
    #x, y = np.squeeze(x, axis=-1), np.squeeze(y, axis=-1)

    thres_matrix   = []
    for this_x, this_y in zip(x, y): 
        this_diff = diff(this_x, this_y)
        if this_diff.shape[-1] != 1: 
            this_diff = np.sum(this_diff, axis=-1)
        else: 
            this_diff = np.squeeze(this_diff, axis=-1)
        thres_matrix.append( this_diff )
    thres_matrix = np.array(thres_matrix, dtype=np.int)

    thres_matrix = thres_matrix.ravel(order='C')
    y_GT         = np.array(mask.ravel(order='C'), dtype=np.int)

    # Compute ROC Curves
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(thres_matrix, kind="mergesort")[::-1]
    y_score            = thres_matrix[desc_score_indices]
    y_copy             = np.copy(y_GT)
    y_labels           = y_copy[desc_score_indices]

    # Keep transition thresholds
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs         = np.r_[distinct_value_indices, y_labels.size - 1]

    tps = np.cumsum(y_labels)[threshold_idxs]

    fps = 1 + threshold_idxs - tps

    tps = [0] + (tps/tps[-1]).tolist()
    fps = [0] + (fps/fps[-1]).tolist()
    
    #fps, tps, _ = roc_curve(y_GT, thres_matrix)

    return tps, fps


""" -----------------------------------------------------------------------------------------
Computes FP and TP of a combination of two classifiers
INPUT: 
    - x1: input images from the classifier 1
    - y1 : output images from the classifier 1
    - x2: input images from the classifier 2
    - y2 : output images from the classifier 2
    - y_labels: binary labels for classification
OUTPUT: 
    - TP and FP arrays for the ROC curve 
----------------------------------------------------------------------------------------- """
def compute_combinedROC(x1, y1, x2, y2, binary_test_labels):
    from sklearn.metrics import auc
    # Compute all thresholds
    thres_matrix1   = {'l0': [], 'l1': [], 'l2': [], 'linf': []}
    thres_matrix2   = {'l0': [], 'l1': [], 'l2': [], 'linf': []}
    for this_x1, this_y1, this_x2, this_y2 in zip(x1, y1, x2, y2): 
        this_diff1    = diff(this_x1, this_y1)
        this_diff2    = diff(this_x2, this_y2)
        for this_norm in ['l0', 'l1', 'l2', 'linf']:
            thres_matrix1[this_norm].append(l_metric(this_diff1, this_norm))
            thres_matrix2[this_norm].append(l_metric(this_diff2, this_norm))

    # Compute ROC Curves for all norms
    all_TP, all_FP = [], []
    for this_norm in ['l0', 'l1', 'l2', 'linf']: 
    #for this_norm in ['l0']: 
        y_score1 = np.array(thres_matrix1[this_norm])
        y_score2 = np.array(thres_matrix2[this_norm])
        # sort scores and corresponding truth values
        desc_score_indices1 = np.argsort(y_score1, kind="mergesort")[::-1]
        y_score1            = y_score1[desc_score_indices1]
        y_copy1             = np.copy(binary_test_labels)
        y_label1            = y_copy1[desc_score_indices1]

        desc_score_indices2 = np.argsort(y_score2, kind="mergesort")[::-1]
        y_score2            = y_score2[desc_score_indices2]
        y_copy2             = np.copy(binary_test_labels)
        y_label2            = y_copy2[desc_score_indices2]

        # Keep transition thresholds
        distinct_value_indices1 = np.where(np.diff(y_score1))[0]
        threshold_idxs1         = np.r_[distinct_value_indices1, y_label1.size - 1]

        distinct_value_indices2 = np.where(np.diff(y_score2))[0]
        threshold_idxs2         = np.r_[distinct_value_indices2, y_label2.size - 1]

        combined_TP, combined_FP = [], []
        eligible1, idx1          = [], 0
        max_auc                  = 0
        for pos1, tresh1 in enumerate(threshold_idxs1): 
            # Add new selected item by thres1
            while idx1 < len(y_score1) and (y_score1[idx1] > tresh1):
                eligible1.append(desc_score_indices1[idx1])
                idx1 += 1

            eligible2, idx2 = [], 0 
            combined_labels = []
            for pos2, tresh2 in enumerate(threshold_idxs2):
                while idx2 < len(y_score2) and (y_score2[idx2] > tresh2):
                    if desc_score_indices2[idx2] in eligible1: 
                        combined_labels.append(y_label2[idx2])
                    idx2 += 1
            if len(combined_labels) != 0: 
                distinct_value_indices = np.where(np.diff(y_score2))[0]
                threshold_idxs         = np.r_[distinct_value_indices, y_label2.size - 1]
                tps = np.cumsum(combined_labels)[threshold_idxs]
                fps = 1 + threshold_idxs - tps
                tps = [0] + (tps/tps[-1]).tolist()
                fps = [0] + (fps/fps[-1]).tolist()
            else: 
                tps = [0,1]
                fps = [0,1]

            this_auc = round(auc(fps, tps),2)
            print(this_norm, this_auc, fps)
            if this_auc > max_auc:
                max_auc = this_auc
                temp_fps = fps
                temp_tps = tps
        all_TP.append(temp_tps)
        all_FP.append(temp_fps)

    return all_TP, all_FP

""" -----------------------------------------------------------------------------------------
Computes FP and TP of a combination of two classifiers
INPUT: 
    - x1: input images from the classifier 1
    - y1 : output images from the classifier 1
    - x2: input images from the classifier 2
    - y2 : output images from the classifier 2
    - y_labels: binary labels for classification
OUTPUT: 
    - TP and FP arrays for the ROC curve 
----------------------------------------------------------------------------------------- """
def compute_2DcombinedROC(x1, y1, x2, y2, binary_test_labels):
    from sklearn.metrics import auc
    # Compute all thresholds
    thres_matrix1   = {'l0': [], 'l1': [], 'l2': [], 'linf': []}
    thres_matrix2   = {'l0': [], 'l1': [], 'l2': [], 'linf': []}
    for this_x1, this_y1, this_x2, this_y2 in zip(x1, y1, x2, y2): 
        this_diff1    = diff(this_x1, this_y1)
        this_diff2    = diff(this_x2, this_y2)
        for this_norm in ['l0', 'l1', 'l2', 'linf']:
            thres_matrix1[this_norm].append(l_metric(this_diff1, this_norm))
            thres_matrix2[this_norm].append(l_metric(this_diff2, this_norm))

    # Compute ROC Curves for all norms
    all_TP, all_FP = [], []
    for this_norm in ['l0', 'l1', 'l2', 'linf']: 
    #for this_norm in ['linf']: 
        y_score1 = np.array(thres_matrix1[this_norm])
        y_score2 = np.array(thres_matrix2[this_norm])
        # sort scores and corresponding truth values
        asc_score_indices1  = np.argsort(y_score1, kind="mergesort")
        y_score1            = y_score1[asc_score_indices1]
        y_copy1             = np.copy(binary_test_labels)
        y_label1            = y_copy1[asc_score_indices1]

        asc_score_indices2  = np.argsort(y_score2, kind="mergesort")
        y_score2            = y_score2[asc_score_indices2]
        y_copy2             = np.copy(binary_test_labels)
        y_label2            = y_copy2[asc_score_indices2]

        # Keep transition thresholds
        distinct_value_indices1 = np.where(np.diff(y_score1))[0]
        threshold_idxs1         = np.r_[distinct_value_indices1, y_label1.size - 1]

        distinct_value_indices2 = np.where(np.diff(y_score2))[0]
        threshold_idxs2         = np.r_[distinct_value_indices2, y_label2.size - 1]

        combined_TP, combined_FP = [], [] 
        eligible1, idx1          = [], 0
        max_auc                  = 0
        temp_tps, temp_fps       = [], []

        for pos1, tresh1 in enumerate(y_score1[threshold_idxs1]): 
            # Add new selected item by thres1
            while idx1 < len(y_score1) and (y_score1[idx1] < tresh1):
                eligible1.append(asc_score_indices1[idx1])
                idx1 += 1

            eligible2, idx2  = [], 0 
            combined_labels  = []
            tps, fps         = [0], [0]
            for pos2, tresh2 in enumerate(y_score2[threshold_idxs2]):
                while idx2 < len(y_score2) and (y_score2[idx2] < tresh2):
                    if asc_score_indices2[idx2] in eligible1: 
                        combined_labels.append(y_label2[idx2])
                    idx2 += 1

                fp = np.sum(combined_labels)
                tp = np.sum(y_label2) - fp 
                tn = len(combined_labels) - fp 
                fn = len(y_label2) - np.sum(y_label2) - tn

                tps.append(tp/(tp+fn) if tp+fn != 0 else 0)
                fps.append(fp/(tn+fp) if tn+fp != 0 else 0)
            tps.append(1)
            fps.append(1)

            if len(temp_fps) == 0:
                temp_tps = [tps]
                temp_fps = [fps]
            else:
                temp_tps += [tps]
                temp_fps += [fps]

  
        all_TP.append(temp_tps)
        all_FP.append(temp_fps)

    return all_TP, all_FP