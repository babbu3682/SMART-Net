from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np
import torch


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred)

def f1_metric(y_true, y_pred):
    return f1_score(y_true, y_pred)

def specificity(y_true, y_pred):
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    return true_negative / (true_negative + false_positive)

def binary_dice_score(y_true, y_pred, smooth=0.0, eps=1e-7):
    bs = y_true.size(0)
    y_true = y_true.view(bs, 1, -1)
    y_pred = y_pred.view(bs, 1, -1)

    intersection = torch.sum(y_true * y_pred, dim=(0, 2))
    cardinality  = torch.sum(y_true + y_pred, dim=(0, 2))

    dice_score = (2. * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score.mean()

def fp_rate_score(y_true, y_pred):
    false_positive = ((y_true == 0) & (y_pred == 1)).sum()
    true_negative  = ((y_true == 0) & (y_pred == 0)).sum()
    fp_count       = false_positive.float()
    max_fp_count   = (false_positive + true_negative).float()
    numerator      = torch.log1p(fp_count)           # log(1 + fp_count)
    denominator    = torch.log1p(max_fp_count)       # log(1 + max_fp_count)
    log_scaled_fp  = numerator / denominator
    return (1 - log_scaled_fp).mean()


def binary_dice_score_np(y_true, y_pred, smooth=0.0, eps=1e-7):
    bs = y_true.shape[0]
    y_true = y_true.reshape(bs, 1, -1)
    y_pred = y_pred.reshape(bs, 1, -1)

    intersection = np.sum(y_true * y_pred, axis=(0, 2))
    cardinality  = np.sum(y_true + y_pred, axis=(0, 2))

    dice_score = (2. * intersection + smooth) / np.clip(cardinality + smooth, eps, None)
    return dice_score.mean()

def fp_rate_score_np(y_true, y_pred):
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    true_negative  = np.sum((y_true == 0) & (y_pred == 0))
    
    fp_count = false_positive.astype(np.float32)
    max_fp_count = (false_positive + true_negative).astype(np.float32)
    
    numerator = np.log1p(fp_count)           # log(1 + fp_count)
    denominator = np.log1p(max_fp_count)     # log(1 + max_fp_count)
    
    log_scaled_fp = numerator / denominator
    return (1 - log_scaled_fp).mean()